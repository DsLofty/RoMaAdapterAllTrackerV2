"""RoMa final-stage adapter V2 wrapper for the VOT folderpython protocol.

This runtime keeps AllTracker and RoMa frozen. It tracks each VOT query group
with AllTracker, builds sparse RoMa candidates at final-stage target frames, and
uses the learned risk/accept adapter to decide when to replace AllTracker final
coordinates with RoMa coordinates.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

from vot.region import Point  # noqa: E402
from vot.region.io import parse_region  # noqa: E402

from alltracker_runtime_utils import expand_path, extract_feat8_sequence, load_alltracker  # noqa: E402
from final_stage_cache_utils import load_selector  # noqa: E402
from export_final_stage_adapter_data import (  # noqa: E402
    build_features,
    deep_offsets,
    sample_deep_corr_grid,
    sample_feat_points,
)
from final_stage_adapter_model import FINAL_STAGE_FEATURE_NAMES, feature_indices  # noqa: E402
from matchers.roma_wrapper import RoMaMatcher  # noqa: E402
from vot_folder_io import (  # noqa: E402
    MAX_SIZE,
    collect_vot_roma_inputs,
    frames_to_video_tensor,
    load_frames,
    resize_frames,
)
from v2_runtime_config import (  # noqa: E402
    FINAL_STAGE_ACCEPT_THRESHOLD,
    FINAL_STAGE_DEEP_CORR_RADIUS8,
    FINAL_STAGE_DEFAULT_CHECKPOINT,
    FINAL_STAGE_RISK_THRESHOLD,
    FINAL_STAGE_TARGET_FRAME_STRIDE,
    INFERENCE_ITERS,
)


DEFAULT_ALLTRACKER_CKPT = ROOT / "ckpt" / "alltracker.pth"
DEFAULT_SELECTOR_CKPT = ROOT / FINAL_STAGE_DEFAULT_CHECKPOINT
FALLBACK_ALLTRACKER_CKPT = Path("/home/zanghan/Pyproject/vot/alltracker/ckpt/alltracker.pth")
FALLBACK_SELECTOR_CKPT = Path("/home/zanghan/Pyproject/vot/alltracker") / FINAL_STAGE_DEFAULT_CHECKPOINT


def _resolve_path(env_name: str, default_path: Path, fallback_path: Path) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        return expand_path(value)
    if default_path.exists():
        return str(default_path)
    return str(fallback_path)


def _device() -> torch.device:
    requested = os.environ.get("VOT_ROMA_ADAPTER_DEVICE", "cuda").strip()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def make_runtime_args() -> SimpleNamespace:
    return SimpleNamespace(
        ckpt_path=_resolve_path("ALLTRACKER_CKPT", DEFAULT_ALLTRACKER_CKPT, FALLBACK_ALLTRACKER_CKPT),
        final_stage_ckpt=_resolve_path(
            "FINAL_STAGE_ADAPTER_CKPT",
            DEFAULT_SELECTOR_CKPT,
            FALLBACK_SELECTOR_CKPT,
        ),
        inference_iters=INFERENCE_ITERS,
        roma_model=os.environ.get("ROMA_MODEL", "outdoor"),
        roma_device=os.environ.get("ROMA_DEVICE", "cuda"),
        roma_input_size=None,
        roma_allow_online_download=bool(int(os.environ.get("ROMA_ALLOW_ONLINE_DOWNLOAD", "0"))),
        roma_cache_warps=False,
        roma_disable_custom_corr=True,
        roma_allow_slow_corr=True,
        roma_pair_batch_size=int(os.environ.get("ROMA_PAIR_BATCH_SIZE", "1")),
        roma_sample_mode="bilinear",
        roma_fail_fast=False,
        target_frame_stride=int(os.environ.get("FINAL_STAGE_TARGET_FRAME_STRIDE", str(FINAL_STAGE_TARGET_FRAME_STRIDE))),
        target_frame_include=os.environ.get("FINAL_STAGE_TARGET_FRAME_INCLUDE", "stride,last"),
        target_points_per_frame=int(os.environ.get("FINAL_STAGE_TARGET_POINTS_PER_FRAME", "4096")),
        online_max_rows_per_sequence=int(os.environ.get("FINAL_STAGE_ONLINE_MAX_ROWS", "0")),
        risk_threshold=float(os.environ.get("FINAL_STAGE_RISK_THRESHOLD", str(FINAL_STAGE_RISK_THRESHOLD))),
        accept_threshold=float(os.environ.get("FINAL_STAGE_ACCEPT_THRESHOLD", str(FINAL_STAGE_ACCEPT_THRESHOLD))),
        deep_corr_radius8=int(os.environ.get("FINAL_STAGE_DEEP_CORR_RADIUS8", str(FINAL_STAGE_DEEP_CORR_RADIUS8))),
        chunk_points=int(os.environ.get("FINAL_STAGE_CHUNK_POINTS", "512")),
        heuristic_roma_certainty_thr=0.75,
    )


def flow_to_tracks(flow: torch.Tensor, visconf: torch.Tensor, query_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, num_frames, _, height, width = flow.shape
    query_int = query_points.round().long()
    query_int[:, 0].clamp_(0, width - 1)
    query_int[:, 1].clamp_(0, height - 1)
    point_count = int(query_int.shape[0])
    xy = torch.zeros((num_frames, point_count, 2), dtype=torch.float32, device=flow.device)
    visible = torch.zeros((num_frames, point_count), dtype=torch.float32, device=flow.device)
    for point_i in range(point_count):
        x = int(query_int[point_i, 0].item())
        y = int(query_int[point_i, 1].item())
        base = query_int[point_i].float().to(flow.device)
        xy[:, point_i] = flow[0, :, :, y, x] + base.reshape(1, 2)
        if visconf.ndim == 5:
            visible[:, point_i] = visconf[0, :, 0, y, x].float()
        elif visconf.ndim == 4:
            visible[:, point_i] = visconf[0, :, y, x].float()
        else:
            visible[:, point_i] = 1.0
    return xy, visible


def build_roma_maps(inputs: dict[str, torch.Tensor] | None, T: int, N: int, device: torch.device):
    roma_xy = torch.full((T, N, 2), float("nan"), dtype=torch.float32, device=device)
    roma_cert = torch.zeros((T, N), dtype=torch.float32, device=device)
    roma_valid = torch.zeros((T, N), dtype=torch.bool, device=device)
    if inputs is None:
        return roma_xy, roma_cert, roma_valid
    frame = inputs["target_frame"].long().to(device)
    point = inputs["point_index"].long().to(device)
    valid = inputs["roma_valid"].reshape(-1).float().to(device) > 0.5
    xy = inputs["roma_xy8"].float().to(device) * 8.0
    cert = inputs["roma_certainty"].reshape(-1).float().to(device)
    finite = torch.isfinite(xy).all(dim=1)
    keep = valid & finite & (frame >= 0) & (frame < T) & (point >= 0) & (point < N)
    if bool(keep.any().item()):
        roma_xy[frame[keep], point[keep]] = xy[keep]
        roma_cert[frame[keep], point[keep]] = cert[keep]
        roma_valid[frame[keep], point[keep]] = True
    return roma_xy, roma_cert, roma_valid


def compute_deep_corr_for_runtime(video_device, model, base_xy, query_xy, roma_xy, roma_valid, radius8, device):
    T, N, _ = base_xy.shape
    offsets8 = deep_offsets(int(radius8), device)
    K = int(offsets8.shape[0])
    feat8 = extract_feat8_sequence(model=model, rgbs=video_device, device=device)
    feat8 = F.normalize(feat8.float(), dim=1)
    query_xy8 = query_xy.float().to(device) / 8.0
    query_feat = sample_feat_points(feat8[0], query_xy8)
    query_feat = F.normalize(query_feat.float(), dim=-1)
    roma_corr = torch.zeros((T, N, K), dtype=torch.float32, device=device)
    baseline_corr = torch.zeros((T, N, K), dtype=torch.float32, device=device)
    for t in range(T):
        roma_corr[t] = sample_deep_corr_grid(feat8[t], query_feat, roma_xy[t] / 8.0, offsets8)
        baseline_corr[t] = sample_deep_corr_grid(feat8[t], query_feat, base_xy[t] / 8.0, offsets8)
        roma_corr[t] = roma_corr[t].masked_fill(~roma_valid[t].reshape(N, 1), 0.0)
    return {
        "roma_corr_grid": roma_corr.detach().cpu(),
        "baseline_corr_grid": baseline_corr.detach().cpu(),
        "grid_offsets8": offsets8.detach().cpu(),
        "radius8": int(radius8),
        "stride": 8,
    }


@torch.no_grad()
def predict_risk_accept(selector, normalizer, selected_feature_names, sample, device, chunk_points):
    source_names = list(sample["feature_names"])
    indices = feature_indices(source_names, selected_feature_names)
    features = sample["features"].float()[..., indices]
    x_all = features.permute(1, 0, 2).contiguous()
    deep = sample["deep_corr"]
    risk_parts = []
    accept_parts = []
    for start in range(0, int(x_all.shape[0]), int(chunk_points)):
        end = min(start + int(chunk_points), int(x_all.shape[0]))
        x = normalizer(x_all[start:end].to(device)).float()
        roma = deep["roma_corr_grid"].float().permute(1, 0, 2).contiguous()[start:end]
        baseline = deep["baseline_corr_grid"].float().permute(1, 0, 2).contiguous()[start:end]
        aux = torch.cat([roma, baseline], dim=-1).to(device).float()
        outputs = selector(x, aux)
        risk_parts.append(torch.sigmoid(outputs["risk_logits"]).cpu())
        accept_parts.append(torch.sigmoid(outputs["accept_logits"]).cpu())
    risk = torch.cat(risk_parts, dim=0).permute(1, 0).contiguous().to(device)
    accept = torch.cat(accept_parts, dim=0).permute(1, 0).contiguous().to(device)
    return risk, accept


def run_group_v2(
    model,
    selector,
    normalizer,
    selected_feature_names,
    roma_matcher,
    args: SimpleNamespace,
    resized_frames: list[np.ndarray],
    offset: int,
    query_list: list[tuple[str, int, int]],
    sx: float,
    sy: float,
    device: torch.device,
) -> dict[str, list[tuple[float, float]]]:
    active_frames = resized_frames[int(offset) :]
    if not active_frames:
        return {oid: [] for oid, _, _ in query_list}
    query_points = torch.as_tensor([[qx, qy] for _, qx, qy in query_list], dtype=torch.float32)
    video = frames_to_video_tensor(active_frames)
    video_device = video.to(device=device, non_blocking=True)

    if video_device.shape[1] > 128:
        flow, visconf, _, _ = model.forward_sliding(
            video_device,
            iters=int(args.inference_iters),
            sw=None,
            is_training=False,
        )
    else:
        flow, visconf, _, _ = model(
            video_device,
            iters=int(args.inference_iters),
            sw=None,
            is_training=False,
        )

    base_xy, visible = flow_to_tracks(flow, visconf, query_points.to(device))
    T, N, _ = base_xy.shape

    inputs = collect_vot_roma_inputs(
        seq_id="v2_offset_%06d" % int(offset),
        frames=active_frames,
        source_points=query_points,
        roma_matcher=roma_matcher,
        args=args,
        device=device,
    )
    roma_xy, roma_cert, roma_valid = build_roma_maps(inputs, T, N, device)
    candidate = roma_valid & torch.isfinite(roma_xy).all(dim=-1)
    coarse_gate = torch.zeros((T, N), dtype=torch.bool, device=device)
    reference = {
        "trajs_e": base_xy,
        "pred_visible_score": visible,
        "first_positive_inds": torch.zeros((N,), dtype=torch.long, device=device),
    }
    patch_stats = {
        "query_patch_texture": torch.zeros((T, N), dtype=torch.float32, device=device),
        "baseline_patch_texture": torch.zeros((T, N), dtype=torch.float32, device=device),
        "roma_patch_texture": torch.zeros((T, N), dtype=torch.float32, device=device),
        "baseline_patch_ncc": torch.zeros((T, N), dtype=torch.float32, device=device),
        "roma_patch_ncc": torch.zeros((T, N), dtype=torch.float32, device=device),
        "patch_ncc_gap": torch.zeros((T, N), dtype=torch.float32, device=device),
        "roma_local_ncc_peak": torch.zeros((T, N), dtype=torch.float32, device=device),
        "roma_local_ncc_margin": torch.zeros((T, N), dtype=torch.float32, device=device),
        "roma_local_peak_offset_px": torch.zeros((T, N), dtype=torch.float32, device=device),
    }
    features = build_features(reference, roma_xy, roma_cert, roma_valid, coarse_gate, patch_stats, args, device)
    deep_corr = compute_deep_corr_for_runtime(
        video_device=video_device,
        model=model,
        base_xy=base_xy,
        query_xy=query_points.to(device),
        roma_xy=roma_xy,
        roma_valid=roma_valid,
        radius8=int(args.deep_corr_radius8),
        device=device,
    )
    sample = {
        "feature_names": list(FINAL_STAGE_FEATURE_NAMES),
        "features": features.detach().cpu(),
        "deep_corr": deep_corr,
    }
    risk_prob, accept_prob = predict_risk_accept(
        selector,
        normalizer,
        selected_feature_names,
        sample,
        device,
        int(args.chunk_points),
    )
    risk_prob = risk_prob.to(device=base_xy.device)
    accept_prob = accept_prob.to(device=base_xy.device)
    roma_xy = roma_xy.to(device=base_xy.device)
    roma_valid = roma_valid.to(device=base_xy.device)
    candidate = candidate.to(device=base_xy.device)
    gate = (
        candidate
        & (risk_prob >= float(args.risk_threshold))
        & (accept_prob >= float(args.accept_threshold))
    ).to(device=base_xy.device)
    final_xy = base_xy.clone()
    if bool(gate.any().item()):
        final_xy[gate] = roma_xy[gate]
    final_xy = final_xy.detach().cpu()
    out = {}
    for point_i, (oid, _, _) in enumerate(query_list):
        track = []
        for t in range(T):
            track.append((float(final_xy[t, point_i, 0].item()) / sx, float(final_xy[t, point_i, 1].item()) / sy))
        out[oid] = track
    print(
        "V2 final-stage gate %.5f risk_thr %.3f accept_thr %.3f"
        % (float(gate.float().mean().item()), float(args.risk_threshold), float(args.accept_threshold)),
        flush=True,
    )
    return out


def load_runtime(args: SimpleNamespace, device: torch.device):
    selector_path = Path(expand_path(args.final_stage_ckpt))
    if not selector_path.exists():
        raise FileNotFoundError(
            "Final-stage adapter checkpoint not found: %s. "
            "Set FINAL_STAGE_ADAPTER_CKPT or place the checkpoint in checkpoints_final_stage_adapter/."
            % selector_path
        )
    print("Loading AllTracker: %s" % args.ckpt_path, flush=True)
    model = load_alltracker(args, device)
    print("Loading final-stage adapter: %s" % selector_path, flush=True)
    selector, normalizer, ckpt, feature_names = load_selector(str(selector_path), device)
    print("Loading RoMa model=%s device=%s" % (args.roma_model, args.roma_device), flush=True)
    roma_matcher = RoMaMatcher(
        model_type=args.roma_model,
        device=args.roma_device,
        input_size=args.roma_input_size,
        allow_online_download=args.roma_allow_online_download,
        cache_warps=args.roma_cache_warps,
        use_custom_corr=not bool(args.roma_disable_custom_corr),
        allow_slow_corr=bool(args.roma_allow_slow_corr),
    )
    return model, selector, normalizer, feature_names, roma_matcher


def main() -> None:
    frame_files = sorted(f for f in os.listdir(".") if f.startswith("frames_") and f.endswith(".txt"))
    if not frame_files:
        raise RuntimeError("No frames_*.txt found in current directory")
    with open(frame_files[0], "r", encoding="utf-8") as fp:
        frame_paths = [line.strip() for line in fp if line.strip()]
    total_frames = len(frame_paths)

    query_files = sorted(f for f in os.listdir(".") if f.startswith("query_") and f.endswith(".txt"))
    if not query_files:
        raise RuntimeError("No query_*.txt found in current directory")

    queries = []
    for query_file in query_files:
        oid = query_file[len("query_") : -len(".txt")]
        with open(query_file, "r", encoding="utf-8") as fp:
            lines = [line.strip() for line in fp if line.strip()]
        offset = int(lines[0])
        state = parse_region(lines[1])
        queries.append((oid, offset, state))

    frames = load_frames(frame_paths)
    resized, height, width, sx, sy = resize_frames(frames, MAX_SIZE)
    print("Frames: %d, model size: %dx%d, scale %.4f %.4f" % (total_frames, height, width, sx, sy), flush=True)

    args = make_runtime_args()
    device = _device()
    model, selector, normalizer, feature_names, roma_matcher = load_runtime(args, device)

    by_offset = defaultdict(list)
    for oid, offset, state in queries:
        by_offset[int(offset)].append((oid, state))

    results = {}
    for offset, group in sorted(by_offset.items()):
        query_list = []
        for oid, state in group:
            if isinstance(state, Point):
                qx = float(state.x) * sx
                qy = float(state.y) * sy
            else:
                qx = width / 2.0
                qy = height / 2.0
            qx_i = min(max(int(round(qx)), 0), width - 1)
            qy_i = min(max(int(round(qy)), 0), height - 1)
            query_list.append((oid, qx_i, qy_i))
        print("Tracking %d queries from offset=%d with final-stage V2." % (len(query_list), int(offset)), flush=True)
        group_tracks = run_group_v2(
            model,
            selector,
            normalizer,
            feature_names,
            roma_matcher,
            args,
            resized,
            int(offset),
            query_list,
            sx=sx,
            sy=sy,
            device=device,
        )
        for oid, _state in group:
            results[oid] = [None] * int(offset) + group_tracks.get(oid, [])

    for oid, positions in results.items():
        with open("output_%s.txt" % oid, "w", encoding="utf-8") as fp:
            for pos in positions[:total_frames]:
                if pos is None:
                    fp.write("0\n")
                else:
                    fp.write("%.6f,%.6f\n" % (float(pos[0]), float(pos[1])))
    print("Done. V2 tracked %d objects over %d frames." % (len(results), total_frames), flush=True)


if __name__ == "__main__":
    main()
