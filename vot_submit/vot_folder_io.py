"""Lightweight VOT folder-protocol IO helpers for RoMaAdapterAllTrackerV2."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch


MAX_SIZE = int(os.environ.get("VOT_ROMA_ADAPTER_MAX_SIZE", "1024"))
MAX_TOKENS = int(os.environ.get("VOT_ROMA_ADAPTER_MAX_TOKENS", "9216"))
FIXED_IMAGE_SIZE = os.environ.get("VOT_ROMA_ADAPTER_IMAGE_SIZE", "").strip()

PROPOSAL_SOURCE_WINDOW = 0
PROPOSAL_SOURCE_ROMA_CERTAINTY = 3
EVENT_TYPE_WINDOW_START = 0


def load_frames(paths: list[str]) -> list[np.ndarray]:
    imgs = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError("Could not read frame: %s" % path)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def parse_fixed_image_size(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    parts = value.lower().replace("x", ",").split(",")
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) != 2:
        raise ValueError("VOT_ROMA_ADAPTER_IMAGE_SIZE must be formatted as H,W, for example 448,768")
    height = int(parts[0])
    width = int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError("VOT_ROMA_ADAPTER_IMAGE_SIZE must contain positive H,W values")
    height = max(8, (height // 8) * 8)
    width = max(8, (width // 8) * 8)
    return height, width


def resize_frames(imgs: list[np.ndarray], max_size: int) -> tuple[list[np.ndarray], int, int, float, float]:
    h0, w0 = imgs[0].shape[:2]
    fixed_size = parse_fixed_image_size(FIXED_IMAGE_SIZE)
    if fixed_size is not None:
        h, w = fixed_size
        resized = [cv2.resize(img, (w, h)) for img in imgs]
        return resized, h, w, w / float(w0), h / float(h0)

    scale = min(max_size / float(h0), max_size / float(w0), 1.0)
    token_scale = (float(MAX_TOKENS) * 64.0 / max(float(h0 * w0), 1.0)) ** 0.5
    scale = min(scale, token_scale)
    h = max(8, (int(h0 * scale) // 8) * 8)
    w = max(8, (int(w0 * scale) // 8) * 8)
    resized = [cv2.resize(img, (w, h)) for img in imgs]
    return resized, h, w, w / float(w0), h / float(h0)


def frames_to_video_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack(frames, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0)


def target_frames_for_vot(num_frames: int, args: SimpleNamespace) -> list[int]:
    if num_frames <= 1:
        return []
    stride = max(1, int(args.target_frame_stride))
    include = str(args.target_frame_include)
    values = list(range(1, int(num_frames), stride))
    if "last" in include.split(",") and (num_frames - 1) not in values:
        values.append(num_frames - 1)
    return sorted(set(int(v) for v in values if 0 < int(v) < int(num_frames)))


def _proposal_source_bit(source_id: int) -> int:
    return int(1 << int(source_id))


def _invalid_mapping(count: int) -> dict[str, torch.Tensor]:
    return {
        "points1": torch.full((count, 2), float("nan"), dtype=torch.float32),
        "certainty": torch.zeros((count,), dtype=torch.float32),
        "valid": torch.zeros((count,), dtype=torch.bool),
    }


def _to_cpu_mapping(mapping: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
        for key, value in mapping.items()
    }


def collect_vot_roma_inputs(
    seq_id: str,
    frames: list[np.ndarray],
    source_points: torch.Tensor,
    roma_matcher: Any,
    args: SimpleNamespace,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """Build final-stage RoMa candidate rows from VOT frames and query points only."""

    del seq_id
    num_frames = len(frames)
    num_points = int(source_points.shape[0])
    if num_frames <= 1 or num_points <= 0:
        return None

    max_points = int(args.target_points_per_frame)
    target_frames = target_frames_for_vot(num_frames, args)
    if not target_frames:
        return None

    rows = {
        "roma_xy8": [],
        "source_xy": [],
        "target_frame": [],
        "point_index": [],
        "event_frame": [],
        "proposal_source": [],
        "event_type": [],
        "event_score": [],
        "proposal_source_mask": [],
        "baseline_patch_ncc": [],
        "roma_patch_ncc": [],
        "patch_ncc_gap": [],
        "patch_mismatch_score": [],
        "query_patch_texture": [],
        "roma_certainty_event": [],
        "roma_valid": [],
        "roma_certainty": [],
        "need_only": [],
        "baseline_risk_prob": [],
        "baseline_corr_margin": [],
        "baseline_update_norm": [],
        "query_anchor_age_norm": [],
        "source_is_query_anchor": [],
        "normalized_frame_index": [],
        "normalized_window_start_index": [],
    }

    all_point_ids = torch.arange(num_points, dtype=torch.long)
    source_image = frames[0]
    jobs = []
    for target_t in target_frames:
        point_ids = all_point_ids
        if max_points > 0 and int(point_ids.numel()) > max_points:
            select = torch.linspace(0, int(point_ids.numel()) - 1, steps=max_points).round().long()
            point_ids = point_ids.index_select(0, select)
        jobs.append(
            {
                "target_t": int(target_t),
                "point_ids": point_ids,
                "source_points": source_points.index_select(0, point_ids).float(),
                "target_image": frames[int(target_t)],
            }
        )

    def append_job(job, mapping):
        target_t = int(job["target_t"])
        point_ids = job["point_ids"].long()
        points0 = job["source_points"].float()
        count = int(point_ids.numel())
        if count <= 0:
            return

        roma_xy = mapping["points1"].float()
        roma_certainty = torch.nan_to_num(mapping["certainty"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        roma_valid = mapping["valid"].bool() & torch.isfinite(roma_xy).all(dim=1)
        certainty_event = (roma_valid & (roma_certainty >= float(args.heuristic_roma_certainty_thr))).float()
        source_mask = torch.full((count,), _proposal_source_bit(PROPOSAL_SOURCE_WINDOW), dtype=torch.long)
        source_mask = torch.where(
            certainty_event > 0.5,
            source_mask | int(_proposal_source_bit(PROPOSAL_SOURCE_ROMA_CERTAINTY)),
            source_mask,
        )
        norm_frame = float(target_t / max(num_frames - 1, 1))

        rows["roma_xy8"].append(roma_xy / 8.0)
        rows["source_xy"].append(points0)
        rows["target_frame"].append(torch.full((count,), target_t, dtype=torch.long))
        rows["point_index"].append(point_ids)
        rows["event_frame"].append(torch.full((count,), target_t, dtype=torch.long))
        rows["proposal_source"].append(torch.full((count,), PROPOSAL_SOURCE_WINDOW, dtype=torch.long))
        rows["event_type"].append(torch.full((count,), EVENT_TYPE_WINDOW_START, dtype=torch.long))
        rows["event_score"].append(certainty_event.float())
        rows["proposal_source_mask"].append(source_mask.long())
        rows["baseline_patch_ncc"].append(torch.full((count,), float("nan"), dtype=torch.float32))
        rows["roma_patch_ncc"].append(torch.full((count,), float("nan"), dtype=torch.float32))
        rows["patch_ncc_gap"].append(torch.full((count,), float("nan"), dtype=torch.float32))
        rows["patch_mismatch_score"].append(torch.zeros((count,), dtype=torch.float32))
        rows["query_patch_texture"].append(torch.full((count,), float("nan"), dtype=torch.float32))
        rows["roma_certainty_event"].append(certainty_event.float())
        rows["roma_valid"].append(roma_valid.float())
        rows["roma_certainty"].append(roma_certainty.float())
        rows["need_only"].append(torch.zeros((count,), dtype=torch.float32))
        rows["baseline_risk_prob"].append(torch.zeros((count,), dtype=torch.float32))
        rows["baseline_corr_margin"].append(torch.zeros((count,), dtype=torch.float32))
        rows["baseline_update_norm"].append(torch.zeros((count,), dtype=torch.float32))
        rows["query_anchor_age_norm"].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows["source_is_query_anchor"].append(torch.ones((count,), dtype=torch.float32))
        rows["normalized_frame_index"].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows["normalized_window_start_index"].append(torch.full((count,), norm_frame, dtype=torch.float32))

    pair_batch_size = max(1, int(args.roma_pair_batch_size))
    for start in range(0, len(jobs), pair_batch_size):
        chunk = jobs[start : start + pair_batch_size]
        mappings = None
        if pair_batch_size > 1 and len(chunk) > 1 and hasattr(roma_matcher, "map_points_batch"):
            try:
                mappings = roma_matcher.map_points_batch(
                    source_image,
                    [job["target_image"] for job in chunk],
                    [job["source_points"] for job in chunk],
                    sample_mode=args.roma_sample_mode,
                    cache_keys=None,
                )
                mappings = [_to_cpu_mapping(mapping) for mapping in mappings]
            except Exception:
                mappings = None
        if mappings is None:
            mappings = []
            for job in chunk:
                try:
                    mapping = roma_matcher.map_points(
                        source_image,
                        job["target_image"],
                        job["source_points"],
                        sample_mode=args.roma_sample_mode,
                        cache_key=None,
                    )
                    mappings.append(_to_cpu_mapping(mapping))
                except Exception:
                    if bool(args.roma_fail_fast):
                        raise
                    mappings.append(_invalid_mapping(int(job["point_ids"].numel())))
        for job, mapping in zip(chunk, mappings):
            append_job(job, mapping)

    if not rows["roma_xy8"]:
        return None
    out = {}
    for key, values in rows.items():
        out[key] = torch.cat(values, dim=0).to(device=device, non_blocking=True)
    out["image_hw"] = (int(frames[0].shape[0]), int(frames[0].shape[1]))
    out["num_frames"] = int(num_frames)

    max_rows = int(args.online_max_rows_per_sequence)
    if max_rows > 0:
        row_count = int(out["target_frame"].numel())
        if row_count > max_rows:
            keep = torch.linspace(0, row_count - 1, steps=max_rows, device=device).round().long()
            for key, value in list(out.items()):
                if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == row_count:
                    out[key] = value.index_select(0, keep)
    return out
