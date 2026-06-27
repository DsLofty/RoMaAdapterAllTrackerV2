"""Shared RoMa branch relocalization helpers.

This module owns the mechanics for proposing RoMa relocalization frames,
mapping 0-anchor query points with RoMa, writing sparse flow8 overrides, and
running the frozen AllTracker branch. It is intentionally independent from
adapter/selector training scripts so export and full eval use the same path.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils.basic
import utils.data
from alltracker_runtime_utils import expand_path
from tapvid_dataset_utils import get_dataset


PROPOSAL_SOURCE_WINDOW = 0
PROPOSAL_SOURCE_BASELINE_EVENT = 1
PROPOSAL_SOURCE_PATCH_EVIDENCE = 2
PROPOSAL_SOURCE_ROMA_CERTAINTY = 3

EVENT_TYPE_WINDOW_START = 0
EVENT_TYPE_VISIBLE_RECOVER = 1
EVENT_TYPE_LOW_CONF_GAP = 2
EVENT_TYPE_MOTION_JUMP = 3
EVENT_TYPE_ACCEL_JUMP = 4
EVENT_TYPE_VIS_DROP = 5
EVENT_TYPE_BASELINE_PATCH_MISMATCH = 6
EVENT_TYPE_ROMA_CERTAINTY = 7


def flow8_grid_shape(H, W):
    H_pad = int(np.ceil(float(H) / 64.0) * 64)
    W_pad = int(np.ceil(float(W) / 64.0) * 64)
    return H_pad // 8, W_pad // 8


def pad_to_multiple_64(x):
    ht, wd = x.shape[-2:]
    pad_ht = (((ht // 64) + 1) * 64 - ht) % 64
    pad_wd = (((wd // 64) + 1) * 64 - wd) % 64
    pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    return F.pad(x, pad, mode='replicate')


def _sample_feat8_rows(feat8_seq, frame_ids, xy8):
    """Sample [T,C,H8,W8] feature maps at per-row frame/xy8 positions."""
    if feat8_seq is None or frame_ids.numel() == 0:
        return None
    T, C, H8, W8 = feat8_seq.shape
    device = feat8_seq.device
    dtype = feat8_seq.dtype
    frame_ids = frame_ids.to(device=device).long().clamp(0, max(T - 1, 0))
    xy8 = xy8.to(device=device, dtype=dtype).float()
    out = torch.zeros((int(frame_ids.numel()), C), device=device, dtype=dtype)
    finite = torch.isfinite(xy8).all(dim=1)
    if not bool(finite.any().item()):
        return out
    for frame_t in torch.unique(frame_ids[finite]).tolist():
        frame_t = int(frame_t)
        mask = finite & (frame_ids == frame_t)
        if not bool(mask.any().item()):
            continue
        coords = xy8[mask]
        if W8 > 1:
            gx = 2.0 * coords[:, 0] / float(W8 - 1) - 1.0
        else:
            gx = torch.zeros_like(coords[:, 0])
        if H8 > 1:
            gy = 2.0 * coords[:, 1] / float(H8 - 1) - 1.0
        else:
            gy = torch.zeros_like(coords[:, 1])
        grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)
        sampled = F.grid_sample(
            feat8_seq[frame_t : frame_t + 1],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )
        out[mask] = sampled[0, :, :, 0].transpose(0, 1)
    return out


def append_visual_features_to_inputs(batch, inputs, model, args, device, preserve_grad=False):
    """Append AllTracker feature embeddings used by the coarse relocation adapter.

    The features are query-frame embedding, baseline-target embedding, RoMa-target
    embedding, plus cosine similarities. This keeps RoMa as a global visual
    proposal while still letting the adapter regress a new coarse position.
    """
    if inputs is None or 'visual_query_feat' in inputs:
        return inputs
    if not bool(getattr(args, 'coarse_visual_features', False)):
        return inputs
    if 'baseline_xy8' not in inputs or 'roma_xy8' not in inputs or 'target_frame' not in inputs:
        return inputs

    rgbs = batch.video.to(device, non_blocking=True).float()
    B, T, C, H, W = rgbs.shape
    if B != 1 or C != 3:
        return inputs
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 1, 3, 1, 1).to(rgbs.dtype)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 1, 3, 1, 1).to(rgbs.dtype)
    images = (rgbs / 255.0 - mean) / std
    images_ = images.reshape(B * T, 3, H, W).contiguous()
    images_ = pad_to_multiple_64(images_)
    grad_enabled = bool(preserve_grad and getattr(args, 'coarse_visual_grad', False))
    with torch.set_grad_enabled(grad_enabled):
        fmaps = model.get_fmaps(images_, B, T, sw=None, is_training=grad_enabled)
    _, C8, H8, W8 = fmaps.shape
    feat8_seq = fmaps.reshape(B, T, C8, H8, W8)[0]
    if not grad_enabled:
        feat8_seq = feat8_seq.detach()

    target_frame = inputs['target_frame'].long().to(device)
    source_xy8 = inputs['source_xy'].float().to(device) / 8.0
    source_frame = torch.zeros_like(target_frame)
    baseline_xy8 = inputs['baseline_xy8'].float().to(device)
    roma_xy8 = inputs['roma_xy8'].float().to(device)

    query_feat = _sample_feat8_rows(feat8_seq, source_frame, source_xy8)
    baseline_feat = _sample_feat8_rows(feat8_seq, target_frame, baseline_xy8)
    roma_feat = _sample_feat8_rows(feat8_seq, target_frame, roma_xy8)
    if query_feat is None or baseline_feat is None or roma_feat is None:
        return inputs

    if bool(getattr(args, 'coarse_visual_l2norm', True)):
        query_feat = F.normalize(query_feat.float(), dim=1, eps=1.0e-6)
        baseline_feat = F.normalize(baseline_feat.float(), dim=1, eps=1.0e-6)
        roma_feat = F.normalize(roma_feat.float(), dim=1, eps=1.0e-6)
    else:
        query_feat = query_feat.float()
        baseline_feat = baseline_feat.float()
        roma_feat = roma_feat.float()

    inputs['visual_query_feat'] = query_feat
    inputs['visual_baseline_feat'] = baseline_feat
    inputs['visual_roma_feat'] = roma_feat
    inputs['visual_baseline_query_cos'] = torch.sum(baseline_feat * query_feat, dim=1)
    inputs['visual_roma_query_cos'] = torch.sum(roma_feat * query_feat, dim=1)
    inputs['visual_roma_baseline_cos'] = torch.sum(roma_feat * baseline_feat, dim=1)
    inputs['visual_roma_query_cos_gain'] = inputs['visual_roma_query_cos'] - inputs['visual_baseline_query_cos']
    return inputs


def window_start_frames(T, seqlen, stride=None):
    if T <= 2:
        return []
    step = int(seqlen) // 2 if stride is None else int(stride)
    starts = []
    start = 0
    while start + int(seqlen) < int(T):
        starts.append(start)
        start += step
    starts.append(start)
    return sorted(set(int(s) for s in starts if int(s) > 0 and int(s) < int(T)))


def _snap_event_to_window_start(event_t, window_starts, snap_mode):
    if str(snap_mode) == 'direct':
        return int(event_t)
    if not window_starts:
        return None
    event_t = int(event_t)
    starts = [int(v) for v in window_starts]
    if str(snap_mode) == 'prev':
        prev = [v for v in starts if v <= event_t]
        return prev[-1] if prev else starts[0]
    if str(snap_mode) == 'next':
        nxt = [v for v in starts if v >= event_t]
        return nxt[0] if nxt else starts[-1]
    return min(starts, key=lambda v: (abs(v - event_t), v))


def _proposal_source_bit(source_id):
    return int(1 << int(source_id))


def _add_relocalization_proposal(
    proposals,
    target_t,
    point_id,
    event_t,
    source_id,
    event_type_id,
    score,
    baseline_patch_ncc=float('nan'),
    patch_mismatch_score=0.0,
    query_patch_texture=float('nan'),
):
    key = (int(target_t), int(point_id))
    value = {
        'target_frame': int(target_t),
        'point_index': int(point_id),
        'event_frame': int(event_t),
        'proposal_source': int(source_id),
        'proposal_source_mask': _proposal_source_bit(source_id),
        'event_type': int(event_type_id),
        'event_score': float(score),
        'baseline_patch_ncc': float(baseline_patch_ncc),
        'patch_mismatch_score': float(patch_mismatch_score),
        'query_patch_texture': float(query_patch_texture),
    }
    old = proposals.get(key)
    if old is not None:
        value['proposal_source_mask'] = int(old.get('proposal_source_mask', 0)) | int(value['proposal_source_mask'])
        if not np.isfinite(value['baseline_patch_ncc']) and np.isfinite(float(old.get('baseline_patch_ncc', float('nan')))):
            value['baseline_patch_ncc'] = float(old.get('baseline_patch_ncc'))
        value['patch_mismatch_score'] = max(float(value['patch_mismatch_score']), float(old.get('patch_mismatch_score', 0.0)))
        if not np.isfinite(value['query_patch_texture']) and np.isfinite(float(old.get('query_patch_texture', float('nan')))):
            value['query_patch_texture'] = float(old.get('query_patch_texture'))
    if old is None or float(value['event_score']) > float(old['event_score']):
        proposals[key] = value
    elif old is not None:
        old['proposal_source_mask'] = value['proposal_source_mask']
        old['patch_mismatch_score'] = value['patch_mismatch_score']
        if np.isfinite(value['baseline_patch_ncc']):
            old['baseline_patch_ncc'] = value['baseline_patch_ncc']
        if np.isfinite(value['query_patch_texture']):
            old['query_patch_texture'] = value['query_patch_texture']


def _image_to_gray(image):
    image = image.detach().float().cpu()
    if image.ndim != 3:
        raise ValueError('expected image with shape [C,H,W]')
    if image.shape[0] == 1:
        return image[0]
    return 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]


def _extract_patch(gray, xy, radius):
    radius = int(radius)
    if radius <= 0:
        return None
    H, W = gray.shape
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if x - radius < 0 or x + radius >= W or y - radius < 0 or y + radius >= H:
        return None
    return gray[y - radius : y + radius + 1, x - radius : x + radius + 1].float()


def patch_ncc(query_gray, target_gray, query_xy, target_xy, radius):
    q = _extract_patch(query_gray, query_xy, radius)
    t = _extract_patch(target_gray, target_xy, radius)
    if q is None or t is None or q.numel() != t.numel():
        return float('nan'), float('nan')
    q = q.reshape(-1)
    t = t.reshape(-1)
    q_std = torch.std(q)
    t_std = torch.std(t)
    q_texture = float(q_std.item())
    if float(q_std.item()) <= 1.0e-6 or float(t_std.item()) <= 1.0e-6:
        return float('nan'), q_texture
    q = (q - torch.mean(q)) / q_std
    t = (t - torch.mean(t)) / t_std
    return float(torch.mean(q * t).item()), q_texture


def build_relocalization_proposals(baseline, args, model, rgbs=None):
    """Build point/frame proposals for 0-anchor RoMa relocalization.

    The default window_start mode preserves historical behavior. event_window
    still writes the AllTracker override at runnable window starts, but chooses
    those starts from nearby baseline events and records the original event
    frame for analysis. mixed_event_patch combines fixed window starts,
    baseline-state events, and image patch mismatch proposals.
    """
    baseline_xy = baseline['trajs_e'].float()
    pred_visible = baseline['pred_visible_score'].float()
    valids = baseline['valids'].float()
    first_positive = baseline['first_positive_inds'].long()
    T, _, _ = baseline_xy.shape
    frame_start = max(1, int(getattr(args, 'proposal_frame_start', 1)))
    frame_end_arg = int(getattr(args, 'proposal_frame_end', 0))
    frame_end = int(T) if frame_end_arg <= 0 else min(int(T), max(0, frame_end_arg))
    if frame_start >= frame_end:
        return []

    def target_in_range(target_t):
        return frame_start <= int(target_t) < frame_end

    source_t = 0
    point_ids = torch.nonzero(first_positive == source_t, as_tuple=False).reshape(-1).long()
    if point_ids.numel() == 0:
        return []

    seqlen = int(getattr(model, 'seqlen', 16))
    window_starts = window_start_frames(T, seqlen)
    if not window_starts:
        return []

    proposals = {}
    mode = str(getattr(args, 'proposal_mode', 'window_start'))
    include_window_start = bool(getattr(args, 'proposal_include_window_start', False))
    if mode == 'window_start':
        include_window_start = True

    if include_window_start:
        for target_t in window_starts:
            if not target_in_range(target_t):
                continue
            valid_point_ids = point_ids[valids[int(target_t), point_ids] > 0.5]
            for point_id in valid_point_ids.tolist():
                _add_relocalization_proposal(
                    proposals,
                    target_t=target_t,
                    point_id=point_id,
                    event_t=target_t,
                    source_id=PROPOSAL_SOURCE_WINDOW,
                    event_type_id=EVENT_TYPE_WINDOW_START,
                    score=0.0,
                )

    if mode == 'window_start':
        return sorted(proposals.values(), key=lambda v: (v['target_frame'], v['point_index']))
    if mode not in ('event_window', 'mixed_event_patch'):
        raise ValueError('unsupported proposal_mode: %s' % str(mode))

    history_len = int(getattr(args, 'history_len', 8))
    conf_thr = float(getattr(args, 'candidate_conf_thr', 0.6))
    mean_conf_thr = float(getattr(args, 'candidate_mean_conf_thr', 0.7))
    low_vis_ratio_thr = float(getattr(args, 'candidate_low_vis_ratio_thr', 0.25))
    motion_jump_thr = float(getattr(args, 'candidate_motion_jump_thr', 16.0))
    accel_thr = float(getattr(args, 'candidate_accel_thr', 16.0))
    gap_thr = int(getattr(args, 'candidate_occlusion_gap_thr', 2))
    min_score = float(getattr(args, 'proposal_event_score_thr', 0.25))
    max_events_per_point = int(getattr(args, 'proposal_max_events_per_point', 4))
    min_separation = int(getattr(args, 'proposal_min_separation', 4))
    snap_mode = str(getattr(args, 'proposal_event_snap', 'nearest'))
    max_rows = int(getattr(args, 'proposal_max_rows_per_sequence', 0))

    for point_id in point_ids.tolist():
        point_events = []
        low_run = 0
        for target_t in range(frame_start, frame_end):
            if float(valids[target_t, point_id]) <= 0.5:
                continue
            hist_start = max(0, int(target_t) - history_len + 1)
            hist_vis = pred_visible[hist_start : target_t + 1, point_id]
            current_vis = float(pred_visible[target_t, point_id])
            prev_vis = float(pred_visible[target_t - 1, point_id])
            min_vis = float(torch.min(hist_vis).item()) if hist_vis.numel() else current_vis
            mean_vis = float(torch.mean(hist_vis).item()) if hist_vis.numel() else current_vis
            low_vis_ratio = float(torch.mean((hist_vis < conf_thr).float()).item()) if hist_vis.numel() else 0.0
            low_run = low_run + 1 if current_vis < conf_thr else 0

            motion_jump = float(torch.linalg.vector_norm(baseline_xy[target_t, point_id] - baseline_xy[target_t - 1, point_id]).item())
            if target_t > 1:
                prev_jump = float(
                    torch.linalg.vector_norm(baseline_xy[target_t - 1, point_id] - baseline_xy[target_t - 2, point_id]).item()
                )
                accel = abs(motion_jump - prev_jump)
            else:
                accel = 0.0

            recover_score = max(0.0, current_vis - min_vis) * max(low_vis_ratio, 0.0)
            gap_score = 0.0
            if low_run == 0 and low_vis_ratio >= low_vis_ratio_thr and current_vis >= mean_conf_thr:
                gap_score = min(1.0, low_vis_ratio + max(0.0, current_vis - min_vis))
            if low_run >= gap_thr:
                gap_score = max(gap_score, min(1.0, float(low_run) / max(1.0, float(history_len))))
            motion_score = min(1.0, motion_jump / max(1.0e-6, motion_jump_thr))
            accel_score = min(1.0, accel / max(1.0e-6, accel_thr))
            drop_score = max(0.0, prev_vis - current_vis)

            typed_scores = [
                (recover_score, EVENT_TYPE_VISIBLE_RECOVER),
                (gap_score, EVENT_TYPE_LOW_CONF_GAP),
                (motion_score, EVENT_TYPE_MOTION_JUMP),
                (accel_score, EVENT_TYPE_ACCEL_JUMP),
                (drop_score, EVENT_TYPE_VIS_DROP),
            ]
            event_score, event_type = max(typed_scores, key=lambda item: item[0])
            suspicious_conf = current_vis < conf_thr or mean_vis < mean_conf_thr or low_vis_ratio > low_vis_ratio_thr
            if event_score < min_score and not suspicious_conf:
                continue
            target_window = _snap_event_to_window_start(target_t, window_starts, snap_mode)
            if target_window is None:
                continue
            if not target_in_range(target_window):
                continue
            if int(target_window) <= 0 or int(target_window) >= T:
                continue
            if float(valids[int(target_window), point_id]) <= 0.5:
                continue
            point_events.append((float(event_score), int(target_window), int(target_t), int(event_type), int(point_id)))

        if not point_events:
            continue
        point_events.sort(key=lambda item: item[0], reverse=True)
        kept_targets = []
        kept = 0
        for event_score, target_window, event_t, event_type, point_id in point_events:
            if any(abs(int(target_window) - int(old_t)) < min_separation for old_t in kept_targets):
                continue
            _add_relocalization_proposal(
                proposals,
                target_t=target_window,
                point_id=point_id,
                event_t=event_t,
                source_id=PROPOSAL_SOURCE_BASELINE_EVENT,
                event_type_id=event_type,
                score=event_score,
            )
            kept_targets.append(int(target_window))
            kept += 1
            if kept >= max_events_per_point:
                break

    if mode == 'mixed_event_patch' and rgbs is not None:
        patch_radius = int(getattr(args, 'proposal_patch_radius', 4))
        patch_stride = max(1, int(getattr(args, 'proposal_patch_stride', 2)))
        patch_ncc_thr = float(getattr(args, 'proposal_patch_baseline_ncc_thr', 0.15))
        patch_texture_thr = float(getattr(args, 'proposal_patch_texture_thr', 1.0))
        patch_max_events = int(getattr(args, 'proposal_patch_max_events_per_point', max_events_per_point))
        query_gray = _image_to_gray(rgbs[source_t])
        gray_cache = {source_t: query_gray}
        source_points = baseline['trajs_g'].float()[source_t]
        for point_id in point_ids.tolist():
            qxy = source_points[point_id, :2]
            patch_events = []
            patch_start = max(1, frame_start)
            patch_start += (1 - patch_start) % patch_stride
            for target_t in range(patch_start, frame_end, patch_stride):
                if float(valids[target_t, point_id]) <= 0.5:
                    continue
                if target_t not in gray_cache:
                    gray_cache[target_t] = _image_to_gray(rgbs[target_t])
                baseline_xy_t = baseline_xy[target_t, point_id, :2]
                ncc, texture = patch_ncc(query_gray, gray_cache[target_t], qxy, baseline_xy_t, patch_radius)
                if not np.isfinite(ncc) or not np.isfinite(texture):
                    continue
                if texture < patch_texture_thr:
                    continue
                mismatch = max(0.0, patch_ncc_thr - float(ncc))
                if mismatch <= 0.0:
                    continue
                score = float(np.clip(mismatch / max(1.0e-6, patch_ncc_thr + 1.0), 0.0, 1.0))
                target_window = _snap_event_to_window_start(target_t, window_starts, snap_mode)
                if target_window is None:
                    continue
                if not target_in_range(target_window):
                    continue
                if int(target_window) <= 0 or int(target_window) >= T:
                    continue
                if float(valids[int(target_window), point_id]) <= 0.5:
                    continue
                patch_events.append((score, int(target_window), int(target_t), int(point_id), float(ncc), score, float(texture)))
            if not patch_events:
                continue
            patch_events.sort(key=lambda item: item[0], reverse=True)
            kept_targets = []
            kept = 0
            for score, target_window, event_t, point_id, ncc, mismatch_score, texture in patch_events:
                if any(abs(int(target_window) - int(old_t)) < min_separation for old_t in kept_targets):
                    continue
                _add_relocalization_proposal(
                    proposals,
                    target_t=target_window,
                    point_id=point_id,
                    event_t=event_t,
                    source_id=PROPOSAL_SOURCE_PATCH_EVIDENCE,
                    event_type_id=EVENT_TYPE_BASELINE_PATCH_MISMATCH,
                    score=score,
                    baseline_patch_ncc=ncc,
                    patch_mismatch_score=mismatch_score,
                    query_patch_texture=texture,
                )
                kept_targets.append(int(target_window))
                kept += 1
                if kept >= patch_max_events:
                    break

    values = sorted(proposals.values(), key=lambda v: (v['target_frame'], v['point_index']))
    if max_rows > 0 and len(values) > max_rows:
        values = sorted(values, key=lambda v: float(v['event_score']), reverse=True)[:max_rows]
        values = sorted(values, key=lambda v: (v['target_frame'], v['point_index']))
    return values


def clear_roma_init(model):
    model.roma_init_enable = False
    model.roma_init_flow8_override = None
    model.roma_init_mask8_override = None
    model.roma_init_preserve_grad = False
    model.roma_init_return_last_only = False
    model.roma_init_skip_visconf_upsample = False
    model._roma_init_last_applied = False
    model._roma_init_window_preserve_grad = False
    if hasattr(model, 'roma_coarse_adapter_enable'):
        model.roma_coarse_adapter_enable = False
        model.roma_coarse_adapter = None
        model.roma_coarse_adapter_inputs = None
        model.roma_coarse_adapter_feature_names = None
        model.roma_coarse_adapter_stats = {}
        model.roma_coarse_adapter_stats_history = []
        model.roma_coarse_adapter_supervision_history = []
        model.roma_coarse_adapter_train_mode = False
        model.roma_coarse_adapter_gt_mix_prob = 0.0
        model.roma_coarse_adapter_gt_mix_ratio = 0.0
        model.roma_coarse_adapter_gt_mix_noise8 = 0.0
        model.roma_coarse_adapter_counterfactual_reject_ratio = 0.0
        model.roma_coarse_adapter_counterfactual_reject_max_rows = 0


def set_roma_init(model, flow8_override, mask8_override, preserve_grad=False, apply_at='window_start'):
    model.roma_init_enable = True
    model.roma_init_apply_at = str(apply_at)
    model.roma_init_flow8_override = flow8_override
    model.roma_init_mask8_override = mask8_override
    model.roma_init_preserve_grad = bool(preserve_grad)
    model.roma_init_return_last_only = bool(preserve_grad)
    model.roma_init_skip_visconf_upsample = bool(preserve_grad)
    model._roma_init_last_applied = False
    model._roma_init_window_preserve_grad = False


def set_roma_coarse_adapter_init(
    model,
    adapter,
    inputs,
    feature_names,
    preserve_grad=False,
    apply_at='window_start',
    gate_mode='none',
    coord_mode='fusion_residual',
    max_frames_per_window=1,
    max_rows_per_frame=128,
    gt_mix_prob=0.0,
    gt_mix_ratio=0.0,
    gt_mix_noise8=0.0,
    relocalize_policy='learned',
    heuristic_roma_certainty_thr=0.7,
    heuristic_baseline_visible_thr=0.5,
    heuristic_baseline_motion_jump8_thr=1.5,
    heuristic_visual_gain_thr=0.05,
    heuristic_visual_cos_thr=0.5,
    heuristic_strong_visual_gain_thr=0.15,
    heuristic_roma_prev_dist8_max=8.0,
    heuristic_min_offset8=0.5,
    safety_gate=False,
    safety_thr=0.6,
    safety_min_gain_px=0.0,
    baseline_need_thr=0.5,
    candidate_quality_thr=0.5,
    candidate_accept_thr=0.5,
    counterfactual_reject_ratio=0.0,
    counterfactual_reject_max_rows=0,
    carry_mode='none',
    carry_max_age=0,
    carry_decay=0.9,
    carry_min_score=0.0,
    carry_max_offset8=8.0,
    carry_require_baseline_suspicious=False,
    carry_apply_strength=1.0,
    carry_refresh_dist8=2.0,
):
    model.roma_init_enable = True
    model.roma_init_apply_at = str(apply_at)
    model.roma_init_flow8_override = None
    model.roma_init_mask8_override = None
    model.roma_init_preserve_grad = bool(preserve_grad)
    model.roma_init_return_last_only = bool(preserve_grad)
    model.roma_init_skip_visconf_upsample = bool(preserve_grad)
    model.roma_coarse_adapter_enable = True
    model.roma_coarse_adapter = adapter
    model.roma_coarse_adapter_inputs = inputs
    model.roma_coarse_adapter_feature_names = list(feature_names)
    model.roma_coarse_adapter_gate_mode = str(gate_mode)
    model.roma_coarse_adapter_coord_mode = str(coord_mode)
    model.roma_coarse_adapter_max_frames_per_window = int(max_frames_per_window)
    model.roma_coarse_adapter_max_rows_per_frame = int(max_rows_per_frame)
    model.roma_coarse_adapter_train_mode = bool(preserve_grad)
    model.roma_coarse_adapter_gt_mix_prob = float(gt_mix_prob)
    model.roma_coarse_adapter_gt_mix_ratio = float(gt_mix_ratio)
    model.roma_coarse_adapter_gt_mix_noise8 = float(gt_mix_noise8)
    model.roma_coarse_adapter_relocalize_policy = str(relocalize_policy)
    model.roma_coarse_adapter_heuristic_roma_certainty_thr = float(heuristic_roma_certainty_thr)
    model.roma_coarse_adapter_heuristic_baseline_visible_thr = float(heuristic_baseline_visible_thr)
    model.roma_coarse_adapter_heuristic_baseline_motion_jump8_thr = float(heuristic_baseline_motion_jump8_thr)
    model.roma_coarse_adapter_heuristic_visual_gain_thr = float(heuristic_visual_gain_thr)
    model.roma_coarse_adapter_heuristic_visual_cos_thr = float(heuristic_visual_cos_thr)
    model.roma_coarse_adapter_heuristic_strong_visual_gain_thr = float(heuristic_strong_visual_gain_thr)
    model.roma_coarse_adapter_heuristic_roma_prev_dist8_max = float(heuristic_roma_prev_dist8_max)
    model.roma_coarse_adapter_heuristic_min_offset8 = float(heuristic_min_offset8)
    model.roma_coarse_adapter_safety_gate = bool(safety_gate)
    model.roma_coarse_adapter_safety_thr = float(safety_thr)
    model.roma_coarse_adapter_safety_min_gain_px = float(safety_min_gain_px)
    model.roma_coarse_adapter_baseline_need_thr = float(baseline_need_thr)
    model.roma_coarse_adapter_candidate_quality_thr = float(candidate_quality_thr)
    model.roma_coarse_adapter_candidate_accept_thr = float(candidate_accept_thr)
    model.roma_coarse_adapter_counterfactual_reject_ratio = float(counterfactual_reject_ratio)
    model.roma_coarse_adapter_counterfactual_reject_max_rows = int(counterfactual_reject_max_rows)
    model.roma_coarse_adapter_carry_mode = str(carry_mode)
    model.roma_coarse_adapter_carry_max_age = int(carry_max_age)
    model.roma_coarse_adapter_carry_decay = float(carry_decay)
    model.roma_coarse_adapter_carry_min_score = float(carry_min_score)
    model.roma_coarse_adapter_carry_max_offset8 = float(carry_max_offset8)
    model.roma_coarse_adapter_carry_require_baseline_suspicious = bool(carry_require_baseline_suspicious)
    model.roma_coarse_adapter_carry_apply_strength = float(carry_apply_strength)
    model.roma_coarse_adapter_carry_refresh_dist8 = float(carry_refresh_dist8)
    model.roma_coarse_adapter_carry_state = {}
    model.roma_coarse_adapter_stats = {}
    model.roma_coarse_adapter_stats_history = []
    model.roma_coarse_adapter_supervision_history = []
    model._roma_init_last_applied = False
    model._roma_init_window_preserve_grad = False


def collect_roma_coarse_adapter_supervision(model):
    history = list(getattr(model, 'roma_coarse_adapter_supervision_history', []) or [])
    if not history:
        return {}
    keys = sorted({key for item in history for key, value in item.items() if torch.is_tensor(value)})
    out = {}
    for key in keys:
        values = [item[key] for item in history if key in item and torch.is_tensor(item[key])]
        if values:
            out[key] = torch.cat(values, dim=0)
    return out


def make_invalid_mapping(num_points):
    return {
        'points1': torch.full((int(num_points), 2), float('nan'), dtype=torch.float32),
        'certainty': torch.full((int(num_points),), float('nan'), dtype=torch.float32),
        'valid': torch.zeros((int(num_points),), dtype=torch.bool),
    }


def to_cpu_mapping(mapping):
    return {
        'points1': mapping['points1'].detach().float().cpu(),
        'certainty': mapping['certainty'].detach().float().cpu(),
        'valid': mapping['valid'].detach().bool().cpu(),
    }


def unpack_batch(batch_pack):
    if isinstance(batch_pack, (tuple, list)) and len(batch_pack) == 2:
        batch, gotit = batch_pack
        if not all(gotit):
            return batch, False
        return batch, True
    return batch_pack, True


def make_tapvid_loader(args, shuffle=False):
    dataset, dataset_root = get_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool(shuffle),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=utils.data.collate_fn_train,
    )
    return loader, dataset_root


def make_kubric_loader(args, shuffle=True):
    from datasets import kubric_movif_dataset

    data_dir = expand_path(args.data_dir)
    if os.path.basename(os.path.normpath(data_dir)) == 'kubric_au':
        data_root = data_dir
    elif os.path.isdir(os.path.join(data_dir, 'kubric_au')):
        data_root = os.path.join(data_dir, 'kubric_au')
    else:
        data_root = data_dir
    def build_dataset(seq_len):
        return kubric_movif_dataset.KubricMovifDataset(
            data_root=data_root,
            crop_size=(int(args.image_size[0]), int(args.image_size[1])),
            seq_len=int(seq_len),
            traj_per_sample=int(args.traj_per_sample),
            traj_max_factor=1,
            use_augs=bool(args.use_augs),
            random_seq_len=False,
            random_first_frame=False,
            random_frame_rate=False,
            random_number_traj=False,
            shuffle_frames=False,
            shuffle=bool(shuffle),
            only_first=True,
            query_rich_crop=bool(getattr(args, 'query_rich_crop', False)) and bool(shuffle),
            query_rich_topk=int(getattr(args, 'query_rich_topk', 8)),
        )

    requested_seq_len = int(args.sequence_len)
    dataset = build_dataset(requested_seq_len)
    if len(dataset) == 0 and requested_seq_len > 24:
        fallback_seq_len = 24
        print(
            'warning: no Kubric videos found for sequence_len=%d under %s; retrying with sequence_len=%d'
            % (requested_seq_len, data_root, fallback_seq_len),
            flush=True,
        )
        args.sequence_len = int(fallback_seq_len)
        dataset = build_dataset(fallback_seq_len)
    if len(dataset) == 0:
        raise FileNotFoundError(
            'No Kubric videos found under %s. Expected folders matching '
            '<split>/<sequence>/ with annot.npy and frames/. Pass --data_dir as '
            'either the parent directory containing kubric_au or the kubric_au '
            'directory itself.' % data_root
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool(shuffle),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=utils.data.collate_fn_train,
    )
    return loader, data_root


def make_loader(args, shuffle=True):
    if str(args.dname).lower() == 'kubric':
        return make_kubric_loader(args, shuffle=shuffle)
    return make_tapvid_loader(args, shuffle=shuffle)


def dense_flow_to_sparse(batch, forward_flow_e, forward_visconf_e, device):
    rgbs = batch.video.to(device, non_blocking=True).float()
    trajs_g = batch.trajs.to(device, non_blocking=True).float()
    vis_g = batch.visibs.to(device, non_blocking=True).float()
    valids = batch.valids.to(device, non_blocking=True).float() if batch.valids is not None else torch.ones_like(vis_g)
    B, T, _, H, W = rgbs.shape
    _, _, N, _ = trajs_g.shape
    assert B == 1
    grid_xy = utils.basic.gridcloud2d(1, H, W, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
    traj_maps_e = forward_flow_e.to(device) + grid_xy
    _, first_positive = torch.max(vis_g, dim=1)
    trajs_e = torch.zeros([B, T, N, 2], device=device, dtype=traj_maps_e.dtype)
    visconfs_e = torch.zeros([B, T, N, 2], device=device, dtype=forward_visconf_e.dtype)

    for first_t_tensor in torch.unique(first_positive):
        first_t = int(first_t_tensor.item())
        point_ids = torch.nonzero(first_positive[0] == first_t, as_tuple=False)[:, 0]
        if point_ids.numel() == 0:
            continue
        if first_t != 0:
            continue
        xyt = trajs_g[:, first_t].round().long()[0, point_ids]
        xyt[:, 0] = torch.clamp(xyt[:, 0], 0, W - 1)
        xyt[:, 1] = torch.clamp(xyt[:, 1], 0, H - 1)
        trajs_chunk = traj_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]].permute(0, 1, 3, 2)
        trajs_e.scatter_add_(
            2,
            point_ids[None, None, :, None].repeat(1, T, 1, 2),
            trajs_chunk,
        )
        vis_chunk = forward_visconf_e[:, :, :, xyt[:, 1], xyt[:, 0]].permute(0, 1, 3, 2)
        visconfs_e.scatter_add_(
            2,
            point_ids[None, None, :, None].repeat(1, T, 1, 2),
            vis_chunk,
        )

    pred_confidence = torch.clamp(visconfs_e[0, :, :, 1], 0.0, 1.0)
    visconfs_e[..., 0] *= visconfs_e[..., 1]
    pred_visible = torch.clamp(visconfs_e[0, :, :, 0], 0.0, 1.0)
    return {
        'trajs_e': trajs_e[0],
        'pred_visible_score': pred_visible,
        'pred_confidence_score': pred_confidence,
        'first_positive_inds': first_positive[0].long(),
        'trajs_g': trajs_g[0],
        'vis_g': vis_g[0],
        'valids': valids[0],
    }


def dense_iter_preds_to_sparse(batch, flow_preds, visconf_preds, device, sequence_len):
    if flow_preds is None or visconf_preds is None:
        return []
    if len(flow_preds) == 0 or len(visconf_preds) == 0:
        return []

    rgbs = batch.video.to(device, non_blocking=True).float()
    B, T, _, H, W = rgbs.shape
    if T <= 2:
        out = []
        for flow_pred, visconf_pred in zip(flow_preds, visconf_preds):
            out.append(dense_flow_to_sparse(batch, flow_pred.to(device), visconf_pred.to(device), device))
        return out

    first_window = flow_preds[0]
    if not isinstance(first_window, (list, tuple)) or len(first_window) == 0:
        return []
    n_iters = int(len(first_window))
    if n_iters <= 0:
        return []
    S = int(sequence_len)
    if S <= 0:
        return []
    step = max(1, S // 2)
    indices = []
    start = 0
    while start + S < T:
        indices.append(int(start))
        start += step
    indices.append(int(start))
    if len(indices) != len(flow_preds):
        return []

    flow_full = [
        torch.zeros((B, T, 2, H, W), dtype=first_window[0].dtype, device=device)
        for _ in range(n_iters)
    ]
    vis_full = [
        torch.zeros((B, T, 2, H, W), dtype=visconf_preds[0][0].dtype, device=device)
        for _ in range(n_iters)
    ]
    for window_idx, ind in enumerate(indices):
        ara = torch.arange(int(ind), int(ind) + S, device=device).long()
        if int(ara[-1].item()) >= T:
            return []
        window_flow_preds = flow_preds[window_idx]
        window_vis_preds = visconf_preds[window_idx]
        if len(window_flow_preds) != n_iters or len(window_vis_preds) != n_iters:
            return []
        for itr in range(n_iters):
            flow_full[itr][:, ara] = window_flow_preds[itr].to(device)
            vis_full[itr][:, ara] = window_vis_preds[itr].to(device)

    return [dense_flow_to_sparse(batch, flow_full[itr], vis_full[itr], device) for itr in range(n_iters)]


def run_alltracker_sparse(batch, model, args, device, override=None, preserve_grad=False):
    rgbs = batch.video.to(device, non_blocking=True).float()
    clear_roma_init(model)
    if override is not None:
        flow8_override, mask8_override = override
        set_roma_init(
            model,
            flow8_override,
            mask8_override,
            preserve_grad=preserve_grad,
            apply_at=str(getattr(args, 'roma_init_apply_at', 'window_start')),
        )
    if rgbs.shape[1] > 128 and bool(preserve_grad):
        clear_roma_init(model)
        raise ValueError('gradient-preserving training path does not support T > 128')
    if rgbs.shape[1] > 128:
        forward_flow_e, forward_visconf_e, flow_preds, visconf_preds = model.forward_sliding(
            rgbs,
            iters=int(args.inference_iters),
            sw=None,
            is_training=False,
        )
    else:
        forward_flow_e, forward_visconf_e, flow_preds, visconf_preds = model(
            rgbs,
            iters=int(args.inference_iters),
            sw=None,
            is_training=False,
        )
    del flow_preds
    del visconf_preds
    sparse = dense_flow_to_sparse(batch, forward_flow_e.to(device), forward_visconf_e.to(device), device)
    clear_roma_init(model)
    return sparse


def run_alltracker_embedded_adapter(
    batch,
    model,
    adapter,
    inputs,
    feature_names,
    args,
    device,
    preserve_grad=False,
):
    rgbs = batch.video.to(device, non_blocking=True).float()
    clear_roma_init(model)
    set_roma_coarse_adapter_init(
        model,
        adapter,
        inputs,
        feature_names,
        preserve_grad=preserve_grad,
        apply_at=str(getattr(args, 'roma_init_apply_at', 'window_start')),
        gate_mode=str(getattr(args, 'gate_mode', 'none')),
        coord_mode=str(getattr(args, 'coord_mode', 'fusion_residual')),
        max_frames_per_window=int(getattr(args, 'embedded_max_frames_per_window', 1)),
        max_rows_per_frame=int(getattr(args, 'embedded_max_rows_per_frame', 128)),
        gt_mix_prob=float(getattr(args, 'gt_init_mix_prob', 0.0)),
        gt_mix_ratio=float(getattr(args, 'gt_init_mix_ratio', 0.0)),
        gt_mix_noise8=float(getattr(args, 'gt_init_noise8', 0.0)),
        relocalize_policy=str(getattr(args, 'relocalize_policy', 'learned')),
        heuristic_roma_certainty_thr=float(getattr(args, 'heuristic_roma_certainty_thr', 0.7)),
        heuristic_baseline_visible_thr=float(getattr(args, 'heuristic_baseline_visible_thr', 0.5)),
        heuristic_baseline_motion_jump8_thr=float(getattr(args, 'heuristic_baseline_motion_jump8_thr', 1.5)),
        heuristic_visual_gain_thr=float(getattr(args, 'heuristic_visual_gain_thr', 0.05)),
        heuristic_visual_cos_thr=float(getattr(args, 'heuristic_visual_cos_thr', 0.5)),
        heuristic_strong_visual_gain_thr=float(getattr(args, 'heuristic_strong_visual_gain_thr', 0.15)),
        heuristic_roma_prev_dist8_max=float(getattr(args, 'heuristic_roma_prev_dist8_max', 8.0)),
        heuristic_min_offset8=float(getattr(args, 'heuristic_min_offset8', 0.5)),
        safety_gate=bool(getattr(args, 'safety_gate', False)),
        safety_thr=float(getattr(args, 'safety_thr', 0.6)),
        safety_min_gain_px=float(getattr(args, 'safety_min_gain_px', 0.0)),
        baseline_need_thr=float(getattr(args, 'baseline_need_thr', 0.5)),
        candidate_quality_thr=float(getattr(args, 'candidate_quality_thr', 0.5)),
        candidate_accept_thr=float(getattr(args, 'candidate_accept_thr', 0.5)),
        counterfactual_reject_ratio=float(getattr(args, 'counterfactual_reject_ratio', 0.0)),
        counterfactual_reject_max_rows=int(getattr(args, 'counterfactual_reject_max_rows', 0)),
        carry_mode=str(getattr(args, 'carry_mode', 'none')),
        carry_max_age=int(getattr(args, 'carry_max_age', 0)),
        carry_decay=float(getattr(args, 'carry_decay', 0.9)),
        carry_min_score=float(getattr(args, 'carry_min_score', 0.0)),
        carry_max_offset8=float(getattr(args, 'carry_max_offset8', 8.0)),
        carry_require_baseline_suspicious=bool(getattr(args, 'carry_require_baseline_suspicious', False)),
        carry_apply_strength=float(getattr(args, 'carry_apply_strength', 1.0)),
        carry_refresh_dist8=float(getattr(args, 'carry_refresh_dist8', 2.0)),
    )
    if rgbs.shape[1] > 128 and bool(preserve_grad):
        clear_roma_init(model)
        raise ValueError('gradient-preserving embedded adapter path does not support T > 128')
    collect_iter_preds = bool(preserve_grad) and (
        float(getattr(args, 'lambda_iter_traj', 0.0)) > 0.0
        or float(getattr(args, 'lambda_iter_visibility', 0.0)) > 0.0
        or float(getattr(args, 'lambda_iter_confidence', 0.0)) > 0.0
    )
    if rgbs.shape[1] > 128:
        forward_flow_e, forward_visconf_e, flow_preds, visconf_preds = model.forward_sliding(
            rgbs,
            iters=int(args.inference_iters),
            sw=None,
            is_training=False,
        )
    else:
        forward_flow_e, forward_visconf_e, flow_preds, visconf_preds = model(
            rgbs,
            iters=int(args.inference_iters),
            sw=None,
            is_training=collect_iter_preds,
        )
    stats = dict(getattr(model, 'roma_coarse_adapter_stats', {}) or {})
    supervision = collect_roma_coarse_adapter_supervision(model)
    if supervision:
        stats['supervision'] = supervision
    sparse = dense_flow_to_sparse(batch, forward_flow_e.to(device), forward_visconf_e.to(device), device)
    if collect_iter_preds:
        iter_sparse = dense_iter_preds_to_sparse(
            batch,
            flow_preds,
            visconf_preds,
            device,
            sequence_len=int(getattr(model, 'seqlen', getattr(args, 'sequence_len', 0))),
        )
        if iter_sparse:
            sparse['iter_trajs_e'] = [item['trajs_e'] for item in iter_sparse]
            sparse['iter_pred_visible_score'] = [item['pred_visible_score'] for item in iter_sparse]
            sparse['iter_pred_confidence_score'] = [item['pred_confidence_score'] for item in iter_sparse]
    del flow_preds
    del visconf_preds
    clear_roma_init(model)
    return sparse, stats


def collect_roma_recurrent_inputs(seq_id, batch, baseline, roma_matcher, args, model, device):
    rgbs = batch.video[0].detach().float().cpu()
    trajs_g = baseline['trajs_g'].float()
    valids = baseline['valids'].float()
    baseline_xy = baseline['trajs_e'].float()
    pred_visible = baseline['pred_visible_score'].float()
    risk_prob = baseline.get('risk_prob', torch.zeros_like(pred_visible)).float()
    risk_prob = torch.nan_to_num(risk_prob, nan=0.0, posinf=0.0, neginf=0.0)
    first_positive = baseline['first_positive_inds'].long()
    T, _, _ = baseline_xy.shape
    _, H, W = rgbs.shape[1:]
    source_t = 0
    point_ids = torch.nonzero(first_positive == source_t, as_tuple=False).reshape(-1).long()
    if point_ids.numel() == 0:
        return None
    if bool((first_positive != source_t).any().item()):
        print(
            'warning: %s has non-zero query frames; recurrent RoMa adapter only uses source=0 anchors.' % str(seq_id),
            flush=True,
        )

    proposals = build_relocalization_proposals(baseline, args, model, rgbs=rgbs)
    if not proposals:
        return None
    proposals_by_frame = {}
    for proposal in proposals:
        proposals_by_frame.setdefault(int(proposal['target_frame']), []).append(proposal)

    rows = {
        'baseline_xy8': [],
        'roma_xy8': [],
        'gt_xy8': [],
        'source_xy': [],
        'target_frame': [],
        'point_index': [],
        'event_frame': [],
        'proposal_source': [],
        'event_type': [],
        'event_score': [],
        'proposal_source_mask': [],
        'baseline_patch_ncc': [],
        'roma_patch_ncc': [],
        'patch_ncc_gap': [],
        'patch_mismatch_score': [],
        'query_patch_texture': [],
        'roma_certainty_event': [],
        'roma_valid': [],
        'roma_certainty': [],
        'prev_baseline_xy8': [],
        'baseline_visible_score': [],
        'baseline_motion_jump8': [],
        'baseline_speed8': [],
        'baseline_accel8': [],
        'baseline_risk_prob': [],
        'baseline_corr_margin': [],
        'baseline_update_norm': [],
        'query_anchor_age_norm': [],
        'source_is_query_anchor': [],
        'normalized_frame_index': [],
        'normalized_window_start_index': [],
    }
    source_points_all = trajs_g[source_t, point_ids, :2]
    point_to_source_row = {int(point_id): i for i, point_id in enumerate(point_ids.tolist())}
    source_image = rgbs[source_t]
    patch_radius = int(getattr(args, 'proposal_patch_radius', 4))
    roma_cert_event_thr = float(getattr(args, 'proposal_roma_cert_event_thr', 0.5))
    query_gray = _image_to_gray(source_image) if patch_radius > 0 else None
    gray_cache = {source_t: query_gray} if query_gray is not None else {}
    for target_t in sorted(proposals_by_frame.keys()):
        frame_proposals = proposals_by_frame[int(target_t)]
        eval_point_ids_list = [
            int(proposal['point_index'])
            for proposal in frame_proposals
            if int(proposal['point_index']) in point_to_source_row and float(valids[int(target_t), int(proposal['point_index'])]) > 0.5
        ]
        if not eval_point_ids_list:
            continue
        eval_point_ids = torch.as_tensor(eval_point_ids_list, dtype=torch.long)
        source_rows = torch.as_tensor([point_to_source_row[int(point_id)] for point_id in eval_point_ids_list], dtype=torch.long)
        source_points = source_points_all[source_rows]
        proposal_lookup = {int(proposal['point_index']): proposal for proposal in frame_proposals}
        target_image = rgbs[target_t]
        try:
            mapping = roma_matcher.map_points(
                source_image,
                target_image,
                source_points,
                sample_mode=args.roma_sample_mode,
                cache_key=(str(seq_id), source_t, int(target_t)) if bool(args.roma_cache_warps) else None,
            )
            mapping = to_cpu_mapping(mapping)
        except Exception as exc:
            if bool(args.roma_fail_fast):
                raise
            print(
                'warning: RoMa match failed for %s source=%d target=%d: %s; marking recurrent samples invalid.'
                % (str(seq_id), source_t, int(target_t), str(exc)),
                flush=True,
            )
            mapping = make_invalid_mapping(int(eval_point_ids.numel()))

        roma_xy = mapping['points1'].float()
        roma_certainty = torch.nan_to_num(mapping['certainty'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        roma_valid = mapping['valid'].bool() & torch.isfinite(roma_xy).all(dim=1)
        if float(args.roma_certainty_thr) >= 0.0:
            roma_valid = roma_valid & (roma_certainty >= float(args.roma_certainty_thr))
        if patch_radius > 0:
            if int(target_t) not in gray_cache:
                gray_cache[int(target_t)] = _image_to_gray(target_image)
            target_gray = gray_cache[int(target_t)]
            baseline_patch_ncc_values = []
            roma_patch_ncc_values = []
            patch_ncc_gap_values = []
            patch_mismatch_values = []
            query_patch_texture_values = []
            roma_certainty_event_values = []
            proposal_source_mask_values = []
            for local_i, point_id in enumerate(eval_point_ids_list):
                proposal = proposal_lookup[int(point_id)]
                qxy = source_points[local_i, :2]
                base_xy_i = baseline_xy[target_t, int(point_id), :2]
                base_ncc, q_texture = patch_ncc(query_gray, target_gray, qxy, base_xy_i, patch_radius)
                if not np.isfinite(base_ncc):
                    base_ncc = float(proposal.get('baseline_patch_ncc', float('nan')))
                if not np.isfinite(q_texture):
                    q_texture = float(proposal.get('query_patch_texture', float('nan')))
                if bool(roma_valid[local_i].item()):
                    roma_ncc, _ = patch_ncc(query_gray, target_gray, qxy, roma_xy[local_i], patch_radius)
                else:
                    roma_ncc = float('nan')
                gap = float(roma_ncc - base_ncc) if np.isfinite(roma_ncc) and np.isfinite(base_ncc) else float('nan')
                mismatch = float(proposal.get('patch_mismatch_score', 0.0))
                if np.isfinite(base_ncc):
                    mismatch = max(mismatch, max(0.0, float(getattr(args, 'proposal_patch_baseline_ncc_thr', 0.15)) - float(base_ncc)))
                roma_cert_event = bool(roma_valid[local_i].item()) and float(roma_certainty[local_i].item()) >= roma_cert_event_thr
                source_mask = int(proposal.get('proposal_source_mask', _proposal_source_bit(int(proposal.get('proposal_source', 0)))))
                if roma_cert_event:
                    source_mask |= _proposal_source_bit(PROPOSAL_SOURCE_ROMA_CERTAINTY)
                baseline_patch_ncc_values.append(float(base_ncc))
                roma_patch_ncc_values.append(float(roma_ncc))
                patch_ncc_gap_values.append(float(gap))
                patch_mismatch_values.append(float(mismatch))
                query_patch_texture_values.append(float(q_texture))
                roma_certainty_event_values.append(float(roma_cert_event))
                proposal_source_mask_values.append(int(source_mask))
        else:
            count_tmp = int(eval_point_ids.numel())
            baseline_patch_ncc_values = [float(proposal_lookup[int(point_id)].get('baseline_patch_ncc', float('nan'))) for point_id in eval_point_ids_list]
            roma_patch_ncc_values = [float('nan')] * count_tmp
            patch_ncc_gap_values = [float('nan')] * count_tmp
            patch_mismatch_values = [float(proposal_lookup[int(point_id)].get('patch_mismatch_score', 0.0)) for point_id in eval_point_ids_list]
            query_patch_texture_values = [float(proposal_lookup[int(point_id)].get('query_patch_texture', float('nan'))) for point_id in eval_point_ids_list]
            roma_certainty_event_values = [float(bool(roma_valid[i].item()) and float(roma_certainty[i].item()) >= roma_cert_event_thr) for i in range(count_tmp)]
            proposal_source_mask_values = [
                int(proposal_lookup[int(point_id)].get('proposal_source_mask', _proposal_source_bit(int(proposal_lookup[int(point_id)].get('proposal_source', 0)))))
                | (_proposal_source_bit(PROPOSAL_SOURCE_ROMA_CERTAINTY) if bool(roma_certainty_event_values[i]) else 0)
                for i, point_id in enumerate(eval_point_ids_list)
            ]
        base_xy = baseline_xy[target_t, eval_point_ids, :2]
        gt_xy = trajs_g[target_t, eval_point_ids, :2]
        if target_t > 0:
            prev_xy = baseline_xy[target_t - 1, eval_point_ids, :2]
            speed8 = torch.linalg.vector_norm((base_xy - prev_xy) / 8.0, dim=1)
        else:
            prev_xy = base_xy
            speed8 = torch.zeros((int(eval_point_ids.numel()),), dtype=torch.float32)
        if target_t > 1:
            prev_speed8 = torch.linalg.vector_norm(
                (baseline_xy[target_t - 1, eval_point_ids, :2] - baseline_xy[target_t - 2, eval_point_ids, :2]) / 8.0,
                dim=1,
            )
            accel8 = torch.abs(speed8 - prev_speed8)
        else:
            accel8 = torch.zeros_like(speed8)

        count = int(eval_point_ids.numel())
        rows['baseline_xy8'].append(base_xy / 8.0)
        rows['roma_xy8'].append(roma_xy / 8.0)
        rows['gt_xy8'].append(gt_xy / 8.0)
        rows['source_xy'].append(source_points)
        rows['target_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['point_index'].append(eval_point_ids.long())
        rows['event_frame'].append(torch.as_tensor([int(proposal_lookup[int(point_id)]['event_frame']) for point_id in eval_point_ids_list], dtype=torch.long))
        rows['proposal_source'].append(torch.as_tensor([int(proposal_lookup[int(point_id)]['proposal_source']) for point_id in eval_point_ids_list], dtype=torch.long))
        rows['event_type'].append(torch.as_tensor([int(proposal_lookup[int(point_id)]['event_type']) for point_id in eval_point_ids_list], dtype=torch.long))
        rows['event_score'].append(torch.as_tensor([float(proposal_lookup[int(point_id)]['event_score']) for point_id in eval_point_ids_list], dtype=torch.float32))
        rows['proposal_source_mask'].append(torch.as_tensor(proposal_source_mask_values, dtype=torch.long))
        rows['baseline_patch_ncc'].append(torch.as_tensor(baseline_patch_ncc_values, dtype=torch.float32))
        rows['roma_patch_ncc'].append(torch.as_tensor(roma_patch_ncc_values, dtype=torch.float32))
        rows['patch_ncc_gap'].append(torch.as_tensor(patch_ncc_gap_values, dtype=torch.float32))
        rows['patch_mismatch_score'].append(torch.as_tensor(patch_mismatch_values, dtype=torch.float32))
        rows['query_patch_texture'].append(torch.as_tensor(query_patch_texture_values, dtype=torch.float32))
        rows['roma_certainty_event'].append(torch.as_tensor(roma_certainty_event_values, dtype=torch.float32))
        rows['roma_valid'].append(roma_valid.float())
        rows['roma_certainty'].append(roma_certainty.float())
        rows['prev_baseline_xy8'].append(prev_xy / 8.0)
        rows['baseline_visible_score'].append(pred_visible[target_t, eval_point_ids].float())
        rows['baseline_motion_jump8'].append(speed8.float())
        rows['baseline_speed8'].append(speed8.float())
        rows['baseline_accel8'].append(accel8.float())
        rows['baseline_risk_prob'].append(risk_prob[target_t, eval_point_ids].float())
        rows['baseline_corr_margin'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_update_norm'].append(torch.zeros((count,), dtype=torch.float32))
        norm_frame = float(target_t / max(T - 1, 1))
        rows['query_anchor_age_norm'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['source_is_query_anchor'].append(torch.ones((count,), dtype=torch.float32))
        rows['normalized_frame_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['normalized_window_start_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))

    if not rows['baseline_xy8']:
        return None
    out = {}
    for key, values in rows.items():
        value = torch.cat(values, dim=0)
        out[key] = value.to(device, non_blocking=True)
    out['image_hw'] = (int(H), int(W))
    out['num_frames'] = int(T)
    out = append_visual_features_to_inputs(
        batch,
        out,
        model,
        args,
        device,
        preserve_grad=bool(getattr(args, 'coarse_visual_grad', False)),
    )
    return out


def _stable_int_seed(value):
    text = str(value)
    seed = 0
    for i, ch in enumerate(text):
        seed = (seed + (i + 1) * ord(ch)) % 2147483647
    return int(seed)


def _target_frame_values(T, args, baseline=None, model=None, rgbs=None):
    frame_start = max(1, int(getattr(args, 'target_frame_start', 1)))
    frame_end_arg = int(getattr(args, 'target_frame_end', 0))
    frame_end = int(T) if frame_end_arg <= 0 else min(int(T), frame_end_arg)
    if frame_start >= frame_end:
        return []
    stride = max(1, int(getattr(args, 'target_frame_stride', 4)))
    include = {
        part.strip().lower()
        for part in str(getattr(args, 'target_frame_include', 'stride,window_start,last')).split(',')
        if part.strip()
    }
    frames = set()
    if 'stride' in include:
        frames.update(range(frame_start, frame_end, stride))
    if 'window_start' in include:
        frames.update(window_start_frames(int(T), int(getattr(args, 'tracklet_len', 12)), stride=stride))
    if 'last' in include and frame_end > frame_start:
        frames.add(frame_end - 1)
    if 'event' in include and baseline is not None and model is not None:
        try:
            for proposal in build_relocalization_proposals(baseline, args, model, rgbs=rgbs):
                target_t = int(proposal.get('target_frame', -1))
                if frame_start <= target_t < frame_end:
                    frames.add(target_t)
        except Exception:
            pass
    return sorted(int(t) for t in frames if frame_start <= int(t) < frame_end and int(t) > 0 and int(t) < int(T))


def collect_roma_target_frame_inputs(seq_id, batch, baseline, roma_matcher, args, model, device):
    """Collect frame-strided point samples for coarse-coordinate regression.

    This path is intentionally not proposal-coupled. RoMa is run once per
    source/target image pair, then points are sampled on the target frame with a
    mix of random supervision and harder baseline-failure supervision.
    """

    rgbs = batch.video[0].detach().float().cpu()
    trajs_g = baseline['trajs_g'].float()
    valids = baseline['valids'].float()
    baseline_xy = baseline['trajs_e'].float()
    pred_visible = baseline['pred_visible_score'].float()
    risk_prob = baseline.get('risk_prob', torch.zeros_like(pred_visible)).float()
    risk_prob = torch.nan_to_num(risk_prob, nan=0.0, posinf=0.0, neginf=0.0)
    first_positive = baseline['first_positive_inds'].long()
    T, _, _ = baseline_xy.shape
    _, H, W = rgbs.shape[1:]
    source_t = 0
    point_ids = torch.nonzero(first_positive == source_t, as_tuple=False).reshape(-1).long()
    if point_ids.numel() == 0:
        return None
    if bool((first_positive != source_t).any().item()):
        print(
            'warning: %s has non-zero query frames; target-frame RoMa coarse export only uses source=0 anchors.' % str(seq_id),
            flush=True,
        )

    target_frames = _target_frame_values(T, args, baseline=baseline, model=model, rgbs=rgbs)
    if not target_frames:
        return None

    rows = {
        'baseline_xy8': [],
        'roma_xy8': [],
        'gt_xy8': [],
        'source_xy': [],
        'target_frame': [],
        'point_index': [],
        'event_frame': [],
        'proposal_source': [],
        'event_type': [],
        'event_score': [],
        'proposal_source_mask': [],
        'baseline_patch_ncc': [],
        'roma_patch_ncc': [],
        'patch_ncc_gap': [],
        'patch_mismatch_score': [],
        'query_patch_texture': [],
        'roma_certainty_event': [],
        'roma_valid': [],
        'roma_certainty': [],
        'prev_baseline_xy8': [],
        'baseline_visible_score': [],
        'baseline_motion_jump8': [],
        'baseline_speed8': [],
        'baseline_accel8': [],
        'baseline_risk_prob': [],
        'baseline_corr_margin': [],
        'baseline_update_norm': [],
        'query_anchor_age_norm': [],
        'source_is_query_anchor': [],
        'normalized_frame_index': [],
        'normalized_window_start_index': [],
    }

    source_points_all = trajs_g[source_t, point_ids, :2]
    source_image = rgbs[source_t]
    patch_radius = int(getattr(args, 'proposal_patch_radius', 4)) if bool(getattr(args, 'target_compute_patch_features', False)) else 0
    roma_cert_event_thr = float(getattr(args, 'proposal_roma_cert_event_thr', 0.5))
    query_gray = _image_to_gray(source_image) if patch_radius > 0 else None
    gray_cache = {source_t: query_gray} if query_gray is not None else {}
    rng = np.random.default_rng(int(getattr(args, 'seed', 123)) + _stable_int_seed(seq_id))
    max_points = int(getattr(args, 'target_points_per_frame', 1024))
    hard_fraction = float(np.clip(float(getattr(args, 'target_hard_fraction', 0.5)), 0.0, 1.0))
    bad_base_px = float(getattr(args, 'bad_baseline_epe_thr', 4.0))
    conf_thr = float(getattr(args, 'candidate_conf_thr', 0.6))
    risk_thr = float(getattr(args, 'target_hard_risk_thr', 0.5))

    for target_t in target_frames:
        valid_mask = (
            (valids[int(target_t), point_ids] > 0.5)
            & torch.isfinite(trajs_g[int(target_t), point_ids, :2]).all(dim=1)
            & torch.isfinite(baseline_xy[int(target_t), point_ids, :2]).all(dim=1)
        )
        candidate_rows = torch.nonzero(valid_mask, as_tuple=False).reshape(-1).long()
        if candidate_rows.numel() == 0:
            continue
        candidate_point_ids = point_ids[candidate_rows]
        base_xy_all = baseline_xy[int(target_t), candidate_point_ids, :2]
        gt_xy_all = trajs_g[int(target_t), candidate_point_ids, :2]
        base_err_px = torch.linalg.vector_norm(base_xy_all - gt_xy_all, dim=1)
        hard_mask = (
            (base_err_px >= bad_base_px)
            | (pred_visible[int(target_t), candidate_point_ids] < conf_thr)
            | (risk_prob[int(target_t), candidate_point_ids] >= risk_thr)
        )

        local_indices = torch.arange(int(candidate_rows.numel()), dtype=torch.long)
        if max_points > 0 and int(local_indices.numel()) > max_points:
            hard_idx = local_indices[hard_mask]
            easy_idx = local_indices[~hard_mask]
            hard_quota = min(int(round(max_points * hard_fraction)), int(hard_idx.numel()))
            easy_quota = max_points - hard_quota
            chosen_parts = []
            if hard_quota > 0:
                chosen_parts.append(torch.as_tensor(rng.choice(hard_idx.numpy(), size=hard_quota, replace=False), dtype=torch.long))
            if easy_quota > 0 and int(easy_idx.numel()) > 0:
                take = min(easy_quota, int(easy_idx.numel()))
                chosen_parts.append(torch.as_tensor(rng.choice(easy_idx.numpy(), size=take, replace=False), dtype=torch.long))
            remaining = max_points - sum(int(part.numel()) for part in chosen_parts)
            if remaining > 0:
                available = np.setdiff1d(local_indices.numpy(), torch.cat(chosen_parts).numpy() if chosen_parts else np.asarray([], dtype=np.int64))
                if available.size > 0:
                    chosen_parts.append(torch.as_tensor(rng.choice(available, size=min(remaining, available.size), replace=False), dtype=torch.long))
            local_indices = torch.sort(torch.cat(chosen_parts, dim=0))[0] if chosen_parts else local_indices[:max_points]

        eval_point_ids = candidate_point_ids[local_indices]
        source_rows = candidate_rows[local_indices]
        source_points = source_points_all[source_rows]
        target_image = rgbs[int(target_t)]
        try:
            mapping = roma_matcher.map_points(
                source_image,
                target_image,
                source_points,
                sample_mode=args.roma_sample_mode,
                cache_key=(str(seq_id), source_t, int(target_t)) if bool(args.roma_cache_warps) else None,
            )
            mapping = to_cpu_mapping(mapping)
        except Exception as exc:
            if bool(args.roma_fail_fast):
                raise
            print(
                'warning: RoMa match failed for %s source=%d target=%d: %s; marking target-frame samples invalid.'
                % (str(seq_id), source_t, int(target_t), str(exc)),
                flush=True,
            )
            mapping = make_invalid_mapping(int(eval_point_ids.numel()))

        roma_xy = mapping['points1'].float()
        roma_certainty = torch.nan_to_num(mapping['certainty'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        roma_valid = mapping['valid'].bool() & torch.isfinite(roma_xy).all(dim=1)
        if float(args.roma_certainty_thr) >= 0.0:
            roma_valid = roma_valid & (roma_certainty >= float(args.roma_certainty_thr))

        base_xy = baseline_xy[int(target_t), eval_point_ids, :2]
        gt_xy = trajs_g[int(target_t), eval_point_ids, :2]
        if int(target_t) > 0:
            prev_xy = baseline_xy[int(target_t) - 1, eval_point_ids, :2]
            speed8 = torch.linalg.vector_norm((base_xy - prev_xy) / 8.0, dim=1)
        else:
            prev_xy = base_xy
            speed8 = torch.zeros((int(eval_point_ids.numel()),), dtype=torch.float32)
        if int(target_t) > 1:
            prev_speed8 = torch.linalg.vector_norm(
                (baseline_xy[int(target_t) - 1, eval_point_ids, :2] - baseline_xy[int(target_t) - 2, eval_point_ids, :2]) / 8.0,
                dim=1,
            )
            accel8 = torch.abs(speed8 - prev_speed8)
        else:
            accel8 = torch.zeros_like(speed8)

        count = int(eval_point_ids.numel())
        base_err_selected = torch.linalg.vector_norm(base_xy - gt_xy, dim=1)
        hard_score = torch.clamp(base_err_selected / max(1.0, bad_base_px), 0.0, 1.0)
        low_vis_score = torch.clamp(1.0 - pred_visible[int(target_t), eval_point_ids].float(), 0.0, 1.0)
        event_score = torch.maximum(hard_score, low_vis_score)
        hard_now = (base_err_selected >= bad_base_px) | (pred_visible[int(target_t), eval_point_ids] < conf_thr) | (risk_prob[int(target_t), eval_point_ids] >= risk_thr)
        source_mask = torch.full((count,), _proposal_source_bit(PROPOSAL_SOURCE_WINDOW), dtype=torch.long)
        source_mask = torch.where(
            hard_now,
            source_mask | int(_proposal_source_bit(PROPOSAL_SOURCE_BASELINE_EVENT)),
            source_mask,
        )
        roma_certainty_event = (roma_valid & (roma_certainty >= roma_cert_event_thr)).float()
        source_mask = torch.where(
            roma_certainty_event > 0.5,
            source_mask | int(_proposal_source_bit(PROPOSAL_SOURCE_ROMA_CERTAINTY)),
            source_mask,
        )

        if patch_radius > 0:
            if int(target_t) not in gray_cache:
                gray_cache[int(target_t)] = _image_to_gray(target_image)
            target_gray = gray_cache[int(target_t)]
            baseline_patch_ncc_values = []
            roma_patch_ncc_values = []
            patch_ncc_gap_values = []
            patch_mismatch_values = []
            query_patch_texture_values = []
            for local_i in range(count):
                qxy = source_points[local_i, :2]
                base_ncc, q_texture = patch_ncc(query_gray, target_gray, qxy, base_xy[local_i], patch_radius)
                roma_ncc = float('nan')
                if bool(roma_valid[local_i].item()):
                    roma_ncc, _ = patch_ncc(query_gray, target_gray, qxy, roma_xy[local_i], patch_radius)
                gap = float(roma_ncc - base_ncc) if np.isfinite(roma_ncc) and np.isfinite(base_ncc) else float('nan')
                mismatch = max(0.0, float(getattr(args, 'proposal_patch_baseline_ncc_thr', 0.15)) - float(base_ncc)) if np.isfinite(base_ncc) else 0.0
                baseline_patch_ncc_values.append(float(base_ncc))
                roma_patch_ncc_values.append(float(roma_ncc))
                patch_ncc_gap_values.append(float(gap))
                patch_mismatch_values.append(float(mismatch))
                query_patch_texture_values.append(float(q_texture))
        else:
            baseline_patch_ncc_values = [float('nan')] * count
            roma_patch_ncc_values = [float('nan')] * count
            patch_ncc_gap_values = [float('nan')] * count
            patch_mismatch_values = [0.0] * count
            query_patch_texture_values = [float('nan')] * count

        norm_frame = float(target_t / max(T - 1, 1))
        rows['baseline_xy8'].append(base_xy / 8.0)
        rows['roma_xy8'].append(roma_xy / 8.0)
        rows['gt_xy8'].append(gt_xy / 8.0)
        rows['source_xy'].append(source_points)
        rows['target_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['point_index'].append(eval_point_ids.long())
        rows['event_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['proposal_source'].append(torch.full((count,), PROPOSAL_SOURCE_WINDOW, dtype=torch.long))
        rows['event_type'].append(torch.full((count,), EVENT_TYPE_WINDOW_START, dtype=torch.long))
        rows['event_score'].append(event_score.float())
        rows['proposal_source_mask'].append(source_mask.long())
        rows['baseline_patch_ncc'].append(torch.as_tensor(baseline_patch_ncc_values, dtype=torch.float32))
        rows['roma_patch_ncc'].append(torch.as_tensor(roma_patch_ncc_values, dtype=torch.float32))
        rows['patch_ncc_gap'].append(torch.as_tensor(patch_ncc_gap_values, dtype=torch.float32))
        rows['patch_mismatch_score'].append(torch.as_tensor(patch_mismatch_values, dtype=torch.float32))
        rows['query_patch_texture'].append(torch.as_tensor(query_patch_texture_values, dtype=torch.float32))
        rows['roma_certainty_event'].append(roma_certainty_event.float())
        rows['roma_valid'].append(roma_valid.float())
        rows['roma_certainty'].append(roma_certainty.float())
        rows['prev_baseline_xy8'].append(prev_xy / 8.0)
        rows['baseline_visible_score'].append(pred_visible[int(target_t), eval_point_ids].float())
        rows['baseline_motion_jump8'].append(speed8.float())
        rows['baseline_speed8'].append(speed8.float())
        rows['baseline_accel8'].append(accel8.float())
        rows['baseline_risk_prob'].append(risk_prob[int(target_t), eval_point_ids].float())
        rows['baseline_corr_margin'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_update_norm'].append(torch.zeros((count,), dtype=torch.float32))
        rows['query_anchor_age_norm'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['source_is_query_anchor'].append(torch.ones((count,), dtype=torch.float32))
        rows['normalized_frame_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['normalized_window_start_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))

    if not rows['baseline_xy8']:
        return None
    out = {}
    for key, values in rows.items():
        out[key] = torch.cat(values, dim=0).to(device, non_blocking=True)
    out['image_hw'] = (int(H), int(W))
    out['num_frames'] = int(T)
    out = append_visual_features_to_inputs(
        batch,
        out,
        model,
        args,
        device,
        preserve_grad=bool(getattr(args, 'coarse_visual_grad', False)),
    )
    return out


def collect_roma_online_coarse_inputs(seq_id, batch, roma_matcher, args, device):
    """Collect RoMa/GT rows without running a baseline trajectory first.

    The returned rows intentionally omit ``baseline_xy8``. The embedded
    AllTracker hook fills baseline/current coarse coordinates from the live
    ``flows8`` state inside each recurrent window, which keeps training aligned
    with the single-forward inference path.
    """

    rgbs = batch.video[0].detach().float().cpu()
    trajs_g = batch.trajs[0].detach().float().cpu()
    vis_g = batch.visibs[0].detach().float().cpu()
    if batch.valids is None:
        valids = torch.ones_like(vis_g)
    else:
        valids = batch.valids[0].detach().float().cpu()
    T, N, _ = trajs_g.shape
    _, H, W = rgbs.shape[1:]
    _, first_positive = torch.max(vis_g, dim=0)
    source_t = 0
    point_ids = torch.nonzero((first_positive == source_t) & (vis_g[source_t] > 0.5) & (valids[source_t] > 0.5), as_tuple=False).reshape(-1).long()
    if point_ids.numel() == 0:
        return None
    if bool((first_positive != source_t).any().item()):
        print(
            'warning: %s has non-zero query frames; online coarse adapter currently uses source=0 anchors.' % str(seq_id),
            flush=True,
        )

    target_frames = _target_frame_values(T, args, baseline=None, model=None, rgbs=rgbs)
    if not target_frames:
        return None

    rows = {
        'roma_xy8': [],
        'gt_xy8': [],
        'source_xy': [],
        'target_frame': [],
        'point_index': [],
        'event_frame': [],
        'proposal_source': [],
        'event_type': [],
        'event_score': [],
        'proposal_source_mask': [],
        'baseline_patch_ncc': [],
        'roma_patch_ncc': [],
        'patch_ncc_gap': [],
        'patch_mismatch_score': [],
        'query_patch_texture': [],
        'roma_certainty_event': [],
        'roma_valid': [],
        'roma_certainty': [],
        'need_only': [],
        'baseline_risk_prob': [],
        'baseline_corr_margin': [],
        'baseline_update_norm': [],
        'query_anchor_age_norm': [],
        'source_is_query_anchor': [],
        'normalized_frame_index': [],
        'normalized_window_start_index': [],
    }

    source_points_all = trajs_g[source_t, point_ids, :2]
    source_image = rgbs[source_t]
    rng = np.random.default_rng(int(getattr(args, 'seed', 123)) + _stable_int_seed(seq_id))
    max_points = int(getattr(args, 'target_points_per_frame', 1024))
    roma_cert_event_thr = float(getattr(args, 'proposal_roma_cert_event_thr', 0.5))
    patch_radius = int(getattr(args, 'proposal_patch_radius', 4)) if bool(getattr(args, 'target_compute_patch_features', False)) else 0
    query_gray = _image_to_gray(source_image) if patch_radius > 0 else None
    gray_cache = {source_t: query_gray} if query_gray is not None else {}

    frame_jobs = []
    need_only_jobs = []
    for target_t in target_frames:
        valid_mask = (
            (valids[int(target_t), point_ids] > 0.5)
            & torch.isfinite(trajs_g[int(target_t), point_ids, :2]).all(dim=1)
            & torch.isfinite(source_points_all).all(dim=1)
        )
        candidate_rows = torch.nonzero(valid_mask, as_tuple=False).reshape(-1).long()
        if candidate_rows.numel() == 0:
            continue
        if max_points > 0 and int(candidate_rows.numel()) > max_points:
            choice = rng.choice(candidate_rows.numpy(), size=max_points, replace=False)
            candidate_rows = torch.sort(torch.as_tensor(choice, dtype=torch.long))[0]

        need_points = int(getattr(args, 'need_uniform_points_per_frame', 0))
        if need_points > 0:
            all_valid_rows = torch.nonzero(valid_mask, as_tuple=False).reshape(-1).long()
            if int(all_valid_rows.numel()) > 0:
                if int(candidate_rows.numel()) > 0:
                    cand_mask = torch.zeros((int(valid_mask.numel()),), dtype=torch.bool)
                    cand_mask[candidate_rows] = True
                    need_rows = all_valid_rows[~cand_mask[all_valid_rows]]
                    if int(need_rows.numel()) == 0:
                        need_rows = all_valid_rows
                else:
                    need_rows = all_valid_rows
                if int(need_rows.numel()) > need_points:
                    choice = rng.choice(need_rows.numpy(), size=need_points, replace=False)
                    need_rows = torch.sort(torch.as_tensor(choice, dtype=torch.long))[0]
                need_only_jobs.append(
                    {
                        'target_t': int(target_t),
                        'eval_point_ids': point_ids[need_rows],
                        'source_points': source_points_all[need_rows],
                    }
                )

        eval_point_ids = point_ids[candidate_rows]
        source_points = source_points_all[candidate_rows]
        target_image = rgbs[int(target_t)]
        frame_jobs.append(
            {
                'target_t': int(target_t),
                'eval_point_ids': eval_point_ids,
                'source_points': source_points,
                'target_image': target_image,
            }
        )

    def _map_single_job(job):
        target_t = int(job['target_t'])
        try:
            mapping = roma_matcher.map_points(
                source_image,
                job['target_image'],
                job['source_points'],
                sample_mode=args.roma_sample_mode,
                cache_key=(str(seq_id), source_t, int(target_t)) if bool(args.roma_cache_warps) else None,
            )
            return to_cpu_mapping(mapping)
        except Exception as exc:
            if bool(args.roma_fail_fast):
                raise
            print(
                'warning: RoMa match failed for %s source=%d target=%d: %s; marking online coarse samples invalid.'
                % (str(seq_id), source_t, int(target_t), str(exc)),
                flush=True,
            )
            return make_invalid_mapping(int(job['eval_point_ids'].numel()))

    def _append_job(job, mapping):
        target_t = int(job['target_t'])
        eval_point_ids = job['eval_point_ids']
        source_points = job['source_points']
        target_image = job['target_image']

        roma_xy = mapping['points1'].float()
        roma_certainty = torch.nan_to_num(mapping['certainty'].float(), nan=0.0, posinf=0.0, neginf=0.0)
        roma_valid = mapping['valid'].bool() & torch.isfinite(roma_xy).all(dim=1)
        if float(args.roma_certainty_thr) >= 0.0:
            roma_valid = roma_valid & (roma_certainty >= float(args.roma_certainty_thr))

        count = int(eval_point_ids.numel())
        gt_xy = trajs_g[int(target_t), eval_point_ids, :2]
        roma_certainty_event = (roma_valid & (roma_certainty >= roma_cert_event_thr)).float()
        source_mask = torch.full((count,), _proposal_source_bit(PROPOSAL_SOURCE_WINDOW), dtype=torch.long)
        source_mask = torch.where(
            roma_certainty_event > 0.5,
            source_mask | int(_proposal_source_bit(PROPOSAL_SOURCE_ROMA_CERTAINTY)),
            source_mask,
        )

        if patch_radius > 0:
            if int(target_t) not in gray_cache:
                gray_cache[int(target_t)] = _image_to_gray(target_image)
            target_gray = gray_cache[int(target_t)]
            roma_patch_ncc_values = []
            query_patch_texture_values = []
            for local_i in range(count):
                if bool(roma_valid[local_i].item()):
                    roma_ncc, q_texture = patch_ncc(query_gray, target_gray, source_points[local_i, :2], roma_xy[local_i], patch_radius)
                else:
                    roma_ncc, q_texture = float('nan'), float('nan')
                roma_patch_ncc_values.append(float(roma_ncc))
                query_patch_texture_values.append(float(q_texture))
            baseline_patch_ncc_values = [float('nan')] * count
            patch_ncc_gap_values = [float('nan')] * count
            patch_mismatch_values = [0.0] * count
        else:
            baseline_patch_ncc_values = [float('nan')] * count
            roma_patch_ncc_values = [float('nan')] * count
            patch_ncc_gap_values = [float('nan')] * count
            patch_mismatch_values = [0.0] * count
            query_patch_texture_values = [float('nan')] * count

        norm_frame = float(target_t / max(T - 1, 1))
        rows['roma_xy8'].append(roma_xy / 8.0)
        rows['gt_xy8'].append(gt_xy / 8.0)
        rows['source_xy'].append(source_points)
        rows['target_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['point_index'].append(eval_point_ids.long())
        rows['event_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['proposal_source'].append(torch.full((count,), PROPOSAL_SOURCE_WINDOW, dtype=torch.long))
        rows['event_type'].append(torch.full((count,), EVENT_TYPE_WINDOW_START, dtype=torch.long))
        rows['event_score'].append(roma_certainty_event.float())
        rows['proposal_source_mask'].append(source_mask.long())
        rows['baseline_patch_ncc'].append(torch.as_tensor(baseline_patch_ncc_values, dtype=torch.float32))
        rows['roma_patch_ncc'].append(torch.as_tensor(roma_patch_ncc_values, dtype=torch.float32))
        rows['patch_ncc_gap'].append(torch.as_tensor(patch_ncc_gap_values, dtype=torch.float32))
        rows['patch_mismatch_score'].append(torch.as_tensor(patch_mismatch_values, dtype=torch.float32))
        rows['query_patch_texture'].append(torch.as_tensor(query_patch_texture_values, dtype=torch.float32))
        rows['roma_certainty_event'].append(roma_certainty_event.float())
        rows['roma_valid'].append(roma_valid.float())
        rows['roma_certainty'].append(roma_certainty.float())
        rows['need_only'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_risk_prob'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_corr_margin'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_update_norm'].append(torch.zeros((count,), dtype=torch.float32))
        rows['query_anchor_age_norm'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['source_is_query_anchor'].append(torch.ones((count,), dtype=torch.float32))
        rows['normalized_frame_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['normalized_window_start_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))

    def _append_need_only_job(job):
        target_t = int(job['target_t'])
        eval_point_ids = job['eval_point_ids']
        source_points = job['source_points']
        count = int(eval_point_ids.numel())
        if count <= 0:
            return
        gt_xy = trajs_g[int(target_t), eval_point_ids, :2]
        norm_frame = float(target_t / max(T - 1, 1))
        rows['roma_xy8'].append(torch.full((count, 2), float('nan'), dtype=torch.float32))
        rows['gt_xy8'].append(gt_xy / 8.0)
        rows['source_xy'].append(source_points)
        rows['target_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['point_index'].append(eval_point_ids.long())
        rows['event_frame'].append(torch.full((count,), int(target_t), dtype=torch.long))
        rows['proposal_source'].append(torch.full((count,), PROPOSAL_SOURCE_WINDOW, dtype=torch.long))
        rows['event_type'].append(torch.full((count,), EVENT_TYPE_WINDOW_START, dtype=torch.long))
        rows['event_score'].append(torch.zeros((count,), dtype=torch.float32))
        rows['proposal_source_mask'].append(torch.full((count,), _proposal_source_bit(PROPOSAL_SOURCE_WINDOW), dtype=torch.long))
        rows['baseline_patch_ncc'].append(torch.full((count,), float('nan'), dtype=torch.float32))
        rows['roma_patch_ncc'].append(torch.full((count,), float('nan'), dtype=torch.float32))
        rows['patch_ncc_gap'].append(torch.full((count,), float('nan'), dtype=torch.float32))
        rows['patch_mismatch_score'].append(torch.zeros((count,), dtype=torch.float32))
        rows['query_patch_texture'].append(torch.full((count,), float('nan'), dtype=torch.float32))
        rows['roma_certainty_event'].append(torch.zeros((count,), dtype=torch.float32))
        rows['roma_valid'].append(torch.zeros((count,), dtype=torch.float32))
        rows['roma_certainty'].append(torch.zeros((count,), dtype=torch.float32))
        rows['need_only'].append(torch.ones((count,), dtype=torch.float32))
        rows['baseline_risk_prob'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_corr_margin'].append(torch.zeros((count,), dtype=torch.float32))
        rows['baseline_update_norm'].append(torch.zeros((count,), dtype=torch.float32))
        rows['query_anchor_age_norm'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['source_is_query_anchor'].append(torch.ones((count,), dtype=torch.float32))
        rows['normalized_frame_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))
        rows['normalized_window_start_index'].append(torch.full((count,), norm_frame, dtype=torch.float32))

    pair_batch_size = max(1, int(getattr(args, 'roma_pair_batch_size', 1)))
    job_start = 0
    while job_start < len(frame_jobs):
        job_chunk = frame_jobs[job_start:job_start + pair_batch_size]
        job_start += pair_batch_size
        mappings = None
        chunk_targets = ','.join(str(int(job['target_t'])) for job in job_chunk)
        chunk_started = time.time()
        if bool(getattr(args, 'roma_log_chunks', False)):
            print(
                'roma match seq=%s targets=%s batch=%d start'
                % (str(seq_id), chunk_targets, len(job_chunk)),
                flush=True,
            )
        if pair_batch_size > 1 and len(job_chunk) > 1 and hasattr(roma_matcher, 'map_points_batch'):
            try:
                cache_keys = [
                    (str(seq_id), source_t, int(job['target_t'])) if bool(args.roma_cache_warps) else None
                    for job in job_chunk
                ]
                mappings = roma_matcher.map_points_batch(
                    source_image,
                    [job['target_image'] for job in job_chunk],
                    [job['source_points'] for job in job_chunk],
                    sample_mode=args.roma_sample_mode,
                    cache_keys=cache_keys,
                )
                mappings = [to_cpu_mapping(mapping) for mapping in mappings]
            except Exception as exc:
                if bool(args.roma_fail_fast):
                    raise
                print(
                    'warning: batched RoMa match failed for %s source=%d targets=%s: %s; falling back to single-pair RoMa.'
                    % (str(seq_id), source_t, ','.join(str(int(job['target_t'])) for job in job_chunk), str(exc)),
                    flush=True,
                )
                mappings = None
        if mappings is None:
            mappings = [_map_single_job(job) for job in job_chunk]
        if bool(getattr(args, 'roma_log_chunks', False)):
            print(
                'roma match seq=%s targets=%s done time=%.1fs'
                % (str(seq_id), chunk_targets, time.time() - chunk_started),
                flush=True,
            )
        for job, mapping in zip(job_chunk, mappings):
            _append_job(job, mapping)

    for job in need_only_jobs:
        _append_need_only_job(job)

    if not rows['roma_xy8']:
        return None
    out = {}
    for key, values in rows.items():
        out[key] = torch.cat(values, dim=0).to(device, non_blocking=True)
    out['image_hw'] = (int(H), int(W))
    out['num_frames'] = int(T)
    return out


def make_sparse_flow8_override(inputs, init_xy8, args):
    T = int(inputs['num_frames'])
    H, W = inputs['image_hw']
    H8, W8 = flow8_grid_shape(H, W)
    device = init_xy8.device
    dtype = init_xy8.dtype
    valid = inputs['roma_valid'].reshape(-1).bool() & torch.isfinite(init_xy8).all(dim=1)
    if not bool(valid.any().item()):
        return None
    source_xy = inputs['source_xy'].float()[valid]
    target_frame = inputs['target_frame'].long()[valid]
    init_xy8_valid = init_xy8[valid]
    source_x8 = torch.clamp(torch.round(source_xy[:, 0] / 8.0).long(), 0, W8 - 1)
    source_y8 = torch.clamp(torch.round(source_xy[:, 1] / 8.0).long(), 0, H8 - 1)
    source_grid_xy8 = torch.stack([source_x8.float(), source_y8.float()], dim=1).to(device=device, dtype=dtype)
    init_flow8 = init_xy8_valid - source_grid_xy8
    flat_idx = target_frame.to(device) * (H8 * W8) + source_y8.to(device) * W8 + source_x8.to(device)
    flat_size = int(T * H8 * W8)
    flat_flow = torch.zeros((flat_size, 2), device=device, dtype=dtype)
    flat_count = torch.zeros((flat_size, 1), device=device, dtype=dtype)
    flat_flow.index_add_(0, flat_idx, init_flow8)
    flat_count.index_add_(0, flat_idx, torch.ones((int(flat_idx.numel()), 1), device=device, dtype=dtype))
    flat_flow = flat_flow / torch.clamp(flat_count, min=1.0)
    mask_flat = flat_count[:, 0] > 0
    flow8 = flat_flow.reshape(T, H8, W8, 2).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    mask8 = mask_flat.reshape(T, H8, W8).unsqueeze(0).contiguous()
    return flow8, mask8
