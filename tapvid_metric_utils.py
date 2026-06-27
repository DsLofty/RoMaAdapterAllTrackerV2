"""Small TAP-Vid metric wrapper used by final-stage V2 diagnostics."""

from __future__ import annotations

import numpy as np
import torch

import utils.misc


def mean_or_nan(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu()
        if values.numel() == 0:
            return float("nan")
        valid = torch.isfinite(values)
        if not bool(valid.any().item()):
            return float("nan")
        return float(values[valid].mean().item())
    arr = np.asarray(values)
    if arr.size == 0:
        return float("nan")
    try:
        arr = arr.astype(np.float32)
    except (TypeError, ValueError):
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def metric_scalar(metrics, key):
    if key not in metrics:
        return float("nan")
    return mean_or_nan(np.asarray(metrics[key], dtype=np.float32))


def tapvid_metrics(trajs_e, pred_visible_score, trajs_g, vis_g, first_positive_inds, image_size):
    T, N, _ = trajs_e.shape
    if N == 0:
        return {"da": float("nan"), "aj": float("nan"), "oa": float("nan")}
    point_ids = torch.arange(N, dtype=torch.long)
    first = first_positive_inds.long().clamp(0, T - 1)
    pts = trajs_g[first, point_ids, :2]
    query_points = torch.cat([first.float()[:, None], pts[:, [1, 0]]], dim=1)[None]
    gt_occluded = (vis_g < 0.5).bool().T[None]
    gt_tracks = trajs_g.permute(1, 0, 2)[None]
    pred_occluded = (pred_visible_score < 0.6).bool().T[None]
    pred_tracks = trajs_e.permute(1, 0, 2)[None]
    metrics = utils.misc.compute_tapvid_metrics(
        query_points=query_points.cpu().numpy(),
        gt_occluded=gt_occluded.cpu().numpy(),
        gt_tracks=gt_tracks.cpu().numpy(),
        pred_occluded=pred_occluded.cpu().numpy(),
        pred_tracks=pred_tracks.cpu().numpy(),
        query_mode="first",
        crop_size=image_size,
    )
    return {
        "da": metric_scalar(metrics, "average_pts_within_thresh"),
        "aj": metric_scalar(metrics, "average_jaccard"),
        "oa": metric_scalar(metrics, "occlusion_accuracy"),
    }
