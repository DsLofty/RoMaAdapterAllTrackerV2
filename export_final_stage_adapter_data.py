"""
Export cached data for a final-stage temporal RoMa selector.

The exported rows are built after the final AllTracker/reference trajectory is
available. The first experiment is deliberately low-dimensional: it learns when
to select RoMa final coordinates over the reference final coordinates.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import utils.data
from alltracker_runtime_utils import dense_baseline_forward, expand_path, extract_feat8_sequence, load_alltracker
from tapvid_dataset_utils import safe_seq_id
from eval_roma_final_stage_heuristic import (
    build_coarse_gate_map,
    build_roma_maps,
    error_map,
    eval_mask,
    motion_jump_px,
    temporal_offset_change_px,
    unpack_batch,
)
from final_stage_adapter_model import FINAL_STAGE_FEATURE_NAMES
from matchers.roma_wrapper import RoMaImportError, RoMaMatcher
from utils.roma_branch_relocalization import collect_roma_online_coarse_inputs, make_loader, run_alltracker_embedded_adapter


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_feature(x, value=0.0):
    return torch.nan_to_num(x.float(), nan=float(value), posinf=float(value), neginf=float(value))


def images_to_gray(batch, device):
    images = batch.video[0].detach().float().to(device)
    if float(images.max().detach().item()) > 2.0:
        images = images / 255.0
    images = torch.clamp(images, 0.0, 1.0)
    if int(images.shape[1]) == 1:
        return images[:, 0]
    return 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]


def sample_patches_from_frame(gray_t, xy, radius):
    n = int(xy.shape[0])
    k = int(radius) * 2 + 1
    if n <= 0:
        return torch.empty((0, k, k), dtype=gray_t.dtype, device=gray_t.device)
    h, w = int(gray_t.shape[0]), int(gray_t.shape[1])
    coords = torch.nan_to_num(xy.float(), nan=-1e6, posinf=-1e6, neginf=-1e6)
    yy, xx = torch.meshgrid(
        torch.arange(-int(radius), int(radius) + 1, device=gray_t.device),
        torch.arange(-int(radius), int(radius) + 1, device=gray_t.device),
        indexing='ij',
    )
    offsets = torch.stack([xx, yy], dim=-1).float()
    grid_xy = coords[:, None, None, :] + offsets[None]
    gx = (grid_xy[..., 0] / max(w - 1, 1)) * 2.0 - 1.0
    gy = (grid_xy[..., 1] / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, n * k, k, 2)
    sampled = F.grid_sample(
        gray_t.reshape(1, 1, h, w),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return sampled.reshape(n, k, k)


def patch_ncc(a, b, eps=1e-6):
    af = a.reshape(a.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    af = af - af.mean(dim=1, keepdim=True)
    bf = bf - bf.mean(dim=1, keepdim=True)
    denom = torch.linalg.vector_norm(af, dim=1) * torch.linalg.vector_norm(bf, dim=1)
    return (af * bf).sum(dim=1) / torch.clamp(denom, min=float(eps))


def sample_query_patches(gray, first, query_xy, radius):
    n = int(query_xy.shape[0])
    k = int(radius) * 2 + 1
    out = torch.zeros((n, k, k), dtype=gray.dtype, device=gray.device)
    for source_t in torch.unique(first):
        source_t = int(source_t.item())
        ids = torch.nonzero(first == source_t, as_tuple=False).reshape(-1)
        if ids.numel() == 0 or source_t < 0 or source_t >= int(gray.shape[0]):
            continue
        out[ids] = sample_patches_from_frame(gray[source_t], query_xy[ids], radius)
    return out


def local_ncc_peak(gray_t, query_patches, roma_xy, patch_radius, search_radius):
    n = int(roma_xy.shape[0])
    if n <= 0 or int(search_radius) <= 0:
        zeros = torch.zeros((n,), dtype=gray_t.dtype, device=gray_t.device)
        return zeros, zeros, zeros
    offsets = []
    for dy in range(-int(search_radius), int(search_radius) + 1):
        for dx in range(-int(search_radius), int(search_radius) + 1):
            offsets.append((float(dx), float(dy)))
    scores = []
    distances = []
    for dx, dy in offsets:
        offset_xy = roma_xy + torch.tensor([dx, dy], dtype=roma_xy.dtype, device=roma_xy.device).view(1, 2)
        patches = sample_patches_from_frame(gray_t, offset_xy, patch_radius)
        scores.append(patch_ncc(query_patches, patches))
        distances.append((dx * dx + dy * dy) ** 0.5)
    score = torch.stack(scores, dim=1)
    sorted_score, sorted_idx = torch.sort(score, dim=1, descending=True)
    peak = sorted_score[:, 0]
    margin = sorted_score[:, 0] - sorted_score[:, 1] if score.shape[1] > 1 else torch.zeros_like(peak)
    dist_values = torch.tensor(distances, dtype=gray_t.dtype, device=gray_t.device)
    peak_offset = dist_values[sorted_idx[:, 0]]
    return peak, margin, peak_offset


def patch_save_dtype(args):
    dtype_name = str(getattr(args, 'patch_tensor_dtype', 'float16')).lower()
    if dtype_name in ('fp32', 'float32'):
        return torch.float32
    if dtype_name in ('fp16', 'float16', 'half'):
        return torch.float16
    raise ValueError('unsupported --patch_tensor_dtype: %s' % dtype_name)


def compute_patch_stats(batch, reference, roma_xy, roma_valid, args, device):
    base_xy = reference['trajs_e'].float().to(device)
    gt = reference['trajs_g'].float().to(device)
    first = reference['first_positive_inds'].long().to(device)
    T, N, _ = base_xy.shape
    gray = images_to_gray(batch, device)
    patch_radius = int(getattr(args, 'final_patch_radius', 4))
    search_radius = int(getattr(args, 'final_patch_search_radius', 2))
    query_xy = gt[first, torch.arange(N, device=device), :2]
    query_patches = sample_query_patches(gray, first, query_xy, patch_radius)
    query_texture_per_point = query_patches.reshape(N, -1).std(dim=1)
    save_patches = bool(getattr(args, 'save_patch_tensors', False))
    patch_tensors = None
    baseline_patch_list = []
    roma_patch_list = []
    if save_patches:
        out_dtype = patch_save_dtype(args)
        patch_tensors = {
            'query_gray': query_patches.detach().to(out_dtype).cpu(),
            'patch_radius': int(patch_radius),
            'search_radius': int(search_radius),
            'dtype': str(out_dtype).replace('torch.', ''),
        }

    zeros = torch.zeros((T, N), dtype=torch.float32, device=device)
    stats = {
        'query_patch_texture': query_texture_per_point.reshape(1, N).expand(T, N).clone(),
        'baseline_patch_texture': zeros.clone(),
        'roma_patch_texture': zeros.clone(),
        'baseline_patch_ncc': zeros.clone(),
        'roma_patch_ncc': zeros.clone(),
        'patch_ncc_gap': zeros.clone(),
        'roma_local_ncc_peak': zeros.clone(),
        'roma_local_ncc_margin': zeros.clone(),
        'roma_local_peak_offset_px': zeros.clone(),
    }
    for t in range(T):
        base_patches = sample_patches_from_frame(gray[t], base_xy[t], patch_radius)
        roma_patches = sample_patches_from_frame(gray[t], roma_xy[t], patch_radius)
        if save_patches:
            baseline_patch_list.append(base_patches.detach().to(out_dtype).cpu())
            roma_patch_list.append(roma_patches.detach().to(out_dtype).cpu())
        base_ncc = patch_ncc(query_patches, base_patches)
        roma_ncc = patch_ncc(query_patches, roma_patches)
        base_tex = base_patches.reshape(N, -1).std(dim=1)
        roma_tex = roma_patches.reshape(N, -1).std(dim=1)
        if search_radius > 0:
            peak, margin, peak_offset = local_ncc_peak(gray[t], query_patches, roma_xy[t], patch_radius, search_radius)
        else:
            peak = roma_ncc
            margin = torch.zeros_like(roma_ncc)
            peak_offset = torch.zeros_like(roma_ncc)
        invalid = ~roma_valid[t]
        roma_ncc = roma_ncc.masked_fill(invalid, 0.0)
        roma_tex = roma_tex.masked_fill(invalid, 0.0)
        peak = peak.masked_fill(invalid, 0.0)
        margin = margin.masked_fill(invalid, 0.0)
        peak_offset = peak_offset.masked_fill(invalid, 0.0)
        stats['baseline_patch_texture'][t] = base_tex
        stats['roma_patch_texture'][t] = roma_tex
        stats['baseline_patch_ncc'][t] = base_ncc
        stats['roma_patch_ncc'][t] = roma_ncc
        stats['patch_ncc_gap'][t] = roma_ncc - base_ncc
        stats['roma_local_ncc_peak'][t] = peak
        stats['roma_local_ncc_margin'][t] = margin
        stats['roma_local_peak_offset_px'][t] = peak_offset
    if save_patches:
        patch_tensors['baseline_gray'] = torch.stack(baseline_patch_list, dim=0)
        patch_tensors['roma_gray'] = torch.stack(roma_patch_list, dim=0)
    return stats, patch_tensors


def deep_corr_save_dtype(args):
    dtype_name = str(getattr(args, 'deep_corr_dtype', 'float16')).lower()
    if dtype_name in ('fp32', 'float32'):
        return torch.float32
    if dtype_name in ('fp16', 'float16', 'half'):
        return torch.float16
    raise ValueError('unsupported --deep_corr_dtype: %s' % dtype_name)


def sample_feat_points(feat_t, xy8):
    xy8 = xy8.to(device=feat_t.device)
    n = int(xy8.shape[0])
    c, h, w = int(feat_t.shape[0]), int(feat_t.shape[1]), int(feat_t.shape[2])
    if n <= 0:
        return torch.empty((0, c), dtype=feat_t.dtype, device=feat_t.device)
    coords = torch.nan_to_num(xy8.float(), nan=-1e6, posinf=-1e6, neginf=-1e6)
    gx = (coords[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
    gy = (coords[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).reshape(1, n, 1, 2)
    sampled = F.grid_sample(
        feat_t.reshape(1, c, h, w),
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
    return sampled.reshape(c, n).transpose(0, 1).contiguous()


def deep_offsets(radius, device):
    offsets = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            offsets.append((float(dx), float(dy)))
    return torch.tensor(offsets, dtype=torch.float32, device=device)


def sample_deep_corr_grid(feat_t, query_feat, center_xy8, offsets8):
    center_xy8 = center_xy8.to(device=feat_t.device)
    offsets8 = offsets8.to(device=feat_t.device)
    query_feat = query_feat.to(device=feat_t.device)
    n = int(center_xy8.shape[0])
    k = int(offsets8.shape[0])
    coords = center_xy8[:, None, :].float() + offsets8[None, :, :]
    local = sample_feat_points(feat_t, coords.reshape(n * k, 2)).reshape(n, k, -1)
    local = F.normalize(local.float(), dim=-1)
    q = F.normalize(query_feat.float(), dim=-1)
    return (local * q[:, None, :]).sum(dim=-1)


def compute_deep_corr_features(batch, reference, roma_xy, roma_valid, candidate, model, args, device):
    if not bool(getattr(args, 'save_deep_corr_features', False)):
        return None
    base_xy = reference['trajs_e'].float().to(device)
    gt = reference['trajs_g'].float().to(device)
    first = reference['first_positive_inds'].long().to(device)
    T, N, _ = base_xy.shape
    radius = int(getattr(args, 'deep_corr_radius8', 2))
    offsets8 = deep_offsets(radius, device)
    K = int(offsets8.shape[0])

    feat8 = extract_feat8_sequence(model=model, rgbs=batch.video, device=device)
    feat8 = F.normalize(feat8.float(), dim=1)
    query_xy8 = gt[first, torch.arange(N, device=device), :2] / 8.0
    query_feat = torch.zeros((N, int(feat8.shape[1])), dtype=torch.float32, device=device)
    for source_t in torch.unique(first):
        source_t = int(source_t.item())
        ids = torch.nonzero(first == source_t, as_tuple=False).reshape(-1)
        if ids.numel() == 0 or source_t < 0 or source_t >= T:
            continue
        query_feat[ids] = sample_feat_points(feat8[source_t], query_xy8[ids])
    query_feat = F.normalize(query_feat.float(), dim=-1)

    roma_corr = torch.zeros((T, N, K), dtype=torch.float32, device=device)
    baseline_corr = torch.zeros((T, N, K), dtype=torch.float32, device=device)
    center_index = int((K - 1) // 2)
    for t in range(T):
        roma_corr[t] = sample_deep_corr_grid(feat8[t], query_feat, roma_xy[t] / 8.0, offsets8)
        baseline_corr[t] = sample_deep_corr_grid(feat8[t], query_feat, base_xy[t] / 8.0, offsets8)
        roma_corr[t] = roma_corr[t].masked_fill(~roma_valid[t].reshape(N, 1), 0.0)

    offset8_float = (gt[..., :2] - roma_xy[..., :2]) / 8.0
    offset8_round = torch.round(offset8_float).long()
    offset_valid = (
        candidate
        & torch.isfinite(offset8_float).all(dim=-1)
        & (offset8_float[..., 0].abs() <= float(radius))
        & (offset8_float[..., 1].abs() <= float(radius))
    )
    dx = torch.clamp(offset8_round[..., 0], min=-radius, max=radius)
    dy = torch.clamp(offset8_round[..., 1], min=-radius, max=radius)
    offset_class = ((dy + radius) * (2 * radius + 1) + (dx + radius)).long()
    offset_class = offset_class.masked_fill(~offset_valid, -1)

    out_dtype = deep_corr_save_dtype(args)
    roma_center = roma_corr[..., center_index]
    base_center = baseline_corr[..., center_index]
    peak, peak_idx = torch.max(roma_corr, dim=-1)
    sorted_corr, _ = torch.sort(roma_corr, dim=-1, descending=True)
    margin = sorted_corr[..., 0] - sorted_corr[..., 1] if K > 1 else torch.zeros_like(peak)
    peak_offset8 = offsets8[peak_idx]
    return {
        'roma_corr_grid': roma_corr.detach().to(out_dtype).cpu(),
        'baseline_corr_grid': baseline_corr.detach().to(out_dtype).cpu(),
        'offset_class': offset_class.detach().cpu(),
        'offset_xy_px': (gt[..., :2] - roma_xy[..., :2]).detach().to(out_dtype).cpu(),
        'offset_valid': offset_valid.detach().cpu(),
        'grid_offsets8': offsets8.detach().cpu(),
        'radius8': int(radius),
        'stride': 8,
        'dtype': str(out_dtype).replace('torch.', ''),
        'roma_center_corr': roma_center.detach().to(out_dtype).cpu(),
        'baseline_center_corr': base_center.detach().to(out_dtype).cpu(),
        'roma_corr_peak': peak.detach().to(out_dtype).cpu(),
        'roma_corr_margin': margin.detach().to(out_dtype).cpu(),
        'roma_corr_peak_offset8': peak_offset8.detach().to(out_dtype).cpu(),
    }


def build_features(reference, roma_xy, roma_cert, roma_valid, coarse_gate, patch_stats, args, device):
    base_xy = reference['trajs_e'].float().to(device)
    visible = reference['pred_visible_score'].float().to(device)
    T, N, _ = base_xy.shape
    first = reference['first_positive_inds'].long().to(device).reshape(1, N).expand(T, N)
    frame_ids = torch.arange(T, device=device).reshape(T, 1).expand(T, N)

    dist = torch.linalg.vector_norm(roma_xy - base_xy, dim=2)
    jump = motion_jump_px(base_xy)
    prev_dist = torch.full((T, N), float('inf'), dtype=torch.float32, device=device)
    if T > 1:
        prev_dist[1:] = torch.linalg.vector_norm(roma_xy[1:] - base_xy[:-1], dim=2)
    offset_change = temporal_offset_change_px(base_xy, roma_xy, roma_valid, args.target_frame_stride)
    offset = roma_xy - base_xy
    step = torch.zeros_like(base_xy)
    if T > 1:
        step[1:] = base_xy[1:] - base_xy[:-1]
    coarse_prev = torch.zeros_like(coarse_gate, dtype=torch.bool)
    coarse_next = torch.zeros_like(coarse_gate, dtype=torch.bool)
    if T > 1:
        coarse_prev[1:] = coarse_gate[:-1]
        coarse_next[:-1] = coarse_gate[1:]
    frame_after_query = (frame_ids - first).float() / float(max(T - 1, 1))
    frame_after_query = torch.clamp(frame_after_query, min=0.0, max=1.0)

    values = {
        'roma_valid': roma_valid.float(),
        'roma_certainty': roma_cert,
        'baseline_visible': visible,
        'baseline_to_roma_dist_px': dist,
        'roma_to_prev_baseline_dist_px': prev_dist,
        'offset_change_px': offset_change,
        'baseline_jump_px': jump,
        'offset_x_px': offset[..., 0],
        'offset_y_px': offset[..., 1],
        'baseline_step_x_px': step[..., 0],
        'baseline_step_y_px': step[..., 1],
        'frame_after_query_norm': frame_after_query,
        'coarse_gate': coarse_gate.float(),
        'coarse_gate_prev': coarse_prev.float(),
        'coarse_gate_next': coarse_next.float(),
    }
    values.update(patch_stats)
    features = [finite_feature(values[name], value=0.0) for name in FINAL_STAGE_FEATURE_NAMES]
    return torch.stack(features, dim=-1)


def build_baseline_risk_labels(reference, base_err, mask, args, device):
    base_xy = reference['trajs_e'].float().to(device)
    visible = reference['pred_visible_score'].float().to(device)
    T, N = base_err.shape
    jump = motion_jump_px(base_xy)
    err_delta = torch.zeros((T, N), dtype=torch.float32, device=device)
    if T > 1:
        err_delta[1:] = base_err[1:] - base_err[:-1]

    finite = mask & torch.isfinite(base_err)
    pos_err = float(args.risk_positive_err_px)
    pos_delta = float(args.risk_err_increase_px)
    pos_delta_min_err = float(args.risk_min_err_for_increase_px)
    pos_jump = float(args.risk_jump_px)
    pos_jump_min_err = float(args.risk_jump_min_err_px)
    neg_err = float(args.risk_negative_err_px)
    neg_jump = float(args.risk_negative_jump_px)
    neg_vis = float(args.risk_negative_visible_thr)

    positive = finite & (
        (base_err >= pos_err)
        | ((err_delta >= pos_delta) & (base_err >= pos_delta_min_err))
        | ((jump >= pos_jump) & (base_err >= pos_jump_min_err))
    )
    negative = finite & (base_err <= neg_err) & (jump <= neg_jump) & (visible >= neg_vis)
    negative = negative & (~positive)

    labels = torch.full((T, N), -1, dtype=torch.long, device=device)
    labels[positive] = 1
    labels[negative] = 0
    meta = {
        'risk_positive_err_px': pos_err,
        'risk_err_increase_px': pos_delta,
        'risk_min_err_for_increase_px': pos_delta_min_err,
        'risk_jump_px': pos_jump,
        'risk_jump_min_err_px': pos_jump_min_err,
        'risk_negative_err_px': neg_err,
        'risk_negative_jump_px': neg_jump,
        'risk_negative_visible_thr': neg_vis,
    }
    return labels, err_delta, jump, meta


def build_roma_accept_labels(risk_labels, candidate, base_err, roma_err, args, device):
    T, N = base_err.shape
    labels = torch.full((T, N), -1, dtype=torch.long, device=device)
    finite = candidate & torch.isfinite(base_err) & torch.isfinite(roma_err)
    if bool(args.accept_require_risk_positive):
        finite = finite & (risk_labels == 1)

    pos_margin = float(args.accept_positive_margin_px)
    neg_margin = float(args.accept_negative_margin_px)
    positive = finite & ((roma_err + pos_margin) < base_err)
    negative = finite & ((base_err + neg_margin) < roma_err)
    if bool(args.accept_tie_as_negative):
        negative = finite & (~positive)
    labels[positive] = 1
    labels[negative] = 0
    meta = {
        'accept_positive_margin_px': pos_margin,
        'accept_negative_margin_px': neg_margin,
        'accept_require_risk_positive': bool(args.accept_require_risk_positive),
        'accept_tie_as_negative': bool(args.accept_tie_as_negative),
    }
    return labels, meta


def export_sequence(seq_id, batch, model, roma_matcher, args, device, adapter=None, feature_names=None):
    alltracker = dense_baseline_forward(batch, model, args, device=device)
    T, N, _ = alltracker['trajs_e'].shape
    inputs = collect_roma_online_coarse_inputs(seq_id, batch, roma_matcher, args, device)
    roma_xy, roma_cert, roma_valid = build_roma_maps(inputs, T, N, device)

    reference = alltracker
    reference_source = 'alltracker'
    coarse_gate = torch.zeros((T, N), dtype=torch.bool, device=device)
    has_coarse_gate = False
    if adapter is not None:
        adapter_ref, embedded_stats = run_alltracker_embedded_adapter(
            batch,
            model,
            adapter,
            inputs,
            feature_names,
            args,
            device,
            preserve_grad=False,
        )
        reference = dict(adapter_ref)
        for key in ('trajs_g', 'vis_g', 'valids', 'first_positive_inds'):
            if key not in reference and key in alltracker:
                reference[key] = alltracker[key]
        reference_source = 'adapter'
        coarse_gate, has_coarse_gate = build_coarse_gate_map(
            embedded_stats.get('supervision', {}) if embedded_stats else {},
            T,
            N,
            device,
        )

    gt = reference['trajs_g'].float().to(device)
    mask = eval_mask(reference, visible_only=bool(args.analysis_visible_only)).to(device)
    base_xy = reference['trajs_e'].float().to(device)
    base_err = error_map(base_xy, gt)
    roma_err = error_map(roma_xy, gt)
    alltracker_err = error_map(alltracker['trajs_e'].float().to(device), gt)
    candidate = mask & roma_valid & torch.isfinite(roma_err) & torch.isfinite(base_err)

    labels = torch.full((T, N), -1, dtype=torch.long, device=device)
    positive = candidate & ((roma_err + float(args.positive_margin_px)) < base_err)
    negative = candidate & ((base_err + float(args.negative_margin_px)) < roma_err)
    if bool(args.tie_as_negative):
        negative = candidate & (~positive)
    labels[positive] = 1
    labels[negative] = 0
    risk_labels, baseline_err_delta, baseline_jump, risk_meta = build_baseline_risk_labels(
        reference,
        base_err,
        mask,
        args,
        device,
    )
    accept_labels, accept_meta = build_roma_accept_labels(
        risk_labels,
        candidate,
        base_err,
        roma_err,
        args,
        device,
    )

    patch_stats, patch_tensors = compute_patch_stats(batch, reference, roma_xy, roma_valid, args, device)
    features = build_features(reference, roma_xy, roma_cert, roma_valid, coarse_gate, patch_stats, args, device)
    sample = {
        'seq_id': str(seq_id),
        'reference_source': reference_source,
        'feature_names': list(FINAL_STAGE_FEATURE_NAMES),
        'features': features.detach().cpu(),
        'labels': labels.detach().cpu(),
        'baseline_risk_labels': risk_labels.detach().cpu(),
        'baseline_risk_meta': dict(risk_meta),
        'roma_accept_labels': accept_labels.detach().cpu(),
        'roma_accept_meta': dict(accept_meta),
        'mask': mask.detach().cpu(),
        'candidate_mask': candidate.detach().cpu(),
        'coarse_gate': coarse_gate.detach().cpu(),
        'has_coarse_gate': bool(has_coarse_gate),
        'reference_xy': base_xy.detach().cpu(),
        'roma_xy': roma_xy.detach().cpu(),
        'gt_xy': gt.detach().cpu(),
        'vis_g': reference['vis_g'].float().detach().cpu(),
        'valids': reference.get('valids', torch.ones_like(reference['vis_g'])).float().detach().cpu(),
        'first_positive_inds': reference['first_positive_inds'].long().detach().cpu(),
        'pred_visible_score': reference['pred_visible_score'].float().detach().cpu(),
        'alltracker_xy': alltracker['trajs_e'].float().detach().cpu(),
        'base_err': base_err.detach().cpu(),
        'baseline_err_delta': baseline_err_delta.detach().cpu(),
        'baseline_jump_px': baseline_jump.detach().cpu(),
        'roma_err': roma_err.detach().cpu(),
        'alltracker_err': alltracker_err.detach().cpu(),
        'image_size': tuple(int(v) for v in args.image_size),
    }
    if patch_tensors is not None:
        sample['patch_tensors'] = patch_tensors
    deep_corr = compute_deep_corr_features(batch, reference, roma_xy, roma_valid, candidate, model, args, device)
    if deep_corr is not None:
        sample['deep_corr'] = deep_corr
    return sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', choices=['kubric', 'dav', 'kin', 'rgb', 'rob'], default='dav')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--only_first', action='store_true', default=True)
    parser.add_argument('--image_size', type=int, nargs='+', default=[448, 768])
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/alltracker.pth')
    parser.add_argument('--adapter_ckpt', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./final_stage_adapter_data')
    parser.add_argument('--exp', type=str, default='dav_best1500_final_stage')
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--inference_iters', type=int, default=4)
    parser.add_argument('--analysis_visible_only', action='store_true', default=True)
    parser.add_argument('--positive_margin_px', type=float, default=0.5)
    parser.add_argument('--negative_margin_px', type=float, default=0.5)
    parser.add_argument('--tie_as_negative', action='store_true', default=False)
    parser.add_argument('--risk_positive_err_px', type=float, default=4.0)
    parser.add_argument('--risk_err_increase_px', type=float, default=3.0)
    parser.add_argument('--risk_min_err_for_increase_px', type=float, default=2.0)
    parser.add_argument('--risk_jump_px', type=float, default=16.0)
    parser.add_argument('--risk_jump_min_err_px', type=float, default=2.5)
    parser.add_argument('--risk_negative_err_px', type=float, default=1.5)
    parser.add_argument('--risk_negative_jump_px', type=float, default=4.0)
    parser.add_argument('--risk_negative_visible_thr', type=float, default=0.6)
    parser.add_argument('--accept_positive_margin_px', type=float, default=1.0)
    parser.add_argument('--accept_negative_margin_px', type=float, default=1.0)
    parser.add_argument('--accept_require_risk_positive', action='store_true', default=True)
    parser.add_argument('--no_accept_require_risk_positive', action='store_false', dest='accept_require_risk_positive')
    parser.add_argument('--accept_tie_as_negative', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--final_patch_radius', type=int, default=4)
    parser.add_argument('--final_patch_search_radius', type=int, default=2)
    parser.add_argument('--save_patch_tensors', action='store_true', default=False)
    parser.add_argument('--patch_tensor_dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--save_deep_corr_features', action='store_true', default=False)
    parser.add_argument('--deep_corr_radius8', type=int, default=2)
    parser.add_argument('--deep_corr_dtype', choices=['float16', 'float32'], default='float16')

    parser.add_argument('--roma_model', choices=['outdoor', 'indoor', 'tiny_outdoor'], default='outdoor')
    parser.add_argument('--roma_device', type=str, default='cuda')
    parser.add_argument('--roma_input_size', type=int, nargs='*', default=None)
    parser.add_argument('--roma_sample_mode', choices=['nearest', 'bilinear'], default='bilinear')
    parser.add_argument('--roma_certainty_thr', type=float, default=-1.0)
    parser.add_argument('--roma_allow_online_download', action='store_true', default=False)
    parser.add_argument('--roma_cache_warps', action='store_true', default=False)
    parser.add_argument('--roma_disable_custom_corr', action='store_true', default=False)
    parser.add_argument('--roma_allow_slow_corr', action='store_true', default=False)
    parser.add_argument('--roma_pair_batch_size', type=int, default=1)
    parser.add_argument('--roma_log_chunks', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--deterministic_sampling', action='store_true', default=True)
    parser.add_argument('--no_deterministic_sampling', action='store_false', dest='deterministic_sampling')

    parser.add_argument('--target_frame_stride', type=int, default=2)
    parser.add_argument('--target_frame_include', type=str, default='stride,last')
    parser.add_argument('--target_points_per_frame', type=int, default=4096)
    parser.add_argument('--need_uniform_points_per_frame', type=int, default=0)
    parser.add_argument('--sequence_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--traj_per_sample', type=int, default=768)
    parser.add_argument('--use_augs', action='store_true', default=False)
    parser.add_argument('--query_rich_crop', '--kubric_query_rich_crop', action='store_true', default=False)
    parser.add_argument('--query_rich_topk', '--kubric_query_rich_topk', type=int, default=8)
    parser.add_argument('--adapter_input_timing', choices=['post_baseline', 'online_coarse'], default='online_coarse')
    parser.add_argument('--embedded_adapter_forward', action='store_true', default=True)
    parser.add_argument('--gate_mode', type=str, default='none')
    parser.add_argument('--coord_mode', type=str, default='two_stage_st_fusion_residual')
    parser.add_argument('--init_mode', type=str, default='coarse_xy')
    parser.add_argument('--relocalize_policy', choices=['learned', 'heuristic', 'learned_accept', 'heuristic_learned_accept'], default='heuristic')
    parser.add_argument('--embedded_max_frames_per_window', type=int, default=24)
    parser.add_argument('--embedded_max_rows_per_frame', type=int, default=64)
    parser.add_argument('--roma_init_apply_at', type=str, default='query_group_start')
    parser.add_argument('--heuristic_roma_certainty_thr', type=float, default=0.75)
    parser.add_argument('--heuristic_baseline_visible_thr', type=float, default=0.5)
    parser.add_argument('--heuristic_baseline_motion_jump8_thr', type=float, default=1.5)
    parser.add_argument('--heuristic_visual_gain_thr', type=float, default=0.05)
    parser.add_argument('--heuristic_visual_cos_thr', type=float, default=0.5)
    parser.add_argument('--heuristic_strong_visual_gain_thr', type=float, default=0.15)
    parser.add_argument('--heuristic_roma_prev_dist8_max', type=float, default=8.0)
    parser.add_argument('--heuristic_min_offset8', type=float, default=0.5)
    parser.add_argument('--safety_gate', action='store_true', default=False)
    parser.add_argument('--coarse_visual_features', action='store_true', default=False)
    parser.add_argument('--local_corr_supplement', action='store_true', default=False)
    parser.add_argument('--carry_mode', type=str, default='none')

    args = parser.parse_args()
    if len(args.image_size) == 1:
        args.image_size = [int(args.image_size[0]), int(args.image_size[0])]
    else:
        args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
    if args.roma_input_size is not None and len(args.roma_input_size) == 0:
        args.roma_input_size = None
    elif args.roma_input_size is not None:
        args.roma_input_size = [int(v) for v in args.roma_input_size]
    return args


def main():
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = os.path.join(expand_path(args.save_dir), str(args.exp))
    seq_dir = os.path.join(out_dir, 'sequences')
    os.makedirs(seq_dir, exist_ok=True)
    loader, dataset_root = make_loader(args, shuffle=bool(args.shuffle))
    max_sequences = len(loader) if int(args.max_steps) < 0 else min(len(loader), int(args.max_steps))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_alltracker(args, device)
    adapter = None
    feature_names = None
    loaded_alltracker = []
    if str(args.adapter_ckpt):
        raise RuntimeError(
            '--adapter_ckpt is not supported in the V2 release package. '
            'Use the frozen AllTracker + RoMa final-stage risk/accept pipeline.'
        )
    try:
        roma_matcher = RoMaMatcher(
            model_type=args.roma_model,
            device=args.roma_device,
            input_size=args.roma_input_size,
            allow_online_download=args.roma_allow_online_download,
            cache_warps=args.roma_cache_warps,
            use_custom_corr=not bool(args.roma_disable_custom_corr),
            allow_slow_corr=bool(args.roma_allow_slow_corr),
        )
    except RoMaImportError as exc:
        print(str(exc), flush=True)
        raise SystemExit(2)

    print('device:', device, flush=True)
    print('dataset_root:', dataset_root, flush=True)
    print('num_sequences:', max_sequences, flush=True)
    print('seed:', int(args.seed), flush=True)
    print('deterministic_sampling:', bool(args.deterministic_sampling), flush=True)
    print('output:', out_dir, flush=True)
    if adapter is not None:
        print('adapter_ckpt:', expand_path(args.adapter_ckpt), flush=True)
        print('loaded_alltracker_trainable:', int(len(loaded_alltracker)), flush=True)

    manifest = []
    started = time.time()
    exported = 0
    for step, batch_pack in enumerate(loader):
        if exported >= max_sequences:
            break
        batch, gotit = unpack_batch(batch_pack)
        if not gotit:
            continue
        seq_id = safe_seq_id(batch, step)
        sample = export_sequence(seq_id, batch, model, roma_matcher, args, device, adapter=adapter, feature_names=feature_names)
        filename = 'seq_%04d.pt' % int(exported)
        path = os.path.join(seq_dir, filename)
        torch.save(sample, path)
        valid_labels = sample['labels'] >= 0
        positives = sample['labels'] == 1
        risk_valid = sample['baseline_risk_labels'] >= 0
        risk_positive = sample['baseline_risk_labels'] == 1
        risk_negative = sample['baseline_risk_labels'] == 0
        accept_valid = sample['roma_accept_labels'] >= 0
        accept_positive = sample['roma_accept_labels'] == 1
        accept_negative = sample['roma_accept_labels'] == 0
        manifest.append({'seq_id': str(seq_id), 'file': filename})
        exported += 1
        print(
            'seq %04d/%04d %s rows %d pos %d candidate %d risk_rows %d risk_pos %d risk_neg %d accept_rows %d accept_pos %d accept_neg %d ref=%s time %.1fs'
            % (
                exported,
                max_sequences,
                str(seq_id),
                int(valid_labels.sum().item()),
                int(positives.sum().item()),
                int(sample['candidate_mask'].sum().item()),
                int(risk_valid.sum().item()),
                int(risk_positive.sum().item()),
                int(risk_negative.sum().item()),
                int(accept_valid.sum().item()),
                int(accept_positive.sum().item()),
                int(accept_negative.sum().item()),
                str(sample['reference_source']),
                time.time() - started,
            ),
            flush=True,
        )

    torch.save(
        {
            'feature_names': list(FINAL_STAGE_FEATURE_NAMES),
            'manifest': manifest,
            'args': vars(args),
        },
        os.path.join(out_dir, 'manifest.pt'),
    )
    print('saved:', out_dir, flush=True)


if __name__ == '__main__':
    main()
