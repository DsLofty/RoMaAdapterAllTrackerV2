"""
Evaluate final-stage RoMa intervention without an adapter.

This script keeps AllTracker and RoMa frozen:
1. Run AllTracker normally to get baseline final trajectories.
2. Run RoMa from query frame 0 to sparse target frames.
3. Apply a conservative heuristic on the final baseline trajectory.
4. Replace or blend final positions with RoMa only on accepted rows.

GT is used only for evaluation/oracle metrics, never for the heuristic.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils.data
from alltracker_runtime_utils import dense_baseline_forward, expand_path, load_alltracker
from tapvid_dataset_utils import get_dataset, safe_seq_id
from tapvid_metric_utils import tapvid_metrics
from matchers.roma_wrapper import RoMaImportError, RoMaMatcher
from utils.roma_branch_relocalization import collect_roma_online_coarse_inputs, run_alltracker_embedded_adapter


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def mean_or_nan(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu()
        if values.numel() == 0:
            return float('nan')
        valid = torch.isfinite(values)
        if not bool(valid.any().item()):
            return float('nan')
        return float(values[valid].mean().item())
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return float('nan')
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(np.mean(arr))


def percentile_or_nan(values, q):
    if isinstance(values, torch.Tensor):
        values = values.detach().float().cpu()
        if values.numel() == 0:
            return float('nan')
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return float('nan')
        return float(torch.quantile(values, float(q)).item())
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(np.quantile(arr, float(q)))


def eval_mask(baseline, visible_only=True):
    trajs = baseline['trajs_e']
    device = trajs.device
    T, N, _ = trajs.shape
    frame_ids = torch.arange(T, device=device).reshape(T, 1).expand(T, N)
    first = baseline['first_positive_inds'].long().to(device).reshape(1, N).expand(T, N)
    valids = baseline.get('valids', torch.ones((T, N), dtype=torch.float32, device=device)).to(device)
    mask = (valids > 0.5) & (frame_ids > first)
    if visible_only:
        mask = mask & (baseline['vis_g'].to(device) > 0.5)
    return mask


def build_roma_maps(inputs, T, N, device):
    roma_xy = torch.full((T, N, 2), float('nan'), dtype=torch.float32, device=device)
    roma_cert = torch.zeros((T, N), dtype=torch.float32, device=device)
    roma_valid = torch.zeros((T, N), dtype=torch.bool, device=device)
    if inputs is None:
        return roma_xy, roma_cert, roma_valid
    frame = inputs['target_frame'].long().to(device)
    point = inputs['point_index'].long().to(device)
    valid = inputs['roma_valid'].reshape(-1).float().to(device) > 0.5
    xy = inputs['roma_xy8'].float().to(device) * 8.0
    cert = inputs['roma_certainty'].reshape(-1).float().to(device)
    finite = torch.isfinite(xy).all(dim=1)
    keep = valid & finite & (frame >= 0) & (frame < T) & (point >= 0) & (point < N)
    if bool(keep.any().item()):
        roma_xy[frame[keep], point[keep]] = xy[keep]
        roma_cert[frame[keep], point[keep]] = cert[keep]
        roma_valid[frame[keep], point[keep]] = True
    return roma_xy, roma_cert, roma_valid


def build_coarse_gate_map(supervision, T, N, device):
    gate_map = torch.zeros((T, N), dtype=torch.bool, device=device)
    if not supervision:
        return gate_map, False
    required = ('target_frame', 'point_index', 'relocalize_event')
    if any(key not in supervision or not torch.is_tensor(supervision[key]) for key in required):
        return gate_map, False
    frame = supervision['target_frame'].long().to(device).reshape(-1)
    point = supervision['point_index'].long().to(device).reshape(-1)
    event = supervision['relocalize_event'].bool().to(device).reshape(-1)
    count = min(int(frame.numel()), int(point.numel()), int(event.numel()))
    if count <= 0:
        return gate_map, True
    frame = frame[:count]
    point = point[:count]
    event = event[:count]
    keep = event & (frame >= 0) & (frame < T) & (point >= 0) & (point < N)
    if bool(keep.any().item()):
        gate_map[frame[keep], point[keep]] = True
    return gate_map, True


def motion_jump_px(trajs):
    T, N, _ = trajs.shape
    out = torch.zeros((T, N), dtype=trajs.dtype, device=trajs.device)
    if T <= 1:
        return out
    out[1:] = torch.linalg.vector_norm(trajs[1:] - trajs[:-1], dim=2)
    return out


def temporal_offset_change_px(baseline_xy, roma_xy, roma_valid, target_stride):
    T, N, _ = baseline_xy.shape
    out = torch.full((T, N), float('inf'), dtype=baseline_xy.dtype, device=baseline_xy.device)
    stride = max(1, int(target_stride))
    offset = roma_xy - baseline_xy
    for t in range(T):
        prev_t = t - stride
        if prev_t < 0:
            continue
        valid = roma_valid[t] & roma_valid[prev_t]
        if bool(valid.any().item()):
            out[t, valid] = torch.linalg.vector_norm(offset[t, valid] - offset[prev_t, valid], dim=1)
    return out


def config_value(args, config, name):
    if config is not None and name in config:
        return config[name]
    return getattr(args, name)


def final_stage_gate(baseline, roma_xy, roma_cert, roma_valid, args, device, config=None):
    base_xy = baseline['trajs_e'].float().to(device)
    visible = baseline['pred_visible_score'].float().to(device)
    T, N, _ = base_xy.shape
    first = baseline['first_positive_inds'].long().to(device).reshape(1, N).expand(T, N)
    frame_ids = torch.arange(T, device=device).reshape(T, 1).expand(T, N)

    dist = torch.linalg.vector_norm(roma_xy - base_xy, dim=2)
    jump = motion_jump_px(base_xy)
    prev_dist = torch.full((T, N), float('inf'), dtype=torch.float32, device=device)
    if T > 1:
        prev_dist[1:] = torch.linalg.vector_norm(roma_xy[1:] - base_xy[:-1], dim=2)
    offset_change = temporal_offset_change_px(base_xy, roma_xy, roma_valid, args.target_frame_stride)

    baseline_suspicious = (
        (visible <= float(config_value(args, config, 'final_baseline_visible_thr')))
        | (jump >= float(config_value(args, config, 'final_baseline_motion_jump_thr_px')))
        | (dist >= float(config_value(args, config, 'final_strong_offset_thr_px')))
    )
    gate = (
        roma_valid
        & (frame_ids > first)
        & (roma_cert >= float(config_value(args, config, 'final_roma_certainty_thr')))
        & (dist >= float(config_value(args, config, 'final_min_offset_px')))
        & (dist <= float(config_value(args, config, 'final_max_offset_px')))
        & (prev_dist <= float(config_value(args, config, 'final_roma_prev_dist_px')))
    )
    if bool(config_value(args, config, 'final_require_baseline_suspicious')):
        gate = gate & baseline_suspicious
    offset_change_max = float(config_value(args, config, 'final_offset_change_max_px'))
    if offset_change_max >= 0.0:
        warmup = frame_ids <= (first + max(1, int(args.target_frame_stride)))
        gate = gate & (warmup | (offset_change <= offset_change_max))
    return gate, {
        'baseline_suspicious': baseline_suspicious,
        'visible': visible,
        'dist_px': dist,
        'jump_px': jump,
        'prev_dist_px': prev_dist,
        'offset_change_px': offset_change,
    }


def make_sweep_configs(args):
    presets = [p.strip() for p in str(args.final_sweep_presets).split(',') if p.strip()]
    preset_map = {
        'base': {},
        'loose1': {
            'final_roma_certainty_thr': 0.70,
            'final_baseline_motion_jump_thr_px': 8.0,
            'final_strong_offset_thr_px': 8.0,
            'final_min_offset_px': 2.0,
            'final_max_offset_px': 96.0,
            'final_roma_prev_dist_px': 96.0,
            'final_offset_change_max_px': 48.0,
            'final_require_baseline_suspicious': True,
        },
        'loose2': {
            'final_roma_certainty_thr': 0.65,
            'final_baseline_motion_jump_thr_px': 6.0,
            'final_strong_offset_thr_px': 6.0,
            'final_min_offset_px': 1.0,
            'final_max_offset_px': 128.0,
            'final_roma_prev_dist_px': 128.0,
            'final_offset_change_max_px': 96.0,
            'final_require_baseline_suspicious': True,
        },
        'loose3': {
            'final_roma_certainty_thr': 0.60,
            'final_baseline_motion_jump_thr_px': 4.0,
            'final_strong_offset_thr_px': 4.0,
            'final_min_offset_px': 1.0,
            'final_max_offset_px': 160.0,
            'final_roma_prev_dist_px': 160.0,
            'final_offset_change_max_px': -1.0,
            'final_require_baseline_suspicious': True,
        },
        'nosusp1': {
            'final_roma_certainty_thr': 0.75,
            'final_min_offset_px': 4.0,
            'final_max_offset_px': 64.0,
            'final_roma_prev_dist_px': 64.0,
            'final_offset_change_max_px': 24.0,
            'final_require_baseline_suspicious': False,
        },
        'nosusp2': {
            'final_roma_certainty_thr': 0.70,
            'final_min_offset_px': 2.0,
            'final_max_offset_px': 96.0,
            'final_roma_prev_dist_px': 96.0,
            'final_offset_change_max_px': 48.0,
            'final_require_baseline_suspicious': False,
        },
    }
    configs = []
    for name in presets:
        if name not in preset_map:
            raise ValueError('unknown --final_sweep_presets entry: %s' % name)
        config = dict(preset_map[name])
        config['name'] = name
        configs.append(config)
    if not configs:
        configs.append({'name': 'base'})
    return configs


def replace_or_blend(base_xy, roma_xy, gate, alpha):
    out = base_xy.clone()
    if bool(gate.any().item()):
        out[gate] = base_xy[gate] + float(alpha) * (roma_xy[gate] - base_xy[gate])
    return out


def error_map(trajs, gt):
    return torch.linalg.vector_norm(trajs.float() - gt.float(), dim=2)


def fill_reference_metadata(reference, alltracker):
    out = dict(reference)
    for key in ('trajs_g', 'vis_g', 'valids', 'first_positive_inds'):
        if key not in out and key in alltracker:
            out[key] = alltracker[key]
    return out


def strategy_metrics_row(
    seq_id,
    strategy,
    gate,
    baseline,
    baseline_metrics,
    base_xy_d,
    roma_xy,
    roma_cert,
    base_err,
    roma_err,
    gt,
    gt_d,
    vis,
    first,
    mask,
    image_size,
    args,
    gate_debug,
    coarse_gate=None,
    has_coarse_gate=False,
    alltracker_err=None,
):
    replaced_xy = replace_or_blend(base_xy_d, roma_xy, gate, 1.0).detach().cpu()
    replace_err = error_map(replaced_xy.to(base_xy_d.device), gt_d)
    replace_metrics = tapvid_metrics(
        replaced_xy,
        baseline['pred_visible_score'],
        gt,
        vis,
        first,
        image_size,
    )
    blend_xy = replace_or_blend(base_xy_d, roma_xy, gate, float(args.final_blend_alpha)).detach().cpu()
    blend_err = error_map(blend_xy.to(base_xy_d.device), gt_d)
    blend_metrics = tapvid_metrics(
        blend_xy,
        baseline['pred_visible_score'],
        gt,
        vis,
        first,
        image_size,
    )
    row = {
        'seq_id': str(seq_id),
        'strategy': str(strategy),
        'baseline_da': baseline_metrics['da'],
        'baseline_aj': baseline_metrics['aj'],
        'baseline_oa': baseline_metrics['oa'],
        'replace_da': replace_metrics['da'],
        'replace_aj': replace_metrics['aj'],
        'replace_oa': replace_metrics['oa'],
        'replace_delta_da': replace_metrics['da'] - baseline_metrics['da'],
        'replace_delta_aj': replace_metrics['aj'] - baseline_metrics['aj'],
        'replace_delta_oa': replace_metrics['oa'] - baseline_metrics['oa'],
        'blend_da': blend_metrics['da'],
        'blend_aj': blend_metrics['aj'],
        'blend_oa': blend_metrics['oa'],
        'blend_delta_da': blend_metrics['da'] - baseline_metrics['da'],
        'blend_delta_aj': blend_metrics['aj'] - baseline_metrics['aj'],
        'blend_delta_oa': blend_metrics['oa'] - baseline_metrics['oa'],
        'baseline_epe_mean': mean_or_nan(base_err[mask]),
        'replace_epe_mean': mean_or_nan(replace_err[mask]),
        'blend_epe_mean': mean_or_nan(blend_err[mask]),
        'replace_delta_epe': mean_or_nan(replace_err[mask] - base_err[mask]),
        'blend_delta_epe': mean_or_nan(blend_err[mask] - base_err[mask]),
        'gate_ratio': mean_or_nan(gate[mask].float()),
        'gate_rows': int(gate.sum().item()),
        'accepted_delta_epe': mean_or_nan((roma_err - base_err)[gate]),
        'rejected_delta_epe': mean_or_nan((roma_err - base_err)[mask & torch.isfinite(roma_err) & (~gate)]),
        'accepted_dist_px': mean_or_nan(gate_debug['dist_px'][gate]),
        'accepted_certainty': mean_or_nan(roma_cert[gate]),
        'accepted_offset_change_px': mean_or_nan(gate_debug['offset_change_px'][gate]),
        'accepted_prev_dist_px': mean_or_nan(gate_debug['prev_dist_px'][gate]),
    }
    if coarse_gate is not None and bool(has_coarse_gate):
        coarse_gate = coarse_gate.to(device=gate.device).bool() & mask
        final_gate = gate.bool() & mask
        both = coarse_gate & final_gate
        coarse_only = coarse_gate & (~final_gate)
        final_only = (~coarse_gate) & final_gate
        neither = mask & (~coarse_gate) & (~final_gate)
        valid_ref = mask
        denom = torch.clamp(valid_ref.float().sum(), min=1.0)

        def _ratio(m):
            return float((m & valid_ref).float().sum().item() / float(denom.item()))

        def _delta(m):
            return mean_or_nan((roma_err - base_err)[m & torch.isfinite(roma_err)])

        def _ref_delta_vs_alltracker(m):
            if alltracker_err is None:
                return float('nan')
            m = m & torch.isfinite(base_err) & torch.isfinite(alltracker_err)
            return mean_or_nan(base_err[m] - alltracker_err[m])

        def _replace_delta_vs_alltracker(m):
            if alltracker_err is None:
                return float('nan')
            m = m & torch.isfinite(replace_err.to(base_err.device)) & torch.isfinite(alltracker_err)
            return mean_or_nan(replace_err.to(base_err.device)[m] - alltracker_err[m])

        row.update({
            'coarse_gate_available': 1.0,
            'coarse_gate_ratio': _ratio(coarse_gate),
            'coarse_gate_rows': int(coarse_gate.sum().item()),
            'overlap_ratio': _ratio(both),
            'overlap_rows': int(both.sum().item()),
            'coarse_only_ratio': _ratio(coarse_only),
            'coarse_only_rows': int(coarse_only.sum().item()),
            'final_only_ratio': _ratio(final_only),
            'final_only_rows': int(final_only.sum().item()),
            'neither_ratio': _ratio(neither),
            'both_delta_epe': _delta(both),
            'coarse_only_delta_epe': _delta(coarse_only),
            'final_only_delta_epe': _delta(final_only),
            'neither_delta_epe': _delta(neither),
            'both_reference_delta_vs_alltracker': _ref_delta_vs_alltracker(both),
            'both_replace_delta_vs_alltracker': _replace_delta_vs_alltracker(both),
            'coarse_only_reference_delta_vs_alltracker': _ref_delta_vs_alltracker(coarse_only),
            'coarse_only_replace_delta_vs_alltracker': _replace_delta_vs_alltracker(coarse_only),
            'final_only_reference_delta_vs_alltracker': _ref_delta_vs_alltracker(final_only),
            'final_only_replace_delta_vs_alltracker': _replace_delta_vs_alltracker(final_only),
            'neither_reference_delta_vs_alltracker': _ref_delta_vs_alltracker(neither),
            'neither_replace_delta_vs_alltracker': _replace_delta_vs_alltracker(neither),
            'final_gate_on_coarse_ratio': (
                float(both.float().sum().item() / max(float(coarse_gate.float().sum().item()), 1.0))
            ),
            'coarse_gate_in_final_ratio': (
                float(both.float().sum().item() / max(float(final_gate.float().sum().item()), 1.0))
            ),
        })
    else:
        row.update({
            'coarse_gate_available': 0.0,
            'coarse_gate_ratio': float('nan'),
            'coarse_gate_rows': float('nan'),
            'overlap_ratio': float('nan'),
            'overlap_rows': float('nan'),
            'coarse_only_ratio': float('nan'),
            'coarse_only_rows': float('nan'),
            'final_only_ratio': float('nan'),
            'final_only_rows': float('nan'),
            'neither_ratio': float('nan'),
            'both_delta_epe': float('nan'),
            'coarse_only_delta_epe': float('nan'),
            'final_only_delta_epe': float('nan'),
            'neither_delta_epe': float('nan'),
            'both_reference_delta_vs_alltracker': float('nan'),
            'both_replace_delta_vs_alltracker': float('nan'),
            'coarse_only_reference_delta_vs_alltracker': float('nan'),
            'coarse_only_replace_delta_vs_alltracker': float('nan'),
            'final_only_reference_delta_vs_alltracker': float('nan'),
            'final_only_replace_delta_vs_alltracker': float('nan'),
            'neither_reference_delta_vs_alltracker': float('nan'),
            'neither_replace_delta_vs_alltracker': float('nan'),
            'final_gate_on_coarse_ratio': float('nan'),
            'coarse_gate_in_final_ratio': float('nan'),
        })
    return row


def feature_group_stats(seq_id, group, mask, features):
    count = int(mask.sum().item())
    row = {'seq_id': str(seq_id), 'group': str(group), 'count': count}
    for name, values in features.items():
        selected = values[mask]
        row[name + '_mean'] = mean_or_nan(selected)
        row[name + '_p25'] = percentile_or_nan(selected, 0.25)
        row[name + '_p50'] = percentile_or_nan(selected, 0.50)
        row[name + '_p75'] = percentile_or_nan(selected, 0.75)
        row[name + '_p90'] = percentile_or_nan(selected, 0.90)
    return row


def oracle_feature_rows(seq_id, oracle_gate, candidate_mask, base_err, roma_err, roma_cert, gate_debug):
    features = {
        'roma_certainty': roma_cert,
        'baseline_to_roma_dist_px': gate_debug['dist_px'],
        'offset_change_px': gate_debug['offset_change_px'],
        'roma_to_prev_baseline_dist_px': gate_debug['prev_dist_px'],
        'baseline_motion_jump_px': gate_debug['jump_px'],
        'baseline_visible_score': gate_debug['visible'],
        'baseline_final_epe': base_err,
        'roma_epe': roma_err,
        'roma_gain_px': base_err - roma_err,
    }
    non_oracle = candidate_mask & (~oracle_gate)
    return [
        feature_group_stats(seq_id, 'oracle', oracle_gate, features),
        feature_group_stats(seq_id, 'non_oracle_candidate', non_oracle, features),
        feature_group_stats(seq_id, 'all_candidate', candidate_mask, features),
    ]


def evaluate_sequence(seq_id, batch, model, roma_matcher, args, device, adapter=None, feature_names=None):
    t0 = time.time()
    alltracker = dense_baseline_forward(batch, model, args, device=device)
    T, N, _ = alltracker['trajs_e'].shape
    gt = alltracker['trajs_g'].float()
    vis = alltracker['vis_g'].float()
    first = alltracker['first_positive_inds'].long()
    image_size = tuple(int(v) for v in args.image_size)

    inputs = collect_roma_online_coarse_inputs(seq_id, batch, roma_matcher, args, device)
    roma_xy, roma_cert, roma_valid = build_roma_maps(inputs, T, N, device)

    reference_source = 'alltracker'
    reference = alltracker
    embedded_stats = {}
    coarse_gate_map = torch.zeros((T, N), dtype=torch.bool, device=device)
    has_coarse_gate = False
    if adapter is not None:
        if inputs is None:
            raise RuntimeError('RoMa inputs are required for --adapter_ckpt combination evaluation.')
        if feature_names is None:
            raise RuntimeError('feature_names are required when adapter is provided.')
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
        reference = fill_reference_metadata(adapter_ref, alltracker)
        reference_source = 'adapter'
        coarse_gate_map, has_coarse_gate = build_coarse_gate_map(
            embedded_stats.get('supervision', {}) if embedded_stats else {},
            T,
            N,
            device,
        )

    base_xy_d = reference['trajs_e'].float().to(device)
    gt_d = gt.to(device)
    mask = eval_mask(reference, visible_only=bool(args.analysis_visible_only)).to(device)
    base_gate, gate_debug = final_stage_gate(reference, roma_xy, roma_cert, roma_valid, args, device)
    base_gate = base_gate & mask

    base_err = error_map(base_xy_d, gt_d)
    roma_err = error_map(roma_xy, gt_d)
    baseline_metrics = tapvid_metrics(
        reference['trajs_e'],
        reference['pred_visible_score'],
        gt,
        vis,
        first,
        image_size,
    )
    alltracker_err = error_map(alltracker['trajs_e'].float().to(device), gt_d)
    alltracker_metrics = tapvid_metrics(
        alltracker['trajs_e'],
        alltracker['pred_visible_score'],
        gt,
        vis,
        first,
        image_size,
    )

    oracle_gate = mask & roma_valid & torch.isfinite(roma_err) & ((roma_err + float(args.oracle_margin_px)) < base_err)
    oracle_xy = replace_or_blend(base_xy_d, roma_xy, oracle_gate, 1.0).detach().cpu()
    oracle_err = error_map(oracle_xy.to(device), gt_d)
    oracle_metrics = tapvid_metrics(
        oracle_xy,
        reference['pred_visible_score'],
        gt,
        vis,
        first,
        image_size,
    )
    sweep_rows = []
    for config in make_sweep_configs(args):
        gate, _ = final_stage_gate(reference, roma_xy, roma_cert, roma_valid, args, device, config=config)
        gate = gate & mask
        sweep_rows.append(
            strategy_metrics_row(
                seq_id,
                config['name'],
                gate,
                reference,
                baseline_metrics,
                base_xy_d,
                roma_xy,
                roma_cert,
                base_err,
                roma_err,
                gt,
                gt_d,
                vis,
                first,
                mask,
                image_size,
                args,
                gate_debug,
                coarse_gate=coarse_gate_map,
                has_coarse_gate=has_coarse_gate,
                alltracker_err=alltracker_err,
            )
        )
    base_strategy = next((r for r in sweep_rows if r['strategy'] == 'base'), sweep_rows[0])

    candidate_mask = mask & roma_valid & torch.isfinite(roma_err) & torch.isfinite(gate_debug['dist_px'])
    oracle_rows = oracle_feature_rows(seq_id, oracle_gate, candidate_mask, base_err, roma_err, roma_cert, gate_debug)

    row = {
        'seq_id': str(seq_id),
        'reference_source': reference_source,
        'alltracker_da': alltracker_metrics['da'],
        'alltracker_aj': alltracker_metrics['aj'],
        'alltracker_oa': alltracker_metrics['oa'],
        'alltracker_epe_mean': mean_or_nan(alltracker_err[mask]),
        'reference_delta_epe_vs_alltracker': mean_or_nan(base_err[mask] - alltracker_err[mask]),
        'baseline_da': baseline_metrics['da'],
        'baseline_aj': baseline_metrics['aj'],
        'baseline_oa': baseline_metrics['oa'],
        'replace_da': base_strategy['replace_da'],
        'replace_aj': base_strategy['replace_aj'],
        'replace_oa': base_strategy['replace_oa'],
        'replace_delta_da': base_strategy['replace_delta_da'],
        'replace_delta_aj': base_strategy['replace_delta_aj'],
        'replace_delta_oa': base_strategy['replace_delta_oa'],
        'blend_da': base_strategy['blend_da'],
        'blend_aj': base_strategy['blend_aj'],
        'blend_oa': base_strategy['blend_oa'],
        'blend_delta_da': base_strategy['blend_delta_da'],
        'blend_delta_aj': base_strategy['blend_delta_aj'],
        'blend_delta_oa': base_strategy['blend_delta_oa'],
        'oracle_da': oracle_metrics['da'],
        'oracle_aj': oracle_metrics['aj'],
        'oracle_oa': oracle_metrics['oa'],
        'oracle_delta_da': oracle_metrics['da'] - baseline_metrics['da'],
        'oracle_delta_aj': oracle_metrics['aj'] - baseline_metrics['aj'],
        'oracle_delta_oa': oracle_metrics['oa'] - baseline_metrics['oa'],
        'baseline_epe_mean': mean_or_nan(base_err[mask]),
        'replace_epe_mean': base_strategy['replace_epe_mean'],
        'blend_epe_mean': base_strategy['blend_epe_mean'],
        'oracle_epe_mean': mean_or_nan(oracle_err[mask]),
        'replace_delta_epe': base_strategy['replace_delta_epe'],
        'blend_delta_epe': base_strategy['blend_delta_epe'],
        'oracle_delta_epe': mean_or_nan(oracle_err[mask] - base_err[mask]),
        'gate_ratio': base_strategy['gate_ratio'],
        'gate_rows': int(base_gate.sum().item()),
        'oracle_ratio': mean_or_nan(oracle_gate[mask].float()),
        'oracle_rows': int(oracle_gate.sum().item()),
        'roma_valid_ratio': mean_or_nan((roma_valid & mask).float()[mask]),
        'accepted_delta_epe': base_strategy['accepted_delta_epe'],
        'rejected_delta_epe': base_strategy['rejected_delta_epe'],
        'accepted_dist_px': base_strategy['accepted_dist_px'],
        'accepted_certainty': base_strategy['accepted_certainty'],
        'coarse_gate_available': base_strategy['coarse_gate_available'],
        'coarse_gate_ratio': base_strategy['coarse_gate_ratio'],
        'coarse_gate_rows': base_strategy['coarse_gate_rows'],
        'overlap_ratio': base_strategy['overlap_ratio'],
        'overlap_rows': base_strategy['overlap_rows'],
        'coarse_only_ratio': base_strategy['coarse_only_ratio'],
        'coarse_only_rows': base_strategy['coarse_only_rows'],
        'final_only_ratio': base_strategy['final_only_ratio'],
        'final_only_rows': base_strategy['final_only_rows'],
        'final_gate_on_coarse_ratio': base_strategy['final_gate_on_coarse_ratio'],
        'coarse_gate_in_final_ratio': base_strategy['coarse_gate_in_final_ratio'],
        'both_delta_epe': base_strategy['both_delta_epe'],
        'coarse_only_delta_epe': base_strategy['coarse_only_delta_epe'],
        'final_only_delta_epe': base_strategy['final_only_delta_epe'],
        'both_reference_delta_vs_alltracker': base_strategy['both_reference_delta_vs_alltracker'],
        'both_replace_delta_vs_alltracker': base_strategy['both_replace_delta_vs_alltracker'],
        'coarse_only_reference_delta_vs_alltracker': base_strategy['coarse_only_reference_delta_vs_alltracker'],
        'coarse_only_replace_delta_vs_alltracker': base_strategy['coarse_only_replace_delta_vs_alltracker'],
        'final_only_reference_delta_vs_alltracker': base_strategy['final_only_reference_delta_vs_alltracker'],
        'final_only_replace_delta_vs_alltracker': base_strategy['final_only_replace_delta_vs_alltracker'],
        'neither_reference_delta_vs_alltracker': base_strategy['neither_reference_delta_vs_alltracker'],
        'neither_replace_delta_vs_alltracker': base_strategy['neither_replace_delta_vs_alltracker'],
        'time_s': time.time() - t0,
    }
    return row, sweep_rows, oracle_rows


def summarize(rows):
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out = {'num_sequences': len(rows)}
    for key in keys:
        if key == 'seq_id':
            continue
        vals = []
        for row in rows:
            try:
                vals.append(float(row.get(key, float('nan'))))
            except (TypeError, ValueError):
                pass
        if vals:
            out[key] = mean_or_nan(np.asarray(vals, dtype=np.float32))
    return out


def summarize_by_key(rows, group_key):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key, '')), []).append(row)
    out = []
    for key, group_rows in grouped.items():
        summary = summarize(group_rows)
        summary[group_key] = key
        out.append(summary)
    return out


def summarize_oracle_features(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row.get('group', '')), []).append(row)
    out = []
    metric_suffixes = ('_mean', '_p25', '_p50', '_p75', '_p90')
    for group, group_rows in grouped.items():
        summary = {'group': group, 'num_sequences': len(group_rows)}
        total_count = sum(int(r.get('count', 0) or 0) for r in group_rows)
        summary['count'] = total_count
        if total_count <= 0:
            for key in group_rows[0].keys():
                if key.endswith(metric_suffixes):
                    summary[key] = float('nan')
            out.append(summary)
            continue
        keys = sorted({key for r in group_rows for key in r.keys() if key.endswith(metric_suffixes)})
        for key in keys:
            weighted_sum = 0.0
            weight_sum = 0
            for row in group_rows:
                count = int(row.get('count', 0) or 0)
                try:
                    value = float(row.get(key, float('nan')))
                except (TypeError, ValueError):
                    value = float('nan')
                if count > 0 and np.isfinite(value):
                    weighted_sum += value * count
                    weight_sum += count
            summary[key] = weighted_sum / weight_sum if weight_sum > 0 else float('nan')
        out.append(summary)
    return out


def unpack_batch(batch_pack):
    if isinstance(batch_pack, tuple) and len(batch_pack) == 2:
        batch, gotit = batch_pack
        if isinstance(gotit, (list, tuple)):
            ok = all(bool(v) for v in gotit)
        elif torch.is_tensor(gotit):
            ok = bool(gotit.bool().all().item())
        else:
            ok = bool(gotit)
        return batch, ok
    return batch_pack, True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', choices=['dav', 'kin', 'rgb', 'rob'], default='dav')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--only_first', action='store_true', default=True)
    parser.add_argument('--image_size', type=int, nargs='+', default=[448, 768])
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/alltracker.pth')
    parser.add_argument(
        '--adapter_ckpt',
        type=str,
        default='',
        help='Optional RoMa coarse adapter checkpoint. If set, final-stage RoMa postprocess is applied on adapter final tracks.',
    )
    parser.add_argument('--save_dir', type=str, default='./roma_final_stage_eval_outputs')
    parser.add_argument('--exp', type=str, default='dav_final_stage_roma_heuristic')
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--inference_iters', type=int, default=4)
    parser.add_argument('--analysis_visible_only', action='store_true', default=True)
    parser.add_argument('--save_per_sequence', action='store_true', default=True)

    parser.add_argument('--roma_model', choices=['outdoor', 'indoor', 'tiny_outdoor'], default='outdoor')
    parser.add_argument('--roma_device', type=str, default='cuda')
    parser.add_argument('--roma_input_size', type=int, nargs='*', default=None)
    parser.add_argument('--roma_sample_mode', choices=['nearest', 'bilinear'], default='bilinear')
    parser.add_argument('--roma_certainty_thr', type=float, default=-1.0)
    parser.add_argument('--roma_allow_online_download', action='store_true', default=False)
    parser.add_argument('--roma_cache_warps', action='store_true', default=False)
    parser.add_argument('--roma_disable_custom_corr', action='store_true', default=False)
    parser.add_argument('--roma_allow_slow_corr', action='store_true', default=False)
    parser.add_argument('--roma_fail_fast', action='store_true', default=False)
    parser.add_argument('--roma_pair_batch_size', type=int, default=1)
    parser.add_argument('--roma_log_chunks', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--target_frame_stride', type=int, default=2)
    parser.add_argument('--target_frame_include', type=str, default='stride,last')
    parser.add_argument('--target_points_per_frame', type=int, default=4096)
    parser.add_argument('--need_uniform_points_per_frame', type=int, default=0)
    parser.add_argument('--sequence_len', type=int, default=24)

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
    parser.add_argument('--safety_thr', type=float, default=0.6)
    parser.add_argument('--safety_min_gain_px', type=float, default=0.0)
    parser.add_argument('--baseline_need_thr', type=float, default=0.5)
    parser.add_argument('--candidate_quality_thr', type=float, default=0.5)
    parser.add_argument('--candidate_accept_thr', type=float, default=0.5)
    parser.add_argument('--counterfactual_reject_ratio', type=float, default=0.0)
    parser.add_argument('--counterfactual_reject_max_rows', type=int, default=0)
    parser.add_argument('--carry_mode', type=str, default='none')
    parser.add_argument('--carry_max_age', type=int, default=0)
    parser.add_argument('--carry_decay', type=float, default=0.9)
    parser.add_argument('--carry_min_score', type=float, default=0.0)
    parser.add_argument('--carry_max_offset8', type=float, default=8.0)
    parser.add_argument('--carry_require_baseline_suspicious', action='store_true', default=False)
    parser.add_argument('--carry_apply_strength', type=float, default=1.0)
    parser.add_argument('--carry_refresh_dist8', type=float, default=2.0)
    parser.add_argument('--local_corr_supplement', action='store_true', default=False)
    parser.add_argument('--local_corr_supplement_certainty_thr', type=float, default=0.65)
    parser.add_argument('--local_corr_supplement_roma_peak_thr', type=float, default=0.0)
    parser.add_argument('--local_corr_supplement_margin_thr', type=float, default=0.02)
    parser.add_argument('--local_corr_supplement_gain_thr', type=float, default=0.05)
    parser.add_argument('--local_corr_supplement_peak_offset8_max', type=float, default=2.0)
    parser.add_argument('--local_corr_supplement_min_offset8', type=float, default=0.5)
    parser.add_argument('--local_corr_supplement_max_offset8', type=float, default=6.0)
    parser.add_argument('--local_corr_supplement_roma_prev_dist8_max', type=float, default=-1.0)
    parser.add_argument('--local_corr_supplement_visual_cos_thr', type=float, default=-2.0)
    parser.add_argument('--local_corr_supplement_require_baseline_suspicious', action='store_true', default=True)

    parser.add_argument('--final_blend_alpha', type=float, default=0.5)
    parser.add_argument('--final_roma_certainty_thr', type=float, default=0.75)
    parser.add_argument('--final_baseline_visible_thr', type=float, default=0.5)
    parser.add_argument('--final_baseline_motion_jump_thr_px', type=float, default=12.0)
    parser.add_argument('--final_strong_offset_thr_px', type=float, default=16.0)
    parser.add_argument('--final_min_offset_px', type=float, default=4.0)
    parser.add_argument('--final_max_offset_px', type=float, default=64.0)
    parser.add_argument('--final_roma_prev_dist_px', type=float, default=48.0)
    parser.add_argument('--final_offset_change_max_px', type=float, default=24.0)
    parser.add_argument('--final_require_baseline_suspicious', action='store_true', default=True)
    parser.add_argument('--no_final_require_baseline_suspicious', action='store_false', dest='final_require_baseline_suspicious')
    parser.add_argument(
        '--final_sweep_presets',
        type=str,
        default='base,loose1,loose2,loose3,nosusp1,nosusp2',
        help='Comma-separated final-stage heuristic presets to evaluate with one RoMa pass.',
    )
    parser.add_argument('--oracle_margin_px', type=float, default=0.0)

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
    out_dir = os.path.join(expand_path(args.save_dir), str(args.exp))
    os.makedirs(out_dir, exist_ok=True)
    dataset, dataset_root = get_dataset(args)
    max_sequences = len(dataset) if int(args.max_steps) < 0 else min(len(dataset), int(args.max_steps))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=utils.data.collate_fn_train,
    )
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
    print('alltracker_ckpt:', expand_path(args.ckpt_path), flush=True)
    if adapter is not None:
        print('adapter_ckpt:', expand_path(args.adapter_ckpt), flush=True)
        print('loaded_alltracker_trainable:', int(len(loaded_alltracker)), flush=True)
    print('output:', out_dir, flush=True)
    print('final_sweep_presets:', args.final_sweep_presets, flush=True)

    rows = []
    sweep_rows = []
    oracle_feature_rows_all = []
    started = time.time()
    evaluated = 0
    for step, batch_pack in enumerate(loader):
        if evaluated >= max_sequences:
            break
        batch, gotit = unpack_batch(batch_pack)
        if not gotit:
            print('warning: skipping sequence %d because dataset returned gotit=False' % int(step), flush=True)
            continue
        seq_id = safe_seq_id(batch, step)
        row, seq_sweep_rows, seq_oracle_feature_rows = evaluate_sequence(
            seq_id,
            batch,
            model,
            roma_matcher,
            args,
            device,
            adapter=adapter,
            feature_names=feature_names,
        )
        rows.append(row)
        sweep_rows.extend(seq_sweep_rows)
        oracle_feature_rows_all.extend(seq_oracle_feature_rows)
        evaluated += 1
        best_sweep = min(
            seq_sweep_rows,
            key=lambda r: float(r.get('blend_delta_epe', float('inf'))) if np.isfinite(float(r.get('blend_delta_epe', float('inf')))) else float('inf'),
        )
        print(
            'seq %04d/%04d %s ref=%s base_epe %.4f replace_epe %.4f d_replace %.4f '
            'blend_epe %.4f d_blend %.4f oracle_d %.4f gate %.4f rows %d '
            'coarse %.4f overlap %.4f final_only %.4f '
            'best_sweep=%s best_blend_d %.4f best_gate %.4f time %.1fs'
            % (
                evaluated,
                max_sequences,
                str(seq_id),
                str(row.get('reference_source', '')),
                row['baseline_epe_mean'],
                row['replace_epe_mean'],
                row['replace_delta_epe'],
                row['blend_epe_mean'],
                row['blend_delta_epe'],
                row['oracle_delta_epe'],
                row['gate_ratio'],
                row['gate_rows'],
                float(row.get('coarse_gate_ratio', float('nan'))),
                float(row.get('overlap_ratio', float('nan'))),
                float(row.get('final_only_ratio', float('nan'))),
                str(best_sweep.get('strategy', '')),
                float(best_sweep.get('blend_delta_epe', float('nan'))),
                float(best_sweep.get('gate_ratio', float('nan'))),
                time.time() - started,
            ),
            flush=True,
        )

    if not rows:
        raise RuntimeError('No sequences evaluated.')
    fieldnames = list(rows[0].keys())
    write_csv(os.path.join(out_dir, 'per_sequence.csv'), rows, fieldnames)
    if sweep_rows:
        sweep_fieldnames = list(sweep_rows[0].keys())
        write_csv(os.path.join(out_dir, 'per_sequence_sweep.csv'), sweep_rows, sweep_fieldnames)
        sweep_summary = summarize_by_key(sweep_rows, 'strategy')
        sweep_summary = sorted(sweep_summary, key=lambda r: str(r.get('strategy', '')))
        sweep_summary_keys = ['strategy'] + [k for k in sweep_summary[0].keys() if k != 'strategy']
        write_csv(os.path.join(out_dir, 'summary_sweep.csv'), sweep_summary, sweep_summary_keys)
    if oracle_feature_rows_all:
        oracle_fieldnames = list(oracle_feature_rows_all[0].keys())
        write_csv(os.path.join(out_dir, 'per_sequence_oracle_features.csv'), oracle_feature_rows_all, oracle_fieldnames)
        oracle_summary = summarize_oracle_features(oracle_feature_rows_all)
        oracle_summary_keys = ['group'] + [k for k in oracle_summary[0].keys() if k != 'group']
        write_csv(os.path.join(out_dir, 'oracle_feature_summary.csv'), oracle_summary, oracle_summary_keys)
    summary = summarize(rows)
    write_csv(os.path.join(out_dir, 'summary.csv'), [summary], list(summary.keys()))
    print(
        'SUMMARY dname=%s num_sequences=%d baseline_epe_mean=%.5g replace_epe_mean=%.5g '
        'replace_delta_epe=%.5g blend_epe_mean=%.5g blend_delta_epe=%.5g '
        'oracle_delta_epe=%.5g gate_ratio=%.5g gate_rows=%.3f saved=%s'
        % (
            args.dname,
            int(summary.get('num_sequences', len(rows))),
            float(summary.get('baseline_epe_mean', float('nan'))),
            float(summary.get('replace_epe_mean', float('nan'))),
            float(summary.get('replace_delta_epe', float('nan'))),
            float(summary.get('blend_epe_mean', float('nan'))),
            float(summary.get('blend_delta_epe', float('nan'))),
            float(summary.get('oracle_delta_epe', float('nan'))),
            float(summary.get('gate_ratio', float('nan'))),
            float(summary.get('gate_rows', float('nan'))),
            out_dir,
        ),
        flush=True,
    )
    if sweep_rows:
        print('SWEEP SUMMARY', flush=True)
        for srow in sorted(summarize_by_key(sweep_rows, 'strategy'), key=lambda r: str(r.get('strategy', ''))):
            print(
                '  %s gate %.5f rows %.2f replace_d %.5g blend_d %.5g acc_d %.5g '
                'coarse %.5f overlap %.5f final_only %.5f both_d %.5g final_only_d %.5g '
                'both_ref_at %.5g both_rep_at %.5g coarse_ref_at %.5g final_ref_at %.5g '
                'cert %.4f dist %.3f offchg %.3f'
                % (
                    str(srow.get('strategy', '')),
                    float(srow.get('gate_ratio', float('nan'))),
                    float(srow.get('gate_rows', float('nan'))),
                    float(srow.get('replace_delta_epe', float('nan'))),
                    float(srow.get('blend_delta_epe', float('nan'))),
                    float(srow.get('accepted_delta_epe', float('nan'))),
                    float(srow.get('coarse_gate_ratio', float('nan'))),
                    float(srow.get('overlap_ratio', float('nan'))),
                    float(srow.get('final_only_ratio', float('nan'))),
                    float(srow.get('both_delta_epe', float('nan'))),
                    float(srow.get('final_only_delta_epe', float('nan'))),
                    float(srow.get('both_reference_delta_vs_alltracker', float('nan'))),
                    float(srow.get('both_replace_delta_vs_alltracker', float('nan'))),
                    float(srow.get('coarse_only_reference_delta_vs_alltracker', float('nan'))),
                    float(srow.get('final_only_reference_delta_vs_alltracker', float('nan'))),
                    float(srow.get('accepted_certainty', float('nan'))),
                    float(srow.get('accepted_dist_px', float('nan'))),
                    float(srow.get('accepted_offset_change_px', float('nan'))),
                ),
                flush=True,
            )
    if oracle_feature_rows_all:
        print('ORACLE FEATURE SUMMARY', flush=True)
        for orow in summarize_oracle_features(oracle_feature_rows_all):
            print(
                '  %s count %d cert %.4f dist %.3f offchg %.3f prevdist %.3f jump %.3f visible %.3f gain %.3f'
                % (
                    str(orow.get('group', '')),
                    int(orow.get('count', 0) or 0),
                    float(orow.get('roma_certainty_mean', float('nan'))),
                    float(orow.get('baseline_to_roma_dist_px_mean', float('nan'))),
                    float(orow.get('offset_change_px_mean', float('nan'))),
                    float(orow.get('roma_to_prev_baseline_dist_px_mean', float('nan'))),
                    float(orow.get('baseline_motion_jump_px_mean', float('nan'))),
                    float(orow.get('baseline_visible_score_mean', float('nan'))),
                    float(orow.get('roma_gain_px_mean', float('nan'))),
                ),
                flush=True,
            )


if __name__ == '__main__':
    main()
