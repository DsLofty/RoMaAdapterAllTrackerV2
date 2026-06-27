import argparse
import csv
import os

import numpy as np
import torch

from alltracker_runtime_utils import expand_path
from final_stage_cache_utils import build_deep_corr_input, load_pt, load_selector, select_features, sequence_paths
from eval_roma_final_stage_heuristic import error_map, mean_or_nan
from tapvid_metric_utils import tapvid_metrics


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def finite_sum(x):
    if x.numel() == 0:
        return float('nan')
    finite = torch.isfinite(x)
    if not bool(finite.any().item()):
        return float('nan')
    return float(x[finite].sum().item())


def parse_float_list(text):
    return [float(item.strip()) for item in str(text).split(',') if item.strip()]


@torch.no_grad()
def predict_risk_accept(model, normalizer, features, device, chunk_points, sample=None, patch_mode='none'):
    x_all = features.float().permute(1, 0, 2).contiguous()
    risk_probs = []
    accept_probs = []
    for start in range(0, int(x_all.shape[0]), int(chunk_points)):
        end = min(start + int(chunk_points), int(x_all.shape[0]))
        x = normalizer(x_all[start:end].to(device)).float()
        if str(patch_mode).lower() not in ('deep_corr_risk_accept', 'deepcorr_risk_accept', 'corr_grid_risk_accept'):
            raise ValueError('risk+accept evaluation requires deep_corr_risk_accept patch_mode')
        deep_corr = build_deep_corr_input(sample, start, end).to(device).float()
        outputs = model(x, deep_corr)
        if not isinstance(outputs, dict) or 'risk_logits' not in outputs or 'accept_logits' not in outputs:
            raise ValueError('risk+accept evaluation requires a model with risk_logits and accept_logits')
        risk_probs.append(torch.sigmoid(outputs['risk_logits']).cpu())
        accept_probs.append(torch.sigmoid(outputs['accept_logits']).cpu())
    risk_prob = torch.cat(risk_probs, dim=0).permute(1, 0).contiguous()
    accept_prob = torch.cat(accept_probs, dim=0).permute(1, 0).contiguous()
    return risk_prob, accept_prob


def replace_xy(base_xy, roma_xy, gate):
    out = base_xy.clone()
    if bool(gate.any().item()):
        out[gate] = roma_xy[gate]
    return out


def evaluate_sample(sample, risk_prob, accept_prob, risk_thr, accept_thr):
    base_xy = sample['reference_xy'].float()
    roma_xy = sample['roma_xy'].float()
    gt = sample['gt_xy'].float()
    vis = sample['vis_g'].float()
    first = sample['first_positive_inds'].long()
    pred_visible = sample['pred_visible_score'].float()
    mask = sample['mask'].bool()
    candidate = sample['candidate_mask'].bool()
    gate = mask & candidate & (risk_prob >= float(risk_thr)) & (accept_prob >= float(accept_thr))
    pred_xy = replace_xy(base_xy, roma_xy, gate)
    base_err = error_map(base_xy, gt)
    pred_err = error_map(pred_xy, gt)
    roma_err = error_map(roma_xy, gt)
    image_size = tuple(int(v) for v in sample.get('image_size', (448, 768)))
    base_metrics = tapvid_metrics(base_xy, pred_visible, gt, vis, first, image_size)
    pred_metrics = tapvid_metrics(pred_xy, pred_visible, gt, vis, first, image_size)
    accepted_delta = roma_err - base_err
    rejected = mask & candidate & (~gate)
    oracle = mask & candidate & torch.isfinite(roma_err) & (roma_err < base_err)
    return {
        'seq_id': str(sample.get('seq_id', '')),
        'risk_threshold': float(risk_thr),
        'accept_threshold': float(accept_thr),
        'baseline_da': base_metrics['da'],
        'baseline_aj': base_metrics['aj'],
        'baseline_oa': base_metrics['oa'],
        'selector_da': pred_metrics['da'],
        'selector_aj': pred_metrics['aj'],
        'selector_oa': pred_metrics['oa'],
        'selector_delta_da': pred_metrics['da'] - base_metrics['da'],
        'selector_delta_aj': pred_metrics['aj'] - base_metrics['aj'],
        'selector_delta_oa': pred_metrics['oa'] - base_metrics['oa'],
        'baseline_epe_mean': mean_or_nan(base_err[mask]),
        'selector_epe_mean': mean_or_nan(pred_err[mask]),
        'selector_delta_epe': mean_or_nan(pred_err[mask] - base_err[mask]),
        'accepted_delta_epe': mean_or_nan(accepted_delta[gate]),
        'rejected_delta_epe': mean_or_nan(accepted_delta[rejected]),
        'risk_pred_ratio': mean_or_nan((risk_prob >= float(risk_thr))[mask].float()),
        'accept_pred_ratio': mean_or_nan((accept_prob >= float(accept_thr))[mask & candidate].float()),
        'gate_ratio': mean_or_nan(gate[mask].float()),
        'oracle_ratio': mean_or_nan(oracle[mask].float()),
        'mask_rows': int(mask.sum().item()),
        'candidate_rows': int((mask & candidate).sum().item()),
        'accepted_rows': int(gate.sum().item()),
        'rejected_rows': int(rejected.sum().item()),
        'oracle_rows': int(oracle.sum().item()),
        'baseline_epe_sum': finite_sum(base_err[mask]),
        'selector_epe_sum': finite_sum(pred_err[mask]),
        'selector_delta_epe_sum': finite_sum((pred_err - base_err)[mask]),
        'accepted_delta_epe_sum': finite_sum(accepted_delta[gate]),
        'rejected_delta_epe_sum': finite_sum(accepted_delta[rejected]),
    }


def summarize(rows):
    out = {'num_sequences': len(rows)}
    if not rows:
        return out
    for key in rows[0].keys():
        if key in ('seq_id',):
            continue
        if key.endswith('_sum') or key.endswith('_rows') or key in ('mask_rows', 'candidate_rows'):
            continue
        values = []
        for row in rows:
            try:
                value = float(row.get(key, float('nan')))
            except (TypeError, ValueError):
                value = float('nan')
            if np.isfinite(value):
                values.append(value)
        out[key] = float(np.mean(values)) if values else float('nan')
    return out


def row_weighted_summarize(rows):
    out = {'num_sequences': len(rows)}
    if not rows:
        return out

    def sum_key(key):
        total = 0.0
        ok = False
        for row in rows:
            try:
                value = float(row.get(key, float('nan')))
            except (TypeError, ValueError):
                value = float('nan')
            if np.isfinite(value):
                total += value
                ok = True
        return total if ok else float('nan')

    mask_rows = sum_key('mask_rows')
    accepted_rows = sum_key('accepted_rows')
    rejected_rows = sum_key('rejected_rows')
    out['mask_rows'] = mask_rows
    out['candidate_rows'] = sum_key('candidate_rows')
    out['accepted_rows'] = accepted_rows
    out['rejected_rows'] = rejected_rows
    out['oracle_rows'] = sum_key('oracle_rows')
    out['baseline_epe_mean'] = sum_key('baseline_epe_sum') / max(mask_rows, 1.0)
    out['selector_epe_mean'] = sum_key('selector_epe_sum') / max(mask_rows, 1.0)
    out['selector_delta_epe'] = sum_key('selector_delta_epe_sum') / max(mask_rows, 1.0)
    out['accepted_delta_epe'] = sum_key('accepted_delta_epe_sum') / max(accepted_rows, 1.0)
    out['rejected_delta_epe'] = sum_key('rejected_delta_epe_sum') / max(rejected_rows, 1.0)
    out['gate_ratio'] = accepted_rows / max(mask_rows, 1.0)
    out['oracle_ratio'] = out['oracle_rows'] / max(mask_rows, 1.0)
    return out


def grouped_summaries(rows, summarize_fn):
    summary_rows = []
    keys = sorted({(float(row['risk_threshold']), float(row['accept_threshold'])) for row in rows})
    for risk_thr, accept_thr in keys:
        group = [
            row for row in rows
            if float(row['risk_threshold']) == risk_thr and float(row['accept_threshold']) == accept_thr
        ]
        summary = summarize_fn(group)
        summary['risk_threshold'] = risk_thr
        summary['accept_threshold'] = accept_thr
        summary_rows.append(summary)
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./final_stage_adapter_eval_outputs')
    parser.add_argument('--exp', type=str, default='final_stage_risk_accept')
    parser.add_argument('--risk_thresholds', type=str, default='0.1,0.2,0.3,0.5,0.7,0.9')
    parser.add_argument('--accept_thresholds', type=str, default='0.1,0.2,0.3,0.5,0.7,0.9,0.95,0.98,0.99')
    parser.add_argument('--chunk_points', type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.join(expand_path(args.save_dir), str(args.exp))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, normalizer, ckpt, selected_feature_names = load_selector(args.ckpt, device)
    patch_mode = str(ckpt.get('args', {}).get('patch_mode', 'none'))
    paths = sequence_paths(args.data_dir)
    risk_thresholds = parse_float_list(args.risk_thresholds)
    accept_thresholds = parse_float_list(args.accept_thresholds)
    rows = []
    print('device:', device, flush=True)
    print('data_dir:', expand_path(args.data_dir), flush=True)
    print('ckpt:', expand_path(args.ckpt), flush=True)
    print('num_sequences:', len(paths), flush=True)
    print('patch_mode:', patch_mode, flush=True)
    print('risk_thresholds:', ','.join(str(v) for v in risk_thresholds), flush=True)
    print('accept_thresholds:', ','.join(str(v) for v in accept_thresholds), flush=True)
    for index, path in enumerate(paths, 1):
        sample = load_pt(path)
        features = select_features(sample, selected_feature_names)
        risk_prob, accept_prob = predict_risk_accept(
            model,
            normalizer,
            features,
            device,
            args.chunk_points,
            sample=sample,
            patch_mode=patch_mode,
        )
        seq_rows = []
        for risk_thr in risk_thresholds:
            for accept_thr in accept_thresholds:
                row = evaluate_sample(sample, risk_prob, accept_prob, risk_thr, accept_thr)
                rows.append(row)
                seq_rows.append(row)
        best = min(seq_rows, key=lambda row: float(row['selector_delta_epe']))
        print(
            'seq %04d/%04d %s best risk %.3f accept %.3f delta_epe %.5f gate %.5f accepted %.5f'
            % (
                index,
                len(paths),
                str(sample.get('seq_id', '')),
                float(best['risk_threshold']),
                float(best['accept_threshold']),
                float(best['selector_delta_epe']),
                float(best['gate_ratio']),
                float(best['accepted_delta_epe']),
            ),
            flush=True,
        )
    write_csv(os.path.join(out_dir, 'per_sequence_thresholds.csv'), rows, list(rows[0].keys()))
    summary_rows = grouped_summaries(rows, summarize)
    write_csv(
        os.path.join(out_dir, 'summary_thresholds.csv'),
        summary_rows,
        ['risk_threshold', 'accept_threshold'] + [k for k in summary_rows[0].keys() if k not in ('risk_threshold', 'accept_threshold')],
    )
    weighted_rows = grouped_summaries(rows, row_weighted_summarize)
    write_csv(
        os.path.join(out_dir, 'summary_thresholds_row_weighted.csv'),
        weighted_rows,
        ['risk_threshold', 'accept_threshold'] + [k for k in weighted_rows[0].keys() if k not in ('risk_threshold', 'accept_threshold')],
    )
    best = min(summary_rows, key=lambda row: float(row.get('selector_delta_epe', float('inf'))))
    best_weighted = min(weighted_rows, key=lambda row: float(row.get('selector_delta_epe', float('inf'))))
    write_csv(os.path.join(out_dir, 'summary_best.csv'), [best], list(best.keys()))
    write_csv(os.path.join(out_dir, 'summary_best_row_weighted.csv'), [best_weighted], list(best_weighted.keys()))
    print(
        'BEST risk %.3f accept %.3f delta_epe %.5f gate %.5f accepted %.5f da_delta %.6f aj_delta %.6f saved=%s'
        % (
            float(best['risk_threshold']),
            float(best['accept_threshold']),
            float(best.get('selector_delta_epe', float('nan'))),
            float(best.get('gate_ratio', float('nan'))),
            float(best.get('accepted_delta_epe', float('nan'))),
            float(best.get('selector_delta_da', float('nan'))),
            float(best.get('selector_delta_aj', float('nan'))),
            out_dir,
        ),
        flush=True,
    )
    print(
        'BEST_ROW_WEIGHTED risk %.3f accept %.3f delta_epe %.5f gate %.5f accepted %.5f saved=%s'
        % (
            float(best_weighted['risk_threshold']),
            float(best_weighted['accept_threshold']),
            float(best_weighted.get('selector_delta_epe', float('nan'))),
            float(best_weighted.get('gate_ratio', float('nan'))),
            float(best_weighted.get('accepted_delta_epe', float('nan'))),
            out_dir,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
