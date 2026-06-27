import argparse
import csv
import os

import numpy as np
import torch

from alltracker_runtime_utils import expand_path
from final_stage_cache_utils import load_pt, load_selector, predict_risk, select_features, sequence_paths
from eval_roma_final_stage_heuristic import mean_or_nan


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def parse_thresholds(text):
    return [float(part.strip()) for part in str(text).split(',') if part.strip()]


def finite_sum(x):
    if x.numel() == 0:
        return float('nan')
    finite = torch.isfinite(x)
    if not bool(finite.any().item()):
        return float('nan')
    return float(x[finite].sum().item())


def evaluate_risk_sample(sample, prob, threshold):
    labels = sample['baseline_risk_labels'].long()
    valid = labels >= 0
    target = labels == 1
    pred = valid & (prob >= float(threshold))
    tp = int((pred & target).sum().item())
    fp = int((pred & (~target) & valid).sum().item())
    fn = int(((~pred) & target).sum().item())
    tn = int(((~pred) & (~target) & valid).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    base_err = sample['base_err'].float()
    return {
        'seq_id': str(sample.get('seq_id', '')),
        'threshold': float(threshold),
        'valid_rows': int(valid.sum().item()),
        'risk_rows': int(target.sum().item()),
        'pred_rows': int(pred.sum().item()),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'risk_ratio': mean_or_nan(target[valid].float()),
        'pred_ratio': mean_or_nan(pred[valid].float()),
        'baseline_err_valid_mean': mean_or_nan(base_err[valid]),
        'baseline_err_risk_mean': mean_or_nan(base_err[target]),
        'baseline_err_pred_mean': mean_or_nan(base_err[pred]),
        'baseline_err_tp_mean': mean_or_nan(base_err[pred & target]),
        'baseline_err_fp_mean': mean_or_nan(base_err[pred & (~target) & valid]),
        'baseline_err_valid_sum': finite_sum(base_err[valid]),
        'baseline_err_pred_sum': finite_sum(base_err[pred]),
    }


def summarize(rows):
    total = {key: 0.0 for key in ('valid_rows', 'risk_rows', 'pred_rows', 'tp', 'fp', 'fn', 'tn')}
    for row in rows:
        for key in total:
            total[key] += float(row.get(key, 0.0))
    tp, fp, fn, tn = total['tp'], total['fp'], total['fn'], total['tn']
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    out = {
        'num_sequences': len(rows),
        **total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'risk_ratio': total['risk_rows'] / max(total['valid_rows'], 1.0),
        'pred_ratio': total['pred_rows'] / max(total['valid_rows'], 1.0),
    }
    for key in (
        'baseline_err_valid_mean',
        'baseline_err_risk_mean',
        'baseline_err_pred_mean',
        'baseline_err_tp_mean',
        'baseline_err_fp_mean',
    ):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./final_stage_adapter_eval_outputs')
    parser.add_argument('--exp', type=str, default='final_stage_baseline_risk_eval')
    parser.add_argument('--thresholds', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99')
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
    thresholds = parse_thresholds(args.thresholds)
    rows = []
    print('device:', device, flush=True)
    print('data_dir:', expand_path(args.data_dir), flush=True)
    print('ckpt:', expand_path(args.ckpt), flush=True)
    print('num_sequences:', len(paths), flush=True)
    print('patch_mode:', patch_mode, flush=True)
    for index, path in enumerate(paths, 1):
        sample = load_pt(path)
        features = select_features(sample, selected_feature_names)
        prob = predict_risk(model, normalizer, features, device, args.chunk_points, sample=sample, patch_mode=patch_mode)
        for threshold in thresholds:
            rows.append(evaluate_risk_sample(sample, prob, threshold))
        best = max(
            [row for row in rows if row['seq_id'] == str(sample.get('seq_id', ''))],
            key=lambda row: row['f1'],
        )
        print(
            'seq %04d/%04d %s best_thr %.3f f1 %.4f precision %.4f recall %.4f pred %.4f'
            % (
                index,
                len(paths),
                str(sample.get('seq_id', '')),
                best['threshold'],
                best['f1'],
                best['precision'],
                best['recall'],
                best['pred_ratio'],
            ),
            flush=True,
        )
    write_csv(os.path.join(out_dir, 'per_sequence_thresholds.csv'), rows, list(rows[0].keys()))
    summary_rows = []
    for threshold in thresholds:
        group = [row for row in rows if abs(float(row['threshold']) - float(threshold)) < 1e-9]
        summary = summarize(group)
        summary['threshold'] = float(threshold)
        summary_rows.append(summary)
    summary_rows = sorted(summary_rows, key=lambda row: float(row['threshold']))
    write_csv(os.path.join(out_dir, 'summary_thresholds.csv'), summary_rows, ['threshold'] + [k for k in summary_rows[0].keys() if k != 'threshold'])
    best_summary = max(summary_rows, key=lambda row: float(row.get('f1', 0.0)))
    write_csv(os.path.join(out_dir, 'summary_best.csv'), [best_summary], list(best_summary.keys()))
    print(
        'BEST threshold %.3f f1 %.4f precision %.4f recall %.4f pred_ratio %.4f saved=%s'
        % (
            float(best_summary['threshold']),
            float(best_summary.get('f1', float('nan'))),
            float(best_summary.get('precision', float('nan'))),
            float(best_summary.get('recall', float('nan'))),
            float(best_summary.get('pred_ratio', float('nan'))),
            out_dir,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
