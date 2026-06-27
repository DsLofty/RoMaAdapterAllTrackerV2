import argparse
import os
import shutil

import torch

from alltracker_runtime_utils import expand_path


def load_pt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def sequence_paths(cache_dir):
    seq_dir = os.path.join(expand_path(cache_dir), 'sequences')
    if not os.path.isdir(seq_dir):
        raise RuntimeError('missing sequence directory: %s' % seq_dir)
    paths = [
        os.path.join(seq_dir, name)
        for name in sorted(os.listdir(seq_dir))
        if name.endswith('.pt')
    ]
    if not paths:
        raise RuntimeError('no .pt files found in %s' % seq_dir)
    return paths


def relabel_sample(sample, args):
    base_err = sample['base_err'].float()
    roma_err = sample['roma_err'].float()
    candidate = sample['candidate_mask'].bool()
    mask = sample.get('mask', torch.ones_like(candidate)).bool()
    risk_labels = sample.get('baseline_risk_labels')
    if risk_labels is not None:
        risk_labels = risk_labels.long()

    valid = mask & candidate & torch.isfinite(base_err) & torch.isfinite(roma_err)
    if bool(args.require_risk_positive):
        if risk_labels is None:
            raise KeyError('sample is missing baseline_risk_labels; cannot use --require_risk_positive')
        valid = valid & (risk_labels == 1)

    gain = base_err - roma_err
    cost = roma_err - base_err

    positive = (
        valid
        & (base_err >= float(args.pos_base_min_px))
        & (roma_err <= float(args.pos_roma_max_px))
        & (gain >= float(args.pos_gain_min_px))
    )
    negative_good_base_bad_roma = (
        valid
        & (base_err <= float(args.neg_base_max_px))
        & (roma_err >= float(args.neg_roma_min_px))
    )
    negative_large_cost = valid & (cost >= float(args.neg_cost_min_px))
    negative = (negative_good_base_bad_roma | negative_large_cost) & (~positive)

    labels = torch.full(base_err.shape, -1, dtype=torch.long)
    labels[positive] = 1
    labels[negative] = 0
    out = dict(sample)
    out['roma_accept_labels'] = labels
    out['roma_accept_meta'] = {
        'label_version': str(args.label_version),
        'require_risk_positive': bool(args.require_risk_positive),
        'pos_base_min_px': float(args.pos_base_min_px),
        'pos_roma_max_px': float(args.pos_roma_max_px),
        'pos_gain_min_px': float(args.pos_gain_min_px),
        'neg_base_max_px': float(args.neg_base_max_px),
        'neg_roma_min_px': float(args.neg_roma_min_px),
        'neg_cost_min_px': float(args.neg_cost_min_px),
    }
    stats = {
        'rows': int((labels >= 0).sum().item()),
        'pos': int((labels == 1).sum().item()),
        'neg': int((labels == 0).sum().item()),
        'candidate': int(candidate.sum().item()),
        'risk_pos': int((risk_labels == 1).sum().item()) if risk_labels is not None else -1,
    }
    return out, stats


def main():
    args = parse_args()
    src = expand_path(args.src)
    dst = expand_path(args.dst)
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError('--src and --dst must be different')
    src_seq_paths = sequence_paths(src)
    dst_seq_dir = os.path.join(dst, 'sequences')
    os.makedirs(dst_seq_dir, exist_ok=True)

    total = {'rows': 0, 'pos': 0, 'neg': 0, 'candidate': 0, 'risk_pos': 0}
    manifest = []
    for index, path in enumerate(src_seq_paths, 1):
        sample = load_pt(path)
        relabeled, stats = relabel_sample(sample, args)
        filename = os.path.basename(path)
        out_path = os.path.join(dst_seq_dir, filename)
        torch.save(relabeled, out_path)
        manifest.append({'seq_id': str(relabeled.get('seq_id', '')), 'file': filename})
        for key in total:
            total[key] += int(stats[key])
        print(
            'seq %04d/%04d %s accept_rows %d accept_pos %d accept_neg %d candidate %d risk_pos %d'
            % (
                index,
                len(src_seq_paths),
                str(relabeled.get('seq_id', '')),
                stats['rows'],
                stats['pos'],
                stats['neg'],
                stats['candidate'],
                stats['risk_pos'],
            ),
            flush=True,
        )

    src_manifest = os.path.join(src, 'manifest.pt')
    if os.path.exists(src_manifest):
        manifest_obj = load_pt(src_manifest)
        manifest_obj = dict(manifest_obj)
        manifest_obj['manifest'] = manifest
        manifest_obj['accept_relabel_meta'] = {
            'src': src,
            'dst': dst,
            'label_version': str(args.label_version),
            'require_risk_positive': bool(args.require_risk_positive),
            'pos_base_min_px': float(args.pos_base_min_px),
            'pos_roma_max_px': float(args.pos_roma_max_px),
            'pos_gain_min_px': float(args.pos_gain_min_px),
            'neg_base_max_px': float(args.neg_base_max_px),
            'neg_roma_min_px': float(args.neg_roma_min_px),
            'neg_cost_min_px': float(args.neg_cost_min_px),
        }
        torch.save(manifest_obj, os.path.join(dst, 'manifest.pt'))
    else:
        torch.save({'manifest': manifest, 'accept_relabel_meta': vars(args)}, os.path.join(dst, 'manifest.pt'))

    for name in ('README.md', 'summary.csv'):
        src_file = os.path.join(src, name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dst, name))

    print(
        'SUMMARY src=%s dst=%s rows=%d pos=%d neg=%d candidate=%d risk_pos=%d'
        % (src, dst, total['rows'], total['pos'], total['neg'], total['candidate'], total['risk_pos']),
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--label_version', type=str, default='strong_v2')
    parser.add_argument('--require_risk_positive', action='store_true', default=True)
    parser.add_argument('--no_require_risk_positive', action='store_false', dest='require_risk_positive')
    parser.add_argument('--pos_base_min_px', type=float, default=8.0)
    parser.add_argument('--pos_roma_max_px', type=float, default=4.0)
    parser.add_argument('--pos_gain_min_px', type=float, default=8.0)
    parser.add_argument('--neg_base_max_px', type=float, default=4.0)
    parser.add_argument('--neg_roma_min_px', type=float, default=12.0)
    parser.add_argument('--neg_cost_min_px', type=float, default=8.0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
