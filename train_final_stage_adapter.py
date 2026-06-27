import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from alltracker_runtime_utils import expand_path
from final_stage_adapter_model import FeatureNormalizer, feature_indices, feature_names_for_profile, make_model


def load_pt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def split_data_dirs(data_dir):
    parts = []
    for item in str(data_dir).split(','):
        item = item.strip()
        if item:
            parts.append(item)
    if not parts:
        raise ValueError('--data_dir is empty')
    return parts


def sequence_paths_one(data_dir):
    seq_dir = os.path.join(expand_path(data_dir), 'sequences')
    paths = [
        os.path.join(seq_dir, name)
        for name in sorted(os.listdir(seq_dir))
        if name.endswith('.pt')
    ]
    if not paths:
        raise RuntimeError('no .pt sequence files found in %s' % seq_dir)
    return paths


def sequence_paths(data_dir):
    out = []
    for item in split_data_dirs(data_dir):
        out.extend(sequence_paths_one(item))
    if not out:
        raise RuntimeError('no .pt sequence files found in %s' % str(data_dir))
    return out


def split_paths_global(data_dir, train_ratio, seed):
    paths = sequence_paths(data_dir)
    rng = random.Random(int(seed))
    rng.shuffle(paths)
    split = max(1, int(round(len(paths) * float(train_ratio))))
    return paths[:split], paths[split:], []


def split_paths_per_source(data_dir, train_ratio, seed):
    train_paths = []
    val_paths = []
    summaries = []
    ratio = float(train_ratio)
    for source_idx, source in enumerate(split_data_dirs(data_dir)):
        paths = sequence_paths_one(source)
        rng = random.Random(int(seed) + source_idx * 1009)
        rng.shuffle(paths)
        if ratio >= 1.0 or len(paths) <= 1:
            train_count = len(paths)
        else:
            train_count = int(round(len(paths) * ratio))
            train_count = max(1, min(train_count, len(paths) - 1))
        source_train = paths[:train_count]
        source_val = paths[train_count:]
        train_paths.extend(source_train)
        val_paths.extend(source_val)
        summaries.append(
            {
                'source': expand_path(source),
                'total': len(paths),
                'train': len(source_train),
                'val': len(source_val),
            }
        )
    rng = random.Random(int(seed) + 7919)
    rng.shuffle(train_paths)
    rng.shuffle(val_paths)
    if not train_paths:
        raise RuntimeError('no training sequence files selected')
    return train_paths, val_paths, summaries


def split_train_val_paths(data_dir, train_ratio, seed, split_mode):
    mode = str(split_mode).strip().lower()
    if mode == 'global':
        return split_paths_global(data_dir, train_ratio, seed)
    if mode in ('per_source', 'source'):
        return split_paths_per_source(data_dir, train_ratio, seed)
    raise ValueError('unknown split_mode: %s' % str(split_mode))


def select_features(sample, selected_feature_names):
    source_names = list(sample.get('feature_names', []))
    if not source_names:
        raise ValueError('sample is missing feature_names; re-export final-stage adapter data')
    indices = feature_indices(source_names, selected_feature_names)
    return sample['features'].float()[..., indices]


def select_labels(sample, target_mode):
    mode = str(target_mode).strip().lower()
    if mode in ('risk_accept_joint', 'joint_risk_accept'):
        if 'roma_accept_labels' not in sample:
            raise KeyError('sample is missing roma_accept_labels; re-export data with the updated exporter')
        return sample['roma_accept_labels'].long()
    if mode in ('baseline_risk', 'risk'):
        if 'baseline_risk_labels' not in sample:
            raise KeyError('sample is missing baseline_risk_labels; re-export data with the updated exporter')
        return sample['baseline_risk_labels'].long()
    raise ValueError('unknown target_mode: %s' % str(target_mode))


def select_risk_labels(sample):
    if 'baseline_risk_labels' not in sample:
        raise KeyError('sample is missing baseline_risk_labels; re-export data with the updated exporter')
    return sample['baseline_risk_labels'].long()


def label_counts(paths, label_fn):
    rows = 0
    pos = 0
    neg = 0
    for path in paths:
        sample = load_pt(path)
        labels = label_fn(sample)
        valid = labels >= 0
        if not bool(valid.any().item()):
            continue
        rows += int(valid.sum().item())
        pos += int((labels[valid] == 1).sum().item())
        neg += int((labels[valid] == 0).sum().item())
    return {'rows': rows, 'pos': pos, 'neg': neg, 'pos_weight': float(max(neg, 1) / max(pos, 1))}


def compute_normalizer(paths, selected_feature_names, target_mode):
    total = 0
    sum_x = None
    sum_x2 = None
    pos = 0
    neg = 0
    for path in paths:
        sample = load_pt(path)
        features = select_features(sample, selected_feature_names)
        labels = select_labels(sample, target_mode)
        valid = labels >= 0
        if not bool(valid.any().item()):
            continue
        x = features[valid]
        if sum_x is None:
            sum_x = x.sum(dim=0)
            sum_x2 = (x * x).sum(dim=0)
        else:
            sum_x += x.sum(dim=0)
            sum_x2 += (x * x).sum(dim=0)
        total += int(x.shape[0])
        pos += int((labels[valid] == 1).sum().item())
        neg += int((labels[valid] == 0).sum().item())
    if total <= 0:
        raise RuntimeError('no labeled rows found in exported data')
    mean = sum_x / float(total)
    var = torch.clamp(sum_x2 / float(total) - mean * mean, min=1e-6)
    normalizer = FeatureNormalizer(mean.float(), torch.sqrt(var).float())
    pos_weight = float(max(neg, 1) / max(pos, 1))
    return normalizer, pos_weight, {'rows': total, 'pos': pos, 'neg': neg}


def is_deep_corr_mode(patch_mode):
    return str(patch_mode).lower() in (
        'deep_corr_gate',
        'deepcorr_gate',
        'corr_grid_gate',
        'deep_corr_risk_accept',
        'deepcorr_risk_accept',
        'corr_grid_risk_accept',
    )


def build_deep_corr_input(sample, start, end):
    deep_corr = sample.get('deep_corr')
    if not isinstance(deep_corr, dict):
        raise ValueError('sample is missing deep_corr; re-export data with --save_deep_corr_features')
    roma = deep_corr['roma_corr_grid'].float().permute(1, 0, 2).contiguous()[start:end]
    baseline = deep_corr['baseline_corr_grid'].float().permute(1, 0, 2).contiguous()[start:end]
    return torch.cat([roma, baseline], dim=-1)

def iter_chunks(features, labels, chunk_points, sample=None, patch_mode='none'):
    # Exported feature shape is (T, N, F); GRU expects (N, T, F).
    x = features.float().permute(1, 0, 2).contiguous()
    y = labels.long().permute(1, 0).contiguous()
    n = int(x.shape[0])
    for start in range(0, n, int(chunk_points)):
        end = min(start + int(chunk_points), n)
        if not is_deep_corr_mode(patch_mode):
            raise ValueError('V2 release training requires deep-corr patch_mode, got %s' % str(patch_mode))
        aux = build_deep_corr_input(sample, start, end)
        yield x[start:end], y[start:end], aux, start, end


def build_sample_weight(sample, start, end, y, args):
    loss_mode = str(args.loss_mode).lower()
    if loss_mode == 'bce':
        return None
    if str(args.target_mode).lower() in ('baseline_risk', 'risk'):
        if loss_mode == 'risk_error_weighted':
            base_err = sample['base_err'].float().permute(1, 0).contiguous()[start:end]
            scale = float(args.gain_weight_scale)
            weights = torch.ones_like(base_err, dtype=torch.float32)
            weights = torch.where(y == 1, 1.0 + scale * torch.clamp(base_err, min=0.0), weights)
            weights = torch.clamp(weights, min=float(args.min_sample_weight), max=float(args.max_sample_weight))
            if bool(args.normalize_sample_weight):
                valid = y >= 0
                if bool(valid.any().item()):
                    weights = weights / torch.clamp(weights[valid].mean(), min=1.0e-6)
            return weights
        raise ValueError('loss_mode %s is not supported for baseline_risk target' % str(args.loss_mode))
    base_err = sample['base_err'].float().permute(1, 0).contiguous()[start:end]
    roma_err = sample['roma_err'].float().permute(1, 0).contiguous()[start:end]
    gain = torch.nan_to_num(base_err - roma_err, nan=0.0, posinf=0.0, neginf=0.0)
    pos_gain = torch.clamp(gain, min=0.0)
    neg_cost = torch.clamp(-gain, min=0.0)
    scale = float(args.gain_weight_scale)
    min_weight = float(args.min_sample_weight)
    max_weight = float(args.max_sample_weight)
    weights = torch.ones_like(gain, dtype=torch.float32)
    if loss_mode == 'gain_weighted':
        weights = torch.where(y == 1, 1.0 + scale * pos_gain, weights)
        weights = torch.where(y == 0, 1.0 + scale * neg_cost, weights)
    elif loss_mode == 'negative_cost_weighted':
        weights = torch.where(y == 0, 1.0 + scale * neg_cost, weights)
    else:
        raise ValueError('unknown loss_mode: %s' % str(args.loss_mode))
    weights = torch.clamp(weights, min=min_weight, max=max_weight)
    if bool(args.normalize_sample_weight):
        valid = y >= 0
        if bool(valid.any().item()):
            weights = weights / torch.clamp(weights[valid].mean(), min=1.0e-6)
    return weights


def logits_for_target(outputs, target_mode):
    if not isinstance(outputs, dict):
        return outputs
    mode = str(target_mode).strip().lower()
    if mode in ('baseline_risk', 'risk') and 'risk_logits' in outputs:
        return outputs['risk_logits']
    raise ValueError('model output does not contain risk_logits for target_mode=%s' % str(target_mode))


def load_initial_risk_state(model, path):
    if not str(path):
        return {'missing': [], 'unexpected': []}
    ckpt = load_pt(expand_path(path))
    state = ckpt.get('model', ckpt)
    result = model.load_state_dict(state, strict=False)
    return {
        'missing': list(result.missing_keys),
        'unexpected': list(result.unexpected_keys),
    }


def load_checkpoint_normalizer(path, selected_feature_names):
    ckpt = load_pt(expand_path(path))
    ckpt_feature_names = list(ckpt.get('feature_names', []))
    if ckpt_feature_names and list(ckpt_feature_names) != list(selected_feature_names):
        raise ValueError(
            'init checkpoint feature_names do not match selected features: %s vs %s'
            % (','.join(ckpt_feature_names), ','.join(selected_feature_names))
        )
    return FeatureNormalizer.from_state_dict(ckpt['normalizer'])


def freeze_risk_backbone(model):
    for name, param in model.named_parameters():
        if name.startswith('deep_encoder.') or name.startswith('temporal.'):
            param.requires_grad_(False)


def freeze_deep_encoder(model):
    for name, param in model.named_parameters():
        if name.startswith('deep_encoder.'):
            param.requires_grad_(False)


def bce_logits_loss(logits, labels, pos_weight, sample_weight=None):
    valid = labels >= 0
    if not bool(valid.any().item()):
        return None, valid
    target = labels.float()
    loss_per_row = F.binary_cross_entropy_with_logits(
        logits[valid],
        target[valid],
        reduction='none',
        pos_weight=pos_weight,
    )
    if sample_weight is not None:
        loss_per_row = loss_per_row * sample_weight[valid]
    return loss_per_row.mean(), valid


def run_epoch(model, paths, normalizer, optimizer, device, args, selected_feature_names, train=True):
    model.train(bool(train))
    total_loss = 0.0
    total_rows = 0
    total_correct = 0
    total_seen = 0
    pos_weight = torch.tensor(float(args.pos_weight), dtype=torch.float32, device=device)
    for path in paths:
        sample = load_pt(path)
        features = select_features(sample, selected_feature_names)
        labels = select_labels(sample, args.target_mode)
        for x_cpu, y_cpu, patch_cpu, start, end in iter_chunks(
            features,
            labels,
            args.chunk_points,
            sample=sample,
            patch_mode=args.patch_mode,
        ):
            x = normalizer(x_cpu.to(device)).float()
            y = y_cpu.to(device)
            patches = patch_cpu.to(device).float() if patch_cpu is not None else None
            sample_weight_cpu = build_sample_weight(sample, start, end, y_cpu, args)
            sample_weight = sample_weight_cpu.to(device).float() if sample_weight_cpu is not None else None
            valid = y >= 0
            outputs = model(x, patches)
            mode = str(args.target_mode).strip().lower()
            if mode in ('risk_accept_joint', 'joint_risk_accept'):
                if not isinstance(outputs, dict) or 'accept_logits' not in outputs or 'risk_logits' not in outputs:
                    raise ValueError('risk_accept_joint requires outputs with accept_logits and risk_logits')
                accept_loss, accept_valid = bce_logits_loss(
                    outputs['accept_logits'],
                    y,
                    pos_weight,
                    sample_weight=sample_weight,
                )
                risk_y_cpu = select_risk_labels(sample).permute(1, 0).contiguous()[start:end]
                risk_y = risk_y_cpu.to(device)
                risk_pos_weight = torch.tensor(float(args.risk_pos_weight), dtype=torch.float32, device=device)
                risk_loss, risk_valid = bce_logits_loss(
                    outputs['risk_logits'],
                    risk_y,
                    risk_pos_weight,
                    sample_weight=None,
                )
                losses = []
                if accept_loss is not None:
                    losses.append(accept_loss)
                if risk_loss is not None and float(args.joint_risk_loss_weight) > 0:
                    losses.append(float(args.joint_risk_loss_weight) * risk_loss)
                if not losses:
                    continue
                loss = sum(losses)
                logits = outputs['accept_logits']
                valid = accept_valid
            else:
                if not bool(valid.any().item()):
                    continue
                logits = logits_for_target(outputs, args.target_mode)
                gate_loss, valid = bce_logits_loss(logits, y, pos_weight, sample_weight=sample_weight)
                loss = gate_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.grad_clip))
                optimizer.step()
            with torch.no_grad():
                pred = logits >= 0.0
                total_correct += int((pred[valid] == (y[valid] == 1)).sum().item())
                total_seen += int(valid.sum().item())
                total_loss += float(loss.item()) * int(valid.sum().item())
                total_rows += int(valid.sum().item())
    return {
        'loss': total_loss / max(total_rows, 1),
        'acc': float(total_correct / max(total_seen, 1)),
        'rows': int(total_rows),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='cache directory, or comma-separated cache directories')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_final_stage_adapter')
    parser.add_argument('--exp', type=str, default='final_stage_v2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--chunk_points', type=int, default=256)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--split_mode', choices=['global', 'per_source'], default='global')
    parser.add_argument('--best_metric', choices=['val_loss', 'val_acc', 'train_loss', 'train_acc'], default='val_loss')
    parser.add_argument('--save_epoch_every', type=int, default=0)
    parser.add_argument('--pos_weight', type=float, default=-1.0)
    parser.add_argument('--feature_profile', choices=['lowdim'], default='lowdim')
    parser.add_argument(
        '--target_mode',
        choices=['baseline_risk', 'risk_accept_joint'],
        default='baseline_risk',
    )
    parser.add_argument(
        '--patch_mode',
        choices=['deep_corr_gate', 'deep_corr_risk_accept'],
        default='deep_corr_gate',
    )
    parser.add_argument('--deep_embed_dim', type=int, default=64)
    parser.add_argument('--init_risk_ckpt', type=str, default='')
    parser.add_argument('--init_normalizer_from_ckpt', action='store_true', default=False)
    parser.add_argument('--freeze_risk_backbone', action='store_true', default=False)
    parser.add_argument('--freeze_deep_encoder', action='store_true', default=False)
    parser.add_argument('--joint_risk_loss_weight', type=float, default=0.2)
    parser.add_argument('--risk_pos_weight', type=float, default=-1.0)
    parser.add_argument('--loss_mode', choices=['bce', 'gain_weighted', 'negative_cost_weighted', 'risk_error_weighted'], default='bce')
    parser.add_argument('--gain_weight_scale', type=float, default=0.25)
    parser.add_argument('--min_sample_weight', type=float, default=0.1)
    parser.add_argument('--max_sample_weight', type=float, default=10.0)
    parser.add_argument('--normalize_sample_weight', action='store_true', default=True)
    parser.add_argument('--no_normalize_sample_weight', action='store_false', dest='normalize_sample_weight')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    train_paths, val_paths, split_summaries = split_train_val_paths(
        args.data_dir,
        args.train_ratio,
        args.seed,
        args.split_mode,
    )
    selected_feature_names = feature_names_for_profile(args.feature_profile)
    normalizer, auto_pos_weight, counts = compute_normalizer(train_paths, selected_feature_names, args.target_mode)
    risk_counts = None
    if str(args.target_mode).strip().lower() in ('risk_accept_joint', 'joint_risk_accept'):
        risk_counts = label_counts(train_paths, select_risk_labels)
        if float(args.risk_pos_weight) <= 0:
            args.risk_pos_weight = float(risk_counts['pos_weight'])
    if bool(args.init_normalizer_from_ckpt):
        if not str(args.init_risk_ckpt):
            raise ValueError('--init_normalizer_from_ckpt requires --init_risk_ckpt')
        normalizer = load_checkpoint_normalizer(args.init_risk_ckpt, selected_feature_names)
    if float(args.pos_weight) <= 0:
        args.pos_weight = float(auto_pos_weight)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(
        in_dim=len(selected_feature_names),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        patch_mode=str(args.patch_mode),
        deep_corr_dim=50,
        deep_embed_dim=int(args.deep_embed_dim),
    ).to(device)
    init_info = load_initial_risk_state(model, args.init_risk_ckpt) if str(args.init_risk_ckpt) else None
    if bool(args.freeze_risk_backbone):
        freeze_risk_backbone(model)
    if bool(args.freeze_deep_encoder):
        freeze_deep_encoder(model)
    normalizer = normalizer.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError('no trainable parameters remain after freezing')
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    out_dir = expand_path(args.save_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_metric = float('-inf') if str(args.best_metric).endswith('_acc') else float('inf')
    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    best_train_loss = float('inf')
    best_train_acc = float('-inf')
    started = time.time()
    print('device:', device, flush=True)
    print('data_dir:', ','.join(expand_path(p) for p in split_data_dirs(args.data_dir)), flush=True)
    print(
        'split_mode:',
        str(args.split_mode),
        'train_ratio:',
        float(args.train_ratio),
        'train_sequences:',
        len(train_paths),
        'val_sequences:',
        len(val_paths),
        flush=True,
    )
    if split_summaries:
        for item in split_summaries:
            print(
                'source_split:',
                item['source'],
                'total',
                item['total'],
                'train',
                item['train'],
                'val',
                item['val'],
                flush=True,
            )
    print('feature_profile:', str(args.feature_profile), 'num_features:', len(selected_feature_names), flush=True)
    print('target_mode:', str(args.target_mode), flush=True)
    if init_info is not None:
        print(
            'init_risk_ckpt:',
            expand_path(args.init_risk_ckpt),
            'missing:',
            len(init_info['missing']),
            'unexpected:',
            len(init_info['unexpected']),
            flush=True,
        )
    print('init_normalizer_from_ckpt:', bool(args.init_normalizer_from_ckpt), flush=True)
    print(
        'freeze_risk_backbone:',
        bool(args.freeze_risk_backbone),
        'freeze_deep_encoder:',
        bool(args.freeze_deep_encoder),
        'trainable_params:',
        sum(p.numel() for p in trainable_params),
        flush=True,
    )
    print(
        'patch_mode:',
        str(args.patch_mode),
        'deep_embed_dim:',
        int(args.deep_embed_dim),
        flush=True,
    )
    print(
        'loss_mode:',
        str(args.loss_mode),
        'gain_weight_scale:',
        float(args.gain_weight_scale),
        'sample_weight_range:',
        float(args.min_sample_weight),
        float(args.max_sample_weight),
        'normalize_sample_weight:',
        bool(args.normalize_sample_weight),
        flush=True,
    )
    print('feature_names:', ','.join(selected_feature_names), flush=True)
    print('rows:', counts['rows'], 'pos:', counts['pos'], 'neg:', counts['neg'], 'pos_weight:', args.pos_weight, flush=True)
    print('best_metric:', str(args.best_metric), 'save_epoch_every:', int(args.save_epoch_every), flush=True)
    if risk_counts is not None:
        print(
            'risk_rows:',
            risk_counts['rows'],
            'risk_pos:',
            risk_counts['pos'],
            'risk_neg:',
            risk_counts['neg'],
            'risk_pos_weight:',
            args.risk_pos_weight,
            'joint_risk_loss_weight:',
            args.joint_risk_loss_weight,
            flush=True,
        )
    for epoch in range(1, int(args.epochs) + 1):
        train_stats = run_epoch(model, train_paths, normalizer, optimizer, device, args, selected_feature_names, train=True)
        if val_paths:
            with torch.no_grad():
                val_stats = run_epoch(model, val_paths, normalizer, optimizer, device, args, selected_feature_names, train=False)
        else:
            val_stats = {'loss': float('nan'), 'acc': float('nan'), 'rows': 0}
        if val_paths:
            metric_values = {
                'val_loss': float(val_stats['loss']),
                'val_acc': float(val_stats['acc']),
                'train_loss': float(train_stats['loss']),
                'train_acc': float(train_stats['acc']),
            }
        else:
            metric_values = {
                'val_loss': float(train_stats['loss']),
                'val_acc': float(train_stats['acc']),
                'train_loss': float(train_stats['loss']),
                'train_acc': float(train_stats['acc']),
            }
        metric = metric_values[str(args.best_metric)]
        print(
            'epoch %03d train_loss %.5f train_acc %.4f val_loss %.5f val_acc %.4f time %.1fs'
            % (
                epoch,
                train_stats['loss'],
                train_stats['acc'],
                val_stats['loss'],
                val_stats['acc'],
                time.time() - started,
            ),
            flush=True,
        )
        ckpt = {
            'model': model.state_dict(),
            'normalizer': normalizer.state_dict(),
            'feature_names': list(selected_feature_names),
            'feature_profile': str(args.feature_profile),
            'target_mode': str(args.target_mode),
            'args': vars(args),
            'epoch': int(epoch),
            'train_stats': train_stats,
            'val_stats': val_stats,
        }
        latest_path = os.path.join(out_dir, '%s_latest.pth' % str(args.exp))
        torch.save(ckpt, latest_path)
        if int(args.save_epoch_every) > 0 and epoch % int(args.save_epoch_every) == 0:
            epoch_path = os.path.join(out_dir, '%s_epoch%03d.pth' % (str(args.exp), int(epoch)))
            torch.save(ckpt, epoch_path)
            print('saved epoch:', epoch_path, flush=True)
        if val_paths and float(val_stats['loss']) < best_val_loss:
            best_val_loss = float(val_stats['loss'])
            best_val_loss_path = os.path.join(out_dir, '%s_best_val_loss.pth' % str(args.exp))
            torch.save(ckpt, best_val_loss_path)
            print('saved best_val_loss:', best_val_loss_path, flush=True)
        if val_paths and float(val_stats['acc']) > best_val_acc:
            best_val_acc = float(val_stats['acc'])
            best_val_acc_path = os.path.join(out_dir, '%s_best_val_acc.pth' % str(args.exp))
            torch.save(ckpt, best_val_acc_path)
            print('saved best_val_acc:', best_val_acc_path, flush=True)
        if float(train_stats['loss']) < best_train_loss:
            best_train_loss = float(train_stats['loss'])
            best_train_loss_path = os.path.join(out_dir, '%s_best_train_loss.pth' % str(args.exp))
            torch.save(ckpt, best_train_loss_path)
        if float(train_stats['acc']) > best_train_acc:
            best_train_acc = float(train_stats['acc'])
            best_train_acc_path = os.path.join(out_dir, '%s_best_train_acc.pth' % str(args.exp))
            torch.save(ckpt, best_train_acc_path)
        better = metric > best_metric if str(args.best_metric).endswith('_acc') else metric < best_metric
        if better:
            best_metric = float(metric)
            best_path = os.path.join(out_dir, '%s_best.pth' % str(args.exp))
            torch.save(ckpt, best_path)
            print('saved best_%s:' % str(args.best_metric), best_path, flush=True)


if __name__ == '__main__':
    main()
