import os

import torch

from alltracker_runtime_utils import expand_path
from final_stage_adapter_model import FeatureNormalizer, FINAL_STAGE_FEATURE_NAMES, feature_indices, make_model


def load_pt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def sequence_paths(data_dir):
    seq_dir = os.path.join(expand_path(data_dir), 'sequences')
    paths = [
        os.path.join(seq_dir, name)
        for name in sorted(os.listdir(seq_dir))
        if name.endswith('.pt')
    ]
    if not paths:
        raise RuntimeError('no .pt sequence files found in %s' % seq_dir)
    return paths


def load_selector(path, device):
    ckpt = load_pt(expand_path(path))
    feature_names = list(ckpt.get('feature_names', FINAL_STAGE_FEATURE_NAMES))
    args = ckpt.get('args', {})
    model = make_model(
        in_dim=len(feature_names),
        hidden_dim=int(args.get('hidden_dim', 64)),
        num_layers=int(args.get('num_layers', 1)),
        dropout=float(args.get('dropout', 0.0)),
        patch_mode=str(args.get('patch_mode', 'deep_corr_risk_accept')),
        deep_corr_dim=50,
        deep_embed_dim=int(args.get('deep_embed_dim', 64)),
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    normalizer = FeatureNormalizer.from_state_dict(ckpt['normalizer']).to(device)
    return model, normalizer, ckpt, feature_names


def select_features(sample, selected_feature_names):
    source_names = list(sample.get('feature_names', []))
    if not source_names:
        raise ValueError('sample is missing feature_names; re-export final-stage adapter data')
    indices = feature_indices(source_names, selected_feature_names)
    return sample['features'].float()[..., indices]


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


@torch.no_grad()
def predict_risk(model, normalizer, features, device, chunk_points, sample=None, patch_mode='deep_corr_gate'):
    x_all = features.float().permute(1, 0, 2).contiguous()
    probs = []
    for start in range(0, int(x_all.shape[0]), int(chunk_points)):
        end = min(start + int(chunk_points), int(x_all.shape[0]))
        x = normalizer(x_all[start:end].to(device)).float()
        if not is_deep_corr_mode(patch_mode):
            raise ValueError('V2 release models require deep-corr patch_mode, got %s' % str(patch_mode))
        deep_corr = build_deep_corr_input(sample, start, end).to(device).float()
        outputs = model(x, deep_corr)
        if isinstance(outputs, dict):
            logits = outputs.get('risk_logits')
            if logits is None:
                raise ValueError('model output is missing risk_logits')
        else:
            logits = outputs
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs, dim=0).permute(1, 0).contiguous()
