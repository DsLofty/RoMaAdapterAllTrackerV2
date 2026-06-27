import torch
import torch.nn as nn


BASELINE_RISK_FEATURE_NAMES = [
    'pred_visible_score',
    'corr_best',
    'corr_second',
    'corr_margin',
    'corr_entropy',
    'update_norm',
    'last_flow_update_norm',
    'motion_jump',
    'acceleration_norm',
    'flow_residual_norm',
    'iteration_delta_norm_mean',
    'iteration_delta_norm_last',
    'baseline_speed',
    'baseline_accel',
    'local_conf_change',
    'visibility_score_change',
]

BASELINE_RISK_FEATURE_SETS = {
    'full': BASELINE_RISK_FEATURE_NAMES,
    'core': [
        'pred_visible_score',
        'corr_margin',
        'corr_entropy',
        'update_norm',
        'last_flow_update_norm',
        'motion_jump',
        'acceleration_norm',
        'iteration_delta_norm_last',
    ],
    'minimal': [
        'pred_visible_score',
        'corr_margin',
        'update_norm',
        'motion_jump',
    ],
}


def get_risk_feature_names(feature_set='full'):
    if feature_set not in BASELINE_RISK_FEATURE_SETS:
        raise ValueError('unknown risk feature_set: %s' % feature_set)
    return list(BASELINE_RISK_FEATURE_SETS[feature_set])


class BaselineRiskPredictor(nn.Module):
    """Small MLP for localization error risk, not visibility prediction.

    The target is whether the current baseline xy has large localization error.
    Visibility/confidence is only one input feature; it is not the supervised
    target for this model.
    """

    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _get_scalar(risk_info, key, count, device=None):
    value = risk_info.get(key)
    if value is None:
        return torch.zeros(count, device=device)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    elif device is not None:
        value = value.to(device=device)
    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim > 1:
        value = value.reshape(value.shape[0], -1)[:, 0]
    return value.float()


def build_risk_input(risk_info, feature_names=None, feature_set='full', device=None):
    """Build [M,D] features from exported AllTracker risk_info.

    Missing or future fields are filled with zero. NaN/Inf is clamped to zero so
    partial feature exports remain trainable and checkpoint feature_names stay
    stable across experiments.
    """
    if feature_names is None:
        feature_names = get_risk_feature_names(feature_set)
    feature_names = list(feature_names)
    if not risk_info or 'xy8' not in risk_info:
        return torch.empty(0, len(feature_names), device=device), feature_names
    count = int(risk_info['xy8'].shape[0])
    features = [_get_scalar(risk_info, name, count, device=device) for name in feature_names]
    x = torch.stack(features, dim=1).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, feature_names
