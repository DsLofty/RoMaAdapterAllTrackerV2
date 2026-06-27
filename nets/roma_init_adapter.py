import torch
import torch.nn as nn


def get_roma_adapter_feature_names():
    return [
        'roma_valid',
        'roma_certainty',
        'roma_offset_x',
        'roma_offset_y',
        'roma_offset_norm',
        'roma_to_prev_baseline_dist',
        'roma_temporal_jump',
        'baseline_visible_score',
        'baseline_confidence',
        'baseline_motion_jump',
        'baseline_speed',
        'baseline_accel',
        'baseline_risk_prob',
        'baseline_corr_margin',
        'baseline_update_norm',
        'query_anchor_age',
        'source_is_query_anchor',
        'query_to_baseline_offset_norm',
        'frame_index_norm',
    ]


def get_roma_recurrent_adapter_feature_names():
    return [
        'roma_valid',
        'roma_certainty',
        'roma_offset_x8',
        'roma_offset_y8',
        'roma_offset_norm8',
        'roma_to_prev_baseline_dist8',
        'roma_temporal_jump8',
        'baseline_visible_score',
        'baseline_motion_jump8',
        'baseline_speed8',
        'baseline_accel8',
        'baseline_risk_prob',
        'baseline_corr_margin',
        'baseline_update_norm',
        'query_anchor_age_norm',
        'source_is_query_anchor',
        'normalized_frame_index',
        'normalized_window_start_index',
    ]


def _as_tensor(batch, key, device=None, default=None, dtype=torch.float32):
    value = batch.get(key, default)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device)
    return value


def _optional_field(batch, key, length, device, default_value=0.0):
    value = _as_tensor(batch, key, device=device)
    if value is None:
        return torch.full((length,), float(default_value), device=device, dtype=torch.float32)
    value = value.reshape(length).float()
    return torch.nan_to_num(value, nan=float(default_value), posinf=float(default_value), neginf=float(default_value))


def build_roma_adapter_features(batch, feature_names=None, device=None):
    baseline_xy = _as_tensor(batch, 'baseline_xy', device=device)
    roma_xy = _as_tensor(batch, 'roma_xy', device=device)
    if baseline_xy is None or roma_xy is None:
        raise ValueError('batch must contain baseline_xy and roma_xy')
    if baseline_xy.ndim != 2 or baseline_xy.shape[-1] != 2:
        raise ValueError('baseline_xy must have shape [B, 2]')
    B = int(baseline_xy.shape[0])

    roma_valid = _optional_field(batch, 'roma_valid', B, baseline_xy.device, 0.0)
    roma_certainty = _optional_field(batch, 'roma_certainty', B, baseline_xy.device, 0.0)
    offset = torch.nan_to_num(roma_xy.float() - baseline_xy.float(), nan=0.0, posinf=0.0, neginf=0.0)
    offset_norm = torch.linalg.vector_norm(offset, dim=-1)

    frame_index = _optional_field(batch, 'frame_index', B, baseline_xy.device, 0.0)
    if 'frame_index_norm' in batch:
        frame_index_norm = _optional_field(batch, 'frame_index_norm', B, baseline_xy.device, 0.0)
    else:
        denom = torch.clamp(torch.max(frame_index), min=1.0)
        frame_index_norm = frame_index / denom

    features = {
        'roma_valid': roma_valid,
        'roma_certainty': roma_certainty,
        'roma_offset_x': offset[:, 0],
        'roma_offset_y': offset[:, 1],
        'roma_offset_norm': offset_norm,
        'roma_to_prev_baseline_dist': _optional_field(batch, 'roma_to_prev_baseline_dist', B, baseline_xy.device, 0.0),
        'roma_temporal_jump': _optional_field(batch, 'roma_temporal_jump', B, baseline_xy.device, 0.0),
        'baseline_visible_score': _optional_field(
            batch,
            'baseline_visible_score' if 'baseline_visible_score' in batch else 'pred_visible_score',
            B,
            baseline_xy.device,
            0.0,
        ),
        'baseline_confidence': _optional_field(batch, 'baseline_confidence', B, baseline_xy.device, 0.0),
        'baseline_motion_jump': _optional_field(batch, 'baseline_motion_jump', B, baseline_xy.device, 0.0),
        'baseline_speed': _optional_field(batch, 'baseline_speed', B, baseline_xy.device, 0.0),
        'baseline_accel': _optional_field(batch, 'baseline_accel', B, baseline_xy.device, 0.0),
        'baseline_risk_prob': _optional_field(batch, 'baseline_risk_prob', B, baseline_xy.device, 0.0),
        'baseline_corr_margin': _optional_field(batch, 'baseline_corr_margin', B, baseline_xy.device, 0.0),
        'baseline_update_norm': _optional_field(batch, 'baseline_update_norm', B, baseline_xy.device, 0.0),
        'query_anchor_age': _optional_field(batch, 'query_anchor_age', B, baseline_xy.device, 0.0),
        'source_is_query_anchor': _optional_field(batch, 'source_is_query_anchor', B, baseline_xy.device, 1.0),
        'query_to_baseline_offset_norm': _optional_field(batch, 'query_to_baseline_offset_norm', B, baseline_xy.device, 0.0),
        'frame_index_norm': frame_index_norm,
    }

    names = get_roma_adapter_feature_names() if feature_names is None else list(feature_names)
    x = torch.stack([features[name] for name in names], dim=-1).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


def build_roma_recurrent_adapter_features(batch, feature_names=None, device=None):
    baseline_xy8 = _as_tensor(batch, 'baseline_xy8', device=device)
    roma_xy8 = _as_tensor(batch, 'roma_xy8', device=device)
    if baseline_xy8 is None or roma_xy8 is None:
        raise ValueError('batch must contain baseline_xy8 and roma_xy8')
    if baseline_xy8.ndim != 2 or baseline_xy8.shape[-1] != 2:
        raise ValueError('baseline_xy8 must have shape [B, 2]')
    B = int(baseline_xy8.shape[0])
    device = baseline_xy8.device

    roma_valid = _optional_field(batch, 'roma_valid', B, device, 0.0)
    roma_certainty = _optional_field(batch, 'roma_certainty', B, device, 0.0)
    offset8 = torch.nan_to_num(roma_xy8.float() - baseline_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)
    offset_norm8 = torch.linalg.vector_norm(offset8, dim=-1)

    if 'prev_baseline_xy8' in batch and 'roma_to_prev_baseline_dist8' not in batch:
        prev_xy8 = _as_tensor(batch, 'prev_baseline_xy8', device=device)
        roma_to_prev8 = torch.linalg.vector_norm(
            torch.nan_to_num(roma_xy8.float() - prev_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
    else:
        roma_to_prev8 = _optional_field(batch, 'roma_to_prev_baseline_dist8', B, device, 0.0)

    features = {
        'roma_valid': roma_valid,
        'roma_certainty': roma_certainty,
        'roma_offset_x8': offset8[:, 0],
        'roma_offset_y8': offset8[:, 1],
        'roma_offset_norm8': offset_norm8,
        'roma_to_prev_baseline_dist8': roma_to_prev8,
        'roma_temporal_jump8': _optional_field(batch, 'roma_temporal_jump8', B, device, 0.0),
        'baseline_visible_score': _optional_field(
            batch,
            'baseline_visible_score' if 'baseline_visible_score' in batch else 'pred_visible_score',
            B,
            device,
            0.0,
        ),
        'baseline_motion_jump8': _optional_field(batch, 'baseline_motion_jump8', B, device, 0.0),
        'baseline_speed8': _optional_field(batch, 'baseline_speed8', B, device, 0.0),
        'baseline_accel8': _optional_field(batch, 'baseline_accel8', B, device, 0.0),
        'baseline_risk_prob': _optional_field(batch, 'baseline_risk_prob', B, device, 0.0),
        'baseline_corr_margin': _optional_field(batch, 'baseline_corr_margin', B, device, 0.0),
        'baseline_update_norm': _optional_field(batch, 'baseline_update_norm', B, device, 0.0),
        'query_anchor_age_norm': _optional_field(batch, 'query_anchor_age_norm', B, device, 0.0),
        'source_is_query_anchor': _optional_field(batch, 'source_is_query_anchor', B, device, 1.0),
        'normalized_frame_index': _optional_field(batch, 'normalized_frame_index', B, device, 0.0),
        'normalized_window_start_index': _optional_field(batch, 'normalized_window_start_index', B, device, 0.0),
    }

    names = get_roma_recurrent_adapter_feature_names() if feature_names is None else list(feature_names)
    x = torch.stack([features[name] for name in names], dim=-1).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


class RoMaInitAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, max_delta=32.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(self.hidden_dim, 2)
        self.gate_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h = self.mlp(x.float())
        raw_delta = self.delta_head(h)
        delta_xy = self.max_delta * torch.tanh(raw_delta)
        gate_logit = self.gate_head(h)
        return delta_xy, gate_logit
