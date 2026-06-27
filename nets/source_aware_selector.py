import torch
import torch.nn as nn


SOURCE_BASELINE = 0
SOURCE_FEAT8_QUERY = 1
SOURCE_FEAT8_MEMORY = 2
SOURCE_ROMA_QUERY = 3
SOURCE_ROMA_MEMORY = 4
SOURCE_OTHER = 5


SOURCE_TYPE_NAMES = {
    SOURCE_BASELINE: 'baseline',
    SOURCE_FEAT8_QUERY: 'feat8_query',
    SOURCE_FEAT8_MEMORY: 'feat8_memory',
    SOURCE_ROMA_QUERY: 'roma_query',
    SOURCE_ROMA_MEMORY: 'roma_memory',
    SOURCE_OTHER: 'other',
}


def source_type_name(source_type):
    return SOURCE_TYPE_NAMES.get(int(source_type), 'other')


def get_source_aware_feature_names():
    return [
        'is_baseline',
        'is_feat8_query',
        'is_feat8_memory',
        'is_roma_query',
        'is_roma_memory',
        'candidate_score',
        'candidate_margin',
        'candidate_certainty',
        'offset_x',
        'offset_y',
        'offset_norm',
        'candidate_to_baseline_dist',
        'candidate_source_frame',
        'candidate_memory_age',
        'candidate_anchor_reliability',
        'candidate_anchor_type_norm',
        'is_anchor_query',
        'is_anchor_last_confident',
        'is_anchor_recent_stable',
        'is_anchor_pre_occlusion',
        'baseline_visible_score',
        'baseline_risk_prob',
        'baseline_motion_jump',
        'baseline_update_norm',
        'baseline_corr_margin',
        'query_memory_roma_agreement',
        'candidate_temporal_consistency',
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


def _optional_candidate_field(batch, key, shape, device, fill=0.0):
    value = _as_tensor(batch, key, device=device)
    if value is None:
        return torch.full(shape, float(fill), device=device)
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return value


def _optional_baseline_field(batch, key, batch_size, device):
    value = _as_tensor(batch, key, device=device)
    if value is None:
        return torch.zeros((batch_size,), device=device)
    value = torch.nan_to_num(value.reshape(batch_size).float(), nan=0.0, posinf=0.0, neginf=0.0)
    return value


def build_source_aware_features(batch, feature_names=None, device=None):
    candidate_xy = _as_tensor(batch, 'candidate_xy', device=device)
    candidate_valid = _as_tensor(batch, 'candidate_valid', device=device, dtype=torch.bool)
    source_type = _as_tensor(batch, 'candidate_source_type', device=device, dtype=torch.long)
    baseline_xy = _as_tensor(batch, 'baseline_xy', device=device)
    if candidate_xy is None or candidate_valid is None or source_type is None or baseline_xy is None:
        raise ValueError('batch must contain candidate_xy, candidate_valid, candidate_source_type, and baseline_xy')

    B, C, _ = candidate_xy.shape
    if baseline_xy.ndim == 2:
        baseline_xy = baseline_xy[:, None, :]
    offset = torch.nan_to_num(candidate_xy.float() - baseline_xy.float(), nan=0.0, posinf=0.0, neginf=0.0)
    offset_norm = torch.linalg.vector_norm(offset, dim=-1)

    candidate_score = _optional_candidate_field(batch, 'candidate_score', (B, C), candidate_xy.device)
    candidate_margin = _optional_candidate_field(batch, 'candidate_margin', (B, C), candidate_xy.device)
    candidate_certainty = _optional_candidate_field(batch, 'candidate_certainty', (B, C), candidate_xy.device)
    candidate_source_frame = _optional_candidate_field(batch, 'candidate_source_frame', (B, C), candidate_xy.device)
    candidate_memory_age = _optional_candidate_field(batch, 'candidate_memory_age', (B, C), candidate_xy.device)
    candidate_anchor_reliability = _optional_candidate_field(batch, 'candidate_anchor_reliability', (B, C), candidate_xy.device)
    candidate_anchor_type = _optional_candidate_field(batch, 'candidate_anchor_type', (B, C), candidate_xy.device, fill=-1.0)
    candidate_offset_norm = _optional_candidate_field(batch, 'candidate_offset_norm', (B, C), candidate_xy.device)
    candidate_to_baseline_dist = torch.where(candidate_offset_norm > 0.0, candidate_offset_norm, offset_norm)

    baseline_visible = _optional_baseline_field(batch, 'baseline_visible_score', B, candidate_xy.device)
    baseline_risk = _optional_baseline_field(batch, 'baseline_risk_prob', B, candidate_xy.device)
    baseline_motion = _optional_baseline_field(batch, 'baseline_motion_jump', B, candidate_xy.device)
    baseline_update = _optional_baseline_field(batch, 'baseline_update_norm', B, candidate_xy.device)
    baseline_corr_margin = _optional_baseline_field(batch, 'baseline_corr_margin', B, candidate_xy.device)
    agreement = _optional_candidate_field(batch, 'query_memory_roma_agreement', (B, C), candidate_xy.device)
    temporal = _optional_candidate_field(batch, 'candidate_temporal_consistency', (B, C), candidate_xy.device)

    features = {
        'is_baseline': (source_type == SOURCE_BASELINE).float(),
        'is_feat8_query': (source_type == SOURCE_FEAT8_QUERY).float(),
        'is_feat8_memory': (source_type == SOURCE_FEAT8_MEMORY).float(),
        'is_roma_query': (source_type == SOURCE_ROMA_QUERY).float(),
        'is_roma_memory': (source_type == SOURCE_ROMA_MEMORY).float(),
        'candidate_score': candidate_score,
        'candidate_margin': candidate_margin,
        'candidate_certainty': candidate_certainty,
        'offset_x': offset[..., 0],
        'offset_y': offset[..., 1],
        'offset_norm': offset_norm,
        'candidate_to_baseline_dist': candidate_to_baseline_dist,
        'candidate_source_frame': candidate_source_frame,
        'candidate_memory_age': candidate_memory_age,
        'candidate_anchor_reliability': candidate_anchor_reliability,
        'candidate_anchor_type_norm': torch.clamp(candidate_anchor_type, min=-1.0, max=3.0) / 3.0,
        'is_anchor_query': (candidate_anchor_type == 0).float(),
        'is_anchor_last_confident': (candidate_anchor_type == 1).float(),
        'is_anchor_recent_stable': (candidate_anchor_type == 2).float(),
        'is_anchor_pre_occlusion': (candidate_anchor_type == 3).float(),
        'baseline_visible_score': baseline_visible[:, None].expand(B, C),
        'baseline_risk_prob': baseline_risk[:, None].expand(B, C),
        'baseline_motion_jump': baseline_motion[:, None].expand(B, C),
        'baseline_update_norm': baseline_update[:, None].expand(B, C),
        'baseline_corr_margin': baseline_corr_margin[:, None].expand(B, C),
        'query_memory_roma_agreement': agreement,
        'candidate_temporal_consistency': temporal,
    }
    names = get_source_aware_feature_names() if feature_names is None else list(feature_names)
    x = torch.stack([features[name] for name in names], dim=-1).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, candidate_valid.bool(), names


class SourceAwareSelector(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x, candidate_valid):
        B, C, D = x.shape
        logits = self.mlp(x.reshape(B * C, D)).reshape(B, C)
        logits = logits.masked_fill(~candidate_valid.bool(), -1.0e9)
        return logits
