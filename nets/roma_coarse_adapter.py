import math

import torch
import torch.nn as nn
import torch.nn.functional as F


COARSE_FEATURE_NAMES = [
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
    'event_score',
    'proposal_has_event',
    'proposal_has_patch',
    'proposal_has_roma_certainty',
    'baseline_patch_ncc',
    'roma_patch_ncc',
    'patch_ncc_gap',
    'patch_mismatch_score',
    'query_patch_texture',
    'roma_certainty_event',
]

COARSE_VISUAL_FEATURE_NAMES = [
    'visual_query_feat',
    'visual_baseline_feat',
    'visual_roma_feat',
    'visual_prev_feat',
    'visual_baseline_query_cos',
    'visual_roma_query_cos',
    'visual_prev_query_cos',
    'visual_roma_baseline_cos',
    'visual_roma_prev_cos',
    'visual_roma_query_cos_gain',
    'visual_prev_query_cos_gain',
]

COARSE_VECTOR_FEATURE_NAMES = {
    'visual_query_feat',
    'visual_baseline_feat',
    'visual_roma_feat',
    'visual_prev_feat',
}

LOCAL_CORR_FEATURE_NAMES = (
    'corr_baseline_5x5',
    'corr_roma_5x5',
    'corr_prev_5x5',
)

LOCAL_CORR_VECTOR_FEATURE_NAMES = set(LOCAL_CORR_FEATURE_NAMES)

CLEAN_LOCAL_CORR_FEATURE_NAMES = [
    'baseline_to_roma_dx8',
    'baseline_to_roma_dy8',
    'baseline_to_roma_norm8',
    'baseline_to_prev_dx8',
    'baseline_to_prev_dy8',
    'baseline_to_prev_norm8',
    'roma_to_prev_dx8',
    'roma_to_prev_dy8',
    'roma_to_prev_norm8',
    'roma_valid',
    'roma_certainty',
    'baseline_visible_score',
    'baseline_corr_margin',
    'baseline_motion_jump8',
    'baseline_speed8',
    'baseline_accel8',
    'query_anchor_age_norm',
    'corr_baseline_5x5',
    'corr_roma_5x5',
    'corr_prev_5x5',
]

CLEAN_CENTER_VISUAL_FEATURE_NAMES = CLEAN_LOCAL_CORR_FEATURE_NAMES + [
    'visual_query_feat',
    'visual_baseline_feat',
    'visual_roma_feat',
    'visual_prev_feat',
    'visual_baseline_query_cos',
    'visual_roma_query_cos',
    'visual_prev_query_cos',
    'visual_roma_baseline_cos',
    'visual_roma_prev_cos',
    'visual_roma_query_cos_gain',
    'visual_prev_query_cos_gain',
]


def get_roma_coarse_feature_names(include_visual=False):
    names = list(COARSE_FEATURE_NAMES)
    if bool(include_visual):
        names.extend(COARSE_VISUAL_FEATURE_NAMES)
    return names


def get_roma_clean_coarse_feature_names():
    return list(CLEAN_LOCAL_CORR_FEATURE_NAMES)


def get_roma_clean_center_visual_feature_names():
    return list(CLEAN_CENTER_VISUAL_FEATURE_NAMES)


def infer_roma_coarse_feature_dim(feature_names, visual_dim=256):
    """Return the flattened feature dimension for coarse adapter rows.

    Most coarse adapter inputs are scalar row features. The three visual feature
    entries are sampled fmap descriptors and therefore expand to ``visual_dim``
    channels each.
    """
    dim = 0
    for name in list(feature_names):
        if str(name) in COARSE_VECTOR_FEATURE_NAMES:
            dim += int(visual_dim)
        elif str(name) in LOCAL_CORR_VECTOR_FEATURE_NAMES:
            dim += 25
        else:
            dim += 1
    return int(dim)


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


def _optional_vector_field(batch, key, length, width, device, default_value=0.0):
    value = _as_tensor(batch, key, device=device)
    if value is None:
        return torch.full((length, width), float(default_value), device=device, dtype=torch.float32)
    value = value.float()
    if value.ndim == 1:
        value = value.reshape(length, width)
    elif value.ndim != 2 or int(value.shape[0]) != int(length):
        value = value.reshape(length, -1)
    if int(value.shape[1]) != int(width):
        raise ValueError('%s must have shape [B, %d]' % (str(key), int(width)))
    return torch.nan_to_num(value, nan=float(default_value), posinf=float(default_value), neginf=float(default_value))


def _source_mask_bit(batch, bit_index, length, device):
    mask = _as_tensor(batch, 'proposal_source_mask', device=device, dtype=torch.long)
    if mask is None:
        return torch.zeros((length,), device=device, dtype=torch.float32)
    mask = mask.reshape(length).long()
    return ((mask & int(1 << int(bit_index))) != 0).float()


def build_roma_coarse_features(batch, feature_names=None, device=None):
    baseline_xy8 = _as_tensor(batch, 'baseline_xy8', device=device)
    roma_xy8 = _as_tensor(batch, 'roma_xy8', device=device)
    if baseline_xy8 is None or roma_xy8 is None:
        raise ValueError('batch must contain baseline_xy8 and roma_xy8')
    if baseline_xy8.ndim != 2 or baseline_xy8.shape[-1] != 2:
        raise ValueError('baseline_xy8 must have shape [B, 2]')
    B = int(baseline_xy8.shape[0])
    device = baseline_xy8.device
    offset8 = torch.nan_to_num(roma_xy8.float() - baseline_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)
    offset_norm8 = torch.linalg.vector_norm(offset8, dim=-1)
    prev_xy8 = _as_tensor(batch, 'prev_baseline_xy8', device=device)
    if prev_xy8 is None:
        prev_xy8 = baseline_xy8
    prev_xy8 = torch.nan_to_num(prev_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)
    baseline_to_prev8 = torch.nan_to_num(prev_xy8.float() - baseline_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)
    roma_to_prev_vec8 = torch.nan_to_num(prev_xy8.float() - roma_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)

    if 'prev_baseline_xy8' in batch and 'roma_to_prev_baseline_dist8' not in batch:
        roma_to_prev8 = torch.linalg.vector_norm(
            torch.nan_to_num(roma_xy8.float() - prev_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
    else:
        roma_to_prev8 = _optional_field(batch, 'roma_to_prev_baseline_dist8', B, device, 0.0)

    baseline_patch_ncc = _optional_field(batch, 'baseline_patch_ncc', B, device, 0.0)
    roma_patch_ncc = _optional_field(batch, 'roma_patch_ncc', B, device, 0.0)
    patch_gap = _optional_field(batch, 'patch_ncc_gap', B, device, 0.0)
    query_patch_texture = _optional_field(batch, 'query_patch_texture', B, device, 0.0)

    features = {
        'roma_valid': _optional_field(batch, 'roma_valid', B, device, 0.0),
        'roma_certainty': _optional_field(batch, 'roma_certainty', B, device, 0.0),
        'roma_offset_x8': offset8[:, 0],
        'roma_offset_y8': offset8[:, 1],
        'roma_offset_norm8': offset_norm8,
        'baseline_to_roma_dx8': offset8[:, 0],
        'baseline_to_roma_dy8': offset8[:, 1],
        'baseline_to_roma_norm8': offset_norm8,
        'baseline_to_prev_dx8': baseline_to_prev8[:, 0],
        'baseline_to_prev_dy8': baseline_to_prev8[:, 1],
        'baseline_to_prev_norm8': torch.linalg.vector_norm(baseline_to_prev8, dim=-1),
        'roma_to_prev_dx8': roma_to_prev_vec8[:, 0],
        'roma_to_prev_dy8': roma_to_prev_vec8[:, 1],
        'roma_to_prev_norm8': torch.linalg.vector_norm(roma_to_prev_vec8, dim=-1),
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
        'event_score': _optional_field(batch, 'event_score', B, device, 0.0),
        'proposal_has_event': _source_mask_bit(batch, 1, B, device),
        'proposal_has_patch': _source_mask_bit(batch, 2, B, device),
        'proposal_has_roma_certainty': _source_mask_bit(batch, 3, B, device),
        'baseline_patch_ncc': torch.nan_to_num(baseline_patch_ncc, nan=0.0),
        'roma_patch_ncc': torch.nan_to_num(roma_patch_ncc, nan=0.0),
        'patch_ncc_gap': torch.nan_to_num(patch_gap, nan=0.0),
        'patch_mismatch_score': _optional_field(batch, 'patch_mismatch_score', B, device, 0.0),
        'query_patch_texture': torch.nan_to_num(query_patch_texture, nan=0.0),
        'roma_certainty_event': _optional_field(batch, 'roma_certainty_event', B, device, 0.0),
    }
    for key in ('visual_query_feat', 'visual_baseline_feat', 'visual_roma_feat', 'visual_prev_feat'):
        value = _as_tensor(batch, key, device=device)
        if value is not None:
            if value.ndim != 2 or int(value.shape[0]) != B:
                raise ValueError('%s must have shape [B, C]' % str(key))
            features[key] = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    for key in (
        'visual_baseline_query_cos',
        'visual_roma_query_cos',
        'visual_prev_query_cos',
        'visual_roma_baseline_cos',
        'visual_roma_prev_cos',
        'visual_roma_query_cos_gain',
        'visual_prev_query_cos_gain',
    ):
        if key in batch:
            features[key] = _optional_field(batch, key, B, device, 0.0)
    for key in LOCAL_CORR_FEATURE_NAMES:
        if key in batch:
            features[key] = _optional_vector_field(batch, key, B, 25, device, 0.0)
        else:
            features[key] = torch.zeros((B, 25), device=device, dtype=torch.float32)

    names = get_roma_coarse_feature_names() if feature_names is None else list(feature_names)
    missing = [name for name in names if name not in features]
    if missing:
        raise KeyError('unknown coarse adapter feature(s): %s' % ','.join(missing))
    pieces = []
    for name in names:
        value = features[name]
        if value.ndim == 1:
            value = value[:, None]
        elif value.ndim != 2:
            value = value.reshape(B, -1)
        pieces.append(value.float())
    x = torch.cat(pieces, dim=-1).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


class RoMaCoarseAdapter(nn.Module):
    """Point-level RoMa coarse relocation adapter.

    ``coord_mode='residual'`` keeps the old behavior and predicts a bounded
    baseline-relative offset. ``coord_mode='fusion_residual'`` keeps the legacy
    soft coordinate fusion path. ``coord_mode='hard_fusion_residual'`` chooses
    exactly one of baseline/RoMa/history in the forward pass with a
    straight-through one-hot estimator, then adds a learned bounded residual.
    ``coord_mode='roma_residual'`` uses RoMa itself as the coarse anchor and
    adds a learned bounded residual.
    """

    COORD_MODES = (
        'residual',
        'fusion_residual',
        'hard_fusion_residual',
        'two_stage_st_fusion_residual',
        'error_hard_fusion_residual',
        'event_hard_fusion_residual',
        'roma_residual',
    )

    def __init__(
        self,
        in_dim,
        hidden_dim=192,
        max_delta8=4.0,
        dropout=0.0,
        coord_mode='residual',
        temporal_mode='none',
        temporal_bidirectional=False,
        zero_init_delta=False,
        basis_init='none',
        basis_init_bias=1.5,
        candidate_switch_margin8=0.25,
        candidate_baseline_bad_thr8=0.5,
        candidate_error_init8=0.5,
        candidate_error_init_margin8=0.5,
        selection_temperature=1.0,
        relocalize_conf_thr=0.6,
        relocalize_enter_count=2,
        relocalize_exit_count=2,
        relocalize_cooldown=0,
        relocalize_init_prob=0.1,
        safety_hidden_dim=64,
        safety_gain_clip_px=8.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta8 = float(max_delta8)
        self.coord_mode = str(coord_mode)
        if self.coord_mode not in self.COORD_MODES:
            raise ValueError('unsupported coord_mode: %s' % self.coord_mode)
        self.temporal_mode = str(temporal_mode or 'none')
        if self.temporal_mode not in ('none', 'gru'):
            raise ValueError('unsupported temporal_mode: %s' % self.temporal_mode)
        self.temporal_bidirectional = bool(temporal_bidirectional)
        layers = [
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        self.trunk = nn.Sequential(*layers)
        if self.temporal_mode == 'gru':
            self.temporal_gru = nn.GRU(
                self.hidden_dim,
                self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=self.temporal_bidirectional,
            )
            temporal_out_dim = self.hidden_dim * (2 if self.temporal_bidirectional else 1)
            self.temporal_proj = nn.Linear(temporal_out_dim, self.hidden_dim)
            self.temporal_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.temporal_gru = None
            self.temporal_proj = None
            self.temporal_norm = None
        self.delta_head = nn.Linear(self.hidden_dim, 2)
        self.zero_init_delta = bool(zero_init_delta)
        if self.zero_init_delta:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.quality_head = nn.Linear(self.hidden_dim, 1)
        self.baseline_need_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_quality_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_accept_head = nn.Linear(self.hidden_dim, 1)
        self.basis_head = nn.Linear(self.hidden_dim, 3)
        self.candidate_error_head = nn.Linear(self.hidden_dim, 3)
        self.relocalize_head = nn.Linear(self.hidden_dim, 1)
        safety_hidden_dim = max(int(safety_hidden_dim), 1)
        self.safety_feature_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.safety_trunk = nn.Sequential(
            nn.Linear(self.hidden_dim, safety_hidden_dim),
            nn.GELU(),
        )
        self.safety_head = nn.Linear(safety_hidden_dim, 1)
        self.safety_gain_head = nn.Linear(safety_hidden_dim, 1)
        self.safety_gain_clip_px = max(float(safety_gain_clip_px), 1.0e-4)
        nn.init.zeros_(self.safety_feature_proj.weight)
        nn.init.zeros_(self.safety_feature_proj.bias)
        nn.init.zeros_(self.safety_head.weight)
        nn.init.zeros_(self.safety_head.bias)
        nn.init.zeros_(self.safety_gain_head.weight)
        nn.init.zeros_(self.safety_gain_head.bias)
        self.candidate_switch_margin8 = max(float(candidate_switch_margin8), 0.0)
        self.candidate_baseline_bad_thr8 = max(float(candidate_baseline_bad_thr8), 0.0)
        self.candidate_error_init8 = max(float(candidate_error_init8), 1.0e-4)
        self.candidate_error_init_margin8 = max(float(candidate_error_init_margin8), 0.0)
        self.selection_temperature = max(float(selection_temperature), 1.0e-4)
        nn.init.zeros_(self.candidate_error_head.weight)
        nn.init.zeros_(self.relocalize_head.weight)
        with torch.no_grad():
            base_raw = math.log(math.expm1(self.candidate_error_init8))
            alt_error = self.candidate_error_init8 + self.candidate_error_init_margin8
            alt_raw = math.log(math.expm1(max(alt_error, 1.0e-4)))
            self.candidate_error_head.bias.copy_(
                torch.tensor([base_raw, alt_raw, alt_raw], dtype=self.candidate_error_head.bias.dtype)
            )
            init_prob = min(max(float(relocalize_init_prob), 1.0e-4), 1.0 - 1.0e-4)
            self.relocalize_head.bias.fill_(math.log(init_prob / (1.0 - init_prob)))
        self.relocalize_conf_thr = min(max(float(relocalize_conf_thr), 0.0), 1.0)
        self.relocalize_enter_count = max(int(relocalize_enter_count), 1)
        self.relocalize_exit_count = max(int(relocalize_exit_count), 1)
        self.relocalize_cooldown = max(int(relocalize_cooldown), 0)
        self.basis_init = str(basis_init or 'none')
        self.basis_init_bias = float(basis_init_bias)
        if self.basis_init not in ('none', 'baseline'):
            raise ValueError('unsupported basis_init: %s' % self.basis_init)
        if self.basis_init == 'baseline':
            nn.init.zeros_(self.basis_head.weight)
            nn.init.zeros_(self.basis_head.bias)
            with torch.no_grad():
                bias = abs(float(self.basis_init_bias))
                self.basis_head.bias[0] = bias
                self.basis_head.bias[1] = -bias
                self.basis_head.bias[2] = -bias

    def _apply_sparse_temporal(self, h, batch=None):
        if self.temporal_gru is None or batch is None:
            return h
        point_index = batch.get('point_index', None)
        target_frame = batch.get('target_frame', None)
        if point_index is None or target_frame is None:
            return h
        if not torch.is_tensor(point_index):
            point_index = torch.as_tensor(point_index, device=h.device)
        if not torch.is_tensor(target_frame):
            target_frame = torch.as_tensor(target_frame, device=h.device)
        point_index = point_index.to(device=h.device).reshape(-1).long()
        target_frame = target_frame.to(device=h.device).reshape(-1).long()
        if int(point_index.numel()) != int(h.shape[0]) or int(target_frame.numel()) != int(h.shape[0]):
            return h
        valid = point_index >= 0
        if not bool(valid.any().detach().cpu().item()):
            return h

        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)
        p_valid = point_index.index_select(0, valid_idx)
        t_valid = target_frame.index_select(0, valid_idx)
        t_min = int(t_valid.min().detach().cpu().item()) if int(t_valid.numel()) > 0 else 0
        t_span = int((t_valid.max() - t_valid.min()).detach().cpu().item()) + 2 if int(t_valid.numel()) > 0 else 1
        sort_key = p_valid * max(t_span, 1) + (t_valid - int(t_min))
        order_local = torch.argsort(sort_key)
        sorted_idx = valid_idx.index_select(0, order_local)
        sorted_points = point_index.index_select(0, sorted_idx)
        h_sorted = h.index_select(0, sorted_idx)
        unique_points, counts = torch.unique_consecutive(sorted_points, return_counts=True)
        if int(unique_points.numel()) == 0:
            return h

        out_sorted = h_sorted.clone()
        start = 0
        for count_value in counts.detach().cpu().tolist():
            count = int(count_value)
            end = start + count
            if count > 1:
                seq = h_sorted[start:end].unsqueeze(0)
                temporal_seq, _ = self.temporal_gru(seq)
                temporal_seq = self.temporal_proj(temporal_seq)
                out_sorted[start:end] = self.temporal_norm((seq + temporal_seq).squeeze(0))
            start = end

        out = h.clone()
        out.index_copy_(0, sorted_idx, out_sorted)
        return out

    def encode_features(self, x, batch=None):
        h = self.trunk(x.float())
        return self._apply_sparse_temporal(h, batch=batch)

    def forward(self, x, batch=None):
        h = self.encode_features(x, batch=batch)
        delta_xy8 = self.max_delta8 * torch.tanh(self.delta_head(h))
        gate_logit = self.gate_head(h)
        quality_logit = self.quality_head(h)
        baseline_need_logit = self.baseline_need_head(h)
        candidate_quality_logit = self.candidate_quality_head(h)
        candidate_accept_logit = self.candidate_accept_head(h)
        basis_logits = self.basis_head(h)
        candidate_error8 = F.softplus(self.candidate_error_head(h))
        relocalize_logit = self.relocalize_head(h)
        safety_context = h + self.safety_feature_proj(x.float())
        safety_h = self.safety_trunk(safety_context)
        safety_logit = self.safety_head(safety_h)
        gain_pred_px = self.safety_gain_clip_px * torch.tanh(self.safety_gain_head(safety_h))
        return (
            delta_xy8,
            gate_logit,
            quality_logit,
            basis_logits,
            candidate_error8,
            relocalize_logit,
            safety_logit,
            gain_pred_px,
            h,
            candidate_accept_logit,
            baseline_need_logit,
            candidate_quality_logit,
        )


class RoMaCleanCoarseAdapter(RoMaCoarseAdapter):
    """Coarse adapter with clean local-correlation inputs and no quality/safety heads."""

    def __init__(
        self,
        in_dim,
        hidden_dim=192,
        max_delta8=4.0,
        dropout=0.0,
        coord_mode='residual',
        temporal_mode='none',
        temporal_bidirectional=False,
        zero_init_delta=False,
        basis_init='none',
        basis_init_bias=1.5,
        candidate_switch_margin8=0.25,
        candidate_baseline_bad_thr8=0.5,
        candidate_error_init8=0.5,
        candidate_error_init_margin8=0.5,
        selection_temperature=1.0,
        relocalize_conf_thr=0.6,
        relocalize_enter_count=2,
        relocalize_exit_count=2,
        relocalize_cooldown=0,
        relocalize_init_prob=0.1,
        **unused_kwargs,
    ):
        nn.Module.__init__(self)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta8 = float(max_delta8)
        self.coord_mode = str(coord_mode)
        if self.coord_mode not in self.COORD_MODES:
            raise ValueError('unsupported coord_mode: %s' % self.coord_mode)
        self.temporal_mode = str(temporal_mode or 'none')
        if self.temporal_mode not in ('none', 'gru'):
            raise ValueError('unsupported temporal_mode: %s' % self.temporal_mode)
        self.temporal_bidirectional = bool(temporal_bidirectional)
        layers = [
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        self.trunk = nn.Sequential(*layers)
        if self.temporal_mode == 'gru':
            self.temporal_gru = nn.GRU(
                self.hidden_dim,
                self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=self.temporal_bidirectional,
            )
            temporal_out_dim = self.hidden_dim * (2 if self.temporal_bidirectional else 1)
            self.temporal_proj = nn.Linear(temporal_out_dim, self.hidden_dim)
            self.temporal_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.temporal_gru = None
            self.temporal_proj = None
            self.temporal_norm = None
        self.delta_head = nn.Linear(self.hidden_dim, 2)
        self.zero_init_delta = bool(zero_init_delta)
        if self.zero_init_delta:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        self.basis_head = nn.Linear(self.hidden_dim, 3)
        self.baseline_need_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_quality_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_accept_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_error_head = nn.Linear(self.hidden_dim, 3)
        self.relocalize_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_switch_margin8 = max(float(candidate_switch_margin8), 0.0)
        self.candidate_baseline_bad_thr8 = max(float(candidate_baseline_bad_thr8), 0.0)
        self.candidate_error_init8 = max(float(candidate_error_init8), 1.0e-4)
        self.candidate_error_init_margin8 = max(float(candidate_error_init_margin8), 0.0)
        self.selection_temperature = max(float(selection_temperature), 1.0e-4)
        nn.init.zeros_(self.candidate_error_head.weight)
        nn.init.zeros_(self.relocalize_head.weight)
        with torch.no_grad():
            base_raw = math.log(math.expm1(self.candidate_error_init8))
            alt_error = self.candidate_error_init8 + self.candidate_error_init_margin8
            alt_raw = math.log(math.expm1(max(alt_error, 1.0e-4)))
            self.candidate_error_head.bias.copy_(
                torch.tensor([base_raw, alt_raw, alt_raw], dtype=self.candidate_error_head.bias.dtype)
            )
            init_prob = min(max(float(relocalize_init_prob), 1.0e-4), 1.0 - 1.0e-4)
            self.relocalize_head.bias.fill_(math.log(init_prob / (1.0 - init_prob)))
        self.relocalize_conf_thr = min(max(float(relocalize_conf_thr), 0.0), 1.0)
        self.relocalize_enter_count = max(int(relocalize_enter_count), 1)
        self.relocalize_exit_count = max(int(relocalize_exit_count), 1)
        self.relocalize_cooldown = max(int(relocalize_cooldown), 0)
        self.basis_init = str(basis_init or 'none')
        self.basis_init_bias = float(basis_init_bias)
        if self.basis_init not in ('none', 'baseline'):
            raise ValueError('unsupported basis_init: %s' % self.basis_init)
        if self.basis_init == 'baseline':
            nn.init.zeros_(self.basis_head.weight)
            nn.init.zeros_(self.basis_head.bias)
            with torch.no_grad():
                bias = abs(float(self.basis_init_bias))
                self.basis_head.bias[0] = bias
                self.basis_head.bias[1] = -bias
                self.basis_head.bias[2] = -bias

    def forward(self, x, batch=None):
        h = self.encode_features(x, batch=batch)
        delta_xy8 = self.max_delta8 * torch.tanh(self.delta_head(h))
        zero_logit = delta_xy8[:, :1] * 0.0
        basis_logits = self.basis_head(h)
        baseline_need_logit = self.baseline_need_head(h)
        candidate_quality_logit = self.candidate_quality_head(h)
        candidate_accept_logit = self.candidate_accept_head(h)
        candidate_error8 = F.softplus(self.candidate_error_head(h))
        relocalize_logit = self.relocalize_head(h)
        return (
            delta_xy8,
            zero_logit,
            zero_logit,
            basis_logits,
            candidate_error8,
            relocalize_logit,
            zero_logit,
            zero_logit,
            h,
            candidate_accept_logit,
            baseline_need_logit,
            candidate_quality_logit,
        )


class RoMaCoordOnlyCoarseAdapter(RoMaCoarseAdapter):
    """Coarse adapter that only predicts coordinate init terms.

    Decision heads are intentionally absent. Heuristic/carry policy outside the
    module decides whether the predicted candidate is injected into AllTracker.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=192,
        max_delta8=4.0,
        dropout=0.0,
        coord_mode='residual',
        temporal_mode='none',
        temporal_bidirectional=False,
        zero_init_delta=False,
        basis_init='none',
        basis_init_bias=1.5,
        candidate_switch_margin8=0.25,
        candidate_baseline_bad_thr8=0.5,
        candidate_error_init8=0.5,
        candidate_error_init_margin8=0.5,
        selection_temperature=1.0,
        relocalize_conf_thr=0.6,
        relocalize_enter_count=2,
        relocalize_exit_count=2,
        relocalize_cooldown=0,
        relocalize_init_prob=0.1,
        **unused_kwargs,
    ):
        nn.Module.__init__(self)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta8 = float(max_delta8)
        self.coord_mode = str(coord_mode)
        if self.coord_mode not in self.COORD_MODES:
            raise ValueError('unsupported coord_mode: %s' % self.coord_mode)
        self.temporal_mode = str(temporal_mode or 'none')
        if self.temporal_mode not in ('none', 'gru'):
            raise ValueError('unsupported temporal_mode: %s' % self.temporal_mode)
        self.temporal_bidirectional = bool(temporal_bidirectional)
        layers = [
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            ]
        )
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        self.trunk = nn.Sequential(*layers)
        if self.temporal_mode == 'gru':
            self.temporal_gru = nn.GRU(
                self.hidden_dim,
                self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=self.temporal_bidirectional,
            )
            temporal_out_dim = self.hidden_dim * (2 if self.temporal_bidirectional else 1)
            self.temporal_proj = nn.Linear(temporal_out_dim, self.hidden_dim)
            self.temporal_norm = nn.LayerNorm(self.hidden_dim)
        else:
            self.temporal_gru = None
            self.temporal_proj = None
            self.temporal_norm = None
        self.delta_head = nn.Linear(self.hidden_dim, 2)
        self.zero_init_delta = bool(zero_init_delta)
        if self.zero_init_delta:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        self.basis_head = nn.Linear(self.hidden_dim, 3)
        self.candidate_switch_margin8 = max(float(candidate_switch_margin8), 0.0)
        self.candidate_baseline_bad_thr8 = max(float(candidate_baseline_bad_thr8), 0.0)
        self.candidate_error_init8 = max(float(candidate_error_init8), 1.0e-4)
        self.candidate_error_init_margin8 = max(float(candidate_error_init_margin8), 0.0)
        self.selection_temperature = max(float(selection_temperature), 1.0e-4)
        self.relocalize_conf_thr = min(max(float(relocalize_conf_thr), 0.0), 1.0)
        self.relocalize_enter_count = max(int(relocalize_enter_count), 1)
        self.relocalize_exit_count = max(int(relocalize_exit_count), 1)
        self.relocalize_cooldown = max(int(relocalize_cooldown), 0)
        self.basis_init = str(basis_init or 'none')
        self.basis_init_bias = float(basis_init_bias)
        if self.basis_init not in ('none', 'baseline'):
            raise ValueError('unsupported basis_init: %s' % self.basis_init)
        if self.basis_init == 'baseline':
            nn.init.zeros_(self.basis_head.weight)
            nn.init.zeros_(self.basis_head.bias)
            with torch.no_grad():
                bias = abs(float(self.basis_init_bias))
                self.basis_head.bias[0] = bias
                self.basis_head.bias[1] = -bias
                self.basis_head.bias[2] = -bias

    def forward(self, x, batch=None):
        h = self.encode_features(x, batch=batch)
        delta_xy8 = self.max_delta8 * torch.tanh(self.delta_head(h))
        zero_logit = delta_xy8[:, :1] * 0.0
        basis_logits = self.basis_head(h)
        return {
            'coord_param_xy8': delta_xy8,
            'gate_logit': zero_logit,
            'quality_logit': zero_logit,
            'basis_logits': basis_logits,
            'candidate_error8': None,
            'relocalize_logit': zero_logit,
            'safety_logit': None,
            'gain_pred_px': None,
            'adapter_hidden': h,
            'candidate_accept_logit': None,
            'baseline_need_logit': None,
            'candidate_quality_logit': None,
        }


def parse_roma_coarse_output(output):
    if isinstance(output, dict):
        return {
            'coord_param_xy8': output['coord_param_xy8'],
            'gate_logit': output['gate_logit'],
            'quality_logit': output.get('quality_logit', torch.zeros_like(output['gate_logit'])),
            'basis_logits': output.get('basis_logits', None),
            'candidate_error8': output.get('candidate_error8', None),
            'relocalize_logit': output.get('relocalize_logit', None),
            'safety_logit': output.get('safety_logit', None),
            'gain_pred_px': output.get('gain_pred_px', None),
            'adapter_hidden': output.get('adapter_hidden', None),
            'candidate_accept_logit': output.get('candidate_accept_logit', None),
            'baseline_need_logit': output.get('baseline_need_logit', None),
            'candidate_quality_logit': output.get('candidate_quality_logit', None),
        }
    if isinstance(output, tuple):
        if len(output) < 2:
            raise ValueError('coarse adapter output tuple must contain at least coord and gate tensors')
        coord_param_xy8 = output[0]
        gate_logit = output[1]
        quality_logit = output[2] if len(output) >= 3 else torch.zeros_like(gate_logit)
        basis_logits = output[3] if len(output) >= 4 else None
        candidate_error8 = output[4] if len(output) >= 5 else None
        relocalize_logit = output[5] if len(output) >= 6 else None
        safety_logit = output[6] if len(output) >= 7 else None
        gain_pred_px = output[7] if len(output) >= 8 else None
        adapter_hidden = output[8] if len(output) >= 9 else None
        candidate_accept_logit = output[9] if len(output) >= 10 else None
        baseline_need_logit = output[10] if len(output) >= 11 else None
        candidate_quality_logit = output[11] if len(output) >= 12 else None
        return {
            'coord_param_xy8': coord_param_xy8,
            'gate_logit': gate_logit,
            'quality_logit': quality_logit,
            'basis_logits': basis_logits,
            'candidate_error8': candidate_error8,
            'relocalize_logit': relocalize_logit,
            'safety_logit': safety_logit,
            'gain_pred_px': gain_pred_px,
            'adapter_hidden': adapter_hidden,
            'candidate_accept_logit': candidate_accept_logit,
            'baseline_need_logit': baseline_need_logit,
            'candidate_quality_logit': candidate_quality_logit,
        }
    raise ValueError('unsupported coarse adapter output type: %s' % type(output).__name__)


def _apply_temporal_event_hysteresis(raw_event, batch, enter_count=2, exit_count=2, cooldown=0):
    """Apply a causal per-point event state machine over sparse adapter rows."""
    raw_event = raw_event.reshape(-1).bool()
    point_index = batch.get('point_index', None)
    target_frame = batch.get('target_frame', None)
    if point_index is None or target_frame is None or int(raw_event.numel()) == 0:
        return raw_event
    point_index = torch.as_tensor(point_index, device=raw_event.device).reshape(-1).long()
    target_frame = torch.as_tensor(target_frame, device=raw_event.device).reshape(-1).long()
    if int(point_index.numel()) != int(raw_event.numel()) or int(target_frame.numel()) != int(raw_event.numel()):
        return raw_event

    result = torch.zeros_like(raw_event)
    valid_points = torch.unique(point_index[point_index >= 0]).detach().cpu().tolist()
    for point in valid_points:
        indices = torch.nonzero(point_index == int(point), as_tuple=False).reshape(-1)
        if int(indices.numel()) == 0:
            continue
        order = torch.argsort(target_frame.index_select(0, indices))
        indices = indices.index_select(0, order)
        active = False
        enter_streak = 0
        exit_streak = 0
        cooldown_left = 0
        for row_index in indices.detach().cpu().tolist():
            evidence = bool(raw_event[int(row_index)].detach().cpu().item())
            if active:
                if evidence:
                    exit_streak = 0
                else:
                    exit_streak += 1
                    if exit_streak >= max(int(exit_count), 1):
                        active = False
                        exit_streak = 0
                        cooldown_left = max(int(cooldown), 0)
            else:
                if cooldown_left > 0:
                    cooldown_left -= 1
                    enter_streak = 0
                elif evidence:
                    enter_streak += 1
                    if enter_streak >= max(int(enter_count), 1):
                        active = True
                        enter_streak = 0
                else:
                    enter_streak = 0
            result[int(row_index)] = active
    return result


def predict_roma_coarse_xy8(adapter, batch, features, coord_mode=None):
    """Predict a coarse 1/8-resolution coordinate from adapter features.

    The returned ``pred_xy8`` is the actual coarse relocation proposal. In
    fusion mode it is not constrained to be baseline plus RoMa offset; the
    network learns weights over baseline, RoMa, and the previous baseline state.
    """

    parsed = parse_roma_coarse_output(adapter(features, batch=batch))
    baseline_xy8 = _as_tensor(batch, 'baseline_xy8', device=features.device)
    roma_xy8 = _as_tensor(batch, 'roma_xy8', device=features.device)
    if baseline_xy8 is None or roma_xy8 is None:
        raise ValueError('batch must contain baseline_xy8 and roma_xy8')
    baseline_xy8 = baseline_xy8.float()
    roma_xy8 = torch.nan_to_num(roma_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)
    prev_xy8 = _as_tensor(batch, 'prev_baseline_xy8', device=features.device)
    if prev_xy8 is None:
        prev_xy8 = baseline_xy8
    prev_xy8 = torch.nan_to_num(prev_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)

    coord_param_xy8 = parsed['coord_param_xy8'].float()
    mode = str(coord_mode or getattr(adapter, 'coord_mode', 'residual'))
    fusion_weights = None
    raw_relocalize_event = None
    relocalize_event = None
    basis_xy8 = baseline_xy8
    residual_xy8 = coord_param_xy8
    if mode == 'fusion_residual':
        basis_logits = parsed['basis_logits']
        if basis_logits is None:
            basis_logits = torch.zeros((baseline_xy8.shape[0], 3), device=baseline_xy8.device, dtype=baseline_xy8.dtype)
        else:
            basis_logits = basis_logits.float()
        roma_valid = _as_tensor(batch, 'roma_valid', device=features.device)
        if roma_valid is not None:
            roma_valid = roma_valid.reshape(-1).float() > 0.5
            basis_logits = basis_logits.clone()
            basis_logits[:, 1] = torch.where(
                roma_valid,
                basis_logits[:, 1],
                torch.full_like(basis_logits[:, 1], -1.0e4),
            )
        fusion_weights = torch.softmax(basis_logits, dim=-1)
        basis_xy8 = (
            fusion_weights[:, 0:1] * baseline_xy8
            + fusion_weights[:, 1:2] * roma_xy8
            + fusion_weights[:, 2:3] * prev_xy8
        )
        pred_xy8 = basis_xy8 + residual_xy8
    elif mode == 'hard_fusion_residual':
        basis_logits = parsed['basis_logits']
        if basis_logits is None:
            basis_logits = torch.zeros((baseline_xy8.shape[0], 3), device=baseline_xy8.device, dtype=baseline_xy8.dtype)
        else:
            basis_logits = basis_logits.float()
        roma_valid = _as_tensor(batch, 'roma_valid', device=features.device)
        if roma_valid is not None:
            roma_valid = roma_valid.reshape(-1).float() > 0.5
            basis_logits = basis_logits.clone()
            basis_logits[:, 1] = torch.where(
                roma_valid,
                basis_logits[:, 1],
                torch.full_like(basis_logits[:, 1], -1.0e4),
            )
        soft_weights = torch.softmax(basis_logits / float(adapter.selection_temperature), dim=-1)
        hard_index = torch.argmax(soft_weights, dim=-1)
        hard_weights = F.one_hot(hard_index, num_classes=3).to(dtype=soft_weights.dtype, device=soft_weights.device)
        fusion_weights = hard_weights.detach() - soft_weights.detach() + soft_weights
        basis_xy8 = (
            fusion_weights[:, 0:1] * baseline_xy8
            + fusion_weights[:, 1:2] * roma_xy8
            + fusion_weights[:, 2:3] * prev_xy8
        )
        pred_xy8 = basis_xy8 + residual_xy8
    elif mode == 'two_stage_st_fusion_residual':
        basis_logits = parsed['basis_logits']
        relocalize_logit = parsed.get('relocalize_logit', None)
        if basis_logits is None or relocalize_logit is None:
            raise ValueError('two_stage_st_fusion_residual requires basis_logits and relocalize_logit')
        basis_logits = basis_logits.float()
        alt_logits = basis_logits[:, 1:3].clone()
        roma_valid = _as_tensor(batch, 'roma_valid', device=features.device)
        if roma_valid is None:
            roma_valid = torch.isfinite(roma_xy8).all(dim=1)
        else:
            roma_valid = roma_valid.reshape(-1).float() > 0.5
        alt_logits[:, 0] = torch.where(
            roma_valid,
            alt_logits[:, 0],
            torch.full_like(alt_logits[:, 0], -1.0e4),
        )
        alt_soft = torch.softmax(alt_logits / float(adapter.selection_temperature), dim=-1)
        alt_index = torch.argmax(alt_soft, dim=-1)
        alt_hard = F.one_hot(alt_index, num_classes=2).to(dtype=alt_soft.dtype, device=alt_soft.device)
        alt_st = alt_hard.detach() - alt_soft.detach() + alt_soft
        alternative_xy8 = alt_st[:, 0:1] * roma_xy8 + alt_st[:, 1:2] * prev_xy8

        relocalize_prob = torch.sigmoid(relocalize_logit.reshape(-1).float())
        intervene_hard = (relocalize_prob > float(adapter.relocalize_conf_thr)).to(relocalize_prob.dtype)
        intervene_st = intervene_hard.detach() - relocalize_prob.detach() + relocalize_prob
        candidate_xy8 = alternative_xy8 + residual_xy8
        pred_xy8 = baseline_xy8 + intervene_st[:, None] * (candidate_xy8 - baseline_xy8)
        basis_xy8 = alternative_xy8
        fusion_weights = torch.stack(
            [
                1.0 - intervene_st,
                intervene_st * alt_st[:, 0],
                intervene_st * alt_st[:, 1],
            ],
            dim=1,
        )
        raw_relocalize_event = relocalize_prob > float(adapter.relocalize_conf_thr)
        relocalize_event = intervene_hard > 0.5
    elif mode == 'error_hard_fusion_residual':
        candidate_error8 = parsed.get('candidate_error8', None)
        if candidate_error8 is None:
            raise ValueError('error_hard_fusion_residual requires candidate_error8 output')
        candidate_error8 = torch.nan_to_num(
            candidate_error8.float(), nan=1.0e4, posinf=1.0e4, neginf=0.0
        )
        roma_valid = _as_tensor(batch, 'roma_valid', device=features.device)
        if roma_valid is not None:
            roma_valid = roma_valid.reshape(-1).float() > 0.5
            candidate_error8 = candidate_error8.clone()
            candidate_error8[:, 1] = torch.where(
                roma_valid,
                candidate_error8[:, 1],
                torch.full_like(candidate_error8[:, 1], 1.0e4),
            )
        alternative_error8, alternative_index = candidate_error8[:, 1:].min(dim=1)
        baseline_bad = candidate_error8[:, 0] > float(adapter.candidate_baseline_bad_thr8)
        candidate_better = alternative_error8 + float(adapter.candidate_switch_margin8) < candidate_error8[:, 0]
        switch = baseline_bad & candidate_better
        hard_index = torch.where(switch, alternative_index + 1, torch.zeros_like(alternative_index))
        fusion_weights = F.one_hot(hard_index, num_classes=3).to(
            dtype=baseline_xy8.dtype, device=baseline_xy8.device
        )
        basis_xy8 = (
            fusion_weights[:, 0:1] * baseline_xy8
            + fusion_weights[:, 1:2] * roma_xy8
            + fusion_weights[:, 2:3] * prev_xy8
        )
        pred_xy8 = basis_xy8 + residual_xy8
    elif mode == 'event_hard_fusion_residual':
        candidate_error8 = parsed.get('candidate_error8', None)
        relocalize_logit = parsed.get('relocalize_logit', None)
        if candidate_error8 is None or relocalize_logit is None:
            raise ValueError('event_hard_fusion_residual requires candidate_error8 and relocalize_logit outputs')
        candidate_error8 = torch.nan_to_num(
            candidate_error8.float(), nan=1.0e4, posinf=1.0e4, neginf=0.0
        )
        relocalize_prob = torch.sigmoid(relocalize_logit.reshape(-1).float())
        roma_valid = _as_tensor(batch, 'roma_valid', device=features.device)
        if roma_valid is None:
            roma_valid = torch.isfinite(roma_xy8).all(dim=1)
        else:
            roma_valid = roma_valid.reshape(-1).float() > 0.5
        predicted_gain8 = candidate_error8[:, 0] - candidate_error8[:, 1]
        raw_relocalize_event = (
            roma_valid
            & (candidate_error8[:, 0] > float(adapter.candidate_baseline_bad_thr8))
            & (predicted_gain8 > float(adapter.candidate_switch_margin8))
            & (relocalize_prob > float(adapter.relocalize_conf_thr))
        )
        relocalize_event = _apply_temporal_event_hysteresis(
            raw_relocalize_event,
            batch,
            enter_count=int(adapter.relocalize_enter_count),
            exit_count=int(adapter.relocalize_exit_count),
            cooldown=int(adapter.relocalize_cooldown),
        )
        hard_index = relocalize_event.long()
        fusion_weights = F.one_hot(hard_index, num_classes=3).to(
            dtype=baseline_xy8.dtype, device=baseline_xy8.device
        )
        basis_xy8 = torch.where(relocalize_event[:, None], roma_xy8, baseline_xy8)
        pred_xy8 = basis_xy8 + residual_xy8
    elif mode == 'roma_residual':
        roma_valid = _as_tensor(batch, 'roma_valid', device=features.device)
        if roma_valid is None:
            valid_roma = torch.isfinite(roma_xy8).all(dim=1)
        else:
            valid_roma = roma_valid.reshape(-1).float() > 0.5
        basis_xy8 = torch.where(valid_roma[:, None], roma_xy8, baseline_xy8)
        pred_xy8 = basis_xy8 + residual_xy8
    elif mode == 'residual':
        pred_xy8 = baseline_xy8 + coord_param_xy8
    else:
        raise ValueError('unsupported coord_mode: %s' % mode)

    pred_xy8 = torch.nan_to_num(pred_xy8.float(), nan=0.0, posinf=0.0, neginf=0.0)
    delta_xy8 = pred_xy8 - baseline_xy8
    return {
        'pred_xy8': pred_xy8,
        'delta_xy8': delta_xy8,
        'coord_param_xy8': coord_param_xy8,
        'residual_xy8': residual_xy8,
        'basis_xy8': basis_xy8,
        'alternative_xy8': alternative_xy8 if mode == 'two_stage_st_fusion_residual' else None,
        'candidate_xy8': candidate_xy8 if mode == 'two_stage_st_fusion_residual' else None,
        'fusion_weights': fusion_weights,
        'gate_logit': parsed['gate_logit'],
        'quality_logit': parsed['quality_logit'],
        'basis_logits': parsed['basis_logits'],
        'candidate_error8': parsed.get('candidate_error8', None),
        'relocalize_logit': parsed.get('relocalize_logit', None),
        'safety_logit': parsed.get('safety_logit', None),
        'gain_pred_px': parsed.get('gain_pred_px', None),
        'adapter_hidden': parsed.get('adapter_hidden', None),
        'candidate_accept_logit': parsed.get('candidate_accept_logit', None),
        'baseline_need_logit': parsed.get('baseline_need_logit', None),
        'candidate_quality_logit': parsed.get('candidate_quality_logit', None),
        'relocalize_prob': (
            torch.sigmoid(parsed['relocalize_logit'].reshape(-1))
            if parsed.get('relocalize_logit', None) is not None else None
        ),
        'raw_relocalize_event': raw_relocalize_event,
        'relocalize_event': relocalize_event,
        'coord_mode': mode,
    }
