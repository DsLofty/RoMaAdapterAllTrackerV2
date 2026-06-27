import torch
import torch.nn as nn


FINAL_STAGE_FEATURE_NAMES = [
    'roma_valid',
    'roma_certainty',
    'baseline_visible',
    'baseline_to_roma_dist_px',
    'roma_to_prev_baseline_dist_px',
    'offset_change_px',
    'baseline_jump_px',
    'offset_x_px',
    'offset_y_px',
    'baseline_step_x_px',
    'baseline_step_y_px',
    'frame_after_query_norm',
    'coarse_gate',
    'coarse_gate_prev',
    'coarse_gate_next',
]


def feature_names_for_profile(profile):
    profile = str(profile).strip().lower()
    if profile in ('lowdim', 'base', 'no_visual', 'geometry', 'all'):
        return list(FINAL_STAGE_FEATURE_NAMES)
    raise ValueError('V2 release only supports lowdim final-stage features, got: %s' % profile)


def feature_indices(source_names, selected_names):
    source_to_index = {str(name): int(i) for i, name in enumerate(source_names)}
    missing = [str(name) for name in selected_names if str(name) not in source_to_index]
    if missing:
        raise KeyError('missing final-stage feature(s): %s' % ','.join(missing))
    return [source_to_index[str(name)] for name in selected_names]


class FeatureNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = torch.clamp(std, min=1e-6)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def __call__(self, features):
        return (features - self.mean.view(1, 1, -1)) / self.std.view(1, 1, -1)

    def state_dict(self):
        return {'mean': self.mean.detach().cpu(), 'std': self.std.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['mean'].float(), state['std'].float())


def make_deep_corr_encoder(in_dim=50, embed_dim=64):
    return nn.Sequential(
        nn.Linear(int(in_dim), int(embed_dim)),
        nn.LayerNorm(int(embed_dim)),
        nn.GELU(),
        nn.Linear(int(embed_dim), int(embed_dim)),
        nn.GELU(),
    )


class TemporalRiskBackbone(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.in_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward_features(self, x):
        y, _ = self.gru(x)
        return y

    def forward(self, x):
        y = self.forward_features(x)
        return self.head(y).squeeze(-1)


class FinalStageDeepCorrGateSelector(nn.Module):
    """Stage A baseline-risk model used to initialize the V2 risk branch."""

    def __init__(
        self,
        scalar_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        deep_corr_dim=50,
        deep_embed_dim=64,
    ):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.deep_corr_dim = int(deep_corr_dim)
        self.deep_embed_dim = int(deep_embed_dim)
        self.deep_encoder = make_deep_corr_encoder(self.deep_corr_dim, self.deep_embed_dim)
        self.temporal = TemporalRiskBackbone(
            in_dim=self.scalar_dim + self.deep_embed_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    def forward(self, x, patches=None, deep_corr=None):
        if deep_corr is None:
            deep_corr = patches
        if deep_corr is None:
            raise ValueError('deep-corr gate selector requires deep_corr input')
        deep_feat = self.deep_encoder(deep_corr)
        return self.temporal(torch.cat([x, deep_feat], dim=-1))


class FinalStageDeepCorrRiskAcceptSelector(nn.Module):
    """Final V2 model: risk head plus RoMa accept head on deep-corr features."""

    def __init__(
        self,
        scalar_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        deep_corr_dim=50,
        deep_embed_dim=64,
    ):
        super().__init__()
        self.scalar_dim = int(scalar_dim)
        self.deep_corr_dim = int(deep_corr_dim)
        self.deep_embed_dim = int(deep_embed_dim)
        self.deep_encoder = make_deep_corr_encoder(self.deep_corr_dim, self.deep_embed_dim)
        self.temporal = TemporalRiskBackbone(
            in_dim=self.scalar_dim + self.deep_embed_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )
        head_dim = int(hidden_dim) * 2
        self.accept_head = nn.Sequential(
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, x, patches=None, deep_corr=None):
        if deep_corr is None:
            deep_corr = patches
        if deep_corr is None:
            raise ValueError('deep-corr risk/accept selector requires deep_corr input')
        deep_feat = self.deep_encoder(deep_corr)
        y = self.temporal.forward_features(torch.cat([x, deep_feat], dim=-1))
        return {
            'risk_logits': self.temporal.head(y).squeeze(-1),
            'accept_logits': self.accept_head(y).squeeze(-1),
        }


def make_model(
    in_dim,
    hidden_dim=64,
    num_layers=1,
    dropout=0.0,
    patch_mode='deep_corr_risk_accept',
    patch_embed_dim=32,
    patch_channels=6,
    deep_corr_dim=50,
    deep_embed_dim=64,
    offset_classes=25,
):
    del patch_embed_dim, patch_channels, offset_classes
    patch_mode = str(patch_mode).strip().lower()
    if patch_mode in ('deep_corr_gate', 'deepcorr_gate', 'corr_grid_gate'):
        return FinalStageDeepCorrGateSelector(
            scalar_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            deep_corr_dim=int(deep_corr_dim),
            deep_embed_dim=int(deep_embed_dim),
        )
    if patch_mode in ('deep_corr_risk_accept', 'deepcorr_risk_accept', 'corr_grid_risk_accept'):
        return FinalStageDeepCorrRiskAcceptSelector(
            scalar_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            deep_corr_dim=int(deep_corr_dim),
            deep_embed_dim=int(deep_embed_dim),
        )
    raise ValueError('V2 release only supports deep_corr_gate and deep_corr_risk_accept, got: %s' % patch_mode)
