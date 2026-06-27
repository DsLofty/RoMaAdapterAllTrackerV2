import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.misc
import numpy as np
import math

from nets.blocks import CNBlockConfig, ConvNeXt, conv1x1, RelUpdateBlock, InputPadder, CorrBlock, BasicEncoder
from nets.roma_coarse_adapter import build_roma_coarse_features, predict_roma_coarse_xy8

class Net(nn.Module):
    def __init__(
            self,
            seqlen,
            use_attn=True,
            use_mixer=False,
            use_conv=False,
            use_convb=False,
            use_basicencoder=False,
            use_sinmotion=False,
            use_relmotion=False,
            use_sinrelmotion=False,
            use_feats8=False,
            no_time=False,
            no_space=False,
            no_split=False,
            no_ctx=False,
            full_split=False,
            corr_levels=5,
            corr_radius=4,
            num_blocks=3,
            dim=128,
            hdim=128,
            init_weights=True,
    ):
        super(Net, self).__init__()

        self.dim = dim
        self.hdim = hdim

        self.no_time = no_time
        self.no_space = no_space
        self.seqlen = seqlen
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.corr_channel = self.corr_levels * (self.corr_radius * 2 + 1) ** 2
        self.num_blocks = num_blocks
        self.corr_reloc_enable = False
        self.corr_reloc_last_only = True
        self.corr_reloc_alpha = 0.25
        self.corr_reloc_score_thr = 0.35
        self.corr_reloc_margin_thr = 0.05
        self.corr_reloc_max_disp = 2.0
        self.corr_reloc_min_disp = 0.5
        self.corr_reloc_center_margin_thr = 0.03
        self.corr_reloc_vis_thr = 0.3
        self.corr_reloc_level = 0
        self.reset_corr_reloc_debug()
        self.adaptive_iters_enable = False
        self.adaptive_extra_iters = 1
        self.adaptive_base_conf_thr = 0.5
        self.adaptive_disable_low_conf_gate = False
        self.adaptive_conf_gain_thr = 0.05
        self.adaptive_min_update_norm = 0.0
        self.adaptive_max_update_norm = 2.0
        self.reset_adaptive_iters_debug()
        self.patch_memory_enable = False
        self.patch_memory_size = 3
        self.patch_memory_patch_size = 3
        self.patch_memory_search_radius = 4
        self.patch_memory_alpha = 0.25
        self.patch_memory_score_mode = 'absolute'
        self.patch_memory_score_thr = 0.45
        self.patch_memory_gain_thr = 0.05
        self.patch_memory_margin_thr = 0.08
        self.patch_memory_max_disp = 4.0
        self.patch_memory_temporal_improve_thr = 0.0
        self.patch_memory_write_conf_thr = 0.8
        self.patch_memory_write_margin_thr = 0.08
        self.patch_memory_write_jump_thr = 4.0
        self.patch_memory_trigger_jump_thr = 4.0
        self.patch_memory_trigger_margin_thr = 0.04
        self.patch_memory_trigger_update_thr = 4.0
        self.patch_memory_debug = False
        self.patch_memory_export_candidates = False
        self.patch_memory_export_topk = 5
        self.patch_memory_ranker_ckpt = ''
        self.patch_memory_ranker_thr = 0.7
        self.patch_memory_ranker_margin = 0.1
        self.patch_memory_apply_corrections = True
        self.patch_memory_max_export_candidates = 20000
        self.patch_memory_candidate_info_chunks = []
        self.patch_memory_reset()
        self.reset_patch_memory_debug()
        self.export_risk_features = False
        self.max_risk_samples_per_batch = 50000
        self.risk_info_chunks = []
        self.baseline_risk_ckpt = ''
        self.baseline_risk_thr = 0.7
        # Optional sparse dense-flow initialization prior used only by external
        # evaluation scripts. When roma_init_flow8_override is None, model
        # behavior is unchanged.
        self.roma_init_flow8_override = None
        self.roma_init_mask8_override = None
        self.roma_init_enable = False
        self.roma_init_apply_at = 'window_start'
        # Default-off training switch for recurrent-in-the-loop adapter
        # experiments. When False, the historical detach behavior is preserved.
        self.roma_init_preserve_grad = False
        self.roma_init_return_last_only = False
        self.roma_init_skip_visconf_upsample = False
        self.roma_coarse_adapter_enable = False
        self.roma_coarse_adapter = None
        self.roma_coarse_adapter_inputs = None
        self.roma_coarse_adapter_feature_names = None
        self.roma_coarse_adapter_gate_mode = 'none'
        self.roma_coarse_adapter_coord_mode = 'fusion_residual'
        self.roma_coarse_adapter_max_frames_per_window = 1
        self.roma_coarse_adapter_max_rows_per_frame = 128
        self.roma_coarse_adapter_train_mode = False
        self.roma_coarse_adapter_gt_mix_prob = 0.0
        self.roma_coarse_adapter_gt_mix_ratio = 0.0
        self.roma_coarse_adapter_gt_mix_noise8 = 0.0
        self.roma_coarse_adapter_counterfactual_reject_ratio = 0.0
        self.roma_coarse_adapter_counterfactual_reject_max_rows = 0
        self.roma_coarse_adapter_stats = {}
        self.roma_coarse_adapter_stats_history = []
        self.roma_coarse_adapter_supervision_history = []
        self.reloc_head_enable = False
        self.reloc_decision_mask8_override = None
        self._reloc_decision_last_applied = False

        self.use_feats8 = use_feats8
        self.use_basicencoder = use_basicencoder
        self.use_sinmotion = use_sinmotion
        self.use_relmotion = use_relmotion
        self.use_sinrelmotion = use_sinrelmotion
        self.no_split = no_split
        self.no_ctx = no_ctx
        self.full_split = full_split

        if use_basicencoder:
            if self.full_split:
                self.fnet = BasicEncoder(input_dim=3, output_dim=self.dim, stride=8)
                self.cnet = BasicEncoder(input_dim=3, output_dim=self.dim, stride=8)
            else:
                if self.no_split:
                    self.fnet = BasicEncoder(input_dim=3, output_dim=self.dim, stride=8)
                else:
                    self.fnet = BasicEncoder(input_dim=3, output_dim=self.dim*2, stride=8)
        else:
            block_setting = [
                CNBlockConfig(96, 192, 3, True), # 4x
                CNBlockConfig(192, 384, 3, False), # 8x
                CNBlockConfig(384, None, 9, False), # 8x
            ]
            self.cnn = ConvNeXt(block_setting, stochastic_depth_prob=0.0, init_weights=init_weights)
            if self.no_split:
                self.dot_conv = conv1x1(384, dim)
            else:
                self.dot_conv = conv1x1(384, dim*2)
            
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim, 2, kernel_size=3, padding=1)
        )
        self.visconf_head = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim, 2, kernel_size=3, padding=1)
        )

        if self.use_sinrelmotion:
            self.pdim = 84 # 32*2
        elif self.use_relmotion:
            self.pdim = 4
        elif self.use_sinmotion:
            self.pdim = 42
        else:
            self.pdim = 2
            
        self.update_block = RelUpdateBlock(self.corr_channel, self.num_blocks, cdim=dim, hdim=hdim, pdim=self.pdim,
                                           use_attn=use_attn, use_mixer=use_mixer, use_conv=use_conv, use_convb=use_convb,
                                           use_layer_scale=True, no_time=no_time, no_space=no_space,
                                           no_ctx=no_ctx)

        time_line = torch.linspace(0, seqlen-1, seqlen).reshape(1, seqlen, 1)
        self.register_buffer("time_emb", utils.misc.get_1d_sincos_pos_embed_from_grid(self.dim, time_line[0])) # 1,S,C

        
    def fetch_time_embed(self, t, dtype, is_training=False):
        S = self.time_emb.shape[1]
        if t == S:
            return self.time_emb.to(dtype)
        elif t==1:
            if is_training:
                ind = np.random.choice(S)
                return self.time_emb[:,ind:ind+1].to(dtype)
            else:
                return self.time_emb[:,1:2].to(dtype)
        else:
            time_emb = self.time_emb.float()
            time_emb = F.interpolate(time_emb.permute(0, 2, 1), size=t, mode="linear").permute(0, 2, 1) 
            return time_emb.to(dtype)
    
    def coords_grid(self, batch, ht, wd, device, dtype):
        coords = torch.meshgrid(torch.arange(ht, device=device, dtype=dtype), torch.arange(wd, device=device, dtype=dtype), indexing='ij')
        coords = torch.stack(coords[::-1], dim=0)
        return coords[None].repeat(batch, 1, 1, 1)

    def reset_corr_reloc_debug(self):
        self.corr_reloc_attempt_count = 0
        self.corr_reloc_accept_count = 0
        self.corr_reloc_best_score_sum = 0.0
        self.corr_reloc_margin_sum = 0.0
        self.corr_reloc_center_margin_sum = 0.0
        self.corr_reloc_offset_norm_sum = 0.0

    def get_corr_reloc_debug(self):
        attempts = int(getattr(self, 'corr_reloc_attempt_count', 0))
        accepts = int(getattr(self, 'corr_reloc_accept_count', 0))
        if accepts > 0:
            best_score_mean = float(getattr(self, 'corr_reloc_best_score_sum', 0.0)) / accepts
            margin_mean = float(getattr(self, 'corr_reloc_margin_sum', 0.0)) / accepts
            center_margin_mean = float(getattr(self, 'corr_reloc_center_margin_sum', 0.0)) / accepts
            offset_norm_mean = float(getattr(self, 'corr_reloc_offset_norm_sum', 0.0)) / accepts
        else:
            best_score_mean = float('nan')
            margin_mean = float('nan')
            center_margin_mean = float('nan')
            offset_norm_mean = float('nan')
        return {
            'corr_reloc_enabled': bool(getattr(self, 'corr_reloc_enable', False)),
            'corr_reloc_attempt_count': attempts,
            'corr_reloc_accept_count': accepts,
            'corr_reloc_accept_ratio': float(accepts) / float(attempts) if attempts > 0 else 0.0,
            'corr_reloc_best_score_mean': best_score_mean,
            'corr_reloc_margin_mean': margin_mean,
            'corr_reloc_center_margin_mean': center_margin_mean,
            'corr_reloc_offset_norm_mean': offset_norm_mean,
        }

    def reset_adaptive_iters_debug(self):
        self.adaptive_iters_attempt_count = 0
        self.adaptive_iters_accept_count = 0
        self.adaptive_iters_base_conf_sum = 0.0
        self.adaptive_iters_conf_gain_sum = 0.0
        self.adaptive_iters_update_norm_sum = 0.0

    def get_adaptive_iters_debug(self):
        attempts = int(getattr(self, 'adaptive_iters_attempt_count', 0))
        accepts = int(getattr(self, 'adaptive_iters_accept_count', 0))
        if attempts > 0:
            base_conf_mean = float(getattr(self, 'adaptive_iters_base_conf_sum', 0.0)) / attempts
        else:
            base_conf_mean = float('nan')
        if accepts > 0:
            conf_gain_mean = float(getattr(self, 'adaptive_iters_conf_gain_sum', 0.0)) / accepts
            update_norm_mean = float(getattr(self, 'adaptive_iters_update_norm_sum', 0.0)) / accepts
        else:
            conf_gain_mean = float('nan')
            update_norm_mean = float('nan')
        return {
            'adaptive_iters_enabled': bool(getattr(self, 'adaptive_iters_enable', False)),
            'adaptive_iters_low_conf_gate_disabled': bool(getattr(self, 'adaptive_disable_low_conf_gate', False)),
            'adaptive_iters_attempt_count': attempts,
            'adaptive_iters_accept_count': accepts,
            'adaptive_iters_accept_ratio': float(accepts) / float(attempts) if attempts > 0 else 0.0,
            'adaptive_iters_base_conf_mean': base_conf_mean,
            'adaptive_iters_conf_gain_mean': conf_gain_mean,
            'adaptive_iters_update_norm_mean': update_norm_mean,
            'adaptive_iters_min_update_norm': float(getattr(self, 'adaptive_min_update_norm', 0.0)),
            'adaptive_iters_max_update_norm': float(getattr(self, 'adaptive_max_update_norm', 2.0)),
        }

    def patch_memory_reset(self):
        self.patch_memory_items = []

    def reset_patch_memory_candidate_info(self):
        self.patch_memory_candidate_info_chunks = []

    def _patch_memory_export_count(self):
        count = 0
        for chunk in getattr(self, 'patch_memory_candidate_info_chunks', []):
            if isinstance(chunk, dict) and 'baseline_xy8' in chunk:
                count += int(chunk['baseline_xy8'].shape[0])
        return count

    def _append_patch_memory_candidate_info(self, info):
        if not bool(getattr(self, 'patch_memory_export_candidates', False)):
            return
        if not isinstance(info, dict) or 'baseline_xy8' not in info:
            return
        count = int(info['baseline_xy8'].shape[0])
        if count <= 0:
            return
        max_count = int(getattr(self, 'patch_memory_max_export_candidates', 20000))
        if max_count > 0:
            remaining = max_count - self._patch_memory_export_count()
            if remaining <= 0:
                return
            if count > remaining:
                info = {k: v[:remaining] if torch.is_tensor(v) and v.shape[0] == count else v for k, v in info.items()}
        cpu_info = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                cpu_info[key] = value.detach().float().cpu()
            else:
                cpu_info[key] = value
        self.patch_memory_candidate_info_chunks.append(cpu_info)

    def get_patch_memory_candidate_info(self):
        chunks = getattr(self, 'patch_memory_candidate_info_chunks', [])
        if not chunks:
            return {}
        keys = sorted({key for chunk in chunks for key in chunk.keys() if torch.is_tensor(chunk.get(key))})
        merged = {}
        for key in keys:
            vals = [chunk[key] for chunk in chunks if key in chunk and torch.is_tensor(chunk[key])]
            if vals:
                merged[key] = torch.cat(vals, dim=0)
        return merged

    def reset_risk_info(self):
        self.risk_info_chunks = []

    def _risk_export_count(self):
        count = 0
        for chunk in getattr(self, 'risk_info_chunks', []):
            if isinstance(chunk, dict) and 'xy8' in chunk:
                count += int(chunk['xy8'].shape[0])
        return count

    def _append_risk_info(self, info):
        if not bool(getattr(self, 'export_risk_features', False)):
            return
        if not isinstance(info, dict) or 'xy8' not in info:
            return
        count = int(info['xy8'].shape[0])
        if count <= 0:
            return
        max_count = int(getattr(self, 'max_risk_samples_per_batch', 50000))
        if max_count > 0:
            remaining = max_count - self._risk_export_count()
            if remaining <= 0:
                return
            if count > remaining:
                info = {k: v[:remaining] if torch.is_tensor(v) and v.shape[0] == count else v for k, v in info.items()}
        cpu_info = {}
        for key, value in info.items():
            if torch.is_tensor(value):
                cpu_info[key] = value.detach().float().cpu()
            else:
                cpu_info[key] = value
        self.risk_info_chunks.append(cpu_info)

    def get_risk_info(self):
        chunks = getattr(self, 'risk_info_chunks', [])
        if not chunks:
            return {}
        keys = sorted({key for chunk in chunks for key in chunk.keys() if torch.is_tensor(chunk.get(key))})
        merged = {}
        for key in keys:
            vals = [chunk[key] for chunk in chunks if key in chunk and torch.is_tensor(chunk[key])]
            if vals:
                merged[key] = torch.cat(vals, dim=0)
        return merged

    def reset_patch_memory_debug(self):
        self.patch_memory_trigger_count = 0
        self.patch_memory_search_count = 0
        self.patch_memory_accept_count = 0
        self.patch_memory_reject_score_count = 0
        self.patch_memory_reject_margin_count = 0
        self.patch_memory_reject_disp_count = 0
        self.patch_memory_reject_temporal_count = 0
        self.patch_memory_write_count = 0
        self.patch_memory_best_score_sum = 0.0
        self.patch_memory_margin_sum = 0.0
        self.patch_memory_offset_norm_sum = 0.0
        self.patch_memory_search_best_score_sum = 0.0
        self.patch_memory_search_current_score_sum = 0.0
        self.patch_memory_search_score_gain_sum = 0.0
        self.patch_memory_search_margin_sum = 0.0
        self.patch_memory_search_offset_norm_sum = 0.0
        self.patch_memory_search_best_score_max = -float('inf')

    def get_patch_memory_debug(self):
        searches = int(getattr(self, 'patch_memory_search_count', 0))
        accepts = int(getattr(self, 'patch_memory_accept_count', 0))
        if accepts > 0:
            best_score_mean = float(getattr(self, 'patch_memory_best_score_sum', 0.0)) / accepts
            margin_mean = float(getattr(self, 'patch_memory_margin_sum', 0.0)) / accepts
            offset_norm_mean = float(getattr(self, 'patch_memory_offset_norm_sum', 0.0)) / accepts
        else:
            best_score_mean = float('nan')
            margin_mean = float('nan')
            offset_norm_mean = float('nan')
        if searches > 0:
            search_best_score_mean = float(getattr(self, 'patch_memory_search_best_score_sum', 0.0)) / searches
            search_current_score_mean = float(getattr(self, 'patch_memory_search_current_score_sum', 0.0)) / searches
            search_score_gain_mean = float(getattr(self, 'patch_memory_search_score_gain_sum', 0.0)) / searches
            search_margin_mean = float(getattr(self, 'patch_memory_search_margin_sum', 0.0)) / searches
            search_offset_norm_mean = float(getattr(self, 'patch_memory_search_offset_norm_sum', 0.0)) / searches
            search_best_score_max = float(getattr(self, 'patch_memory_search_best_score_max', -float('inf')))
        else:
            search_best_score_mean = float('nan')
            search_current_score_mean = float('nan')
            search_score_gain_mean = float('nan')
            search_margin_mean = float('nan')
            search_offset_norm_mean = float('nan')
            search_best_score_max = float('nan')
        return {
            'patch_memory_enabled': bool(getattr(self, 'patch_memory_enable', False)),
            'patch_memory_score_mode': str(getattr(self, 'patch_memory_score_mode', 'absolute')),
            'patch_memory_gain_thr': float(getattr(self, 'patch_memory_gain_thr', 0.05)),
            'patch_memory_trigger_count': int(getattr(self, 'patch_memory_trigger_count', 0)),
            'patch_memory_search_count': searches,
            'patch_memory_accept_count': accepts,
            'patch_memory_reject_score_count': int(getattr(self, 'patch_memory_reject_score_count', 0)),
            'patch_memory_reject_margin_count': int(getattr(self, 'patch_memory_reject_margin_count', 0)),
            'patch_memory_reject_disp_count': int(getattr(self, 'patch_memory_reject_disp_count', 0)),
            'patch_memory_reject_temporal_count': int(getattr(self, 'patch_memory_reject_temporal_count', 0)),
            'patch_memory_write_count': int(getattr(self, 'patch_memory_write_count', 0)),
            'patch_memory_accept_ratio': float(accepts) / float(searches) if searches > 0 else 0.0,
            'patch_memory_best_score_mean': best_score_mean,
            'patch_memory_margin_mean': margin_mean,
            'patch_memory_offset_norm_mean': offset_norm_mean,
            'patch_memory_search_best_score_mean': search_best_score_mean,
            'patch_memory_search_best_score_max': search_best_score_max,
            'patch_memory_search_current_score_mean': search_current_score_mean,
            'patch_memory_search_score_gain_mean': search_score_gain_mean,
            'patch_memory_search_margin_mean': search_margin_mean,
            'patch_memory_search_offset_norm_mean': search_offset_norm_mean,
        }

    def patch_memory_extract_patch_descriptor(self, feature_map, xy8, patch_size):
        """Sample an L2-normalized patch descriptor map at 1/8-grid xy positions.

        feature_map is [N,C,H8,W8], xy8 is [N,2,H8,W8] in x/y feature-grid
        coordinates. The output descriptor is [N,C*K*K,H8,W8]. Positions whose
        full KxK patch would cross the feature map border are marked invalid.
        """
        if feature_map is None or xy8 is None:
            return None, None
        if feature_map.ndim != 4 or xy8.ndim != 4 or xy8.shape[1] != 2:
            return None, None
        if feature_map.shape[0] != xy8.shape[0] or feature_map.shape[-2:] != xy8.shape[-2:]:
            return None, None

        patch_size = max(1, int(patch_size))
        if patch_size % 2 == 0:
            patch_size += 1
        radius = patch_size // 2
        N, _, H8, W8 = feature_map.shape
        fmap = feature_map.float()
        x = xy8[:, 0].float()
        y = xy8[:, 1].float()
        valid = (
            (x >= float(radius))
            & (x <= float(W8 - 1 - radius))
            & (y >= float(radius))
            & (y <= float(H8 - 1 - radius))
        )

        samples = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sx = x + float(dx)
                sy = y + float(dy)
                gx = sx * (2.0 / max(W8 - 1, 1)) - 1.0
                gy = sy * (2.0 / max(H8 - 1, 1)) - 1.0
                grid = torch.stack([gx, gy], dim=-1)
                sample = F.grid_sample(
                    fmap,
                    grid,
                    align_corners=True,
                    padding_mode='border',
                )
                samples.append(sample)

        descriptor = torch.cat(samples, dim=1)
        descriptor = F.normalize(descriptor, dim=1, eps=1e-6)
        descriptor = descriptor * valid[:, None].to(dtype=descriptor.dtype)
        return descriptor, valid

    def patch_memory_write(self, feature_map, xy8, frame_index, reliability=None, confidence=None):
        if feature_map is None or xy8 is None:
            return
        patch_size = int(getattr(self, 'patch_memory_patch_size', 3))
        descriptor, valid = self.patch_memory_extract_patch_descriptor(feature_map, xy8, patch_size)
        if descriptor is None or valid is None:
            return

        if reliability is not None:
            valid = valid & reliability.bool()
        reliability_score = valid.float() if reliability is None else reliability.float() * valid.float()
        confidence_score = None
        if confidence is not None:
            confidence_score = confidence.detach().float().clone()

        write_count = int(valid.sum().detach().cpu().item())
        if write_count == 0:
            return

        item = {
            'frame_index': int(frame_index),
            'xy8': xy8.detach().clone(),
            'descriptor': descriptor.detach().clone(),
            'valid': valid.detach().clone(),
            'reliability': reliability_score.detach().clone(),
            'confidence': confidence_score,
            'recently_used': False,
        }
        self.patch_memory_items.append(item)
        max_items = max(1, int(getattr(self, 'patch_memory_size', 3)))
        if len(self.patch_memory_items) > max_items:
            # Keep the first-frame template as a stable appearance anchor and
            # retain the most recent reliable templates for short-term changes.
            first_item = self.patch_memory_items[0]
            recent_items = self.patch_memory_items[-(max_items - 1):] if max_items > 1 else []
            self.patch_memory_items = [first_item] + recent_items

        self.patch_memory_write_count = int(getattr(self, 'patch_memory_write_count', 0)) + write_count

    def _patch_memory_sample_chunk_descriptors(self, feature_map, candidate_xy8, patch_size):
        """Sample descriptors for candidate points.

        feature_map is [1,C,H8,W8]. candidate_xy8 is [P,M,2]. Output is
        [P,M,C*K*K] plus [P,M] valid flags.
        """
        patch_size = max(1, int(patch_size))
        if patch_size % 2 == 0:
            patch_size += 1
        radius = patch_size // 2
        _, C, H8, W8 = feature_map.shape
        P, M, _ = candidate_xy8.shape
        patch_offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                patch_offsets.append([float(dx), float(dy)])
        patch_offsets = torch.as_tensor(
            patch_offsets,
            device=candidate_xy8.device,
            dtype=candidate_xy8.dtype,
        )
        K2 = patch_offsets.shape[0]
        sample_xy = candidate_xy8[:, :, None, :] + patch_offsets[None, None]
        valid = (
            (sample_xy[..., 0] >= 0.0)
            & (sample_xy[..., 0] <= float(W8 - 1))
            & (sample_xy[..., 1] >= 0.0)
            & (sample_xy[..., 1] <= float(H8 - 1))
        ).all(dim=2)
        sample_xy = sample_xy.reshape(P, M * K2, 2)
        gx = sample_xy[..., 0].float() * (2.0 / max(W8 - 1, 1)) - 1.0
        gy = sample_xy[..., 1].float() * (2.0 / max(H8 - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            feature_map.float(),
            grid,
            align_corners=True,
            padding_mode='border',
        )
        sampled = sampled[0].reshape(C, P, M, K2)
        descriptor = sampled.permute(1, 2, 0, 3).reshape(P, M, C * K2)
        descriptor = F.normalize(descriptor, dim=2, eps=1e-6)
        descriptor = descriptor * valid[:, :, None].to(dtype=descriptor.dtype)
        return descriptor, valid

    def patch_memory_local_search(
            self,
            feature_map,
            pred_xy8,
            suspect_mask,
            prev_xy8=None,
            corr_margin=None,
            update_norm=None,
            motion_jump=None,
            visibility_conf=None,
            frame_index=0,
    ):
        if not getattr(self, 'patch_memory_items', None):
            zero_offset = torch.zeros_like(pred_xy8)
            return zero_offset, torch.zeros_like(suspect_mask, dtype=torch.bool)
        if feature_map is None or pred_xy8 is None or suspect_mask is None:
            zero_offset = torch.zeros_like(pred_xy8)
            return zero_offset, torch.zeros_like(suspect_mask, dtype=torch.bool)

        B, C, H8, W8 = feature_map.shape
        offset_out = torch.zeros_like(pred_xy8)
        accept_out = torch.zeros_like(suspect_mask, dtype=torch.bool)
        radius = max(1, int(getattr(self, 'patch_memory_search_radius', 4)))
        patch_size = int(getattr(self, 'patch_memory_patch_size', 3))
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append([float(dx), float(dy)])
        offsets = torch.as_tensor(offsets, device=feature_map.device, dtype=pred_xy8.dtype)
        max_disp = float(getattr(self, 'patch_memory_max_disp', 4.0))
        score_mode = str(getattr(self, 'patch_memory_score_mode', 'absolute')).lower()
        if score_mode not in ['absolute', 'gain', 'absolute_or_gain', 'absolute_and_gain']:
            score_mode = 'absolute'
        score_thr = float(getattr(self, 'patch_memory_score_thr', 0.45))
        gain_thr = float(getattr(self, 'patch_memory_gain_thr', 0.05))
        margin_thr = float(getattr(self, 'patch_memory_margin_thr', 0.08))
        temporal_thr = float(getattr(self, 'patch_memory_temporal_improve_thr', 0.0))
        chunk_size = 128

        total_search = 0
        total_accept = 0
        total_reject_score = 0
        total_reject_margin = 0
        total_reject_disp = 0
        total_reject_temporal = 0
        best_score_sum = 0.0
        margin_sum = 0.0
        offset_norm_sum = 0.0
        search_best_score_sum = 0.0
        search_current_score_sum = 0.0
        search_score_gain_sum = 0.0
        search_margin_sum = 0.0
        search_offset_norm_sum = 0.0
        search_best_score_max = -float('inf')

        for b in range(B):
            ys, xs = torch.nonzero(suspect_mask[b], as_tuple=True)
            point_count = int(ys.numel())
            if point_count == 0:
                continue
            for start in range(0, point_count, chunk_size):
                end = min(start + chunk_size, point_count)
                y_chunk = ys[start:end]
                x_chunk = xs[start:end]
                P = int(y_chunk.numel())
                if P == 0:
                    continue

                pred_chunk = pred_xy8[b, :, y_chunk, x_chunk].permute(1, 0)
                candidate_xy = pred_chunk[:, None, :] + offsets[None]
                cand_desc, cand_valid = self._patch_memory_sample_chunk_descriptors(
                    feature_map[b:b + 1],
                    candidate_xy,
                    patch_size,
                )
                current_desc, current_valid = self._patch_memory_sample_chunk_descriptors(
                    feature_map[b:b + 1],
                    pred_chunk[:, None, :],
                    patch_size,
                )

                sim_blocks = []
                current_blocks = []
                xy_blocks = []
                age_blocks = []
                reliability_blocks = []
                for item in self.patch_memory_items:
                    if b >= item['descriptor'].shape[0]:
                        continue
                    mem_valid = item['valid'][b, y_chunk, x_chunk].bool()
                    if not bool(mem_valid.any().detach().cpu().item()):
                        continue
                    mem_desc = item['descriptor'][b, :, y_chunk, x_chunk].permute(1, 0).float()
                    if mem_desc.shape[1] != cand_desc.shape[2]:
                        continue
                    sim = torch.sum(cand_desc * mem_desc[:, None, :], dim=2)
                    sim = sim.masked_fill(~cand_valid, -float('inf'))
                    sim = sim.masked_fill(~mem_valid[:, None], -float('inf'))
                    current_sim = torch.sum(current_desc[:, 0] * mem_desc, dim=1)
                    current_sim = current_sim.masked_fill(~current_valid[:, 0], -float('inf'))
                    current_sim = current_sim.masked_fill(~mem_valid, -float('inf'))
                    sim_blocks.append(sim)
                    current_blocks.append(current_sim[:, None].expand_as(sim))
                    xy_blocks.append(candidate_xy)
                    item_frame = int(item.get('frame_index', 0))
                    age_blocks.append(torch.full_like(sim, float(int(frame_index) - item_frame)))
                    if item.get('reliability') is not None:
                        mem_reliability = item['reliability'][b, y_chunk, x_chunk].float()
                    else:
                        mem_reliability = mem_valid.float()
                    reliability_blocks.append(mem_reliability[:, None].expand_as(sim))

                if not sim_blocks:
                    continue

                all_sim = torch.cat(sim_blocks, dim=1)
                all_current = torch.cat(current_blocks, dim=1)
                all_xy = torch.cat(xy_blocks, dim=1)
                all_age = torch.cat(age_blocks, dim=1)
                all_reliability = torch.cat(reliability_blocks, dim=1)
                finite = torch.isfinite(all_sim)
                finite_count = finite.sum(dim=1)
                searchable = finite_count > 0
                if not bool(searchable.any().detach().cpu().item()):
                    continue

                safe_sim = all_sim.masked_fill(~finite, -1.0e9)
                requested_export_topk = max(1, int(getattr(self, 'patch_memory_export_topk', 5)))
                topn = min(max(2, requested_export_topk + 1), int(safe_sim.shape[1]))
                topk = torch.topk(safe_sim, k=topn, dim=1)
                top_values = topk.values
                top_indices = topk.indices
                best_score = top_values[:, 0]
                if top_values.shape[1] >= 2:
                    second_score = top_values[:, 1]
                else:
                    second_score = torch.full_like(best_score, -float('inf'))
                best_idx = top_indices[:, 0]
                best_xy = all_xy[torch.arange(P, device=feature_map.device), best_idx]
                current_score = all_current[torch.arange(P, device=feature_map.device), best_idx]
                memory_age = all_age[torch.arange(P, device=feature_map.device), best_idx]
                memory_reliability = all_reliability[torch.arange(P, device=feature_map.device), best_idx]
                margin = best_score - second_score
                margin_for_stats = torch.where(
                    finite_count >= 2,
                    margin,
                    torch.zeros_like(margin),
                )
                patch_margin = torch.where(
                    finite_count >= 2,
                    margin,
                    torch.full_like(margin, float('nan')),
                )
                score_gain = best_score - current_score
                offset = best_xy - pred_chunk
                offset_norm = torch.sqrt(torch.sum(offset * offset, dim=1))

                absolute_ok = best_score >= score_thr
                gain_ok = torch.isfinite(current_score) & (score_gain >= gain_thr)
                if score_mode == 'gain':
                    score_ok = searchable & gain_ok
                elif score_mode == 'absolute_or_gain':
                    score_ok = searchable & (absolute_ok | gain_ok)
                elif score_mode == 'absolute_and_gain':
                    score_ok = searchable & absolute_ok & gain_ok
                else:
                    score_ok = searchable & absolute_ok
                margin_ok = (finite_count >= 2) & (margin >= margin_thr)
                disp_ok = offset_norm <= max_disp
                if prev_xy8 is not None:
                    prev_chunk = prev_xy8[b, :, y_chunk, x_chunk].permute(1, 0)
                    baseline_jump = torch.sqrt(torch.sum((pred_chunk - prev_chunk) ** 2, dim=1))
                    candidate_jump = torch.sqrt(torch.sum((best_xy - prev_chunk) ** 2, dim=1))
                    temporal_ok = candidate_jump <= (baseline_jump + temporal_thr)
                else:
                    baseline_jump = torch.zeros_like(best_score)
                    candidate_jump = torch.zeros_like(best_score)
                    temporal_ok = torch.ones_like(score_ok)

                if bool(getattr(self, 'patch_memory_export_candidates', False)):
                    export_mask = searchable
                    if bool(export_mask.any().detach().cpu().item()):
                        target_export_topk = requested_export_topk
                        export_topk = min(target_export_topk, int(top_values.shape[1]))
                        row_ids = torch.arange(P, device=feature_map.device)[:, None]
                        memory_topk_idx = top_indices[:, :export_topk]
                        memory_topk_score = top_values[:, :export_topk]
                        memory_topk_valid = torch.isfinite(memory_topk_score) & (memory_topk_score > -1.0e8)
                        memory_topk_xy = all_xy[row_ids, memory_topk_idx]
                        memory_topk_current = all_current[row_ids, memory_topk_idx]
                        memory_topk_age = all_age[row_ids, memory_topk_idx]
                        memory_topk_reliability = all_reliability[row_ids, memory_topk_idx]
                        memory_topk_offset = memory_topk_xy - pred_chunk[:, None, :]
                        memory_topk_offset_norm = torch.sqrt(torch.sum(memory_topk_offset * memory_topk_offset, dim=2))
                        memory_topk_score_safe = torch.where(
                            torch.isfinite(memory_topk_score) & memory_topk_valid,
                            memory_topk_score,
                            torch.zeros_like(memory_topk_score),
                        )
                        memory_topk_current_safe = torch.where(
                            torch.isfinite(memory_topk_current) & memory_topk_valid,
                            memory_topk_current,
                            torch.zeros_like(memory_topk_current),
                        )
                        memory_topk_score_gain = memory_topk_score_safe - memory_topk_current_safe
                        next_values = torch.full_like(memory_topk_score, float('nan'))
                        next_valid = torch.zeros_like(memory_topk_valid)
                        if top_values.shape[1] > 1:
                            next_count = min(export_topk, int(top_values.shape[1]) - 1)
                            if next_count > 0:
                                next_values[:, :next_count] = top_values[:, 1:1 + next_count]
                                next_valid[:, :next_count] = next_values[:, :next_count] > -1.0e8
                        memory_topk_margin = memory_topk_score - next_values
                        memory_topk_margin = torch.where(
                            torch.isfinite(memory_topk_margin) & memory_topk_valid & next_valid,
                            memory_topk_margin,
                            torch.full_like(memory_topk_margin, float('nan')),
                        )
                        if prev_xy8 is not None:
                            prev_chunk_for_export = prev_xy8[b, :, y_chunk, x_chunk].permute(1, 0)
                            memory_topk_jump = torch.sqrt(torch.sum((memory_topk_xy - prev_chunk_for_export[:, None, :]) ** 2, dim=2))
                        else:
                            memory_topk_jump = torch.zeros_like(memory_topk_score_safe)
                        if export_topk < target_export_topk:
                            pad_count = target_export_topk - export_topk
                            pad_xy = pred_chunk[:, None, :].expand(P, pad_count, 2)
                            pad_offset = torch.zeros((P, pad_count, 2), device=feature_map.device, dtype=pred_chunk.dtype)
                            pad_scalar = torch.zeros((P, pad_count), device=feature_map.device, dtype=best_score.dtype)
                            pad_nan = torch.full((P, pad_count), float('nan'), device=feature_map.device, dtype=best_score.dtype)
                            pad_valid = torch.zeros((P, pad_count), device=feature_map.device, dtype=torch.bool)
                            memory_topk_xy = torch.cat([memory_topk_xy, pad_xy], dim=1)
                            memory_topk_offset = torch.cat([memory_topk_offset, pad_offset], dim=1)
                            memory_topk_offset_norm = torch.cat([memory_topk_offset_norm, pad_scalar], dim=1)
                            memory_topk_score_safe = torch.cat([memory_topk_score_safe, pad_scalar], dim=1)
                            memory_topk_score_gain = torch.cat([memory_topk_score_gain, pad_scalar], dim=1)
                            memory_topk_margin = torch.cat([memory_topk_margin, pad_nan], dim=1)
                            memory_topk_jump = torch.cat([memory_topk_jump, pad_scalar], dim=1)
                            memory_topk_age = torch.cat([memory_topk_age, pad_scalar], dim=1)
                            memory_topk_reliability = torch.cat([memory_topk_reliability, pad_scalar], dim=1)
                            memory_topk_valid = torch.cat([memory_topk_valid, pad_valid], dim=1)
                        baseline_xy_all = pred_chunk[:, None, :]
                        baseline_offset_all = torch.zeros_like(baseline_xy_all)
                        baseline_score_all = torch.where(
                            torch.isfinite(current_score),
                            current_score,
                            torch.zeros_like(current_score),
                        )[:, None]
                        baseline_scalar_zeros = torch.zeros_like(baseline_score_all)
                        baseline_valid = torch.ones_like(baseline_score_all, dtype=torch.bool)
                        candidate_xy8_all = torch.cat([baseline_xy_all, memory_topk_xy], dim=1)
                        candidate_offset_all = torch.cat([baseline_offset_all, memory_topk_offset], dim=1)
                        candidate_score_all = torch.cat([baseline_score_all, memory_topk_score_safe], dim=1)
                        # candidate 0 is the baseline coordinate. Its score is
                        # the current-location similarity to the same memory item
                        # as the top-1 match; this is only a diagnostic feature.
                        candidate_source_all = torch.cat([
                            torch.zeros_like(baseline_score_all),
                            torch.ones_like(memory_topk_score_safe),
                        ], dim=1)
                        candidate_score_gain_all = torch.cat([baseline_scalar_zeros, memory_topk_score_gain], dim=1)
                        candidate_margin_all = torch.cat([baseline_scalar_zeros, memory_topk_margin], dim=1)
                        candidate_offset_norm_all = torch.cat([baseline_scalar_zeros, memory_topk_offset_norm], dim=1)
                        candidate_jump_all = torch.cat([baseline_jump[:, None], memory_topk_jump], dim=1)
                        candidate_jump_delta_all = candidate_jump_all - baseline_jump[:, None]
                        memory_age_all = torch.cat([baseline_scalar_zeros, memory_topk_age], dim=1)
                        memory_reliability_all = torch.cat([
                            torch.ones_like(baseline_score_all),
                            memory_topk_reliability,
                        ], dim=1)
                        candidate_valid_all = torch.cat([baseline_valid, memory_topk_valid], dim=1)
                        if corr_margin is not None:
                            corr_margin_chunk = corr_margin[b, y_chunk, x_chunk].to(dtype=best_score.dtype)
                        else:
                            corr_margin_chunk = torch.zeros_like(best_score)
                        if update_norm is not None:
                            update_norm_chunk = update_norm[b, y_chunk, x_chunk].to(dtype=best_score.dtype)
                        else:
                            update_norm_chunk = torch.zeros_like(best_score)
                        if motion_jump is not None:
                            motion_jump_chunk = motion_jump[b, y_chunk, x_chunk].to(dtype=best_score.dtype)
                        else:
                            motion_jump_chunk = baseline_jump
                        if visibility_conf is not None:
                            visibility_chunk = visibility_conf[b, y_chunk, x_chunk].to(dtype=best_score.dtype)
                        else:
                            visibility_chunk = torch.zeros_like(best_score)
                        grid_xy = torch.stack([x_chunk.to(dtype=best_score.dtype), y_chunk.to(dtype=best_score.dtype)], dim=1)
                        current_score_safe = torch.where(
                            torch.isfinite(current_score),
                            current_score,
                            torch.zeros_like(current_score),
                        )
                        score_gain_safe = torch.where(
                            torch.isfinite(score_gain),
                            score_gain,
                            torch.zeros_like(score_gain),
                        )
                        self._append_patch_memory_candidate_info({
                            'baseline_xy8': pred_chunk[export_mask],
                            'candidate_xy8': best_xy[export_mask],
                            'offset': offset[export_mask],
                            'best_score': best_score[export_mask, None],
                            'current_score': current_score_safe[export_mask, None],
                            'score_gain': score_gain_safe[export_mask, None],
                            'patch_margin': patch_margin[export_mask, None],
                            'offset_norm': offset_norm[export_mask, None],
                            'corr_margin': corr_margin_chunk[export_mask, None],
                            'update_norm': update_norm_chunk[export_mask, None],
                            'motion_jump': motion_jump_chunk[export_mask, None],
                            'baseline_jump': baseline_jump[export_mask, None],
                            'candidate_jump': candidate_jump[export_mask, None],
                            'visibility_conf': visibility_chunk[export_mask, None],
                            'memory_age': memory_age[export_mask, None],
                            'memory_reliability': memory_reliability[export_mask, None],
                            'frame_index': torch.full((int(export_mask.sum().detach().cpu().item()), 1), float(frame_index), device=feature_map.device, dtype=best_score.dtype),
                            'batch_index': torch.full((int(export_mask.sum().detach().cpu().item()), 1), float(b), device=feature_map.device, dtype=best_score.dtype),
                            'grid_xy8': grid_xy[export_mask],
                            'grid_y': y_chunk.to(dtype=best_score.dtype)[export_mask, None],
                            'grid_x': x_chunk.to(dtype=best_score.dtype)[export_mask, None],
                            'grid_index': (y_chunk * W8 + x_chunk).to(dtype=best_score.dtype)[export_mask, None],
                            'point_or_grid_index': (y_chunk * W8 + x_chunk).to(dtype=best_score.dtype)[export_mask, None],
                            'candidate_xy8_all': candidate_xy8_all[export_mask],
                            'candidate_offset_all': candidate_offset_all[export_mask],
                            'candidate_score_all': candidate_score_all[export_mask],
                            'candidate_source_all': candidate_source_all[export_mask],
                            'candidate_score_gain_all': candidate_score_gain_all[export_mask],
                            'candidate_margin_all': candidate_margin_all[export_mask],
                            'candidate_offset_norm_all': candidate_offset_norm_all[export_mask],
                            'candidate_jump_all': candidate_jump_all[export_mask],
                            'candidate_jump_delta_all': candidate_jump_delta_all[export_mask],
                            'memory_age_all': memory_age_all[export_mask],
                            'memory_reliability_all': memory_reliability_all[export_mask],
                            'candidate_valid_all': candidate_valid_all[export_mask],
                        })

                reject_score = searchable & (~score_ok)
                reject_margin = searchable & score_ok & (~margin_ok)
                reject_disp = searchable & score_ok & margin_ok & (~disp_ok)
                reject_temporal = searchable & score_ok & margin_ok & disp_ok & (~temporal_ok)
                accept = searchable & score_ok & margin_ok & disp_ok & temporal_ok

                total_search += int(searchable.sum().detach().cpu().item())
                if bool(searchable.any().detach().cpu().item()):
                    search_best_score_sum += float(best_score[searchable].sum().detach().cpu().item())
                    current_score_safe = torch.where(
                        torch.isfinite(current_score),
                        current_score,
                        torch.zeros_like(current_score),
                    )
                    search_current_score_sum += float(current_score_safe[searchable].sum().detach().cpu().item())
                    score_gain_safe = torch.where(
                        torch.isfinite(score_gain),
                        score_gain,
                        torch.zeros_like(score_gain),
                    )
                    search_score_gain_sum += float(score_gain_safe[searchable].sum().detach().cpu().item())
                    search_margin_sum += float(margin_for_stats[searchable].sum().detach().cpu().item())
                    search_offset_norm_sum += float(offset_norm[searchable].sum().detach().cpu().item())
                    search_best_score_max = max(
                        search_best_score_max,
                        float(best_score[searchable].max().detach().cpu().item()),
                    )
                total_reject_score += int(reject_score.sum().detach().cpu().item())
                total_reject_margin += int(reject_margin.sum().detach().cpu().item())
                total_reject_disp += int(reject_disp.sum().detach().cpu().item())
                total_reject_temporal += int(reject_temporal.sum().detach().cpu().item())
                accept_count = int(accept.sum().detach().cpu().item())
                total_accept += accept_count
                if accept_count > 0:
                    best_score_sum += float(best_score[accept].sum().detach().cpu().item())
                    margin_sum += float(margin[accept].sum().detach().cpu().item())
                    offset_norm_sum += float(offset_norm[accept].sum().detach().cpu().item())
                    offset_out[b, :, y_chunk[accept], x_chunk[accept]] = offset[accept].permute(1, 0).to(dtype=offset_out.dtype)
                    accept_out[b, y_chunk[accept], x_chunk[accept]] = True

        self.patch_memory_search_count = int(getattr(self, 'patch_memory_search_count', 0)) + total_search
        self.patch_memory_accept_count = int(getattr(self, 'patch_memory_accept_count', 0)) + total_accept
        self.patch_memory_reject_score_count = int(getattr(self, 'patch_memory_reject_score_count', 0)) + total_reject_score
        self.patch_memory_reject_margin_count = int(getattr(self, 'patch_memory_reject_margin_count', 0)) + total_reject_margin
        self.patch_memory_reject_disp_count = int(getattr(self, 'patch_memory_reject_disp_count', 0)) + total_reject_disp
        self.patch_memory_reject_temporal_count = int(getattr(self, 'patch_memory_reject_temporal_count', 0)) + total_reject_temporal
        self.patch_memory_best_score_sum = float(getattr(self, 'patch_memory_best_score_sum', 0.0)) + best_score_sum
        self.patch_memory_margin_sum = float(getattr(self, 'patch_memory_margin_sum', 0.0)) + margin_sum
        self.patch_memory_offset_norm_sum = float(getattr(self, 'patch_memory_offset_norm_sum', 0.0)) + offset_norm_sum
        self.patch_memory_search_best_score_sum = float(getattr(self, 'patch_memory_search_best_score_sum', 0.0)) + search_best_score_sum
        self.patch_memory_search_current_score_sum = float(getattr(self, 'patch_memory_search_current_score_sum', 0.0)) + search_current_score_sum
        self.patch_memory_search_score_gain_sum = float(getattr(self, 'patch_memory_search_score_gain_sum', 0.0)) + search_score_gain_sum
        self.patch_memory_search_margin_sum = float(getattr(self, 'patch_memory_search_margin_sum', 0.0)) + search_margin_sum
        self.patch_memory_search_offset_norm_sum = float(getattr(self, 'patch_memory_search_offset_norm_sum', 0.0)) + search_offset_norm_sum
        prev_max = float(getattr(self, 'patch_memory_search_best_score_max', -float('inf')))
        self.patch_memory_search_best_score_max = max(prev_max, search_best_score_max)
        return offset_out, accept_out

    def _patch_memory_corr_margin(self, corr, B, S, H8, W8):
        if corr is None or corr.ndim != 4 or corr.shape[0] != B * S:
            return None
        radius = int(getattr(self, 'corr_radius', 4))
        window_area = (2 * radius + 1) ** 2
        if corr.shape[1] < window_area:
            return None
        local_corr = corr[:, :window_area]
        top2 = torch.topk(local_corr, k=2, dim=1)
        margin = top2.values[:, 0] - top2.values[:, 1]
        return margin.reshape(B, S, H8, W8)

    def _risk_corr_stats(self, corr, B, S, H8, W8, dtype):
        zeros = torch.zeros((B, S, H8, W8), device=corr.device if corr is not None else None, dtype=dtype) if corr is not None else None
        if corr is None or corr.ndim != 4 or corr.shape[0] != B * S:
            return None
        radius = int(getattr(self, 'corr_radius', 4))
        window_area = (2 * radius + 1) ** 2
        if corr.shape[1] < window_area:
            return None
        local_corr = corr[:, :window_area].to(dtype)
        top2 = torch.topk(local_corr, k=2, dim=1)
        corr_best = top2.values[:, 0].reshape(B, S, H8, W8)
        corr_second = top2.values[:, 1].reshape(B, S, H8, W8)
        corr_margin = corr_best - corr_second
        prob = torch.softmax(local_corr.float(), dim=1).to(dtype)
        corr_entropy = -(prob * torch.log(torch.clamp(prob, min=1.0e-6))).sum(dim=1).reshape(B, S, H8, W8)
        return {
            'corr_best': corr_best,
            'corr_second': corr_second,
            'corr_margin': corr_margin,
            'corr_entropy': corr_entropy,
        }

    def export_baseline_risk_features(
            self,
            flows8,
            visconfs8,
            corr,
            coords1,
            last_update_norm=None,
            iteration_delta_norm_mean=None,
            iteration_delta_norm_last=None,
    ):
        if not bool(getattr(self, 'export_risk_features', False)):
            return
        if flows8 is None or visconfs8 is None or coords1 is None:
            return
        if flows8.ndim != 4 or coords1.ndim != 4:
            return
        BS, _, H8, W8 = flows8.shape
        if coords1.shape[0] != BS or coords1.shape[-2:] != (H8, W8):
            return
        window_start = int(getattr(self, '_patch_memory_window_start', 0))
        S = int(getattr(self, '_risk_window_s', 1))
        if S <= 0 or BS % S != 0:
            S = 1
        B = BS // S
        dtype = flows8.dtype
        device = flows8.device

        coords2 = coords1 + flows8
        xy_seq = coords2.reshape(B, S, 2, H8, W8)
        flow_seq = flows8.reshape(B, S, 2, H8, W8)
        if visconfs8.ndim == 4 and visconfs8.shape[0] == BS and visconfs8.shape[1] >= 2:
            vis_seq = visconfs8.reshape(B, S, 2, H8, W8)
            pred_visible_score = torch.sigmoid(vis_seq[:, :, 0]) * torch.sigmoid(vis_seq[:, :, 1])
        else:
            pred_visible_score = torch.zeros((B, S, H8, W8), device=device, dtype=dtype)

        if S > 1:
            velocity = xy_seq[:, 1:] - xy_seq[:, :-1]
            speed_tail = torch.sqrt(torch.sum(velocity * velocity, dim=2))
            speed = torch.cat([torch.zeros((B, 1, H8, W8), device=device, dtype=dtype), speed_tail], dim=1)
            if S > 2:
                accel_tail = velocity[:, 1:] - velocity[:, :-1]
                accel_tail = torch.sqrt(torch.sum(accel_tail * accel_tail, dim=2))
                acceleration = torch.cat([torch.zeros((B, 2, H8, W8), device=device, dtype=dtype), accel_tail], dim=1)
            else:
                acceleration = torch.zeros((B, S, H8, W8), device=device, dtype=dtype)
            conf_change_tail = pred_visible_score[:, 1:] - pred_visible_score[:, :-1]
            visibility_score_change = torch.cat([torch.zeros((B, 1, H8, W8), device=device, dtype=dtype), conf_change_tail], dim=1)
        else:
            speed = torch.zeros((B, S, H8, W8), device=device, dtype=dtype)
            acceleration = torch.zeros((B, S, H8, W8), device=device, dtype=dtype)
            visibility_score_change = torch.zeros((B, S, H8, W8), device=device, dtype=dtype)

        corr_stats = self._risk_corr_stats(corr, B, S, H8, W8, dtype)
        zero_scalar = torch.zeros((B, S, H8, W8), device=device, dtype=dtype)
        if corr_stats is None:
            corr_stats = {
                'corr_best': zero_scalar,
                'corr_second': zero_scalar,
                'corr_margin': zero_scalar,
                'corr_entropy': zero_scalar,
            }

        def reshape_stat(value):
            if value is None:
                return zero_scalar
            if value.ndim == 3 and value.shape == (BS, H8, W8):
                return value.reshape(B, S, H8, W8).to(device=device, dtype=dtype)
            if value.ndim == 4 and value.shape == (B, S, H8, W8):
                return value.to(device=device, dtype=dtype)
            return zero_scalar

        update_norm_seq = reshape_stat(last_update_norm)
        iter_mean_seq = reshape_stat(iteration_delta_norm_mean)
        iter_last_seq = reshape_stat(iteration_delta_norm_last)
        flow_residual_norm = torch.sqrt(torch.sum(flow_seq * flow_seq, dim=2))

        ys, xs = torch.meshgrid(
            torch.arange(H8, device=device, dtype=dtype),
            torch.arange(W8, device=device, dtype=dtype),
            indexing='ij',
        )
        grid_xy = torch.stack([xs, ys], dim=-1).reshape(1, 1, H8, W8, 2).expand(B, S, H8, W8, 2)
        frame_ids = torch.arange(S, device=device, dtype=dtype).reshape(1, S, 1, 1).expand(B, S, H8, W8) + float(window_start)
        batch_ids = torch.arange(B, device=device, dtype=dtype).reshape(B, 1, 1, 1).expand(B, S, H8, W8)
        grid_index = (ys * W8 + xs).reshape(1, 1, H8, W8).expand(B, S, H8, W8).to(dtype)

        info = {
            'xy8': xy_seq.permute(0, 1, 3, 4, 2).reshape(-1, 2),
            'grid_xy8': grid_xy.reshape(-1, 2),
            'frame_index': frame_ids.reshape(-1, 1),
            'batch_index': batch_ids.reshape(-1, 1),
            'point_or_grid_index': grid_index.reshape(-1, 1),
            'pred_visible_score': pred_visible_score.reshape(-1, 1),
            'corr_best': corr_stats['corr_best'].reshape(-1, 1),
            'corr_second': corr_stats['corr_second'].reshape(-1, 1),
            'corr_margin': corr_stats['corr_margin'].reshape(-1, 1),
            'corr_entropy': corr_stats['corr_entropy'].reshape(-1, 1),
            'update_norm': update_norm_seq.reshape(-1, 1),
            'last_flow_update_norm': update_norm_seq.reshape(-1, 1),
            'motion_jump': speed.reshape(-1, 1),
            'acceleration_norm': acceleration.reshape(-1, 1),
            'flow_residual_norm': flow_residual_norm.reshape(-1, 1),
            'iteration_delta_norm_mean': iter_mean_seq.reshape(-1, 1),
            'iteration_delta_norm_last': iter_last_seq.reshape(-1, 1),
            'baseline_speed': speed.reshape(-1, 1),
            'baseline_accel': acceleration.reshape(-1, 1),
            'local_conf_change': visibility_score_change.reshape(-1, 1),
            'visibility_score_change': visibility_score_change.reshape(-1, 1),
        }
        self._append_risk_info(info)

    def patch_memory_relocalize(self, flows8, fmaps2, fmap_anchor, corr=None, visconfs8=None, update_norm=None, coords1=None):
        if flows8 is None or fmaps2 is None or fmap_anchor is None or coords1 is None:
            return flows8
        if flows8.ndim != 4 or fmaps2.ndim != 5:
            return flows8
        B, S, C, H8, W8 = fmaps2.shape
        if flows8.shape[0] != B * S or flows8.shape[-2:] != (H8, W8):
            return flows8

        self.patch_memory_reset()
        coords_base = coords1.reshape(B, S, 2, H8, W8)
        flows_seq = flows8.reshape(B, S, 2, H8, W8).clone()
        coords_seq = coords_base + flows_seq
        corr_margin = self._patch_memory_corr_margin(corr, B, S, H8, W8)
        if update_norm is not None and update_norm.ndim == 3 and update_norm.shape[0] == B * S:
            update_norm_seq = update_norm.reshape(B, S, H8, W8)
        else:
            update_norm_seq = torch.zeros((B, S, H8, W8), device=flows8.device, dtype=flows8.dtype)

        if visconfs8 is not None and visconfs8.ndim == 4 and visconfs8.shape[0] == B * S:
            vis_seq = visconfs8.reshape(B, S, 2, H8, W8)
            conf_seq = torch.sigmoid(vis_seq[:, :, 0]) * torch.sigmoid(vis_seq[:, :, 1])
        else:
            conf_seq = torch.ones((B, S, H8, W8), device=flows8.device, dtype=flows8.dtype)

        base_xy = coords_base[:, 0]
        first_reliability = torch.ones((B, H8, W8), device=flows8.device, dtype=torch.bool)
        self.patch_memory_write(
            fmap_anchor,
            base_xy,
            frame_index=0,
            reliability=first_reliability,
            confidence=torch.ones((B, H8, W8), device=flows8.device, dtype=flows8.dtype),
        )

        trigger_jump_thr = float(getattr(self, 'patch_memory_trigger_jump_thr', 4.0))
        trigger_margin_thr = float(getattr(self, 'patch_memory_trigger_margin_thr', 0.04))
        trigger_update_thr = float(getattr(self, 'patch_memory_trigger_update_thr', 4.0))
        write_conf_thr = float(getattr(self, 'patch_memory_write_conf_thr', 0.8))
        write_margin_thr = float(getattr(self, 'patch_memory_write_margin_thr', 0.08))
        write_jump_thr = float(getattr(self, 'patch_memory_write_jump_thr', 4.0))
        alpha = float(getattr(self, 'patch_memory_alpha', 0.25))
        apply_corrections = bool(getattr(self, 'patch_memory_apply_corrections', True))
        window_start = int(getattr(self, '_patch_memory_window_start', 0))

        # First-stage objective: validate whether non-GT history patches can
        # correct local drift without worsening temporal continuity. The module
        # is intentionally strict: low accepts mean trigger/gates are conservative;
        # many accepts with metric drops mean memory matching is not reliable yet.
        start_s = 0 if S == 1 else 1
        for s in range(start_s, S):
            prev_xy = coords_seq[:, s - 1] if s > 0 else None
            pred_xy = coords_seq[:, s]
            if prev_xy is None:
                jump = torch.zeros((B, H8, W8), device=flows8.device, dtype=flows8.dtype)
            else:
                jump = torch.sqrt(torch.sum((pred_xy - prev_xy) ** 2, dim=1))
            if corr_margin is None:
                margin_s = torch.full_like(jump, float('inf'))
            else:
                margin_s = corr_margin[:, s].to(dtype=jump.dtype)
            update_s = update_norm_seq[:, s].to(dtype=jump.dtype)
            suspect = (
                (jump > trigger_jump_thr)
                | (margin_s < trigger_margin_thr)
                | (update_s > trigger_update_thr)
            )
            trigger_count = int(suspect.sum().detach().cpu().item())
            self.patch_memory_trigger_count = int(getattr(self, 'patch_memory_trigger_count', 0)) + trigger_count

            if trigger_count > 0:
                global_frame_index = window_start + s
                offset, accept = self.patch_memory_local_search(
                    fmaps2[:, s],
                    pred_xy,
                    suspect,
                    prev_xy8=prev_xy,
                    corr_margin=margin_s,
                    update_norm=update_s,
                    motion_jump=jump,
                    visibility_conf=conf_seq[:, s],
                    frame_index=global_frame_index,
                )
                if apply_corrections and bool(accept.any().detach().cpu().item()):
                    accept_f = accept[:, None].to(dtype=flows_seq.dtype)
                    flows_seq[:, s] = flows_seq[:, s] + alpha * offset * accept_f
                    coords_seq[:, s] = coords_base[:, s] + flows_seq[:, s]

            if s > 0:
                write_jump = torch.sqrt(torch.sum((coords_seq[:, s] - coords_seq[:, s - 1]) ** 2, dim=1))
            else:
                write_jump = torch.zeros((B, H8, W8), device=flows8.device, dtype=flows8.dtype)
            reliable = (conf_seq[:, s] >= write_conf_thr) & (write_jump <= write_jump_thr)
            if corr_margin is not None:
                reliable = reliable & (corr_margin[:, s] >= write_margin_thr)
            self.patch_memory_write(
                fmaps2[:, s],
                coords_seq[:, s],
                frame_index=window_start + s,
                reliability=reliable,
                confidence=conf_seq[:, s],
            )

        return flows_seq.reshape(B * S, 2, H8, W8)

    def corr_guided_relocalize(self, flows8, corr, visconfs8=None):
        if corr is None or flows8 is None:
            return flows8
        if corr.ndim != 4 or flows8.ndim != 4 or flows8.shape[1] < 2:
            return flows8
        if corr.shape[0] != flows8.shape[0] or corr.shape[-2:] != flows8.shape[-2:]:
            return flows8

        radius = int(getattr(self, 'corr_radius', 4))
        window = 2 * radius + 1
        window_area = window * window
        level = int(getattr(self, 'corr_reloc_level', 0))
        start = level * window_area
        end = start + window_area
        if radius < 1 or start < 0 or end > corr.shape[1] or window_area < 2:
            return flows8

        # CorrBlock returns [B*S, levels*(2r+1)^2, H8, W8]. The level-0
        # local 9x9 window is the first block of channels when corr_radius=4.
        local_corr = corr[:, start:end].to(dtype=flows8.dtype)
        top2 = torch.topk(local_corr, k=2, dim=1)
        best_score = top2.values[:, 0]
        second_score = top2.values[:, 1]
        best_idx = top2.indices[:, 0]
        margin = best_score - second_score
        center_idx = radius * window + radius
        center_score = local_corr[:, center_idx]
        center_margin = best_score - center_score

        row = torch.div(best_idx, window, rounding_mode='floor').to(dtype=flows8.dtype)
        col = (best_idx % window).to(dtype=flows8.dtype)
        # CorrBlock builds delta with meshgrid(dy, dx), then passes it to
        # bilinear_sampler in xy order, so the flattened row indexes x-offset.
        offset_x = row - float(radius)
        offset_y = col - float(radius)
        offset_norm = torch.sqrt(offset_x * offset_x + offset_y * offset_y)

        score_thr = float(getattr(self, 'corr_reloc_score_thr', 0.35))
        margin_thr = float(getattr(self, 'corr_reloc_margin_thr', 0.05))
        max_disp = float(getattr(self, 'corr_reloc_max_disp', 2.0))
        min_disp = float(getattr(self, 'corr_reloc_min_disp', 0.5))
        center_margin_thr = float(getattr(self, 'corr_reloc_center_margin_thr', 0.03))
        gate = (
            (best_score >= score_thr)
            & (margin >= margin_thr)
            & (offset_norm <= max_disp)
            & (offset_norm >= min_disp)
            & (center_margin >= center_margin_thr)
        )

        if visconfs8 is not None and visconfs8.ndim == 4 and visconfs8.shape[1] >= 2:
            if visconfs8.shape[0] == flows8.shape[0] and visconfs8.shape[-2:] == flows8.shape[-2:]:
                visible_score = torch.sigmoid(visconfs8[:, 0]) * torch.sigmoid(visconfs8[:, 1])
                gate = gate & (visible_score >= float(getattr(self, 'corr_reloc_vis_thr', 0.3)))

        attempt_count = int(gate.numel())
        accept_count = int(gate.sum().detach().cpu().item())
        self.corr_reloc_attempt_count = int(getattr(self, 'corr_reloc_attempt_count', 0)) + attempt_count
        self.corr_reloc_accept_count = int(getattr(self, 'corr_reloc_accept_count', 0)) + accept_count
        if accept_count > 0:
            self.corr_reloc_best_score_sum = float(getattr(self, 'corr_reloc_best_score_sum', 0.0)) + float(best_score[gate].sum().detach().cpu().item())
            self.corr_reloc_margin_sum = float(getattr(self, 'corr_reloc_margin_sum', 0.0)) + float(margin[gate].sum().detach().cpu().item())
            self.corr_reloc_center_margin_sum = float(getattr(self, 'corr_reloc_center_margin_sum', 0.0)) + float(center_margin[gate].sum().detach().cpu().item())
            self.corr_reloc_offset_norm_sum = float(getattr(self, 'corr_reloc_offset_norm_sum', 0.0)) + float(offset_norm[gate].sum().detach().cpu().item())

        if accept_count == 0:
            return flows8

        alpha = float(getattr(self, 'corr_reloc_alpha', 0.25))
        gate_f = gate.to(dtype=flows8.dtype)
        flows8_new = flows8.clone()
        flows8_new[:, 0] = flows8_new[:, 0] + alpha * offset_x * gate_f
        flows8_new[:, 1] = flows8_new[:, 1] + alpha * offset_y * gate_f
        return flows8_new

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords2 - coords1"""
        N, C, H, W = img.shape
        coords1 = self.coords_grid(N, H//8, W//8, device=img.device)
        coords2 = self.coords_grid(N, H//8, W//8, device=img.device)
        return coords1, coords2

    def upsample_data(self, flow, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W).to(flow.dtype)

    def _apply_roma_init_override_to_flows8(self, flows8, ara):
        flow_override = getattr(self, 'roma_init_flow8_override', None)
        mask_override = getattr(self, 'roma_init_mask8_override', None)
        if flow_override is None or mask_override is None:
            self._roma_init_last_applied = False
            return flows8
        if flows8 is None or flows8.ndim != 5:
            self._roma_init_last_applied = False
            return flows8
        if not bool(getattr(self, 'roma_init_enable', False)):
            self._roma_init_last_applied = False
            return flows8
        apply_at = str(getattr(self, 'roma_init_apply_at', 'window_start'))
        if apply_at not in ('window_start', 'query_group_start'):
            self._roma_init_last_applied = False
            return flows8
        B, S, _, H8, W8 = flows8.shape
        if flow_override.ndim != 5 or mask_override.ndim != 4:
            self._roma_init_last_applied = False
            return flows8
        if flow_override.shape[0] != B or flow_override.shape[2] != 2:
            self._roma_init_last_applied = False
            return flows8
        if flow_override.shape[-2:] != (H8, W8) or mask_override.shape[-2:] != (H8, W8):
            self._roma_init_last_applied = False
            return flows8
        ara_list = [int(x) for x in list(ara)]
        time_to_local = {t: i for i, t in enumerate(ara_list)}
        chosen_local = None
        if apply_at == 'window_start':
            # For window-start initialization, only write the prior when the
            # requested global target frame is the first local frame of this
            # window. This avoids accidentally applying a start-frame prior to
            # an earlier overlapping window that merely contains the same frame.
            t = ara_list[0]
            if t < int(flow_override.shape[1]) and bool(mask_override[:, t].any().detach().cpu().item()):
                chosen_local = time_to_local[t]
                chosen_time = t
        else:
            for t in ara_list:
                if t < int(flow_override.shape[1]) and bool(mask_override[:, t].any().detach().cpu().item()):
                    chosen_local = time_to_local[t]
                    chosen_time = t
                    break
        if chosen_local is None:
            self._roma_init_last_applied = False
            return flows8
        mask = mask_override[:, chosen_time].to(device=flows8.device).bool()
        decision_mask = getattr(self, 'reloc_decision_mask8_override', None)
        if bool(getattr(self, 'reloc_head_enable', False)) and decision_mask is not None:
            if (
                torch.is_tensor(decision_mask)
                and decision_mask.ndim == 4
                and decision_mask.shape[0] == B
                and chosen_time < int(decision_mask.shape[1])
                and decision_mask.shape[-2:] == (H8, W8)
            ):
                mask = mask & decision_mask[:, chosen_time].to(device=flows8.device).bool()
            else:
                mask = torch.zeros_like(mask, dtype=torch.bool)
        if not bool(mask.any().detach().cpu().item()):
            self._roma_init_last_applied = False
            self._reloc_decision_last_applied = False
            return flows8
        override = flow_override[:, chosen_time].to(device=flows8.device, dtype=flows8.dtype)
        flows8 = flows8.clone()
        flows8[:, chosen_local] = torch.where(mask[:, None], override, flows8[:, chosen_local])
        self._roma_init_last_applied = True
        self._reloc_decision_last_applied = bool(getattr(self, 'reloc_head_enable', False))
        return flows8

    def _sample_roma_coarse_fmap_rows(self, fmaps, frame_ids, xy8, fmap_ara=None):
        """Sample per-row AllTracker feature embeddings at 1/8-resolution xy."""
        if fmaps is None or frame_ids is None or xy8 is None:
            return None
        if fmaps.ndim != 5 or int(fmaps.shape[0]) != 1:
            return None
        B, T, C, H8, W8 = fmaps.shape
        if int(frame_ids.numel()) == 0:
            return torch.zeros((0, C), device=fmaps.device, dtype=torch.float32)
        device = fmaps.device
        frame_ids = frame_ids.to(device=device).long().reshape(-1)
        xy8 = xy8.to(device=device).float().reshape(-1, 2)
        out = torch.zeros((int(frame_ids.numel()), C), device=device, dtype=torch.float32)
        finite = torch.isfinite(xy8).all(dim=1)
        if not bool(finite.any().detach().cpu().item()):
            return out

        if fmap_ara is None:
            local_ids = frame_ids.clamp(0, max(T - 1, 0))
            valid_frame = torch.ones_like(finite, dtype=torch.bool)
        else:
            ara_list = [int(x) for x in list(fmap_ara)]
            frame_to_local = {int(t): i for i, t in enumerate(ara_list)}
            local_vals = [frame_to_local.get(int(t), -1) for t in frame_ids.detach().cpu().tolist()]
            local_ids = torch.as_tensor(local_vals, device=device, dtype=torch.long)
            valid_frame = local_ids >= 0
            local_ids = local_ids.clamp(0, max(T - 1, 0))

        valid = finite & valid_frame
        if not bool(valid.any().detach().cpu().item()):
            return out
        for local_t in torch.unique(local_ids[valid]).detach().cpu().tolist():
            local_t = int(local_t)
            mask = valid & (local_ids == local_t)
            if not bool(mask.any().detach().cpu().item()):
                continue
            coords = xy8[mask]
            if W8 > 1:
                gx = 2.0 * coords[:, 0] / float(W8 - 1) - 1.0
            else:
                gx = torch.zeros_like(coords[:, 0])
            if H8 > 1:
                gy = 2.0 * coords[:, 1] / float(H8 - 1) - 1.0
            else:
                gy = torch.zeros_like(coords[:, 1])
            grid = torch.stack([gx, gy], dim=-1).reshape(1, -1, 1, 2)
            sampled = F.grid_sample(
                fmaps[:, local_t].float(),
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )
            out[mask] = sampled[0, :, :, 0].transpose(0, 1).float()
        return out

    def _update_roma_coarse_adapter_stats(self, frame_stats):
        if not frame_stats:
            return
        history = list(getattr(self, 'roma_coarse_adapter_stats_history', []) or [])
        history.extend(frame_stats)
        self.roma_coarse_adapter_stats_history = history

        total_rows = float(sum(float(stat.get('num_rows', 0.0)) for stat in history))
        if total_rows <= 0.0:
            self.roma_coarse_adapter_stats = {}
            return

        def weighted_mean(key):
            numer = 0.0
            denom = 0.0
            for stat in history:
                weight = float(stat.get('num_rows', 0.0))
                value = float(stat.get(key, float('nan')))
                if np.isfinite(value) and weight > 0.0:
                    numer += weight * value
                    denom += weight
            return float(numer / denom) if denom > 0.0 else float('nan')

        target_frames = [float(stat.get('target_frame', float('nan'))) for stat in history]
        target_frames = [value for value in target_frames if np.isfinite(value)]
        self.roma_coarse_adapter_stats = {
            'applied': True,
            'num_rows': total_rows,
            'num_frames': float(len(history)),
            'target_frame': target_frames[-1] if target_frames else float('nan'),
            'target_frame_min': min(target_frames) if target_frames else float('nan'),
            'target_frame_max': max(target_frames) if target_frames else float('nan'),
            'gate_mean': weighted_mean('gate_mean'),
            'gate_prob_mean': weighted_mean('gate_prob_mean'),
            'effective_delta_norm8_mean': weighted_mean('effective_delta_norm8_mean'),
            'fusion_w_baseline': weighted_mean('fusion_w_baseline'),
            'fusion_w_roma': weighted_mean('fusion_w_roma'),
            'fusion_w_prev': weighted_mean('fusion_w_prev'),
            'gt_mix_frame_ratio': weighted_mean('gt_mix_active'),
            'carry_mean': weighted_mean('carry_mean'),
            'carry_refresh_mean': weighted_mean('carry_refresh_mean'),
        }

    def _apply_roma_coarse_adapter_to_flows8(self, flows8, ara, fmaps=None, fmap_ara=None, visconfs8=None, query_fmap=None):
        """Use an embedded coarse adapter to write sparse flow8 initialization."""
        if flows8 is None or flows8.ndim != 5:
            self._roma_init_last_applied = False
            return flows8
        if not bool(getattr(self, 'roma_init_enable', False)):
            self._roma_init_last_applied = False
            return flows8
        if not bool(getattr(self, 'roma_coarse_adapter_enable', False)):
            self._roma_init_last_applied = False
            return flows8
        adapter = getattr(self, 'roma_coarse_adapter', None)
        inputs = getattr(self, 'roma_coarse_adapter_inputs', None)
        feature_names = getattr(self, 'roma_coarse_adapter_feature_names', None)
        if adapter is None or inputs is None or feature_names is None:
            self._roma_init_last_applied = False
            return flows8
        if int(flows8.shape[0]) != 1:
            self._roma_init_last_applied = False
            return flows8
        if 'target_frame' not in inputs or 'source_xy' not in inputs or 'roma_xy8' not in inputs:
            self._roma_init_last_applied = False
            return flows8

        device = flows8.device
        dtype = flows8.dtype
        _, _, _, H8, W8 = flows8.shape
        target_frame_all = inputs['target_frame']
        if not torch.is_tensor(target_frame_all):
            target_frame_all = torch.as_tensor(target_frame_all)
        target_frame_all = target_frame_all.to(device=device).long().reshape(-1)
        if int(target_frame_all.numel()) == 0:
            self._roma_init_last_applied = False
            return flows8

        ara_list = [int(x) for x in list(ara)]
        time_to_local = {t: i for i, t in enumerate(ara_list)}
        apply_at = str(getattr(self, 'roma_init_apply_at', 'window_start'))
        if apply_at == 'window_start':
            candidate_times = [int(ara_list[0])]
        elif apply_at == 'query_group_start':
            candidate_times = [int(t) for t in ara_list]
        else:
            self._roma_init_last_applied = False
            return flows8
        target_times = []
        for target_t in candidate_times:
            if target_t in time_to_local and bool((target_frame_all == int(target_t)).any().detach().cpu().item()):
                target_times.append(int(target_t))
        max_frames = int(getattr(self, 'roma_coarse_adapter_max_frames_per_window', 1))
        if max_frames > 0:
            target_times = target_times[:max_frames]
        if not target_times:
            self._roma_init_last_applied = False
            return flows8

        total_rows = int(target_frame_all.shape[0])
        max_rows_per_frame = int(getattr(self, 'roma_coarse_adapter_max_rows_per_frame', 128))
        fmap_ara_set = set(int(x) for x in list(fmap_ara)) if fmap_ara is not None else None
        has_query_fmap = query_fmap is not None and torch.is_tensor(query_fmap) and query_fmap.ndim == 4
        has_visual = any(str(name).startswith('visual_') for name in list(feature_names))
        has_local_corr = any(str(name).startswith('corr_') for name in list(feature_names))
        relocalize_policy = str(getattr(self, 'roma_coarse_adapter_relocalize_policy', 'learned'))
        flows_out = flows8
        frame_stats = []
        applied_any = False
        prepared = []
        combined_parts = {}

        def _append_combined(row_batch):
            num_rows = -1
            for value in row_batch.values():
                if torch.is_tensor(value) and value.ndim >= 1:
                    num_rows = int(value.shape[0])
                    break
            if num_rows <= 0:
                return
            for key, value in row_batch.items():
                if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == num_rows:
                    combined_parts.setdefault(key, []).append(value)

        for chosen_time in target_times:
            chosen_local = int(time_to_local[int(chosen_time)])
            row_idx = torch.nonzero(target_frame_all == int(chosen_time), as_tuple=False).reshape(-1)
            if int(row_idx.numel()) == 0:
                continue
            if max_rows_per_frame > 0 and int(row_idx.numel()) > max_rows_per_frame:
                need_only_input = inputs.get('need_only', None)
                if torch.is_tensor(need_only_input) and int(need_only_input.reshape(-1).numel()) == total_rows:
                    need_mask = need_only_input.to(device=device).reshape(-1).index_select(0, row_idx).float() > 0.5
                    decision_idx = row_idx[~need_mask]
                    need_idx = row_idx[need_mask]

                    def _limit_idx(values, cap):
                        if int(values.numel()) <= cap:
                            return values
                        select = torch.linspace(
                            0,
                            int(values.numel()) - 1,
                            steps=cap,
                            device=values.device,
                        ).round().long()
                        return values.index_select(0, select)

                    row_idx = torch.cat(
                        [
                            _limit_idx(decision_idx, max_rows_per_frame),
                            _limit_idx(need_idx, max_rows_per_frame),
                        ],
                        dim=0,
                    )
                    if int(row_idx.numel()) == 0:
                        continue
                else:
                    select = torch.linspace(
                        0,
                        int(row_idx.numel()) - 1,
                        steps=max_rows_per_frame,
                        device=row_idx.device,
                    ).round().long()
                    row_idx = row_idx.index_select(0, select)

            row_batch = {}
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == total_rows:
                    row_batch[key] = value.to(device=device).index_select(0, row_idx)
            if 'source_xy' not in row_batch or 'roma_xy8' not in row_batch:
                continue

            source_xy = row_batch['source_xy'].float()
            source_x8 = torch.clamp(torch.round(source_xy[:, 0] / 8.0).long(), 0, W8 - 1)
            source_y8 = torch.clamp(torch.round(source_xy[:, 1] / 8.0).long(), 0, H8 - 1)
            source_grid_xy8 = torch.stack([source_x8.float(), source_y8.float()], dim=1).to(device=device, dtype=dtype)
            baseline_flow = flows_out[0, chosen_local, :, source_y8, source_x8].permute(1, 0)
            baseline_xy8 = source_grid_xy8 + baseline_flow
            prev_time = int(chosen_time) - 1
            if prev_time in time_to_local:
                prev_local = int(time_to_local[prev_time])
                prev_flow = flows_out[0, prev_local, :, source_y8, source_x8].permute(1, 0)
                prev_baseline_xy8 = source_grid_xy8 + prev_flow
            elif 'prev_baseline_xy8' in row_batch:
                prev_baseline_xy8 = row_batch['prev_baseline_xy8'].to(device=device, dtype=dtype)
            else:
                prev_baseline_xy8 = baseline_xy8

            row_batch['baseline_xy8'] = baseline_xy8.float()
            row_batch['prev_baseline_xy8'] = prev_baseline_xy8.float()
            if 'baseline_motion_jump8' not in row_batch:
                row_batch['baseline_motion_jump8'] = torch.linalg.vector_norm(baseline_xy8.float() - prev_baseline_xy8.float(), dim=1)
            if 'baseline_speed8' not in row_batch:
                row_batch['baseline_speed8'] = row_batch['baseline_motion_jump8'].float()
            if 'baseline_accel8' not in row_batch:
                row_batch['baseline_accel8'] = torch.zeros_like(row_batch['baseline_speed8']).float()
            if 'baseline_visible_score' not in row_batch and visconfs8 is not None and visconfs8.ndim == 5:
                vis_current = visconfs8[0, chosen_local, 0, source_y8, source_x8].float()
                if int(visconfs8.shape[2]) > 1:
                    vis_current = vis_current * visconfs8[0, chosen_local, 1, source_y8, source_x8].float()
                row_batch['baseline_visible_score'] = torch.clamp(vis_current, 0.0, 1.0)
            row_batch['roma_xy8'] = row_batch['roma_xy8'].to(device=device).float()
            if 'roma_valid' not in row_batch:
                row_batch['roma_valid'] = torch.ones((int(row_idx.numel()),), device=device, dtype=torch.float32)
            else:
                row_batch['roma_valid'] = row_batch['roma_valid'].to(device=device).float().reshape(-1)
            if 'need_only' not in row_batch:
                row_batch['need_only'] = torch.zeros_like(row_batch['roma_valid'])
            else:
                row_batch['need_only'] = row_batch['need_only'].to(device=device).float().reshape(-1)
            need_only = row_batch['need_only'] > 0.5
            roma_finite = torch.isfinite(row_batch['roma_xy8']).all(dim=1)
            if bool((need_only & (~roma_finite)).any().detach().cpu().item()):
                row_batch['roma_xy8'] = torch.where(
                    (need_only & (~roma_finite))[:, None],
                    baseline_xy8.float(),
                    row_batch['roma_xy8'],
                )

            can_sample_visual = (
                (has_visual or has_local_corr or relocalize_policy in ('heuristic', 'heuristic_learned_accept'))
                and fmaps is not None
                and (fmap_ara_set is None or int(chosen_time) in fmap_ara_set)
                and (fmap_ara_set is None or 0 in fmap_ara_set or has_query_fmap)
            )
            if can_sample_visual:
                source_frame = torch.zeros_like(row_batch['roma_valid'], dtype=torch.long, device=device)
                target_frame = torch.full_like(source_frame, int(chosen_time))
                source_xy8 = source_xy.to(device=device).float() / 8.0
                if has_query_fmap and (fmap_ara_set is not None and 0 not in fmap_ara_set):
                    query_feat = self._sample_roma_coarse_fmap_rows(
                        query_fmap[:, None],
                        source_frame,
                        source_xy8,
                        fmap_ara=[0],
                    )
                else:
                    query_feat = self._sample_roma_coarse_fmap_rows(fmaps, source_frame, source_xy8, fmap_ara=fmap_ara)
                baseline_feat = self._sample_roma_coarse_fmap_rows(fmaps, target_frame, baseline_xy8, fmap_ara=fmap_ara)
                roma_feat = self._sample_roma_coarse_fmap_rows(fmaps, target_frame, row_batch['roma_xy8'], fmap_ara=fmap_ara)
                prev_feat = self._sample_roma_coarse_fmap_rows(fmaps, target_frame, prev_baseline_xy8, fmap_ara=fmap_ara)
                if query_feat is not None and baseline_feat is not None and roma_feat is not None and prev_feat is not None:
                    query_feat = F.normalize(query_feat.float(), dim=1, eps=1.0e-6)
                    baseline_feat = F.normalize(baseline_feat.float(), dim=1, eps=1.0e-6)
                    roma_feat = F.normalize(roma_feat.float(), dim=1, eps=1.0e-6)
                    prev_feat = F.normalize(prev_feat.float(), dim=1, eps=1.0e-6)
                    row_batch['visual_query_feat'] = query_feat
                    row_batch['visual_baseline_feat'] = baseline_feat
                    row_batch['visual_roma_feat'] = roma_feat
                    row_batch['visual_prev_feat'] = prev_feat
                    row_batch['visual_baseline_query_cos'] = torch.sum(baseline_feat * query_feat, dim=1)
                    row_batch['visual_roma_query_cos'] = torch.sum(roma_feat * query_feat, dim=1)
                    row_batch['visual_prev_query_cos'] = torch.sum(prev_feat * query_feat, dim=1)
                    row_batch['visual_roma_baseline_cos'] = torch.sum(roma_feat * baseline_feat, dim=1)
                    row_batch['visual_roma_prev_cos'] = torch.sum(roma_feat * prev_feat, dim=1)
                    row_batch['visual_roma_query_cos_gain'] = row_batch['visual_roma_query_cos'] - row_batch['visual_baseline_query_cos']
                    row_batch['visual_prev_query_cos_gain'] = row_batch['visual_prev_query_cos'] - row_batch['visual_baseline_query_cos']
                    if has_local_corr:
                        offsets = torch.stack(
                            torch.meshgrid(
                                torch.arange(-2, 3, device=device, dtype=torch.float32),
                                torch.arange(-2, 3, device=device, dtype=torch.float32),
                                indexing='ij',
                            ),
                            dim=-1,
                        ).reshape(25, 2)

                        def _corr_5x5(center_xy8):
                            center_xy8 = center_xy8.to(device=device).float()
                            n = int(center_xy8.shape[0])
                            if n == 0:
                                return torch.zeros((0, 25), device=device, dtype=torch.float32)
                            sample_xy8 = (center_xy8[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
                            sample_frame = target_frame[:, None].expand(n, 25).reshape(-1)
                            sample_feat = self._sample_roma_coarse_fmap_rows(
                                fmaps,
                                sample_frame,
                                sample_xy8,
                                fmap_ara=fmap_ara,
                            )
                            if sample_feat is None:
                                return torch.zeros((n, 25), device=device, dtype=torch.float32)
                            sample_feat = F.normalize(sample_feat.float(), dim=1, eps=1.0e-6).reshape(n, 25, -1)
                            return torch.sum(sample_feat * query_feat[:, None, :], dim=-1)

                        row_batch['corr_baseline_5x5'] = _corr_5x5(baseline_xy8)
                        row_batch['corr_roma_5x5'] = _corr_5x5(row_batch['roma_xy8'])
                        row_batch['corr_prev_5x5'] = _corr_5x5(prev_baseline_xy8)

            roma_valid = row_batch['roma_valid'].reshape(-1) > 0.5
            finite = torch.isfinite(row_batch['roma_xy8']).all(dim=1) & torch.isfinite(baseline_xy8).all(dim=1)
            valid = finite & (roma_valid | need_only)
            if not bool(valid.any().detach().cpu().item()):
                continue

            prepared.append(
                {
                    'chosen_time': int(chosen_time),
                    'chosen_local': int(chosen_local),
                    'row_idx': row_idx,
                    'source_x8': source_x8,
                    'source_y8': source_y8,
                    'source_grid_xy8': source_grid_xy8,
                    'baseline_xy8': baseline_xy8.float(),
                    'prev_baseline_xy8': prev_baseline_xy8.float(),
                    'row_count': int(row_idx.numel()),
                }
            )
            _append_combined(row_batch)

        if not prepared:
            self._roma_init_last_applied = False
            return flows8

        combined_batch = {}
        for key, values in combined_parts.items():
            if values:
                combined_batch[key] = torch.cat(values, dim=0)
        if 'baseline_xy8' not in combined_batch or 'roma_xy8' not in combined_batch:
            self._roma_init_last_applied = False
            return flows8

        features, _ = build_roma_coarse_features(combined_batch, feature_names=list(feature_names), device=device)
        coord_mode = str(getattr(self, 'roma_coarse_adapter_coord_mode', 'fusion_residual'))
        pred_all = predict_roma_coarse_xy8(
            adapter,
            combined_batch,
            features,
            coord_mode=coord_mode,
        )
        if coord_mode == 'two_stage_st_fusion_residual' and pred_all.get('relocalize_logit', None) is not None:
            learned_gate_prob_all = torch.sigmoid(pred_all['relocalize_logit'].reshape(-1))
        else:
            learned_gate_prob_all = torch.sigmoid(pred_all['gate_logit'].reshape(-1))
        gate_mode = str(getattr(self, 'roma_coarse_adapter_gate_mode', 'none'))
        need_only_all = combined_batch.get(
            'need_only',
            torch.zeros_like(combined_batch['roma_valid']),
        ).to(device=device).float().reshape(-1) > 0.5
        if relocalize_policy in ('heuristic', 'heuristic_learned_accept'):
            def _scalar(name, default):
                value = combined_batch.get(name, None)
                if value is None:
                    return torch.full_like(learned_gate_prob_all, float(default))
                return value.to(device=device).float().reshape(-1)

            roma_xy8 = combined_batch['roma_xy8'].to(device=device).float()
            baseline_xy8 = combined_batch['baseline_xy8'].to(device=device).float()
            prev_xy8 = combined_batch['prev_baseline_xy8'].to(device=device).float()
            roma_valid = _scalar('roma_valid', 0.0) > 0.5
            certainty = _scalar('roma_certainty', 0.0)
            visible = _scalar('baseline_visible_score', 1.0)
            motion_jump8 = _scalar('baseline_motion_jump8', 0.0)
            visual_gain = _scalar('visual_roma_query_cos_gain', float('-inf'))
            visual_cos = _scalar('visual_roma_query_cos', float('-inf'))
            offset8 = torch.linalg.vector_norm(roma_xy8 - baseline_xy8, dim=1)
            prev_dist8 = torch.linalg.vector_norm(roma_xy8 - prev_xy8, dim=1)

            visual_supported = (
                (visual_gain >= float(getattr(self, 'roma_coarse_adapter_heuristic_visual_gain_thr', 0.05)))
                & (visual_cos >= float(getattr(self, 'roma_coarse_adapter_heuristic_visual_cos_thr', 0.5)))
            )
            baseline_suspicious = (
                (visible <= float(getattr(self, 'roma_coarse_adapter_heuristic_baseline_visible_thr', 0.5)))
                | (motion_jump8 >= float(getattr(self, 'roma_coarse_adapter_heuristic_baseline_motion_jump8_thr', 1.5)))
                | (visual_gain >= float(getattr(self, 'roma_coarse_adapter_heuristic_strong_visual_gain_thr', 0.15)))
            )
            temporal_safe = torch.ones_like(roma_valid)
            prev_dist8_max = float(getattr(self, 'roma_coarse_adapter_heuristic_roma_prev_dist8_max', 8.0))
            if prev_dist8_max > 0.0:
                temporal_safe = prev_dist8 <= prev_dist8_max
            heuristic_event = (
                roma_valid
                & (~need_only_all)
                & torch.isfinite(roma_xy8).all(dim=1)
                & (certainty >= float(getattr(self, 'roma_coarse_adapter_heuristic_roma_certainty_thr', 0.7)))
                & visual_supported
                & baseline_suspicious
                & temporal_safe
                & (offset8 >= float(getattr(self, 'roma_coarse_adapter_heuristic_min_offset8', 0.5)))
            )
            safety_prob = torch.sigmoid(
                pred_all.get('safety_logit', torch.full_like(learned_gate_prob_all[:, None], -20.0)).reshape(-1)
            )
            gain_pred_px = pred_all.get(
                'gain_pred_px', torch.full_like(learned_gate_prob_all[:, None], float('-inf'))
            ).reshape(-1)
            if bool(getattr(self, 'roma_coarse_adapter_safety_gate', False)):
                heuristic_event = (
                    heuristic_event
                    & (safety_prob >= float(getattr(self, 'roma_coarse_adapter_safety_thr', 0.6)))
                    & (gain_pred_px >= float(getattr(self, 'roma_coarse_adapter_safety_min_gain_px', 0.0)))
                )
            if relocalize_policy == 'heuristic_learned_accept':
                need_logit = pred_all.get('baseline_need_logit', None)
                quality_logit = pred_all.get('candidate_quality_logit', None)
                accept_logit = pred_all.get('candidate_accept_logit', None)
                need_prob = torch.sigmoid(need_logit.reshape(-1)) if need_logit is not None else torch.ones_like(learned_gate_prob_all)
                quality_prob = torch.sigmoid(quality_logit.reshape(-1)) if quality_logit is not None else torch.ones_like(learned_gate_prob_all)
                accept_prob = torch.sigmoid(accept_logit.reshape(-1)) if accept_logit is not None else learned_gate_prob_all
                need_thr = float(getattr(self, 'roma_coarse_adapter_baseline_need_thr', 0.5))
                quality_thr = float(getattr(self, 'roma_coarse_adapter_candidate_quality_thr', 0.5))
                accept_thr = float(getattr(self, 'roma_coarse_adapter_candidate_accept_thr', 0.5))
                learned_accept_event = (
                    (need_prob >= need_thr)
                    & (quality_prob >= quality_thr)
                    & (accept_prob >= accept_thr)
                )
                gate_all = (heuristic_event & learned_accept_event).float()
                gate_prob_all = accept_prob
            else:
                gate_all = heuristic_event.float()
                gate_prob_all = gate_all
        elif relocalize_policy == 'learned_accept':
            need_logit = pred_all.get('baseline_need_logit', None)
            quality_logit = pred_all.get('candidate_quality_logit', None)
            accept_logit = pred_all.get('candidate_accept_logit', None)
            need_prob = torch.sigmoid(need_logit.reshape(-1)) if need_logit is not None else torch.ones_like(learned_gate_prob_all)
            quality_prob = torch.sigmoid(quality_logit.reshape(-1)) if quality_logit is not None else torch.ones_like(learned_gate_prob_all)
            accept_prob = torch.sigmoid(accept_logit.reshape(-1)) if accept_logit is not None else learned_gate_prob_all
            need_thr = float(getattr(self, 'roma_coarse_adapter_baseline_need_thr', 0.5))
            quality_thr = float(getattr(self, 'roma_coarse_adapter_candidate_quality_thr', 0.5))
            accept_thr = float(getattr(self, 'roma_coarse_adapter_candidate_accept_thr', 0.5))
            roma_valid = combined_batch['roma_valid'].to(device=device).reshape(-1).float() > 0.5
            decision_event = (
                roma_valid
                & (~need_only_all)
                & torch.isfinite(combined_batch['roma_xy8'].to(device=device).float()).all(dim=1)
                & (need_prob >= need_thr)
                & (quality_prob >= quality_thr)
                & (accept_prob >= accept_thr)
            )
            gate_all = decision_event.float()
            gate_prob_all = accept_prob
        elif coord_mode == 'two_stage_st_fusion_residual':
            relocalize_event = pred_all.get('relocalize_event', None)
            if relocalize_event is None:
                gate_all = (learned_gate_prob_all > float(getattr(adapter, 'relocalize_conf_thr', 0.5))).float()
            else:
                gate_all = relocalize_event.reshape(-1).to(device=device, dtype=learned_gate_prob_all.dtype)
            gate_prob_all = learned_gate_prob_all
        elif gate_mode == 'none':
            gate_all = torch.ones_like(learned_gate_prob_all)
            gate_prob_all = learned_gate_prob_all
        elif gate_mode == 'hard_straight_through':
            hard = (learned_gate_prob_all > 0.5).float()
            gate_all = hard.detach() - learned_gate_prob_all.detach() + learned_gate_prob_all
            gate_prob_all = learned_gate_prob_all
        else:
            gate_all = learned_gate_prob_all
            gate_prob_all = learned_gate_prob_all

        roma_valid_all = combined_batch['roma_valid'].reshape(-1).float() > 0.5
        finite_all = torch.isfinite(combined_batch['baseline_xy8'].float()).all(dim=1)
        if relocalize_policy in ('heuristic', 'heuristic_learned_accept'):
            finite_all = (
                finite_all
                & torch.isfinite(combined_batch['prev_baseline_xy8'].float()).all(dim=1)
                & torch.isfinite(combined_batch['roma_xy8'].float()).all(dim=1)
            )
            valid_all = roma_valid_all & finite_all
        elif coord_mode == 'two_stage_st_fusion_residual':
            finite_all = finite_all & torch.isfinite(combined_batch['prev_baseline_xy8'].float()).all(dim=1)
            valid_all = finite_all
        else:
            finite_all = finite_all & torch.isfinite(combined_batch['roma_xy8'].float()).all(dim=1)
            valid_all = roma_valid_all & finite_all
        policy_gate_all = gate_all * valid_all.float() * (~need_only_all).float()
        counterfactual_gate_all = torch.zeros_like(policy_gate_all)
        if bool(getattr(self, 'roma_coarse_adapter_train_mode', False)):
            reject_ratio = float(getattr(self, 'roma_coarse_adapter_counterfactual_reject_ratio', 0.0))
            reject_max_rows = int(getattr(self, 'roma_coarse_adapter_counterfactual_reject_max_rows', 0))
            if reject_ratio > 0.0:
                rejected = (policy_gate_all <= 0.5) & valid_all & (~need_only_all)
                rejected_idx = torch.nonzero(rejected, as_tuple=False).reshape(-1)
                if int(rejected_idx.numel()) > 0:
                    take = int(math.ceil(float(rejected_idx.numel()) * min(reject_ratio, 1.0)))
                    if reject_ratio > 1.0:
                        take = int(reject_ratio)
                    if reject_max_rows > 0:
                        take = min(take, reject_max_rows)
                    take = max(0, min(take, int(rejected_idx.numel())))
                    if take > 0:
                        perm = torch.randperm(int(rejected_idx.numel()), device=device)[:take]
                        sampled = rejected_idx.index_select(0, perm)
                        counterfactual_gate_all[sampled] = 1.0
        gate_all = torch.maximum(policy_gate_all, counterfactual_gate_all)
        if relocalize_policy in ('heuristic', 'learned_accept', 'heuristic_learned_accept'):
            residual_xy8 = pred_all.get('residual_xy8', torch.zeros_like(combined_batch['roma_xy8']))
            pred_xy8_all = pred_all.get(
                'candidate_xy8',
                combined_batch['roma_xy8'].to(device=device).float() + residual_xy8.to(device=device).float(),
            ).to(device=device).float()
        else:
            pred_xy8_all = pred_all['pred_xy8'].to(device=device).float()
        gt_mix_active_all = torch.zeros_like(gate_all)
        if (
            bool(getattr(self, 'roma_coarse_adapter_train_mode', False))
            and 'gt_xy8' in combined_batch
            and float(getattr(self, 'roma_coarse_adapter_gt_mix_prob', 0.0)) > 0.0
            and float(getattr(self, 'roma_coarse_adapter_gt_mix_ratio', 0.0)) > 0.0
        ):
            mix_prob = float(getattr(self, 'roma_coarse_adapter_gt_mix_prob', 0.0))
            mix_ratio = float(np.clip(float(getattr(self, 'roma_coarse_adapter_gt_mix_ratio', 0.0)), 0.0, 1.0))
            if float(torch.rand((), device=device).item()) < mix_prob:
                gt_xy8 = combined_batch['gt_xy8'].to(device=device).float()
                if float(getattr(self, 'roma_coarse_adapter_gt_mix_noise8', 0.0)) > 0.0:
                    gt_xy8 = gt_xy8 + torch.randn_like(gt_xy8) * float(getattr(self, 'roma_coarse_adapter_gt_mix_noise8', 0.0))
                gt_finite = torch.isfinite(gt_xy8).all(dim=1) & valid_all & (~need_only_all)
                mixed_pred = (1.0 - mix_ratio) * pred_xy8_all + mix_ratio * gt_xy8
                pred_xy8_all = torch.where(gt_finite[:, None], mixed_pred, pred_xy8_all)
                gt_mix_active_all = gt_finite.float()
        if coord_mode == 'two_stage_st_fusion_residual':
            gate_all = torch.maximum(gate_all, gt_mix_active_all)
        init_xy8_all = combined_batch['baseline_xy8'].float() + gate_all[:, None] * (
            pred_xy8_all - combined_batch['baseline_xy8'].float()
        )
        carry_mode = str(getattr(self, 'roma_coarse_adapter_carry_mode', 'none'))
        carry_applied_all = torch.zeros_like(gate_all, dtype=torch.bool)
        carry_refresh_all = torch.zeros_like(gate_all, dtype=torch.bool)
        if carry_mode not in ('none', '') and 'point_index' in combined_batch:
            point_index_all = combined_batch['point_index'].to(device=device).long().reshape(-1)
            target_frame_idx_all = combined_batch['target_frame'].to(device=device).long().reshape(-1)
            baseline_xy8_all = combined_batch['baseline_xy8'].to(device=device).float()
            carry_state = getattr(self, 'roma_coarse_adapter_carry_state', None)
            if carry_state is None:
                carry_state = {}
                self.roma_coarse_adapter_carry_state = carry_state
            max_age = max(0, int(getattr(self, 'roma_coarse_adapter_carry_max_age', 0)))
            decay = float(getattr(self, 'roma_coarse_adapter_carry_decay', 0.9))
            min_score = float(getattr(self, 'roma_coarse_adapter_carry_min_score', 0.0))
            max_offset8 = float(getattr(self, 'roma_coarse_adapter_carry_max_offset8', 8.0))
            apply_strength = float(getattr(self, 'roma_coarse_adapter_carry_apply_strength', 1.0))
            refresh_dist8 = float(getattr(self, 'roma_coarse_adapter_carry_refresh_dist8', 2.0))
            require_suspicious = bool(getattr(self, 'roma_coarse_adapter_carry_require_baseline_suspicious', False))
            if require_suspicious:
                visible_all = combined_batch.get('baseline_visible_score', None)
                motion_jump_all = combined_batch.get('baseline_motion_jump8', None)
                visual_gain_all = combined_batch.get('visual_roma_query_cos_gain', None)
                visible_all = (
                    visible_all.to(device=device).float().reshape(-1)
                    if torch.is_tensor(visible_all)
                    else torch.ones_like(gate_all, dtype=torch.float32)
                )
                motion_jump_all = (
                    motion_jump_all.to(device=device).float().reshape(-1)
                    if torch.is_tensor(motion_jump_all)
                    else torch.zeros_like(gate_all, dtype=torch.float32)
                )
                visual_gain_all = (
                    visual_gain_all.to(device=device).float().reshape(-1)
                    if torch.is_tensor(visual_gain_all)
                    else torch.full_like(gate_all, float('-inf'), dtype=torch.float32)
                )
                carry_suspicious_all = (
                    (visible_all <= float(getattr(self, 'roma_coarse_adapter_heuristic_baseline_visible_thr', 0.5)))
                    | (motion_jump_all >= float(getattr(self, 'roma_coarse_adapter_heuristic_baseline_motion_jump8_thr', 1.5)))
                    | (visual_gain_all >= float(getattr(self, 'roma_coarse_adapter_heuristic_strong_visual_gain_thr', 0.15)))
                )
            else:
                carry_suspicious_all = torch.ones_like(gate_all, dtype=torch.bool)
            order = torch.argsort(target_frame_idx_all * (point_index_all.max().clamp_min(0) + 1) + point_index_all.clamp_min(0))
            for row_idx_t in order.detach().cpu().tolist():
                row_idx = int(row_idx_t)
                point_id = int(point_index_all[row_idx].detach().cpu().item())
                frame_id = int(target_frame_idx_all[row_idx].detach().cpu().item())
                if point_id < 0:
                    continue
                baseline_xy = baseline_xy8_all[row_idx]
                if not bool(torch.isfinite(baseline_xy).all().item()):
                    continue
                accepted = bool((gate_all[row_idx] > 0.5).detach().cpu().item())
                if accepted:
                    offset = (pred_xy8_all[row_idx] - baseline_xy).detach()
                    offset_norm = float(torch.linalg.vector_norm(offset).detach().cpu().item())
                    if bool(torch.isfinite(offset).all().item()) and (max_offset8 <= 0.0 or offset_norm <= max_offset8):
                        score = 1.0
                        if 'roma_certainty' in combined_batch:
                            score = float(combined_batch['roma_certainty'].reshape(-1)[row_idx].detach().float().cpu().item())
                        allow_refresh = True
                        old_state = carry_state.get(point_id)
                        if carry_mode == 'offset_decay' and old_state and torch.is_tensor(old_state.get('offset', None)):
                            old_offset = old_state['offset'].to(device=device, dtype=offset.dtype)
                            offset_dist = float(torch.linalg.vector_norm(offset - old_offset).detach().cpu().item())
                            allow_refresh = offset_dist <= refresh_dist8
                        if allow_refresh:
                            carry_state[point_id] = {
                                'offset': offset.detach(),
                                'score': float(score),
                                'frame': int(frame_id),
                            }
                            carry_refresh_all[row_idx] = True
                        elif carry_mode == 'offset_decay' and old_state:
                            old_score = float(old_state.get('score', 0.0))
                            old_state['score'] = float(old_score * decay)
                            old_state['frame'] = int(frame_id)
                            carry_state[point_id] = old_state
                    continue
                if not bool(carry_suspicious_all[row_idx].detach().cpu().item()):
                    continue
                state = carry_state.get(point_id)
                if not state:
                    continue
                age = frame_id - int(state.get('frame', frame_id))
                if age < 0 or age > max_age:
                    continue
                score = float(state.get('score', 0.0)) * (decay ** max(age, 0))
                if score < min_score:
                    continue
                offset = state.get('offset')
                if not torch.is_tensor(offset):
                    continue
                offset = offset.to(device=device, dtype=baseline_xy.dtype)
                offset_norm = float(torch.linalg.vector_norm(offset).detach().cpu().item())
                if not bool(torch.isfinite(offset).all().item()) or (max_offset8 > 0.0 and offset_norm > max_offset8):
                    continue
                if carry_mode == 'offset_decay':
                    effective_offset = offset * apply_strength * (decay ** max(age, 0))
                else:
                    effective_offset = offset
                init_xy8_all[row_idx] = baseline_xy + effective_offset
                gate_all[row_idx] = 1.0
                carry_applied_all[row_idx] = True
            self.roma_coarse_adapter_carry_state = carry_state
        if relocalize_policy in ('heuristic', 'learned_accept', 'heuristic_learned_accept'):
            fusion_weights_all = torch.stack(
                [1.0 - gate_all, gate_all, torch.zeros_like(gate_all)], dim=1
            )
        else:
            fusion_weights_all = pred_all.get('fusion_weights', None)

        if 'gt_xy8' in combined_batch:
            supervision = {
                'adapter_features': features.float(),
                'target_frame': combined_batch['target_frame'].to(device=device).long().reshape(-1),
                'point_index': combined_batch.get(
                    'point_index',
                    torch.full_like(combined_batch['target_frame'].to(device=device).long().reshape(-1), -1),
                ).to(device=device).long().reshape(-1),
                'baseline_xy8': combined_batch['baseline_xy8'].float(),
                'prev_baseline_xy8': combined_batch['prev_baseline_xy8'].float(),
                'roma_xy8': combined_batch['roma_xy8'].float(),
                'gt_xy8': combined_batch['gt_xy8'].to(device=device).float(),
                'roma_valid': combined_batch['roma_valid'].float().reshape(-1),
                'need_only': combined_batch.get(
                    'need_only',
                    torch.zeros_like(combined_batch['roma_valid']),
                ).float().reshape(-1),
                'roma_certainty': combined_batch.get(
                    'roma_certainty',
                    torch.zeros_like(combined_batch['roma_valid']),
                ).to(device=device).float().reshape(-1),
                'pred_xy8': pred_xy8_all.float(),
                'delta_xy8': pred_xy8_all.float() - combined_batch['baseline_xy8'].float(),
                'effective_delta_xy8': init_xy8_all.float() - combined_batch['baseline_xy8'].float(),
                'gate_logit': pred_all['gate_logit'].reshape(-1),
                'quality_logit': pred_all['quality_logit'].reshape(-1),
                'gate_prob': gate_prob_all.reshape(-1),
                'learned_gate_prob': learned_gate_prob_all.reshape(-1),
                'gate': gate_all.reshape(-1),
                'policy_relocalize_event': policy_gate_all.reshape(-1).bool(),
                'counterfactual_relocalize_event': counterfactual_gate_all.reshape(-1).bool(),
                'carry_relocalize_event': carry_applied_all.reshape(-1).bool(),
                'carry_refresh_event': carry_refresh_all.reshape(-1).bool(),
                'quality_prob': torch.sigmoid(pred_all['quality_logit'].reshape(-1)),
            }
            safety_logit = pred_all.get('safety_logit', None)
            if safety_logit is not None:
                supervision['safety_logit'] = safety_logit.reshape(-1).float()
                supervision['safety_prob'] = torch.sigmoid(safety_logit.reshape(-1).float())
            gain_pred_px = pred_all.get('gain_pred_px', None)
            if gain_pred_px is not None:
                supervision['gain_pred_px'] = gain_pred_px.reshape(-1).float()
            candidate_accept_logit = pred_all.get('candidate_accept_logit', None)
            if candidate_accept_logit is not None:
                candidate_accept_logit = candidate_accept_logit.reshape(-1).float()
                supervision['candidate_accept_logit'] = candidate_accept_logit
                supervision['candidate_accept_prob'] = torch.sigmoid(candidate_accept_logit)
            baseline_need_logit = pred_all.get('baseline_need_logit', None)
            if baseline_need_logit is not None:
                baseline_need_logit = baseline_need_logit.reshape(-1).float()
                supervision['baseline_need_logit'] = baseline_need_logit
                supervision['baseline_need_prob'] = torch.sigmoid(baseline_need_logit)
            candidate_quality_logit = pred_all.get('candidate_quality_logit', None)
            if candidate_quality_logit is not None:
                candidate_quality_logit = candidate_quality_logit.reshape(-1).float()
                supervision['candidate_quality_logit'] = candidate_quality_logit
                supervision['candidate_quality_prob'] = torch.sigmoid(candidate_quality_logit)
            adapter_hidden = pred_all.get('adapter_hidden', None)
            if adapter_hidden is not None:
                supervision['adapter_hidden'] = adapter_hidden.float()
            for key in (
                'baseline_visible_score',
                'baseline_motion_jump8',
                'visual_roma_query_cos_gain',
                'visual_roma_query_cos',
            ):
                value = combined_batch.get(key, None)
                if torch.is_tensor(value):
                    supervision[key] = value.to(device=device).float().reshape(-1)
            basis_logits = pred_all.get('basis_logits', None)
            if basis_logits is not None:
                supervision['basis_logits'] = basis_logits.float()
            candidate_error8 = pred_all.get('candidate_error8', None)
            if candidate_error8 is not None:
                supervision['candidate_error8'] = candidate_error8.float()
            relocalize_logit = pred_all.get('relocalize_logit', None)
            if relocalize_logit is not None:
                supervision['relocalize_logit'] = relocalize_logit.reshape(-1).float()
            if relocalize_policy in ('heuristic', 'learned_accept', 'heuristic_learned_accept'):
                supervision['raw_relocalize_event'] = policy_gate_all.reshape(-1).bool()
                supervision['relocalize_event'] = gate_all.reshape(-1).bool()
            else:
                for key in ('raw_relocalize_event', 'relocalize_event'):
                    value = pred_all.get(key, None)
                    if value is not None:
                        supervision[key] = value.reshape(-1).bool()
            if fusion_weights_all is not None:
                supervision['fusion_weights'] = fusion_weights_all.float()
            for key in ('residual_xy8', 'alternative_xy8', 'candidate_xy8'):
                value = pred_all.get(key, None)
                if value is not None:
                    supervision[key] = value.float()
            if relocalize_policy in ('heuristic', 'learned_accept', 'heuristic_learned_accept'):
                supervision['candidate_xy8'] = pred_xy8_all.float()
            history = getattr(self, 'roma_coarse_adapter_supervision_history', None)
            if history is None:
                history = []
                self.roma_coarse_adapter_supervision_history = history
            history.append(supervision)

        row_start = 0
        for item in prepared:
            row_end = row_start + int(item['row_count'])
            gate = gate_all[row_start:row_end]
            gate_prob = gate_prob_all[row_start:row_end]
            init_xy8 = init_xy8_all[row_start:row_end]
            baseline_xy8 = item['baseline_xy8']
            source_grid_xy8 = item['source_grid_xy8']
            init_flow8 = init_xy8.to(dtype=dtype) - source_grid_xy8.to(dtype=dtype)
            source_y8 = item['source_y8']
            source_x8 = item['source_x8']
            flat_idx = source_y8 * int(W8) + source_x8
            flat_size = int(H8 * W8)
            flat_flow = torch.zeros((flat_size, 2), device=device, dtype=dtype)
            flat_count = torch.zeros((flat_size, 1), device=device, dtype=dtype)
            weights = gate.to(device=device, dtype=dtype).reshape(-1, 1)
            flat_flow.index_add_(0, flat_idx, init_flow8 * weights)
            flat_count.index_add_(0, flat_idx, weights)
            mask_flat = flat_count[:, 0] > 1.0e-6
            if bool(mask_flat.any().detach().cpu().item()):
                flat_flow = flat_flow / flat_count.clamp_min(1.0e-6)
                flow_map = flat_flow.reshape(H8, W8, 2).permute(2, 0, 1)
                mask = mask_flat.reshape(1, H8, W8)
                if not applied_any:
                    flows_out = flows_out.clone()
                flows_out[0, int(item['chosen_local'])] = torch.where(mask, flow_map, flows_out[0, int(item['chosen_local'])])
                applied_any = True

            fusion_slice = fusion_weights_all[row_start:row_end] if fusion_weights_all is not None else None
            carry_slice = carry_applied_all[row_start:row_end]
            carry_refresh_slice = carry_refresh_all[row_start:row_end]
            frame_stats.append(
                {
                    'target_frame': float(item['chosen_time']),
                    'num_rows': float(item['row_count']),
                    'gate_mean': float(gate.detach().float().mean().item()),
                    'gate_prob_mean': float(gate_prob.detach().float().mean().item()),
                    'effective_delta_norm8_mean': float(torch.linalg.vector_norm((init_xy8 - baseline_xy8.float()).detach(), dim=1).mean().item()),
                    'fusion_w_baseline': float(fusion_slice[:, 0].detach().float().mean().item()) if fusion_slice is not None else float('nan'),
                    'fusion_w_roma': float(fusion_slice[:, 1].detach().float().mean().item()) if fusion_slice is not None else float('nan'),
                    'fusion_w_prev': float(fusion_slice[:, 2].detach().float().mean().item()) if fusion_slice is not None else float('nan'),
                    'gt_mix_active': float(gt_mix_active_all[row_start:row_end].detach().float().mean().item()),
                    'carry_mean': float(carry_slice.detach().float().mean().item()),
                    'carry_refresh_mean': float(carry_refresh_slice.detach().float().mean().item()),
                }
            )
            row_start = row_end

        if not applied_any:
            self._roma_init_last_applied = False
            return flows8
        self._update_roma_coarse_adapter_stats(frame_stats)
        self._roma_init_last_applied = True
        self._reloc_decision_last_applied = False
        return flows_out

    def get_T_padded_images(self, images, T, S, is_training, stride=None, pad=True):
        B,T,C,H,W = images.shape
        indices = None
        if T > 2:
            step = S // 2 if stride is None else stride
            indices = []
            start = 0
            while start + S < T:
                indices.append(start)
                start += step
            indices.append(start)
            Tpad = indices[-1]+S-T
            if pad:
                if is_training:
                    assert Tpad == 0
                else:
                    images = images.reshape(B,1,T,C*H*W)
                    if Tpad > 0:
                        padding_tensor = images[:,:,-1:,:].expand(B,1,Tpad,C*H*W)
                        images = torch.cat([images, padding_tensor], dim=2)
                    images = images.reshape(B,T+Tpad,C,H,W)
                    T = T+Tpad
        else:
            assert T == 2
        return images, T, indices

    def get_fmaps(self, images_, B, T, sw, is_training):
        _, _, H_pad, W_pad = images_.shape # revised HW

        C, H8, W8 = self.dim*2, H_pad//8, W_pad//8
        if self.no_split:
            C = self.dim

        fmaps_chunk_size = 64
        if (not is_training) and (T > fmaps_chunk_size):
            images = images_.reshape(B,T,3,H_pad,W_pad)
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                images_chunk = images[:, t : t + fmaps_chunk_size]
                images_chunk = images_chunk.cuda()
                if self.use_basicencoder:
                    if self.full_split:
                        fmaps_chunk1 = self.fnet(images_chunk.reshape(-1, 3, H_pad, W_pad))
                        fmaps_chunk2 = self.cnet(images_chunk.reshape(-1, 3, H_pad, W_pad))
                        fmaps_chunk = torch.cat([fmaps_chunk1, fmaps_chunk2], axis=1)
                    else:
                        fmaps_chunk = self.fnet(images_chunk.reshape(-1, 3, H_pad, W_pad))
                else:
                    fmaps_chunk = self.cnn(images_chunk.reshape(-1, 3, H_pad, W_pad))
                    if t==0 and sw is not None and sw.save_this:
                        sw.summ_feat('1_model/fmap_raw', fmaps_chunk[0:1])
                    fmaps_chunk = self.dot_conv(fmaps_chunk) # B*T,C,H8,W8
                T_chunk = images_chunk.shape[1]
                fmaps.append(fmaps_chunk.reshape(B, -1, C, H8, W8))
            fmaps_ = torch.cat(fmaps, dim=1).reshape(-1, C, H8, W8)
        else:
            if not is_training:
                # sometimes we need to move things to cuda here
                images_ = images_.cuda()
            if self.use_basicencoder:
                if self.full_split:
                    fmaps1_ = self.fnet(images_)
                    fmaps2_ = self.cnet(images_)
                    fmaps_ = torch.cat([fmaps1_, fmaps2_], axis=1)
                else:
                    fmaps_ = self.fnet(images_)
            else:
                fmaps_ = self.cnn(images_)
                if sw is not None and sw.save_this:
                    sw.summ_feat('1_model/fmap_raw', fmaps_[0:1])
                fmaps_ = self.dot_conv(fmaps_) # B*T,C,H8,W8
        return fmaps_
    
    def forward(self, images, iters=4, sw=None, is_training=False, stride=None):
        B,T,C,H,W = images.shape
        S = self.seqlen
        device = images.device
        dtype = images.dtype
        if bool(getattr(self, 'patch_memory_export_candidates', False)):
            self.reset_patch_memory_candidate_info()
        if bool(getattr(self, 'export_risk_features', False)):
            self.reset_risk_info()

        # images are in [0,255]
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,1,3,1,1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,1,3,1,1).to(images.dtype)
        images = images / 255.0
        images = (images - mean)/std

        T_bak = T
        if stride is not None:
            pad = False
        else:
            pad = True
        images, T, indices = self.get_T_padded_images(images, T, S, is_training, stride=stride, pad=pad)

        images = images.contiguous()
        images_ = images.reshape(B*T,3,H,W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]

        _, _, H_pad, W_pad = images_.shape # revised HW
        C, H8, W8 = self.dim*2, H_pad//8, W_pad//8
        C2 = C//2
        if self.no_split:
            C = self.dim
            C2 = C

        fmaps = self.get_fmaps(images_, B, T, sw, is_training).reshape(B,T,C,H8,W8)
        device = fmaps.device

        fmap_anchor = fmaps[:,0]

        if T<=2 or is_training:
            # note: collecting preds can get expensive on a long video
            all_flow_preds = []
            all_visconf_preds = []
        else:
            all_flow_preds = None
            all_visconf_preds = None

        if T > 2: # multiframe tracking
            
            # we will store our final outputs in these tensors
            full_flows = torch.zeros((B,T,2,H,W), dtype=dtype, device=device)
            full_visconfs = torch.zeros((B,T,2,H,W), dtype=dtype, device=device)
            # 1/8 resolution 
            full_flows8 = torch.zeros((B,T,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            full_visconfs8 = torch.zeros((B,T,2,H_pad//8,W_pad//8), dtype=dtype, device=device)

            if self.use_feats8:
                full_feats8 = torch.zeros((B,T,C2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            visits = np.zeros((T))

            for ii, ind in enumerate(indices):
                ara = np.arange(ind,ind+S)
                if ii < len(indices)-1:
                    next_ind = indices[ii+1]
                    next_ara = np.arange(next_ind,next_ind+S)
                
                # print("torch.cuda.memory_allocated: %.1fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024), 'ara', ara)
                fmaps2 = fmaps[:,ara]
                flows8 = full_flows8[:,ara]
                visconfs8_window = full_visconfs8[:,ara]
                flows8 = self._apply_roma_coarse_adapter_to_flows8(flows8, ara, fmaps=fmaps, visconfs8=visconfs8_window)
                if not bool(getattr(self, '_roma_init_last_applied', False)):
                    flows8 = self._apply_roma_init_override_to_flows8(flows8, ara)
                window_preserve_grad = (
                    bool(getattr(self, 'roma_init_preserve_grad', False))
                    and bool(getattr(self, '_roma_init_last_applied', False))
                )
                flows8 = flows8.reshape(B*(S),2,H_pad//8,W_pad//8)
                if not window_preserve_grad:
                    flows8 = flows8.detach()
                visconfs8 = visconfs8_window.reshape(B*(S),2,H_pad//8,W_pad//8).detach()

                if self.use_feats8:
                    if ind==0:
                        feats8 = None
                    else:
                        feats8 = full_feats8[:,ara].reshape(B*(S),C2,H_pad//8,W_pad//8).detach()
                else:
                    feats8 = None

                self._patch_memory_window_start = int(ind)
                self._roma_init_window_preserve_grad = bool(window_preserve_grad)
                flow_predictions, visconf_predictions, flows8, visconfs8, feats8 = self.forward_window(
                    fmap_anchor, fmaps2, visconfs8, iters=iters, flowfeat=feats8, flows8=flows8,
                    is_training=is_training)

                unpad_flow_predictions = []
                unpad_visconf_predictions = []
                for i in range(len(flow_predictions)):
                    flow_predictions[i] = padder.unpad(flow_predictions[i])
                    unpad_flow_predictions.append(flow_predictions[i].reshape(B,S,2,H,W))
                    visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                    unpad_visconf_predictions.append(visconf_predictions[i].reshape(B,S,2,H,W))

                full_flows[:,ara] = unpad_flow_predictions[-1].reshape(B,S,2,H,W)
                full_flows8[:,ara] = flows8.reshape(B,S,2,H_pad//8,W_pad//8)
                full_visconfs[:,ara] = unpad_visconf_predictions[-1].reshape(B,S,2,H,W)
                full_visconfs8[:,ara] = visconfs8.reshape(B,S,2,H_pad//8,W_pad//8)
                if self.use_feats8:
                    full_feats8[:,ara] = feats8.reshape(B,S,C2,H_pad//8,W_pad//8)
                visits[ara] += 1

                if is_training:
                    all_flow_preds.append(unpad_flow_predictions)
                    all_visconf_preds.append(unpad_visconf_predictions)
                else:
                    del unpad_flow_predictions
                    del unpad_visconf_predictions

                # for the next iter, replace empty data with nearest available preds
                invalid_idx = np.where(visits==0)[0]
                valid_idx = np.where(visits>0)[0]
                for idx in invalid_idx:
                    nearest = valid_idx[np.argmin(np.abs(valid_idx - idx))]
                    # print('replacing %d with %d' % (idx, nearest))
                    full_flows8[:,idx] = full_flows8[:,nearest]
                    full_visconfs8[:,idx] = full_visconfs8[:,nearest]
                    if self.use_feats8:
                        full_feats8[:,idx] = full_feats8[:,nearest]
        else: # flow

            flows8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)

            self._patch_memory_window_start = 1
            flow_predictions, visconf_predictions, flows8, visconfs8, feats8 = self.forward_window(
                fmap_anchor, fmaps[:,1:2], visconfs8, iters=iters, flowfeat=None, flows8=flows8,
                is_training=is_training)
            unpad_flow_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_predictions[i].reshape(B,2,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(visconf_predictions[i].reshape(B,2,H,W))
            full_flows = all_flow_preds[-1].reshape(B,2,H,W)
            full_visconfs = all_visconf_preds[-1].reshape(B,2,H,W)
                
        if (not is_training) and (T > 2):
            full_flows = full_flows[:,:T_bak]
            full_visconfs = full_visconfs[:,:T_bak]
            
        return full_flows, full_visconfs, all_flow_preds, all_visconf_preds
    
    def forward_sliding(self, images, iters=4, sw=None, is_training=False, window_len=None, stride=None):
        B,T,C,H,W = images.shape
        S = self.seqlen if window_len is None else window_len
        device = images.device
        dtype = images.dtype
        stride = S // 2 if stride is None else stride

        # images are in [0,255]
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1,1,3,1,1).to(images.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1,1,3,1,1).to(images.dtype)
        images = images / 255.0
        images = (images - mean)/std

        T_bak = T
        images, T, indices = self.get_T_padded_images(images, T, S, is_training, stride)
        assert stride <= S // 2

        images = images.contiguous()
        images_ = images.reshape(B*T,3,H,W)
        padder = InputPadder(images_.shape)
        images_ = padder.pad(images_)[0]

        _, _, H_pad, W_pad = images_.shape # revised HW
        C, H8, W8 = self.dim*2, H_pad//8, W_pad//8
        C2 = C//2
        if self.no_split:
            C = self.dim
            C2 = C
            
        all_flow_preds = None
        all_visconf_preds = None
        
        if T<=2:
            # note: collecting preds can get expensive on a long video
            all_flow_preds = []
            all_visconf_preds = []
            
            fmaps = self.get_fmaps(images_, B, T, sw, is_training).reshape(B,T,C,H8,W8)
            device = fmaps.device
            
            flows8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
            visconfs8 = torch.zeros((B,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                
            fmap_anchor = fmaps[:,0]
            
            flow_predictions, visconf_predictions, flows8, visconfs8, feats8 = self.forward_window(
                fmap_anchor, fmaps[:,1:2], visconfs8, iters=iters, flowfeat=None, flows8=flows8,
                is_training=is_training)
            unpad_flow_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                all_flow_preds.append(flow_predictions[i].reshape(B,2,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                all_visconf_preds.append(visconf_predictions[i].reshape(B,2,H,W))
            full_flows = all_flow_preds[-1].reshape(B,2,H,W).detach().cpu()
            full_visconfs = all_visconf_preds[-1].reshape(B,2,H,W).detach().cpu()
            
            return full_flows, full_visconfs, all_flow_preds, all_visconf_preds

        assert T > 2 # multiframe tracking
        
        if is_training:
            all_flow_preds = []
            all_visconf_preds = []
            
        # we will store our final outputs in these cpu tensors
        full_flows = torch.zeros((B,T,2,H,W), dtype=dtype, device='cpu')
        full_visconfs = torch.zeros((B,T,2,H,W), dtype=dtype, device='cpu')
        
        images_ = images_.reshape(B,T,3,H_pad,W_pad)
        fmap_anchor = self.get_fmaps(images_[:,:1].reshape(-1,3,H_pad,W_pad), B, 1, sw, is_training).reshape(B,C,H8,W8)
        device = fmap_anchor.device
        full_visited = torch.zeros((T,), dtype=torch.bool, device=device)

        for ii, ind in enumerate(indices):
            ara = np.arange(ind,ind+S)
            if ii == 0:
                flows8 = torch.zeros((B,S,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                visconfs8 = torch.zeros((B,S,2,H_pad//8,W_pad//8), dtype=dtype, device=device)
                fmaps2 = self.get_fmaps(images_[:,ara].reshape(-1,3,H_pad,W_pad), B, S, sw, is_training).reshape(B,S,C,H8,W8)
            else:
                flows8 = torch.cat([flows8[:,stride:stride+S//2], flows8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                visconfs8 = torch.cat([visconfs8[:,stride:stride+S//2], visconfs8[:,stride+S//2-1:stride+S//2].repeat(1,S//2,1,1,1)], dim=1)
                fmaps2 = torch.cat([fmaps2[:,stride:stride+S//2], 
                                    self.get_fmaps(images_[:,np.arange(ind+S//2,ind+S)].reshape(-1,3,H_pad,W_pad), B, S//2, sw, is_training).reshape(B,S//2,C,H8,W8)], dim=1)
            flows8 = self._apply_roma_coarse_adapter_to_flows8(
                flows8,
                ara,
                fmaps=fmaps2,
                fmap_ara=ara,
                visconfs8=visconfs8,
                query_fmap=fmap_anchor,
            )
            if not bool(getattr(self, '_roma_init_last_applied', False)):
                flows8 = self._apply_roma_init_override_to_flows8(flows8, ara)
            window_preserve_grad = (
                bool(getattr(self, 'roma_init_preserve_grad', False))
                and bool(getattr(self, '_roma_init_last_applied', False))
            )
            flows8 = flows8.reshape(B*S,2,H_pad//8,W_pad//8)
            if not window_preserve_grad:
                flows8 = flows8.detach()
            visconfs8 = visconfs8.reshape(B*S,2,H_pad//8,W_pad//8).detach()
            
            self._roma_init_window_preserve_grad = bool(window_preserve_grad)
            flow_predictions, visconf_predictions, flows8, visconfs8, _ = self.forward_window(
                fmap_anchor, fmaps2, visconfs8, iters=iters, flowfeat=None, flows8=flows8,
                is_training=is_training)

            unpad_flow_predictions = []
            unpad_visconf_predictions = []
            for i in range(len(flow_predictions)):
                flow_predictions[i] = padder.unpad(flow_predictions[i])
                unpad_flow_predictions.append(flow_predictions[i].reshape(B,S,2,H,W))
                visconf_predictions[i] = padder.unpad(torch.sigmoid(visconf_predictions[i]))
                unpad_visconf_predictions.append(visconf_predictions[i].reshape(B,S,2,H,W))

            current_visiting = torch.zeros((T,), dtype=torch.bool, device=device)
            current_visiting[ara] = True
            
            to_fill = current_visiting & (~full_visited)
            to_fill_sum = to_fill.sum().item()
            full_flows[:,to_fill] = unpad_flow_predictions[-1].reshape(B,S,2,H,W)[:,-to_fill_sum:].detach().cpu()
            full_visconfs[:,to_fill] = unpad_visconf_predictions[-1].reshape(B,S,2,H,W)[:,-to_fill_sum:].detach().cpu()
            full_visited |= current_visiting

            if is_training:
                all_flow_preds.append(unpad_flow_predictions)
                all_visconf_preds.append(unpad_visconf_predictions)
            else:
                del unpad_flow_predictions
                del unpad_visconf_predictions
                
            flows8 = flows8.reshape(B,S,2,H_pad//8,W_pad//8)
            visconfs8 = visconfs8.reshape(B,S,2,H_pad//8,W_pad//8)
                
        if not is_training:
            full_flows = full_flows[:,:T_bak]
            full_visconfs = full_visconfs[:,:T_bak]
            
        return full_flows, full_visconfs, all_flow_preds, all_visconf_preds
        
    def forward_window(self, fmap1_single, fmaps2, visconfs8, iters=None, flowfeat=None, flows8=None, sw=None, is_training=False):
        B,S,C,H8,W8 = fmaps2.shape
        device = fmaps2.device
        dtype = fmaps2.dtype
        self._risk_window_s = int(S)

        flow_predictions = []
        visconf_predictions = []

        fmap1 = fmap1_single.unsqueeze(1).repeat(1,S,1,1,1) # B,S,C,H,W
        fmap1 = fmap1.reshape(B*(S),C,H8,W8).contiguous()

        fmap2 = fmaps2.reshape(B*(S),C,H8,W8).contiguous()

        visconfs8 = visconfs8.reshape(B*(S),2,H8,W8).contiguous()

        corr_fn = CorrBlock(fmap1, fmap2, self.corr_levels, self.corr_radius)

        coords1 = self.coords_grid(B*(S), H8, W8, device=fmap1.device, dtype=dtype)

        if self.no_split:
            flowfeat, ctxfeat = fmap1.clone(), fmap1.clone()
        else:
            if flowfeat is not None:
                _, ctxfeat = torch.split(fmap1, [self.dim, self.dim], dim=1)
            else:
                flowfeat, ctxfeat = torch.split(fmap1, [self.dim, self.dim], dim=1)
                
        # add pos emb to ctxfeat (and not flowfeat), since ctxfeat is untouched across iters
        time_emb = self.fetch_time_embed(S, ctxfeat.dtype, is_training).reshape(1,S,self.dim,1,1).repeat(B,1,1,1,1)
        ctxfeat = ctxfeat + time_emb.reshape(B*S,self.dim,1,1)

        if self.no_ctx:
            flowfeat = flowfeat + time_emb.reshape(B*S,self.dim,1,1)

        def recurrent_update_step(flows8, visconfs8, flowfeat):
            _, _, H8, W8 = flows8.shape
            if bool(getattr(self, '_roma_init_window_preserve_grad', False)):
                coords2 = coords1 + flows8
            else:
                flows8 = flows8.detach()
                coords2 = (coords1 + flows8).detach() # B*S,2,H,W
            corr = corr_fn(coords2).to(dtype)

            if self.use_relmotion or self.use_sinrelmotion:
                coords_ = coords2.reshape(B,S,2,H8*W8).permute(0,1,3,2) # B,S,H8*W8,2
                rel_coords_forward = coords_[:, :-1] - coords_[:, 1:]
                rel_coords_backward = coords_[:, 1:] - coords_[:, :-1]
                rel_coords_forward = torch.nn.functional.pad(
                    rel_coords_forward, (0, 0, 0, 0, 0, 1) # pad the 3rd-last dim (S) by (0,1)
                )
                rel_coords_backward = torch.nn.functional.pad(
                    rel_coords_backward, (0, 0, 0, 0, 1, 0) # pad the 3rd-last dim (S) by (1,0)
                )
                rel_coords = torch.cat([rel_coords_forward, rel_coords_backward], dim=-1) # B,S,H8*W8,4

                if self.use_sinrelmotion:
                    rel_pos_emb_input = utils.misc.posenc(
                        rel_coords,
                        min_deg=0,
                        max_deg=10,
                    )  # B,S,H*W,pdim
                    motion = rel_pos_emb_input.reshape(B*S,H8,W8,self.pdim).permute(0,3,1,2).to(dtype) # B*S,pdim,H8,W8
                else:
                    motion = rel_coords.reshape(B*S,H8,W8,4).permute(0,3,1,2).to(dtype) # B*S,4,H8,W8
                
            else:
                if self.use_sinmotion:
                    pos_emb_input = utils.misc.posenc(
                        flows8.reshape(B,S,H8*W8,2),
                        min_deg=0,
                        max_deg=10,
                    )  # B,S,H*W,pdim
                    motion = pos_emb_input.reshape(B*S,H8,W8,self.pdim).permute(0,3,1,2).to(dtype) # B*S,pdim,H8,W8
                else:
                    motion = flows8
                    
            flowfeat = self.update_block(flowfeat, ctxfeat, visconfs8, corr, motion, S)
            flow_update = self.flow_head(flowfeat)
            visconf_update = self.visconf_head(flowfeat)
            weight_update = .25 * self.upsample_weight(flowfeat)
            flows8 = flows8 + flow_update
            visconfs8 = visconfs8 + visconf_update
            return flows8, visconfs8, flowfeat, weight_update, corr, flow_update

        last_update_norm = None
        iteration_delta_norm_sum = None
        corr = None
        keep_last_only = (
            bool(getattr(self, '_roma_init_window_preserve_grad', False))
            and bool(getattr(self, 'roma_init_return_last_only', False))
        )
        skip_visconf_upsample = (
            bool(getattr(self, '_roma_init_window_preserve_grad', False))
            and bool(getattr(self, 'roma_init_skip_visconf_upsample', False))
        )
        for itr in range(iters):
            flows8, visconfs8, flowfeat, weight_update, corr, flow_update = recurrent_update_step(flows8, visconfs8, flowfeat)
            last_update_norm = torch.sqrt(torch.sum(flow_update ** 2, dim=1))
            if iteration_delta_norm_sum is None:
                iteration_delta_norm_sum = last_update_norm
            else:
                iteration_delta_norm_sum = iteration_delta_norm_sum + last_update_norm
            if getattr(self, 'corr_reloc_enable', False):
                if (not getattr(self, 'corr_reloc_last_only', True)) or (itr == iters - 1):
                    flows8 = self.corr_guided_relocalize(flows8, corr, visconfs8)
            flow_up = self.upsample_data(flows8, weight_update)
            if keep_last_only:
                flow_predictions = [flow_up]
            else:
                flow_predictions.append(flow_up)
            if skip_visconf_upsample:
                visconf_up = torch.zeros_like(flow_up)
            else:
                visconf_up = self.upsample_data(visconfs8, weight_update)
            if keep_last_only:
                visconf_predictions = [visconf_up]
            else:
                visconf_predictions.append(visconf_up)

        if getattr(self, 'patch_memory_enable', False) and (not is_training):
            flows8 = self.patch_memory_relocalize(
                flows8=flows8,
                fmaps2=fmaps2,
                fmap_anchor=fmap1_single,
                corr=corr,
                visconfs8=visconfs8,
                update_norm=last_update_norm,
                coords1=coords1,
            )
            weight_update = .25 * self.upsample_weight(flowfeat)
            flow_up = self.upsample_data(flows8, weight_update)
            flow_predictions.append(flow_up)
            visconf_up = self.upsample_data(visconfs8, weight_update)
            visconf_predictions.append(visconf_up)

        if getattr(self, 'adaptive_iters_enable', False) and (not is_training):
            extra_iters = int(getattr(self, 'adaptive_extra_iters', 1))
            if extra_iters > 0:
                base_flows8 = flows8
                base_visconfs8 = visconfs8
                base_flowfeat = flowfeat
                extra_flows8 = flows8
                extra_visconfs8 = visconfs8
                extra_flowfeat = flowfeat
                for _ in range(extra_iters):
                    extra_flows8, extra_visconfs8, extra_flowfeat, _, _, _ = recurrent_update_step(
                        extra_flows8,
                        extra_visconfs8,
                        extra_flowfeat,
                    )

                base_conf = torch.sigmoid(base_visconfs8[:, 0]) * torch.sigmoid(base_visconfs8[:, 1])
                extra_conf = torch.sigmoid(extra_visconfs8[:, 0]) * torch.sigmoid(extra_visconfs8[:, 1])
                conf_gain = extra_conf - base_conf
                update_norm = torch.sqrt(torch.sum((extra_flows8 - base_flows8) ** 2, dim=1))

                if bool(getattr(self, 'adaptive_disable_low_conf_gate', False)):
                    attempt_mask = torch.ones_like(base_conf, dtype=torch.bool)
                else:
                    attempt_mask = base_conf < float(getattr(self, 'adaptive_base_conf_thr', 0.5))
                accept = (
                    attempt_mask
                    & (conf_gain >= float(getattr(self, 'adaptive_conf_gain_thr', 0.05)))
                    & (update_norm >= float(getattr(self, 'adaptive_min_update_norm', 0.0)))
                    & (update_norm <= float(getattr(self, 'adaptive_max_update_norm', 2.0)))
                )

                attempt_count = int(attempt_mask.sum().detach().cpu().item())
                accept_count = int(accept.sum().detach().cpu().item())
                self.adaptive_iters_attempt_count = int(getattr(self, 'adaptive_iters_attempt_count', 0)) + attempt_count
                self.adaptive_iters_accept_count = int(getattr(self, 'adaptive_iters_accept_count', 0)) + accept_count
                if attempt_count > 0:
                    self.adaptive_iters_base_conf_sum = float(getattr(self, 'adaptive_iters_base_conf_sum', 0.0)) + float(base_conf[attempt_mask].sum().detach().cpu().item())
                if accept_count > 0:
                    self.adaptive_iters_conf_gain_sum = float(getattr(self, 'adaptive_iters_conf_gain_sum', 0.0)) + float(conf_gain[accept].sum().detach().cpu().item())
                    self.adaptive_iters_update_norm_sum = float(getattr(self, 'adaptive_iters_update_norm_sum', 0.0)) + float(update_norm[accept].sum().detach().cpu().item())

                if accept_count > 0:
                    accept_mask = accept[:, None]
                    flows8 = torch.where(accept_mask, extra_flows8, base_flows8)
                    visconfs8 = torch.where(accept_mask, extra_visconfs8, base_visconfs8)
                    flowfeat = torch.where(accept_mask, extra_flowfeat, base_flowfeat)
                else:
                    flows8 = base_flows8
                    visconfs8 = base_visconfs8
                    flowfeat = base_flowfeat

                weight_update = .25 * self.upsample_weight(flowfeat)
                flow_up = self.upsample_data(flows8, weight_update)
                flow_predictions.append(flow_up)
                visconf_up = self.upsample_data(visconfs8, weight_update)
                visconf_predictions.append(visconf_up)
            
        if bool(getattr(self, 'export_risk_features', False)):
            if iteration_delta_norm_sum is not None and iters is not None and int(iters) > 0:
                iteration_delta_norm_mean = iteration_delta_norm_sum / float(int(iters))
            else:
                iteration_delta_norm_mean = None
            self.export_baseline_risk_features(
                flows8=flows8,
                visconfs8=visconfs8,
                corr=corr,
                coords1=coords1,
                last_update_norm=last_update_norm,
                iteration_delta_norm_mean=iteration_delta_norm_mean,
                iteration_delta_norm_last=last_update_norm,
            )

        return flow_predictions, visconf_predictions, flows8, visconfs8, flowfeat

    
