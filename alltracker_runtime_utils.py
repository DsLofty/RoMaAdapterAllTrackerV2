import os

import numpy as np
import torch
import torch.nn.functional as F

import utils.basic


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def pad_to_multiple_64(x):
    ht, wd = x.shape[-2:]
    pad_ht = (((ht // 64) + 1) * 64 - ht) % 64
    pad_wd = (((wd // 64) + 1) * 64 - wd) % 64
    pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
    return F.pad(x, pad, mode='replicate')


def load_alltracker(args, device):
    from nets.alltracker import Net

    model = Net(16)
    state_dict = torch.load(expand_path(args.ckpt_path), map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def extract_feat8_sequence(model, rgbs, device):
    b, t, c, h, w = rgbs.shape
    assert b == 1 and c == 3
    images = rgbs.to(device).float()
    mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 1, 3, 1, 1).to(images.dtype)
    std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 1, 3, 1, 1).to(images.dtype)
    images = images / 255.0
    images = (images - mean) / std
    images_ = images.reshape(b * t, 3, h, w).contiguous()
    images_ = pad_to_multiple_64(images_)
    with torch.no_grad():
        fmaps = model.get_fmaps(images_, b, t, sw=None, is_training=False)
    _, c8, h8, w8 = fmaps.shape
    return fmaps.reshape(b, t, c8, h8, w8)[0].detach()


def _clear_sparse_init(model):
    model.roma_init_enable = False
    model.roma_init_flow8_override = None
    model.roma_init_mask8_override = None
    model.roma_init_preserve_grad = False
    model.roma_init_return_last_only = False
    model.roma_init_skip_visconf_upsample = False
    model.reloc_head_enable = False
    model.reloc_decision_mask8_override = None
    model._roma_init_last_applied = False
    model._roma_init_window_preserve_grad = False
    model._reloc_decision_last_applied = False


def _set_sparse_init(model, flow8_override, mask8_override, apply_at):
    model.roma_init_enable = True
    model.roma_init_apply_at = str(apply_at)
    model.roma_init_flow8_override = flow8_override
    model.roma_init_mask8_override = mask8_override
    model.roma_init_preserve_grad = False
    model.roma_init_return_last_only = False
    model.roma_init_skip_visconf_upsample = False
    model._roma_init_last_applied = False
    model._roma_init_window_preserve_grad = False


def _flow8_grid_shape(height, width):
    h_pad = int(np.ceil(float(height) / 64.0) * 64)
    w_pad = int(np.ceil(float(width) / 64.0) * 64)
    return h_pad // 8, w_pad // 8


def _make_group_sparse_init(override_rows, trajs_g, first_frame, point_indices, total_frames, height, width, device, dtype):
    if not override_rows:
        return None
    point_set = {int(p) for p in point_indices.detach().long().cpu().tolist()}
    kept = []
    for row in override_rows:
        if not bool(row.get('valid', True)):
            continue
        point_id = int(row.get('point_index', -1))
        target_t = int(row.get('target_frame', -1))
        if point_id not in point_set or target_t <= int(first_frame) or target_t >= int(total_frames):
            continue
        xy8 = torch.as_tensor(row.get('xy8', row.get('xy', [float('nan'), float('nan')])), dtype=dtype, device=device)
        xy8 = xy8.reshape(2)
        if 'xy8' not in row:
            xy8 = xy8 / 8.0
        if not bool(torch.isfinite(xy8).all().detach().cpu().item()):
            continue
        kept.append((point_id, target_t - int(first_frame), xy8))
    if not kept:
        return None

    local_frames = int(total_frames) - int(first_frame)
    h8, w8 = _flow8_grid_shape(height, width)
    flat_size = int(local_frames * h8 * w8)
    flat_flow = torch.zeros((flat_size, 2), dtype=dtype, device=device)
    flat_count = torch.zeros((flat_size, 1), dtype=dtype, device=device)
    flat_indices = []
    init_flows = []
    for point_id, local_t, xy8 in kept:
        source_xy = trajs_g[0, int(first_frame), point_id, :2].to(device=device, dtype=dtype)
        source_x8 = torch.clamp(torch.round(source_xy[0] / 8.0).long(), 0, w8 - 1)
        source_y8 = torch.clamp(torch.round(source_xy[1] / 8.0).long(), 0, h8 - 1)
        source_grid_xy8 = torch.stack([source_x8.float(), source_y8.float()]).to(device=device, dtype=dtype)
        flat_indices.append(int(local_t) * h8 * w8 + int(source_y8.item()) * w8 + int(source_x8.item()))
        init_flows.append(xy8 - source_grid_xy8)
    if not flat_indices:
        return None

    flat_idx = torch.as_tensor(flat_indices, dtype=torch.long, device=device)
    init_flow = torch.stack(init_flows, dim=0)
    flat_flow.index_add_(0, flat_idx, init_flow)
    flat_count.index_add_(0, flat_idx, torch.ones((int(flat_idx.numel()), 1), dtype=dtype, device=device))
    flat_flow = flat_flow / torch.clamp(flat_count, min=1.0)
    mask_flat = flat_count[:, 0] > 0
    flow8 = flat_flow.reshape(local_frames, h8, w8, 2).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    mask8 = mask_flat.reshape(local_frames, h8, w8).unsqueeze(0).contiguous()
    return flow8, mask8


def dense_baseline_forward(
    batch,
    model,
    args,
    risk_model=None,
    risk_feature_names=None,
    device=None,
    override_rows=None,
    override_apply_at=None,
):
    del risk_model, risk_feature_names
    rgbs = batch.video.to(device, non_blocking=True).float()
    trajs_g = batch.trajs.to(device, non_blocking=True).float()
    vis_g = batch.visibs.to(device, non_blocking=True).float()
    valids = batch.valids.to(device, non_blocking=True).float() if batch.valids is not None else torch.ones_like(vis_g)
    b, t, c, h, w = rgbs.shape
    _, _, n, _ = trajs_g.shape
    assert b == 1 and c == 3

    _, first_positive_inds = torch.max(vis_g, dim=1)
    grid_xy = utils.basic.gridcloud2d(1, h, w, norm=False, device=device).float()
    grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, h, w)
    trajs_e = torch.zeros([b, t, n, 2], device=device)
    visconfs_e = torch.zeros([b, t, n, 2], device=device)
    risk_prob = torch.full([t, n], float('nan'), device='cpu')

    old_export = bool(getattr(model, 'export_risk_features', False))
    model.export_risk_features = False
    with torch.no_grad():
        for first_positive_ind_t in torch.unique(first_positive_inds):
            first_positive_ind = int(first_positive_ind_t.item())
            chunk_pt_idxs = torch.nonzero(first_positive_inds[0] == first_positive_ind, as_tuple=False)[:, 0]
            if chunk_pt_idxs.numel() == 0:
                continue

            traj_maps_e = grid_xy.repeat(1, t, 1, 1, 1)
            visconf_maps_e = torch.zeros_like(traj_maps_e)
            if first_positive_ind < t - 1:
                sparse_init = _make_group_sparse_init(
                    override_rows,
                    trajs_g,
                    first_positive_ind,
                    chunk_pt_idxs,
                    t,
                    h,
                    w,
                    device,
                    rgbs.dtype,
                )
                if sparse_init is not None:
                    _set_sparse_init(
                        model,
                        sparse_init[0],
                        sparse_init[1],
                        str(override_apply_at or getattr(args, 'roma_init_apply_at', 'window_start')),
                    )
                else:
                    _clear_sparse_init(model)

                if t > 128:
                    forward_flow_e, forward_visconf_e, forward_flow_preds, forward_visconf_preds = model.forward_sliding(
                        rgbs[:, first_positive_ind:],
                        iters=int(args.inference_iters),
                        sw=None,
                        is_training=False,
                    )
                else:
                    forward_flow_e, forward_visconf_e, forward_flow_preds, forward_visconf_preds = model(
                        rgbs[:, first_positive_ind:],
                        iters=int(args.inference_iters),
                        sw=None,
                        is_training=False,
                    )
                _clear_sparse_init(model)
                del forward_flow_preds, forward_visconf_preds
                forward_traj_maps_e = forward_flow_e.to(device) + grid_xy
                traj_maps_e[:, first_positive_ind:] = forward_traj_maps_e
                visconf_maps_e[:, first_positive_ind:] = forward_visconf_e.to(device)

            xyt = trajs_g[:, first_positive_ind].round().long()[0, chunk_pt_idxs]
            xyt[:, 0] = torch.clamp(xyt[:, 0], 0, w - 1)
            xyt[:, 1] = torch.clamp(xyt[:, 1], 0, h - 1)
            trajs_e_chunk = traj_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]].permute(0, 1, 3, 2)
            trajs_e.scatter_add_(
                2,
                chunk_pt_idxs[None, None, :, None].repeat(1, trajs_e_chunk.shape[1], 1, 2),
                trajs_e_chunk,
            )
            visconfs_e_chunk = visconf_maps_e[:, :, :, xyt[:, 1], xyt[:, 0]].permute(0, 1, 3, 2)
            visconfs_e.scatter_add_(
                2,
                chunk_pt_idxs[None, None, :, None].repeat(1, visconfs_e_chunk.shape[1], 1, 2),
                visconfs_e_chunk,
            )

    model.export_risk_features = old_export
    _clear_sparse_init(model)
    visconfs_e[..., 0] *= visconfs_e[..., 1]
    return {
        'trajs_e': trajs_e[0].detach().cpu(),
        'pred_visible_score': torch.clamp(visconfs_e[0, :, :, 0].detach().cpu(), 0.0, 1.0),
        'risk_prob': risk_prob,
        'first_positive_inds': first_positive_inds[0].detach().cpu().long(),
        'trajs_g': trajs_g[0].detach().cpu(),
        'vis_g': vis_g[0].detach().cpu(),
        'valids': valids[0].detach().cpu(),
    }
