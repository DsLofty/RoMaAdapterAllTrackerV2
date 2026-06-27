import os
import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


class RoMaImportError(RuntimeError):
    pass


def _repo_root():
    return Path(__file__).resolve().parents[1]


def _roma_root():
    return _repo_root() / 'third_party' / 'RoMa'


def ensure_roma_importable():
    roma_root = _roma_root()
    if roma_root.exists():
        roma_path = str(roma_root)
        if roma_path not in sys.path:
            sys.path.insert(0, roma_path)
    try:
        import romatch  # noqa: F401
    except ImportError as exc:
        msg = (
            'RoMa import failed. Please install RoMa in the active AllTracker environment:\n'
            '  cd third_party/RoMa && pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple\n'
            'Missing dependency may include loguru/kornia/timm/poselib/albumentations.\n'
            'For the most common missing packages, run:\n'
            '  pip install loguru kornia timm albumentations\n'
            'Original error: %s'
        ) % str(exc)
        raise RoMaImportError(msg) from exc


def _hub_checkpoint_path(filename):
    hub_dir = Path(torch.hub.get_dir())
    return hub_dir / 'checkpoints' / filename


def check_cached_weights(model_type):
    """Avoid surprise downloads unless --roma_allow_online_download is set."""
    missing = []
    if model_type in ('outdoor', 'indoor'):
        roma_name = 'roma_outdoor.pth' if model_type == 'outdoor' else 'roma_indoor.pth'
        for name in (roma_name, 'dinov2_vitl14_pretrain.pth'):
            if not _hub_checkpoint_path(name).exists():
                missing.append(str(_hub_checkpoint_path(name)))
    elif model_type == 'tiny_outdoor':
        if not _hub_checkpoint_path('tiny_roma_v1_outdoor.pth').exists():
            missing.append(str(_hub_checkpoint_path('tiny_roma_v1_outdoor.pth')))
        # XFeat is loaded through torch.hub.load from a GitHub repo; there is no
        # simple checkpoint-only path check, so require online permission.
        missing.append('XFeat torch.hub repo/cache may be required for tiny_outdoor')
    return missing


def _as_pil_image(image):
    from PIL import Image

    if torch.is_tensor(image):
        x = image.detach().cpu()
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError('RoMa wrapper expects a single image, got batch size %d' % int(x.shape[0]))
            x = x[0]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        if x.ndim != 3 or x.shape[-1] not in (1, 3):
            raise ValueError('Unsupported image tensor shape for RoMa: %s' % (tuple(image.shape),))
        if x.dtype.is_floating_point:
            if float(x.max().item()) <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).byte()
        else:
            x = x.clamp(0, 255).byte()
        arr = x.numpy()
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Image.fromarray(arr).convert('RGB')

    try:
        import numpy as np
        from PIL import Image
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if float(arr.max()) <= 1.5:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert('RGB')
    except ImportError:
        pass

    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert('RGB')
    raise ValueError('Unsupported image input type for RoMa: %s' % type(image))


class RoMaMatcher:
    """Small defensive wrapper around third_party/RoMa.

    RoMa returns a dense warp in normalized coordinates. For the default
    symmetric RoMa models, the first half of the warp width stores source A to
    target B: warp[..., :W, 2:]. The second half stores B to A and is ignored
    by this wrapper.
    """

    def __init__(
        self,
        model_type='outdoor',
        device='cuda',
        input_size=None,
        allow_online_download=False,
        cache_warps=False,
        use_custom_corr=True,
        allow_slow_corr=False,
    ):
        ensure_roma_importable()
        self.model_type = str(model_type)
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == 'cpu' else 'cpu')
        self.input_size = None if input_size is None else tuple(int(v) for v in input_size)
        self.allow_online_download = bool(allow_online_download)
        self.cache_warps = bool(cache_warps)
        self.use_custom_corr = bool(use_custom_corr)
        self.allow_slow_corr = bool(allow_slow_corr)
        if self.use_custom_corr and importlib.util.find_spec('local_corr') is None:
            if not self.allow_slow_corr:
                raise RoMaImportError(
                    'RoMa fused local correlation is unavailable. The native PyTorch fallback is too slow '
                    'for recurrent adapter training. Install it in the active environment with:\n'
                    '  pip install "fused-local-corr>=0.2.2"\n'
                    'Then verify with:\n'
                    '  python -c "import local_corr; print(local_corr.__file__)"\n'
                    'Use --roma_allow_slow_corr only for debugging.'
                )
            self.use_custom_corr = False
        self.model = None
        self.model_output_size = None
        self.cache = {}

    def _resolve_output_size(self, image_size):
        if self.input_size is None:
            return tuple(int(v) for v in image_size)
        if len(self.input_size) == 1:
            return (int(self.input_size[0]), int(self.input_size[0]))
        if len(self.input_size) >= 2:
            return (int(self.input_size[0]), int(self.input_size[1]))
        return tuple(int(v) for v in image_size)

    def _build_model(self, image_size):
        output_size = self._resolve_output_size(image_size)
        if self.model is not None and self.model_output_size == output_size:
            return
        if self.model is not None and self.model_output_size != output_size:
            raise RuntimeError(
                'RoMa model was initialized for output size %s but got %s. '
                'Run one fixed --image_size/--roma_input_size per process.'
                % (str(self.model_output_size), str(output_size))
            )

        if not self.allow_online_download:
            missing = check_cached_weights(self.model_type)
            if missing:
                raise RuntimeError(
                    'RoMa weights are not available in the torch hub cache and '
                    '--roma_allow_online_download was not set. Missing:\n  %s\n'
                    'Either rerun with --roma_allow_online_download or pre-download/copy these weights.'
                    % '\n  '.join(missing)
                )

        torch.set_float32_matmul_precision('highest')
        from romatch import roma_indoor, roma_outdoor, tiny_roma_v1_outdoor

        if self.model_type == 'outdoor':
            self.model = roma_outdoor(
                device=self.device,
                upsample_res=output_size,
                use_custom_corr=self.use_custom_corr,
            )
        elif self.model_type == 'indoor':
            self.model = roma_indoor(
                device=self.device,
                upsample_res=output_size,
                use_custom_corr=self.use_custom_corr,
            )
        elif self.model_type == 'tiny_outdoor':
            self.model = tiny_roma_v1_outdoor(device=self.device)
        else:
            raise ValueError('unsupported RoMa model_type: %s' % self.model_type)

        self.model.eval()
        self.model_output_size = output_size

    @torch.no_grad()
    def match_dense(self, image0, image1, cache_key=None):
        pil0 = _as_pil_image(image0)
        pil1 = _as_pil_image(image1)
        image_size = (int(pil0.height), int(pil0.width))
        if int(pil1.height) != image_size[0] or int(pil1.width) != image_size[1]:
            raise ValueError('RoMa image pair must have the same size, got %s and %s' % (pil0.size, pil1.size))
        self._build_model(image_size)

        if self.cache_warps and cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        warp, certainty = self.model.match(pil0, pil1, device=self.device)
        if warp.ndim == 4:
            warp = warp[0]
        if certainty.ndim == 3:
            certainty = certainty[0]
        warp = warp.detach()
        certainty = certainty.detach()

        H_out = int(warp.shape[0])
        W_all = int(warp.shape[1])
        W_out = int(self.model_output_size[1]) if self.model_output_size is not None else W_all
        if W_all >= 2 * W_out:
            warp_ab = warp[:, :W_out, 2:4]
            certainty_ab = certainty[:, :W_out]
        elif W_all == W_out:
            warp_ab = warp[:, :, 2:4] if warp.shape[-1] >= 4 else warp[:, :, :2]
            certainty_ab = certainty
        else:
            half = W_all // 2 if W_all % 2 == 0 else W_all
            warp_ab = warp[:, :half, 2:4]
            certainty_ab = certainty[:, :half]
            W_out = half

        result = {
            'warp_ab': warp_ab,
            'certainty_ab': certainty_ab,
            'output_size': (H_out, W_out),
            'image_size': image_size,
        }
        if self.cache_warps and cache_key is not None:
            self.cache[cache_key] = result
        return result

    def _extract_ab_from_warp(self, warp, certainty):
        H_out = int(warp.shape[0])
        W_all = int(warp.shape[1])
        W_out = int(self.model_output_size[1]) if self.model_output_size is not None else W_all
        if W_all >= 2 * W_out:
            warp_ab = warp[:, :W_out, 2:4]
            certainty_ab = certainty[:, :W_out]
        elif W_all == W_out:
            warp_ab = warp[:, :, 2:4] if warp.shape[-1] >= 4 else warp[:, :, :2]
            certainty_ab = certainty
        else:
            half = W_all // 2 if W_all % 2 == 0 else W_all
            warp_ab = warp[:, :half, 2:4]
            certainty_ab = certainty[:, :half]
            W_out = half
        return warp_ab, certainty_ab, (H_out, W_out)

    def _tensor_pair_batch(self, pil0_list, pil1_list, resize, clahe=False):
        from romatch.utils import get_tuple_transform_ops

        transform = get_tuple_transform_ops(resize=resize, normalize=True, clahe=clahe)
        im0_tensors = []
        im1_tensors = []
        for pil0, pil1 in zip(pil0_list, pil1_list):
            im0, im1 = transform((pil0, pil1))
            im0_tensors.append(im0)
            im1_tensors.append(im1)
        return torch.stack(im0_tensors, dim=0), torch.stack(im1_tensors, dim=0)

    @torch.no_grad()
    def match_dense_batch(self, image0, image1_list, cache_keys=None):
        """Run RoMa for several source/target image pairs in one micro-batch.

        This keeps the public single-pair API intact while allowing callers to
        amortize the expensive RoMa forward over multiple target frames. Results
        are returned as a list with the same dense-dict schema as match_dense().
        """

        if image1_list is None:
            image1_list = []
        image1_list = list(image1_list)
        if len(image1_list) == 0:
            return []
        if cache_keys is None:
            cache_keys = [None] * len(image1_list)
        cache_keys = list(cache_keys)
        if len(cache_keys) != len(image1_list):
            raise ValueError('cache_keys length must match image1_list length')

        pil0 = _as_pil_image(image0)
        pil1_all = [_as_pil_image(image1) for image1 in image1_list]
        image_size = (int(pil0.height), int(pil0.width))
        for pil1 in pil1_all:
            if int(pil1.height) != image_size[0] or int(pil1.width) != image_size[1]:
                raise ValueError('RoMa image pair must have the same size, got %s and %s' % (pil0.size, pil1.size))
        self._build_model(image_size)

        results = [None] * len(pil1_all)
        uncached_indices = []
        for idx, cache_key in enumerate(cache_keys):
            if self.cache_warps and cache_key is not None and cache_key in self.cache:
                results[idx] = self.cache[cache_key]
            else:
                uncached_indices.append(idx)
        if not uncached_indices:
            return results

        # RoMa's tensor batch path expects already-normalized tensors. For
        # upsampled models we also pass high-res tensors explicitly; otherwise
        # RoMa's PIL-only upsample branch would reject tensor inputs.
        coarse_resize = (int(getattr(self.model, 'h_resized', 560)), int(getattr(self.model, 'w_resized', 560)))
        output_size = tuple(int(v) for v in self.model_output_size)
        source_pils = [pil0 for _ in uncached_indices]
        target_pils = [pil1_all[idx] for idx in uncached_indices]
        im0_low, im1_low = self._tensor_pair_batch(source_pils, target_pils, resize=coarse_resize, clahe=False)
        im0_low = im0_low.to(self.device)
        im1_low = im1_low.to(self.device)

        match_kwargs = {'device': self.device}
        if bool(getattr(self.model, 'upsample_preds', False)):
            im0_high, im1_high = self._tensor_pair_batch(source_pils, target_pils, resize=output_size, clahe=False)
            match_kwargs['im_A_high_res'] = im0_high.to(self.device)
            match_kwargs['im_B_high_res'] = im1_high.to(self.device)

        warp, certainty = self.model.match(im0_low, im1_low, **match_kwargs)
        if warp.ndim == 3:
            warp = warp.unsqueeze(0)
        if certainty.ndim == 2:
            certainty = certainty.unsqueeze(0)
        warp = warp.detach()
        certainty = certainty.detach()

        for batch_i, result_i in enumerate(uncached_indices):
            warp_ab, certainty_ab, dense_output_size = self._extract_ab_from_warp(warp[batch_i], certainty[batch_i])
            result = {
                'warp_ab': warp_ab,
                'certainty_ab': certainty_ab,
                'output_size': dense_output_size,
                'image_size': image_size,
            }
            cache_key = cache_keys[result_i]
            if self.cache_warps and cache_key is not None:
                self.cache[cache_key] = result
            results[result_i] = result
        return results

    def _map_points_from_dense(self, dense, points0, sample_mode='bilinear'):
        warp_ab = dense['warp_ab'].to(self.device)
        certainty_ab = dense['certainty_ab'].to(self.device)
        H_out, W_out = dense['output_size']
        H_img, W_img = dense['image_size']

        pts = torch.as_tensor(points0, device=self.device, dtype=torch.float32)
        if pts.numel() == 0:
            return {
                'points1': torch.empty(0, 2, device=self.device),
                'certainty': torch.empty(0, device=self.device),
                'valid': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        pts = pts.reshape(-1, 2)
        valid0 = (pts[:, 0] >= 0) & (pts[:, 0] < W_img) & (pts[:, 1] >= 0) & (pts[:, 1] < H_img)

        # RoMa uses normalized pixel coordinates x_norm = 2*x/W - 1,
        # y_norm = 2*y/H - 1. This mirrors RegressionMatcher.to_normalized_coordinates.
        pts_out = pts.clone()
        pts_out[:, 0] = pts_out[:, 0] * (float(W_out) / float(W_img))
        pts_out[:, 1] = pts_out[:, 1] * (float(H_out) / float(H_img))
        grid = torch.stack(
            [
                2.0 * pts_out[:, 0] / max(float(W_out), 1.0) - 1.0,
                2.0 * pts_out[:, 1] / max(float(H_out), 1.0) - 1.0,
            ],
            dim=-1,
        ).reshape(1, -1, 1, 2)

        mode = 'nearest' if str(sample_mode) == 'nearest' else 'bilinear'
        sampled_norm = F.grid_sample(
            warp_ab.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode=mode,
            align_corners=False,
            padding_mode='zeros',
        )[0, :, :, 0].T
        sampled_cert = F.grid_sample(
            certainty_ab.unsqueeze(0).unsqueeze(0),
            grid,
            mode=mode,
            align_corners=False,
            padding_mode='zeros',
        )[0, 0, :, 0]

        points1_out = torch.stack(
            [
                float(W_out) * (sampled_norm[:, 0] + 1.0) / 2.0,
                float(H_out) * (sampled_norm[:, 1] + 1.0) / 2.0,
            ],
            dim=-1,
        )
        points1 = points1_out.clone()
        points1[:, 0] = points1[:, 0] * (float(W_img) / float(W_out))
        points1[:, 1] = points1[:, 1] * (float(H_img) / float(H_out))
        valid1 = (
            (sampled_norm[:, 0] >= -1.0)
            & (sampled_norm[:, 0] <= 1.0)
            & (sampled_norm[:, 1] >= -1.0)
            & (sampled_norm[:, 1] <= 1.0)
            & torch.isfinite(points1).all(dim=1)
            & torch.isfinite(sampled_cert)
        )
        return {
            'points1': points1,
            'certainty': sampled_cert.clamp(0.0, 1.0),
            'valid': valid0 & valid1,
        }

    @torch.no_grad()
    def map_points(self, image0, image1, points0, sample_mode='bilinear', cache_key=None):
        dense = self.match_dense(image0, image1, cache_key=cache_key)
        return self._map_points_from_dense(dense, points0, sample_mode=sample_mode)

    @torch.no_grad()
    def map_points_batch(self, image0, image1_list, points0_list, sample_mode='bilinear', cache_keys=None):
        dense_list = self.match_dense_batch(image0, image1_list, cache_keys=cache_keys)
        if len(dense_list) != len(points0_list):
            raise ValueError('points0_list length must match image1_list length')
        return [
            self._map_points_from_dense(dense, points0, sample_mode=sample_mode)
            for dense, points0 in zip(dense_list, points0_list)
        ]
