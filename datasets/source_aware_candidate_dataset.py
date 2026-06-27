import os

import numpy as np
import torch
from torch.utils.data import Dataset


SOURCE_NAME_TO_ID = {
    'baseline': 0,
    'feat8_query': 1,
    'query': 1,
    'feat8_memory': 2,
    'memory': 2,
    'roma_query': 3,
    'roma': 3,
    'roma_memory': 4,
    'other': 5,
}


def load_candidate_npz(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=True)
    data = {key: npz[key] for key in npz.files}
    npz.close()
    return data


def label_margin_from_data(data, fallback=1.0):
    value = data.get('label_margin', None)
    if value is None:
        return float(fallback)
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return float(fallback)
    return float(arr[0])


def parse_allowed_source_types(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() in ('all', '*'):
        return None
    allowed = set()
    for token in text.split(','):
        token = token.strip()
        if token == '':
            continue
        key = token.lower()
        if key in SOURCE_NAME_TO_ID:
            allowed.add(int(SOURCE_NAME_TO_ID[key]))
        else:
            allowed.add(int(token))
    # Baseline is always kept as the safe fallback class.
    allowed.add(0)
    return sorted(allowed)


def apply_allowed_source_filter(data, allowed_source_types=None, label_margin=None):
    allowed = parse_allowed_source_types(allowed_source_types)
    if allowed is None and (label_margin is None or float(label_margin) < 0):
        return data

    out = dict(data)
    candidate_valid = np.asarray(out['candidate_valid']).copy().astype(np.bool_)
    candidate_source_type = np.asarray(out['candidate_source_type']).astype(np.int64)
    if allowed is not None:
        source_allowed = np.isin(candidate_source_type, np.asarray(allowed, dtype=np.int64))
        candidate_valid &= source_allowed
    # Candidate 0 is the baseline fallback. Keep it valid unless the export
    # itself marked it invalid.
    candidate_valid[:, 0] = np.asarray(data['candidate_valid'])[:, 0].astype(np.bool_)
    out['candidate_valid'] = candidate_valid

    candidate_err = np.asarray(out['candidate_err']).astype(np.float32)
    baseline_err = np.asarray(out['baseline_err']).astype(np.float32)
    safe_err = np.where(candidate_valid, candidate_err, np.inf)
    oracle_best_idx = np.argmin(safe_err, axis=1).astype(np.int64)
    oracle_best_err = safe_err[np.arange(safe_err.shape[0]), oracle_best_idx].astype(np.float32)
    no_valid = ~np.isfinite(oracle_best_err)
    oracle_best_idx[no_valid] = 0
    oracle_best_err[no_valid] = baseline_err[no_valid]
    out['oracle_best_idx'] = oracle_best_idx
    out['oracle_best_err'] = oracle_best_err
    out['oracle_delta'] = oracle_best_err - baseline_err

    if label_margin is None or float(label_margin) < 0:
        label_margin = label_margin_from_data(out, fallback=1.0)
    label = np.zeros((safe_err.shape[0],), dtype=np.int64)
    good = (
        np.isfinite(oracle_best_err)
        & np.isfinite(baseline_err)
        & ((oracle_best_err + float(label_margin)) < baseline_err)
    )
    label[good] = oracle_best_idx[good]
    label[~candidate_valid[np.arange(label.shape[0]), label]] = 0
    out['label'] = label
    out['label_margin'] = np.asarray([float(label_margin)], dtype=np.float32)

    baseline_err_matrix = np.broadcast_to(baseline_err[:, None], candidate_err.shape)
    finite_baseline = np.isfinite(baseline_err_matrix)
    finite_candidate = candidate_valid & np.isfinite(candidate_err) & finite_baseline
    nonbaseline = np.ones_like(candidate_valid, dtype=np.bool_)
    nonbaseline[:, 0] = False
    candidate_delta = np.full_like(candidate_err, np.nan, dtype=np.float32)
    candidate_gain = np.full_like(candidate_err, np.nan, dtype=np.float32)
    candidate_delta[finite_candidate] = candidate_err[finite_candidate] - baseline_err_matrix[finite_candidate]
    candidate_gain[finite_candidate] = baseline_err_matrix[finite_candidate] - candidate_err[finite_candidate]
    candidate_safe_positive = (
        nonbaseline
        & finite_candidate
        & ((candidate_err + float(label_margin)) < baseline_err_matrix)
    )
    candidate_false_accept = nonbaseline & finite_candidate & (candidate_err >= baseline_err_matrix)
    candidate_unsafe_negative = nonbaseline & candidate_valid & ~candidate_safe_positive
    out['candidate_delta'] = candidate_delta.astype(np.float32)
    out['candidate_gain'] = candidate_gain.astype(np.float32)
    out['candidate_safe_positive'] = candidate_safe_positive.astype(np.bool_)
    out['candidate_false_accept'] = candidate_false_accept.astype(np.bool_)
    out['candidate_unsafe_negative'] = candidate_unsafe_negative.astype(np.bool_)

    candidate_anchor_type = np.asarray(out.get('candidate_anchor_type', np.full_like(candidate_source_type, -1))).astype(np.int64)
    row_idx = np.arange(label.shape[0])
    out['label_worth_reloc'] = (label != 0).astype(np.bool_)
    out['label_best_anchor_idx'] = label.astype(np.int64)
    label_anchor_type = np.full((label.shape[0],), -1, dtype=np.int64)
    positive_label = label != 0
    label_anchor_type[positive_label] = candidate_anchor_type[row_idx[positive_label], label[positive_label]]
    out['label_best_anchor_type'] = label_anchor_type
    label_gain = np.zeros((label.shape[0],), dtype=np.float32)
    label_gain[positive_label] = candidate_gain[row_idx[positive_label], label[positive_label]]
    label_gain = np.nan_to_num(label_gain, nan=0.0, posinf=0.0, neginf=0.0)
    out['label_accept_gain'] = label_gain

    oracle_anchor_type = np.full((oracle_best_idx.shape[0],), -1, dtype=np.int64)
    positive_oracle = oracle_best_idx != 0
    oracle_anchor_type[positive_oracle] = candidate_anchor_type[row_idx[positive_oracle], oracle_best_idx[positive_oracle]]
    out['oracle_best_anchor_type'] = oracle_anchor_type
    out['oracle_accept_gain'] = (baseline_err - oracle_best_err).astype(np.float32)
    out['oracle_improves_baseline'] = (oracle_best_err < baseline_err).astype(np.bool_)
    out['allowed_source_types'] = np.asarray([] if allowed is None else allowed, dtype=np.int64)
    return out


class SourceAwareCandidateDataset(Dataset):
    def __init__(self, candidate_npz):
        self.data = load_candidate_npz(candidate_npz)
        self.length = int(self.data['candidate_xy'].shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item = {}
        for key, value in self.data.items():
            if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == self.length:
                item[key] = value[index]
        return item


def candidate_collate(batch):
    out = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if isinstance(first, np.ndarray) and first.dtype.kind in ('U', 'S', 'O'):
            out[key] = np.asarray(values)
        elif isinstance(first, (str, bytes)):
            out[key] = np.asarray(values)
        else:
            out[key] = torch.as_tensor(np.stack(values, axis=0))
    return out


def tensor_batch_from_npz(data, indices, device=None):
    batch = {}
    for key, value in data.items():
        if hasattr(value, 'shape') and len(value.shape) > 0 and value.shape[0] == data['candidate_xy'].shape[0]:
            sliced = value[indices]
            if isinstance(sliced, np.ndarray) and sliced.dtype.kind in ('U', 'S', 'O'):
                batch[key] = sliced
            else:
                tensor = torch.as_tensor(sliced)
                if device is not None:
                    tensor = tensor.to(device)
                batch[key] = tensor
    return batch
