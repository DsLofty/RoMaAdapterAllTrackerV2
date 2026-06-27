#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"

SOURCES=(
  "ce24_drivingpt"
  "ce24_fltpt"
  "ce24_monkapt"
  "ce24_springpt"
  "ce64_drivingpt"
  "ce64_kublong"
  "ce64_monkapt"
  "ce64_podlong"
  "ce64_springpt"
)

for name in "${SOURCES[@]}"; do
  src="${BASE}/kubric_train_alltracker_${name}_stride1_margin10_300seq_seed123_deepcorr_risk_accept_v1"
  dst="${BASE}/kubric_train_alltracker_${name}_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  if [[ ! -d "${src}/sequences" ]]; then
    echo "skip missing cache: ${src}"
    continue
  fi
  python relabel_final_stage_accept_cache.py \
    --src "${src}" \
    --dst "${dst}" \
    --label_version strong_v2 \
    --require_risk_positive \
    --pos_base_min_px 8.0 \
    --pos_roma_max_px 4.0 \
    --pos_gain_min_px 8.0 \
    --neg_base_max_px 4.0 \
    --neg_roma_min_px 12.0 \
    --neg_cost_min_px 8.0
done
