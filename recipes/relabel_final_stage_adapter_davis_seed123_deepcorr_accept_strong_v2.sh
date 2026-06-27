#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"
SRC="${BASE}/dav_eval_alltracker_stride1_margin10_seed123_deepcorr_risk_accept_v1"
DST="${BASE}/dav_eval_alltracker_stride1_strongv2_seed123_deepcorr_risk_accept_v1"

if [[ ! -d "${SRC}/sequences" ]]; then
  echo "missing DAVIS risk_accept cache: ${SRC}" >&2
  exit 1
fi

python relabel_final_stage_accept_cache.py \
  --src "${SRC}" \
  --dst "${DST}" \
  --label_version strong_v2 \
  --require_risk_positive \
  --pos_base_min_px 8.0 \
  --pos_roma_max_px 4.0 \
  --pos_gain_min_px 8.0 \
  --neg_base_max_px 4.0 \
  --neg_roma_min_px 12.0 \
  --neg_cost_min_px 8.0
