#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
BASE="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"
SAVE_DIR="${FINAL_STAGE_CKPT_DIR:-${PROJECT_ROOT}/checkpoints_final_stage_adapter}"

CACHES=(
  "${BASE}/kubric_train_alltracker_ce24_drivingpt_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce24_fltpt_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce24_monkapt_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce24_springpt_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce64_drivingpt_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce64_kublong_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce64_monkapt_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce64_podlong_stride2_margin05_seed123_deepcorr_risk_v1"
  "${BASE}/kubric_train_alltracker_ce64_springpt_stride2_margin05_seed123_deepcorr_risk_v1"
)

EXISTING=()
for cache in "${CACHES[@]}"; do
  if [[ -d "${cache}/sequences" ]]; then
    EXISTING+=("${cache}")
  else
    echo "skip missing cache: ${cache}"
  fi
done

if [[ "${#EXISTING[@]}" -eq 0 ]]; then
  echo "no Stage A risk caches found; run recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh first" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"
DATA_DIR="$(IFS=,; echo "${EXISTING[*]}")"

echo "Stage A: training baseline risk head on ${#EXISTING[@]} 100seq risk caches"
echo "save_dir: ${SAVE_DIR}"
echo "${DATA_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python train_final_stage_adapter.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --exp kubric_multisource_baseline_risk_seed123_deepcorr_v1 \
  --target_mode baseline_risk \
  --feature_profile lowdim \
  --patch_mode deep_corr_gate \
  --deep_embed_dim 64 \
  --loss_mode risk_error_weighted \
  --gain_weight_scale 0.25 \
  --max_sample_weight 10.0 \
  --epochs 30 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --hidden_dim 64 \
  --num_layers 1 \
  --chunk_points 256 \
  --train_ratio 1.0 \
  --seed 123
