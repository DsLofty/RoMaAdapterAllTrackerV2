#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
CKPT_DIR="${FINAL_STAGE_CKPT_DIR:-${PROJECT_ROOT}/checkpoints_final_stage_adapter}"
CKPT_SUFFIX="${1:-epoch040}"
CKPT="${CKPT_DIR}/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_${CKPT_SUFFIX}.pth"
EXP="dav_eval_kubric_multisource_risk_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_${CKPT_SUFFIX}_conservative"
DATA_BASE="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"
DATA_DIR="${DAVIS_FINAL_STAGE_DATA_DIR:-${DATA_BASE}/dav_eval_alltracker_stride1_strongv2_seed123_deepcorr_risk_accept_v1}"
SAVE_DIR="${FINAL_STAGE_EVAL_DIR:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_eval_outputs}"

if [[ ! -f "${CKPT}" ]]; then
  echo "checkpoint not found: ${CKPT}" >&2
  echo "usage: bash $0 [epoch030|epoch035|epoch040|epoch050|epoch060|best|latest|best_val_acc|best_val_loss]" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python eval_final_stage_risk_accept.py \
  --data_dir "${DATA_DIR}" \
  --ckpt "${CKPT}" \
  --save_dir "${SAVE_DIR}" \
  --exp "${EXP}" \
  --risk_thresholds 0.2,0.3,0.4,0.5,0.7,0.9 \
  --accept_thresholds 0.99,0.995,0.997,0.999,0.9995,0.9999 \
  --chunk_points 512
