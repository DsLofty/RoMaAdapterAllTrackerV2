#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
CKPT_DIR="${FINAL_STAGE_CKPT_DIR:-${PROJECT_ROOT}/checkpoints_final_stage_adapter}"
CKPT="${RISK_CKPT:-${CKPT_DIR}/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth}"
DATA_BASE="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"
DATA_DIR="${DAVIS_RISK_DATA_DIR:-${DATA_BASE}/dav_eval_alltracker_stride2_margin05_seed123_deepcorr_risk_v1}"
SAVE_DIR="${FINAL_STAGE_EVAL_DIR:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_eval_outputs}"

if [[ ! -f "${CKPT}" ]]; then
  echo "risk checkpoint not found: ${CKPT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python eval_final_stage_risk.py \
  --data_dir "${DATA_DIR}" \
  --ckpt "${CKPT}" \
  --save_dir "${SAVE_DIR}" \
  --exp dav_eval_kubric_multisource_baseline_risk_seed123_deepcorr_v1 \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99 \
  --chunk_points 512
