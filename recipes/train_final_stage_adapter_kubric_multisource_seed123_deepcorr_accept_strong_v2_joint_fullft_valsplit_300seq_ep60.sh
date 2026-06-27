#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROJECT_ROOT="$(pwd)"
BASE="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"
SAVE_DIR="${FINAL_STAGE_CKPT_DIR:-${PROJECT_ROOT}/checkpoints_final_stage_adapter}"

CACHES=(
  "${BASE}/kubric_train_alltracker_ce24_drivingpt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce24_fltpt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce24_monkapt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce24_springpt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce64_drivingpt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce64_kublong_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce64_monkapt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce64_podlong_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
  "${BASE}/kubric_train_alltracker_ce64_springpt_stride1_strongv2_300seq_seed123_deepcorr_risk_accept_v1"
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
  echo "no 300seq strong-v2 caches found; run recipes/relabel_final_stage_adapter_multisource_seed123_deepcorr_accept_strong_v2_300seq.sh first" >&2
  exit 1
fi

DATA_DIR="$(IFS=,; echo "${EXISTING[*]}")"
DEFAULT_RISK_CKPT="${PROJECT_ROOT}/checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth"
FALLBACK_RISK_CKPT="/home/zanghan/Pyproject/vot/alltracker/checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth"
RISK_CKPT="${RISK_CKPT:-${DEFAULT_RISK_CKPT}}"
if [[ ! -f "${RISK_CKPT}" && -f "${FALLBACK_RISK_CKPT}" ]]; then
  RISK_CKPT="${FALLBACK_RISK_CKPT}"
fi

if [[ ! -f "${RISK_CKPT}" ]]; then
  echo "risk init checkpoint not found: ${RISK_CKPT}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

echo "risk-init full fine-tuning risk+accept on ${#EXISTING[@]} 300seq strong-v2 caches"
echo "risk_ckpt: ${RISK_CKPT}"
echo "save_dir: ${SAVE_DIR}"
echo "exp: kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60"
echo "${DATA_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python train_final_stage_adapter.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --exp kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60 \
  --target_mode risk_accept_joint \
  --feature_profile lowdim \
  --patch_mode deep_corr_risk_accept \
  --deep_embed_dim 64 \
  --init_risk_ckpt "${RISK_CKPT}" \
  --init_normalizer_from_ckpt \
  --joint_risk_loss_weight 0.2 \
  --loss_mode bce \
  --epochs 60 \
  --lr 0.00005 \
  --weight_decay 0.0001 \
  --hidden_dim 64 \
  --num_layers 1 \
  --chunk_points 256 \
  --train_ratio 0.9 \
  --split_mode per_source \
  --best_metric val_acc \
  --save_epoch_every 5 \
  --seed 123
