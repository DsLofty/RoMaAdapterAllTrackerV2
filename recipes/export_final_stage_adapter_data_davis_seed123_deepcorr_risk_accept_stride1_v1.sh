#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SAVE_DIR="${FINAL_STAGE_DATA_BASE:-/home/zanghan/Pyproject/vot/alltracker/final_stage_adapter_data}"
ALLTRACKER_CKPT="${ALLTRACKER_CKPT:-/home/zanghan/Pyproject/vot/alltracker/ckpt/alltracker.pth}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
python export_final_stage_adapter_data.py \
  --dname dav \
  --only_first \
  --image_size 448 768 \
  --ckpt_path "${ALLTRACKER_CKPT}" \
  --save_dir "${SAVE_DIR}" \
  --exp dav_eval_alltracker_stride1_margin10_seed123_deepcorr_risk_accept_v1 \
  --seed 123 \
  --deterministic_sampling \
  --num_workers 0 \
  --inference_iters 4 \
  --target_frame_stride 1 \
  --target_frame_include stride,last \
  --target_points_per_frame 4096 \
  --positive_margin_px 0.5 \
  --negative_margin_px 0.5 \
  --risk_positive_err_px 4.0 \
  --risk_err_increase_px 3.0 \
  --risk_min_err_for_increase_px 2.0 \
  --risk_jump_px 16.0 \
  --risk_jump_min_err_px 2.5 \
  --risk_negative_err_px 1.5 \
  --risk_negative_jump_px 4.0 \
  --risk_negative_visible_thr 0.6 \
  --accept_positive_margin_px 1.0 \
  --accept_negative_margin_px 1.0 \
  --accept_require_risk_positive \
  --roma_model outdoor \
  --roma_device cuda \
  --roma_sample_mode bilinear \
  --roma_disable_custom_corr \
  --roma_pair_batch_size 1 \
  --analysis_visible_only \
  --save_deep_corr_features \
  --deep_corr_radius8 2 \
  --deep_corr_dtype float16
