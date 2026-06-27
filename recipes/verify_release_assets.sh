#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

FILES=(
  "ckpt/alltracker.pth"
  "checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth"
  "checkpoints_final_stage_adapter/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth"
  "vot_submit/roma_adapter_folder_tracker_v2"
  "vot_submit/roma_adapter_folder_tracker_v2.py"
  "vot_submit/vot_folder_io.py"
  "vot_submit/trackers_v2.ini.template"
  "vot_submit/v2_runtime_config.py"
  "alltracker_runtime_utils.py"
  "final_stage_cache_utils.py"
  "final_stage_adapter_model.py"
)

missing=0
for path in "${FILES[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "ok: ${path}"
  else
    echo "missing: ${path}"
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  echo "release assets are incomplete" >&2
  exit 1
fi

echo "release assets verified"
