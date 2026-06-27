#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh
bash recipes/export_final_stage_adapter_data_davis_seed123_deepcorr_risk_v1.sh
bash recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh
bash recipes/eval_final_stage_risk_davis_multisource_seed123_deepcorr_v1.sh
