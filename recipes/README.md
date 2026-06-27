# V2 Release Recipe Entrypoints

These scripts reproduce the selected RoMaAdapterAllTrackerV2 release model.
Run them from `release_roma_adapter_v2/` or from the parent repository
using the same relative paths.

## Current Best Flow

```bash
# Stage A: export/train/evaluate 100seq baseline risk head.
bash recipes/run_stage_a_risk_100seq.sh

# Export 300 sequences per Kubric source.
bash recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_accept_stride1_300seq_v1.sh

# Convert accept labels to strong-v2.
bash recipes/relabel_final_stage_adapter_multisource_seed123_deepcorr_accept_strong_v2_300seq.sh

# Export and relabel DAVIS risk+accept eval cache.
bash recipes/export_final_stage_adapter_data_davis_seed123_deepcorr_risk_accept_stride1_v1.sh
bash recipes/relabel_final_stage_adapter_davis_seed123_deepcorr_accept_strong_v2.sh

# Train risk-initialized risk+accept joint model for 60 epochs.
bash recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_accept_strong_v2_joint_fullft_valsplit_300seq_ep60.sh

# Evaluate the selected epoch040 checkpoint.
bash recipes/eval_final_stage_risk_accept_davis_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_conservative.sh epoch040
```

Full one-shot entrypoint:

```bash
bash recipes/run_full_reproduction_from_stage_a.sh
```

If the Stage A checkpoint already exists, use:

```bash
bash recipes/run_current_best_training_and_eval.sh
```

## Selected Runtime Parameters

```text
checkpoint suffix = epoch040
risk threshold    = 0.2
accept threshold  = 0.997
image size        = 448x768 for VOT submission-style evaluation
```

## Important Inputs

The 60-epoch joint training initializes from the 100seq baseline risk head:

```text
checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
```

The final selected checkpoint is:

```text
checkpoints_final_stage_adapter/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
```

## Environment Overrides

```bash
KUBRIC_ROOT=/path/to/kubric/data
FINAL_STAGE_DATA_BASE=/path/to/final_stage_adapter_data
ALLTRACKER_CKPT=/path/to/alltracker.pth
RISK_CKPT=/path/to/risk_init.pth
FINAL_STAGE_CKPT_DIR=/path/to/checkpoints_final_stage_adapter
DAVIS_FINAL_STAGE_DATA_DIR=/path/to/dav_eval_cache
FINAL_STAGE_EVAL_DIR=/path/to/eval_outputs
```

The active release model is the risk-initialized `300seq_ep60_epoch040` model.

Before VOT packaging, verify that the local release directory contains the
required runtime assets:

```bash
bash recipes/verify_release_assets.sh
```
