# RoMaAdapterAllTrackerV2 Experiment Summary

This file records the selected experiment that should be treated as the current
V2 release candidate.

## Selected Runtime

```text
tracker id      = RoMaAdapterAllTrackerV2
final checkpoint= kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
risk threshold  = 0.2
accept threshold= 0.997
target stride   = 1
image size      = 448x768 for VOT submission-style evaluation
```

## Model Design

The runtime uses a two-head final-stage adapter:

```text
risk_head:
  predicts whether the AllTracker coordinate is risky.

accept_head:
  predicts whether the RoMa candidate should be accepted when the baseline is risky.

deployment gate:
  accept RoMa if risk_prob >= 0.2 and accept_prob >= 0.997
```

Inputs are:

```text
15 low-dimensional temporal/geometric features
deep local correlation features sampled around query/baseline/RoMa locations
```

The deployed action is coordinate replacement only:

```text
final_xy = roma_xy where gate is true
final_xy = alltracker_xy otherwise
```

## Data

Stage B uses deterministic Kubric multisource sampling with seed 123 and up to
300 sequences per source:

```text
ce24/drivingpt
ce24/fltpt
ce24/monkapt
ce24/springpt
ce64/drivingpt
ce64/kublong
ce64/monkapt
ce64/podlong
ce64/springpt
```

Export settings:

```text
image_size              = 448 768
sequence_len            = 24
target_frame_stride     = 1
target_frame_include    = stride,last
target_points_per_frame = 4096
deep_corr_radius8       = 2
deep_corr_dtype         = float16
deterministic_sampling  = true
seed                    = 123
```

## Labels

Stage A risk pretraining uses baseline risk labels from the final-stage cache.

Stage B accept labels use strong-v2:

```text
positive:
  risk_label == 1
  baseline_error >= 8 px
  roma_error <= 4 px
  baseline_error - roma_error >= 8 px

negative:
  risk_label == 1
  (
    baseline_error <= 4 px and roma_error >= 12 px
    or roma_error - baseline_error >= 8 px
  )
```

## Training

Stage A:

```text
checkpoint = kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
role       = initializes the risk head and shared representation
data       = multisource 100seq risk cache
```

Stage B:

```text
script              = recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_accept_strong_v2_joint_fullft_valsplit_300seq_ep60.sh
target_mode         = risk_accept_joint
patch_mode          = deep_corr_risk_accept
feature_profile     = lowdim
init_risk_ckpt      = Stage A checkpoint
joint_risk_loss     = 0.2
epochs              = 60
lr                  = 5e-5
weight_decay        = 1e-4
train split         = per_source, train_ratio=0.9
best metric         = val_acc
saved snapshots     = every 5 epochs
selected checkpoint = epoch040
```

## DAVIS Result

Selected epoch040 conservative sweep:

```text
risk threshold          = 0.2
accept threshold        = 0.997
baseline_epe_mean       = 5.76720
selector_delta_epe      = -0.06168
gate_ratio              = 0.08154
accepted_delta_epe      = -0.27403
selector_delta_da       = +0.002685
selector_delta_aj       = +0.002589
row_weighted_delta_epe  = -0.06960
```

Checkpoint comparison:

```text
epoch030 BEST delta_epe = -0.05676
epoch035 BEST delta_epe = -0.05733
epoch040 BEST delta_epe = -0.06168
epoch050 BEST delta_epe = -0.05591
epoch060 BEST delta_epe = -0.05544
```

Epoch040 is selected because it is best on EPE, row-weighted EPE, DA, and AJ in
the current DAVIS diagnostic sweep.

## Reproduction Commands

Full reproduction from Stage A:

```bash
bash recipes/run_full_reproduction_from_stage_a.sh
```

Step-by-step reproduction:

```bash
bash recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh
bash recipes/export_final_stage_adapter_data_davis_seed123_deepcorr_risk_v1.sh
bash recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh
bash recipes/eval_final_stage_risk_davis_multisource_seed123_deepcorr_v1.sh
bash recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_accept_stride1_300seq_v1.sh
bash recipes/relabel_final_stage_adapter_multisource_seed123_deepcorr_accept_strong_v2_300seq.sh
bash recipes/export_final_stage_adapter_data_davis_seed123_deepcorr_risk_accept_stride1_v1.sh
bash recipes/relabel_final_stage_adapter_davis_seed123_deepcorr_accept_strong_v2.sh
bash recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_accept_strong_v2_joint_fullft_valsplit_300seq_ep60.sh
bash recipes/eval_final_stage_risk_accept_davis_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_conservative.sh epoch040
```

## VOT Commands

```bash
cd ~/votsp_workspace

CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VOT_ROMA_ADAPTER_IMAGE_SIZE=448,768 \
vot --debug evaluate --force --persist RoMaAdapterAllTrackerV2

vot pack RoMaAdapterAllTrackerV2
```

The generated archive should retain `identifier: RoMaAdapterAllTrackerV2` in
`manifest.yml`.
