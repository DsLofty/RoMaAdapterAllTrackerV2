# RoMaAdapterAllTrackerV2

This directory is the clean reproducible V2 release package for the final-stage
RoMa adapter on top of AllTracker. It contains the VOT folder-protocol runtime
and the scripts used to reproduce the selected DAVIS/VOT release model.

The V2 tracker keeps AllTracker and RoMa frozen. It runs AllTracker first, builds
sparse RoMa candidates at final-stage frames, and uses a learned two-head
adapter to decide whether RoMa should replace the AllTracker coordinate.

## Selected Release Model

The current selected V2 model is:

```text
checkpoint:
  checkpoints_final_stage_adapter/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth

runtime thresholds:
  risk_prob   >= 0.200
  accept_prob >= 0.997

VOT submission image size:
  runtime default; VOT_ROMA_ADAPTER_IMAGE_SIZE was not set
```

DAVIS diagnostic result for this model:

```text
baseline_epe_mean       = 5.76720
selector_delta_epe      = -0.06168
selector_gate_ratio     = 0.08154
accepted_delta_epe      = -0.27403
selector_delta_da       = +0.002685
selector_delta_aj       = +0.002589
row_weighted_delta_epe  = -0.06960
```

For comparison, the previous release best1500 result was:

```text
delta_epe = -0.03112
delta_da  = +0.00114089
delta_aj  = +0.00009698
```

## Training Lineage

The selected V2 checkpoint is produced by a two-stage training process.

Stage A: baseline risk pretraining

```text
data          = Kubric multisource, 100 sequences per source
target        = baseline risk only
model mode    = deep_corr_gate / baseline_risk
checkpoint    = checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
purpose       = initialize the risk distribution and shared visual/lowdim backbone
```

Stage B: risk+accept joint optimization

```text
data          = Kubric multisource, 300 sequences per source
sources       = ce24/drivingpt, ce24/fltpt, ce24/monkapt, ce24/springpt,
                ce64/drivingpt, ce64/kublong, ce64/monkapt,
                ce64/podlong, ce64/springpt
target stride = 1
labels        = strong_v2 accept labels
init          = Stage A baseline risk checkpoint
model mode    = deep_corr_risk_accept
target mode   = risk_accept_joint
train split   = per_source, train_ratio=0.9
loss          = accept BCE + 0.2 * risk BCE
epochs        = 60
selected      = epoch040
```

Strong-v2 accept label rule:

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

## Directory Layout

```text
vot_submit/                       VOT folder-protocol wrapper
recipes/                          data export, relabel, training, eval scripts
ckpt/                             place alltracker.pth here for self-contained use
checkpoints_final_stage_adapter/  place risk init and selected V2 checkpoints here
datasets/, matchers/, nets/,
third_party/, utils/              runtime dependencies copied from the main project
final_stage_adapter_model.py      final-stage adapter model
alltracker_runtime_utils.py       lightweight AllTracker loading/forward helpers
final_stage_cache_utils.py        final-stage cache/model loading helpers
tapvid_dataset_utils.py           lightweight TAPVID dataset resolver
tapvid_metric_utils.py            lightweight TAPVID metric helper
export_final_stage_adapter_data.py
relabel_final_stage_accept_cache.py
train_final_stage_adapter.py
eval_final_stage_risk.py
eval_final_stage_risk_accept.py
EXPERIMENT_SUMMARY.md             concise reproduction notes
```

## Third-party Dependencies

This release vendors RoMa under:

```text
third_party/RoMa
```

The runtime wrapper `matchers/roma_wrapper.py` expects this path when importing
`romatch`. Keep RoMa's original `LICENSE` and `README.md` files with the
vendored source. RoMa remains a third-party dependency and is governed by its
own license.

After cloning, install the local RoMa package in the same Python environment:

```bash
cd third_party/RoMa
pip install -e .
```

## Required Checkpoints

For a self-contained VOT package, copy these files into this directory:

```text
ckpt/alltracker.pth
checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
checkpoints_final_stage_adapter/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
```

The VOT runtime falls back to the development paths under
`/home/zanghan/Pyproject/vot/alltracker/` if local checkpoint files are absent.
For submission packaging, local checkpoint files are preferred.

Check local release assets before VOT packaging:

```bash
bash recipes/verify_release_assets.sh
```

## Reproduce Data And Training

Run from this directory or from the parent repository path.

### Stage A: 100seq Baseline Risk Head

Export 100seq multisource baseline-risk data:

```bash
bash recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh
```

Export DAVIS risk eval cache:

```bash
bash recipes/export_final_stage_adapter_data_davis_seed123_deepcorr_risk_v1.sh
```

Train the baseline risk head:

```bash
bash recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_risk_100seq_v1.sh
```

Evaluate the Stage A risk head:

```bash
bash recipes/eval_final_stage_risk_davis_multisource_seed123_deepcorr_v1.sh
```

One command for Stage A:

```bash
bash recipes/run_stage_a_risk_100seq.sh
```

### Stage B: 300seq Risk+Accept Joint Model

Export 300seq multisource final-stage data:

```bash
bash recipes/export_final_stage_adapter_data_kubric_multisource_seed123_deepcorr_risk_accept_stride1_300seq_v1.sh
```

Relabel accept samples with strong-v2 labels:

```bash
bash recipes/relabel_final_stage_adapter_multisource_seed123_deepcorr_accept_strong_v2_300seq.sh
```

Export and relabel the DAVIS risk+accept eval cache:

```bash
bash recipes/export_final_stage_adapter_data_davis_seed123_deepcorr_risk_accept_stride1_v1.sh
bash recipes/relabel_final_stage_adapter_davis_seed123_deepcorr_accept_strong_v2.sh
```

Train the selected risk-initialized 60-epoch joint model:

```bash
bash recipes/train_final_stage_adapter_kubric_multisource_seed123_deepcorr_accept_strong_v2_joint_fullft_valsplit_300seq_ep60.sh
```

Evaluate the selected epoch040 checkpoint:

```bash
bash recipes/eval_final_stage_risk_accept_davis_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_conservative.sh epoch040
```

One command for the full current-best flow:

```bash
bash recipes/run_current_best_training_and_eval.sh
```

One command from Stage A through Stage B:

```bash
bash recipes/run_full_reproduction_from_stage_a.sh
```

Useful environment overrides:

```bash
KUBRIC_ROOT=/path/to/kubric/data
FINAL_STAGE_DATA_BASE=/path/to/final_stage_adapter_data
ALLTRACKER_CKPT=/path/to/alltracker.pth
RISK_CKPT=/path/to/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
FINAL_STAGE_CKPT_DIR=/path/to/checkpoints_final_stage_adapter
DAVIS_FINAL_STAGE_DATA_DIR=/path/to/dav_eval_cache
FINAL_STAGE_EVAL_DIR=/path/to/eval_outputs
```

## VOT Evaluation And Packaging

Register `vot_submit/trackers_v2.ini.template` in the VOT workspace, with
`paths` pointing to this directory and `vot_submit`.

Evaluation command used for the submitted V2 package:

```bash
cd ~/votsp_workspace

CUDA_VISIBLE_DEVICES=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vot --debug evaluate --force --persist RoMaAdapterAllTrackerV2
```

Then package:

```bash
vot pack RoMaAdapterAllTrackerV2
```

