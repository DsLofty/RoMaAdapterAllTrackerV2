# RoMaAdapterAllTrackerV2 VOT Wrapper

This directory contains the folder-protocol VOT entrypoint for the selected
RoMaAdapterAllTrackerV2 final-stage release candidate.

## Files

```text
roma_adapter_folder_tracker_v2       executable launcher used by VOT
roma_adapter_folder_tracker_v2.py    final-stage risk/accept runtime
trackers_v2.ini.template             VOT workspace tracker template
v2_runtime_config.py                 fixed candidate metadata
vot_folder_io.py                     lightweight VOT frame IO and RoMa candidate collection
```

The old V1 folder runtime is archived under `archive_v1/` for reference only.
The V2 tracker does not import it.

## Runtime Pipeline

1. Run frozen AllTracker to produce final dense tracks.
2. Run RoMa from query frame to final-stage target frames.
3. Build low-dimensional temporal features and deep-correlation local features.
4. Run the learned risk/accept adapter.
5. Replace AllTracker final coordinates with RoMa only when:

```text
risk_prob >= FINAL_STAGE_RISK_THRESHOLD
accept_prob >= FINAL_STAGE_ACCEPT_THRESHOLD
```

Current default candidate:

```text
checkpoint = checkpoints_final_stage_adapter/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
risk_thr   = 0.2
accept_thr = 0.997
```

You can override these at runtime:

```bash
FINAL_STAGE_ADAPTER_CKPT=/path/to/checkpoint.pth
FINAL_STAGE_RISK_THRESHOLD=0.2
FINAL_STAGE_ACCEPT_THRESHOLD=0.997
```

## Workspace Configuration

Copy `trackers_v2.ini.template` into the VOT workspace and replace `/path/to/...`
with the absolute path to this v2 candidate directory.

```ini
[RoMaAdapterAllTrackerV2]
label = RoMaAdapterAllTrackerV2
command = roma_adapter_folder_tracker_v2
protocol = folderpython
paths = /path/to/release_roma_adapter_v2_candidate:/path/to/release_roma_adapter_v2_candidate/vot_submit
```

## Test/Evaluate

Use the same fixed image size used by the reproduced v1 submission and by the
selected V2 package:

```bash
cd ~/votsp_workspace

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VOT_ROMA_ADAPTER_IMAGE_SIZE=448,768 \
vot --debug test RoMaAdapterAllTrackerV2

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VOT_ROMA_ADAPTER_IMAGE_SIZE=448,768 \
vot --debug evaluate --force --persist RoMaAdapterAllTrackerV2
```

Package after validation:

```bash
cd ~/votsp_workspace
vot pack RoMaAdapterAllTrackerV2
```

## Required Checkpoints

The directory should contain or be given paths to:

```text
ckpt/alltracker.pth
checkpoints_final_stage_adapter/kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
checkpoints_final_stage_adapter/kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
```

The runtime falls back to the development paths under
`/home/zanghan/Pyproject/vot/alltracker/` if local checkpoint files are absent.
For a self-contained package, copy both checkpoints into this directory before
packing.
