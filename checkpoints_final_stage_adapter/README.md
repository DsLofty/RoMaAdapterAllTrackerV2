# Final-Stage Adapter Checkpoints

Place the selected V2 checkpoints here for a self-contained VOT package.

Required for reproducing the current release lineage:

```text
kubric_multisource_baseline_risk_seed123_deepcorr_v1_best.pth
kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
```

The runtime default final-stage checkpoint is:

```text
kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth
```

The runtime also accepts an external checkpoint path:

```bash
FINAL_STAGE_ADAPTER_CKPT=/path/to/final_stage_adapter.pth
```
