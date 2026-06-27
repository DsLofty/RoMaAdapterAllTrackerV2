"""Runtime metadata for the selected RoMaAdapterAllTrackerV2 release."""

TRACKER_ID = "RoMaAdapterAllTrackerV2"

DEFAULT_IMAGE_SIZE = (448, 768)
INFERENCE_ITERS = 4

FINAL_STAGE_DEFAULT_CHECKPOINT = (
    "checkpoints_final_stage_adapter/"
    "kubric_multisource_roma_accept_seed123_deepcorr_strong_v2_joint_fullft_valsplit_300seq_ep60_epoch040.pth"
)

FINAL_STAGE_RISK_THRESHOLD = 0.2
FINAL_STAGE_ACCEPT_THRESHOLD = 0.997
FINAL_STAGE_TARGET_FRAME_STRIDE = 1
FINAL_STAGE_DEEP_CORR_RADIUS8 = 2

DAVIS_CANDIDATE_METRICS = {
    "delta_epe": -0.06168,
    "gate": 0.08154,
    "accepted_delta_epe": -0.27403,
    "delta_da": 0.002685,
    "delta_aj": 0.002589,
    "row_weighted_delta_epe": -0.06960,
}
