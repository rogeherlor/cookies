# -*- coding: utf-8 -*-
"""
Shared configuration for ins_runner.py, ins_genetic.py, and ins_compare.py.

All scripts read from here so they always operate on the same dataset,
outage window, rotation mode, and filter selection.

IMPORTANT — performance conclusions
------------------------------------
Do NOT compare filter outputs from a first run.  The default parameters
here are generic starting points.  Run ins_genetic.py for each filter
variant to obtain tuned FILTER_PARAMS before drawing any conclusions.
"""
import sys
from pathlib import Path

# Allow importing data_loader from the same directory
sys.path.insert(0, str(Path(__file__).parent))
import data_loader
import filter_params as fp


# ── Filter / smoother selection ────────────────────────────────────────────────
# Classical filters:
#   "ekf_vanilla"    — Euler-angle EKF, GPS-only  (Groves 2013)
#   "ekf_enhanced"   — Euler-angle EKF + NHC + ZUPT
#   "eskf_vanilla"   — Quaternion ESKF, GPS-only  (Solà 2017)
#   "eskf_enhanced"  — Quaternion ESKF + NHC + ZUPT
#   "iekf_vanilla"   — Left-invariant EKF, GPS-only  (Barrau & Bonnabel 2017)
#   "iekf_enhanced"  — Left-invariant EKF + NHC + ZUPT
#   "imu_only"       — Pure dead reckoning (no GNSS, no filter)
# Smoothers (see smoothers/):
#   "rts_smoother"   — Rauch-Tung-Striebel batch smoother (uses all GPS — not causal)
#   "isam2"          — iSAM2 online/causal smoother (Kaess et al. IJRR 2012)
#                      Requires: conda install -c conda-forge gtsam
# Deep learning filters (require trained weights in artifacts/ — see dl_filters/):
#   "iekf_ai_imu"    — AI-IMU Dead-Reckoning (Brossard et al. IEEE TIV 2020)
#   "tlio"           — Tight Learned Inertial Odometry (Liu et al. IEEE RA-L 2020)
#   "deep_kf"        — Deep Kalman Filter GNSS+IMU (Hosseinyalamdary MDPI Sensors 2018)
#   "tartan_imu"     — Tartan IMU foundation model (Zhao et al. CVPR 2025)
FILTER = "eskf_enhanced"


# ── Rotation mode ──────────────────────────────────────────────────────────────
# True  → full 3D (roll, pitch, yaw), recommended for vehicles with pitch changes
# False → 2D (yaw only, flat-earth assumption), simpler and faster
MODE_3D = False


# ── GNSS outage simulation ─────────────────────────────────────────────────────
OUTAGE_START    = 200   # [s] — time from start when GPS signal is lost
OUTAGE_DURATION = 200    # [s] — duration of GPS blackout


# ── Dataset ────────────────────────────────────────────────────────────────────
# Only KITTI is used.  Change the sequence ID as needed.
NAV_DATA = data_loader.get_kitti_dataset('01')  # seq 00 = 2011_10_03_drive_0027
# clean test sequences are 01, 04, 06, 07, 08, 09, 10. Sequences 00, 02, 05 have a ~2-second data gap (logging problem) and 03 has no raw data at all.
# train on all except for the one being validated

# ── Dead-reckoning evaluation mode ────────────────────────────────────────────
# True  → disable GPS for ALL filters; initialize from ground-truth state.
#          Produces trel/rrel directly comparable to Brossard et al. 2020 Table I.
#          ins_compare.py and iekf_ai_imu.py both honour this flag.
# False → normal GPS-aided INS with outage simulation (default, main use case)
DR_MODE = False


# ── Ground truth source ────────────────────────────────────────────────────────
# True  → use the RTS smoother (best achievable from IMU+GPS, full-data pass)
#          as ground truth for metrics and plots.
# False → use the raw KITTI GPS reference as ground truth; RTS is shown as an
#          additional purple overlay in trajectory/error plots.
USE_RTS_AS_GT = True


# ── Filter parameters ──────────────────────────────────────────────────────────
# Set to None to use each filter's built-in DEFAULT_PARAMS.
# After running ins_genetic.py, paste the optimised dict here.
#
# Example (ESKF enhanced, KITTI 3D, optimised):
# FILTER_PARAMS = {
#     'Qpos': 2.978e+01, 'Qvel': 1.542e+01, 'QorientXY': 6.2359e-02,
#     'QorientZ': 1.2136e-02, 'Qacc': 4.3830e-02, 'QgyrXY': 1.5068e-05,
#     'QgyrZ': 2.0950e-03, 'Rpos': 10.30, 'beta_acc': -5.388e-08,
#     'beta_gyr': -2.660e-01, 'P_pos_std': 0.32, 'P_vel_std': 2.72,
#     'P_orient_std': 0.156, 'P_acc_std': 1.051e-03, 'P_gyr_std': 7.642e-03,
# }
FILTER_PARAMS = fp.get(FILTER, MODE_3D, NAV_DATA.dataset_name)