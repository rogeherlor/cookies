# -*- coding: utf-8 -*-
"""
Shared EKF run configuration.

Both ekf.py and ekf_genetic.py read from here so the optimizer always
runs on exactly the same dataset / outage / rotation mode.
"""
import data_loader


# ── GNSS outage ───────────────────────────────────────────────────────────────
OUTAGE_START    = 200   # seconds from start
OUTAGE_DURATION = 50   # seconds

# ── Rotation mode ─────────────────────────────────────────────────────────────
USE_3D_ROTATION = False  # True = full 3D (roll/pitch/yaw), False = yaw only

# ── EKF parameters ───────────────────────────────────────────────────────────
# Paste optimized results from ekf_genetic.py here.
# Set to None to use DEFAULT_EKF_PARAMS from ekf_core.py.

if (False):
    # KITTI
    NAV_DATA = data_loader.get_kitti_dataset('10_03_0027')
    EKF_PARAMS = {'Qpos': 5.890e-01, 'Qvel': 7.008e+01, 'QorientXY': 1.5590e-03, 'QorientZ': 8.7920e-02, 'Qacc': 3.3123e-03, 'QgyrXY': 4.1251e-04, 'QgyrZ': 2.3662e-01, 'Rpos': 3.66, 'beta_acc': -1.222e-06, 'beta_gyr': -1.994e-01, 'P_pos_std': 0.19, 'P_vel_std': 0.52, 'P_orient_std': 0.191, 'P_acc_std': 2.227e-02, 'P_gyr_std': 1.046e-03}

else:
    # Cookies
    NAV_DATA = data_loader.get_cookies_dataset(
        'castellana_270226_5',
        log_file='ttyUSB1_2026-02-27_12-01-08.218318811.log'
    )
    EKF_PARAMS = {'Qpos': 1.324e+01, 'Qvel': 1.612e+00, 'QorientXY': 5.3714e-05, 'QorientZ': 2.1049e-01, 'Qacc': 9.2034e-02, 'QgyrXY': 3.8828e-04, 'QgyrZ': 1.0456e-03, 'Rpos': 62.19, 'beta_acc': -5.457e-07, 'beta_gyr': -7.259e-01, 'P_pos_std': 3.38, 'P_vel_std': 2.01, 'P_orient_std': 0.012, 'P_acc_std': 3.881e-02, 'P_gyr_std': 1.254e-03}
