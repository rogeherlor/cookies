"""
Shared EKF run configuration.

Both ekf.py and ekf_genetic.py read from here so the optimizer always
runs on exactly the same dataset / outage / rotation mode.
"""
import data_loader


# ── GNSS outage ───────────────────────────────────────────────────────────────
OUTAGE_START    = 200   # seconds from start
OUTAGE_DURATION = 70   # seconds

# ── Rotation mode ─────────────────────────────────────────────────────────────
USE_3D_ROTATION = True  # True = full 3D (roll/pitch/yaw), False = yaw only

# Note: NHC/ZUPT/level pseudo-constraints were removed intentionally.
# GNSS velocity is always fused when `vel_s`/`cog_s` are present.

# ── EKF parameters ───────────────────────────────────────────────────────────
# Paste optimized results from ekf_genetic.py here.
# Set to None to use DEFAULT_EKF_PARAMS from ekf_core.py.

if (True):
    # KITTI
    NAV_DATA = data_loader.get_kitti_dataset('10_03_0027')
    EKF_PARAMS = {'Qpos': 8.460e-01, 'Qvel': 8.199e+00, 'QorientXY': 1.4378e-03, 'QorientZ': 9.7801e-02, 'Qacc': 4.8729e-02, 'QgyrXY': 1.5092e-04, 'QgyrZ': 1.6406e-02, 'Rpos': 0.15, 'Rvel': 1.439e+00, 'beta_acc': -1.610e-06, 'beta_gyr': -1.479e+00, 'P_pos_std': 29.45, 'P_vel_std': 0.46, 'P_orient_std': 0.070, 'P_acc_std': 1.107e-02, 'P_gyr_std': 5.486e-03, }

else:
    # Cookies
    NAV_DATA = data_loader.get_cookies_dataset(
        'castellana_270226_5',
        log_file='ttyUSB1_2026-02-27_12-01-08.218318811.log'
    )
    EKF_PARAMS = {'Qpos': 7.911e+01, 'Qvel': 6.693e-01, 'QorientXY': 7.3964e-05, 'QorientZ': 6.3778e-02, 'Qacc': 9.7516e-02, 'QgyrXY': 2.0384e-06, 'QgyrZ': 3.6157e-02, 'Rpos': 3.40, 'Rvel': 5.000e-01, 'beta_acc': -3.810e-07, 'beta_gyr': -1.623e-01, 'P_pos_std': 2.73, 'P_vel_std': 0.52, 'P_orient_std': 0.029, 'P_acc_std': 2.852e-03, 'P_gyr_std': 5.651e-03}
