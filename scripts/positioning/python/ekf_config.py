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

# ── Pseudo-measurement constraints ───────────────────────────────────────────
ENABLE_NHC  = True   # Non-Holonomic Constraints (no lateral/vertical slip)
ENABLE_ZUPT = False   # Zero Velocity Updates (detect standstill from IMU)
ENABLE_LEVEL = False   # Leveling constraint (roll/pitch ~ 0, 2D mode only)

# ── EKF parameters ───────────────────────────────────────────────────────────
# Paste optimized results from ekf_genetic.py here.
# Set to None to use DEFAULT_EKF_PARAMS from ekf_core.py.

if (True):
    # KITTI
    NAV_DATA = data_loader.get_kitti_dataset('10_03_0027')
    EKF_PARAMS = {'Qpos': 4.588e-01, 'Qvel': 7.193e+00, 'QorientXY': 9.5376e-05, 'QorientZ': 3.3101e-02, 'Qacc': 2.5540e-03, 'QgyrXY': 3.8952e-04, 'QgyrZ': 6.6613e-02, 'Rpos': 0.87, 'beta_acc': -5.830e-08, 'beta_gyr': -6.349e-01, 'P_pos_std': 0.73, 'P_vel_std': 0.43, 'P_orient_std': 0.085, 'P_acc_std': 4.667e-02, 'P_gyr_std': 4.502e-03, 'Rnhc': 6.948e+00, 'Rzupt': 9.787e-02, 'zupt_accel_threshold': 1.841e-01, 'zupt_gyro_threshold': 2.266e-02, 'Rlevel': 3.790e-02}

else:
    # Cookies
    NAV_DATA = data_loader.get_cookies_dataset(
        'castellana_270226_5',
        log_file='ttyUSB1_2026-02-27_12-01-08.218318811.log'
    )
    EKF_PARAMS = {'Qpos': 7.911e+01, 'Qvel': 6.693e-01, 'QorientXY': 7.3964e-05, 'QorientZ': 6.3778e-02, 'Qacc': 9.7516e-02, 'QgyrXY': 2.0384e-06, 'QgyrZ': 3.6157e-02, 'Rpos': 3.40, 'beta_acc': -3.810e-07, 'beta_gyr': -1.623e-01, 'P_pos_std': 2.73, 'P_vel_std': 0.52, 'P_orient_std': 0.029, 'P_acc_std': 2.852e-03, 'P_gyr_std': 5.651e-03}
