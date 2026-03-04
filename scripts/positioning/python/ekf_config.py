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
ENABLE_NHC  = False   # Non-Holonomic Constraints (no lateral/vertical slip)
ENABLE_ZUPT = False   # Zero Velocity Updates (detect standstill from IMU)
ENABLE_LEVEL = False   # Leveling constraint (roll/pitch ~ 0, 2D mode only)

# ── EKF parameters ───────────────────────────────────────────────────────────
# Paste optimized results from ekf_genetic.py here.
# Set to None to use DEFAULT_EKF_PARAMS from ekf_core.py.

if (True):
    # KITTI
    NAV_DATA = data_loader.get_kitti_dataset('10_03_0027')
    EKF_PARAMS = {'Qpos': 8.857e+01, 'Qvel': 3.796e+00, 'QorientXY': 2.4431e-05, 'QorientZ': 5.8836e-02, 'Qacc': 6.6651e-03, 'QgyrXY': 1.6606e-06, 'QgyrZ': 9.4215e-03, 'Rpos': 1.95, 'beta_acc': -4.710e-06, 'beta_gyr': -9.830e-01, 'P_pos_std': 7.27, 'P_vel_std': 2.51, 'P_orient_std': 0.033, 'P_acc_std': 2.162e-02, 'P_gyr_std': 2.186e-04, 'Rnhc': 1.171e-01, 'Rzupt': 1.141e-04, 'zupt_accel_threshold': 5.875e-02, 'zupt_gyro_threshold': 1.615e-01, 'Rlevel': 4.008e-01}

else:
    # Cookies
    NAV_DATA = data_loader.get_cookies_dataset(
        'castellana_270226_5',
        log_file='ttyUSB1_2026-02-27_12-01-08.218318811.log'
    )
    EKF_PARAMS = {'Qpos': 7.911e+01, 'Qvel': 6.693e-01, 'QorientXY': 7.3964e-05, 'QorientZ': 6.3778e-02, 'Qacc': 9.7516e-02, 'QgyrXY': 2.0384e-06, 'QgyrZ': 3.6157e-02, 'Rpos': 3.40, 'beta_acc': -3.810e-07, 'beta_gyr': -1.623e-01, 'P_pos_std': 2.73, 'P_vel_std': 0.52, 'P_orient_std': 0.029, 'P_acc_std': 2.852e-03, 'P_gyr_std': 5.651e-03}
