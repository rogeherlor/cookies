"""
Shared EKF run configuration.

Both ekf.py and ekf_genetic.py read from here so the optimizer always
runs on exactly the same dataset / outage / rotation mode.
"""
import data_loader


# ── Core selection ────────────────────────────────────────────────────────────
# Name of the EKF core implementation to use.
#   "ekf"  -> legacy Euler-based core  (ekf_core.py)
#   "eskf" -> quaternion error-state core (eskf_core.py)
# In future you can add more cores and route them in ekf.py / ekf_genetic.py.
CORE_NAME = "eskf"


# ── GNSS outage ───────────────────────────────────────────────────────────────
OUTAGE_START    = 200   # seconds from start
OUTAGE_DURATION = 70   # seconds

# ── Rotation mode ─────────────────────────────────────────────────────────────
USE_3D_ROTATION = True  # True = full 3D (roll/pitch/yaw), False = yaw only


# ── EKF parameters ───────────────────────────────────────────────────────────
# Paste optimized results from ekf_genetic.py here.
# Set to None to use DEFAULT_EKF_PARAMS from ekf_core.py.

if (True):
    # KITTI
    NAV_DATA = data_loader.get_kitti_dataset('10_03_0027')
    # 2D 
    # EKF_PARAMS = {'Qpos': 1.080e+01, 'Qvel': 1.171e+00, 'QorientXY': 4.3530e-03, 'QorientZ': 1.0393e-02, 'Qacc': 4.4156e-01, 'QgyrXY': 3.9277e-06, 'QgyrZ': 9.2825e-03, 'Rpos': 2.11, 'Rvel': 6.832e-02, 'beta_acc': -5.101e-06, 'beta_gyr': -1.313e+00, 'P_pos_std': 24.38, 'P_vel_std': 0.32, 'P_orient_std': 0.017, 'P_acc_std': 6.960e-03, 'P_gyr_std': 2.744e-03, }
    # 3D
    EKF_PARAMS = {'Qpos': 2.831e+01, 'Qvel': 1.304e+01, 'QorientXY': 1.1277e-02, 'QorientZ': 6.0520e-02, 'Qacc': 1.1967e-03, 'QgyrXY': 2.3574e-03, 'QgyrZ': 2.7707e-01, 'Rpos': 1.89, 'Rvel': 2.208e-01, 'beta_acc': -2.816e-06, 'beta_gyr': -6.023e-02, 'P_pos_std': 7.76, 'P_vel_std': 0.22, 'P_orient_std': 0.011, 'P_acc_std': 2.509e-02, 'P_gyr_std': 1.861e-03, }


else:
    # Cookies
    NAV_DATA = data_loader.get_cookies_dataset(
        'castellana_270226_5',
        log_file='ttyUSB1_2026-02-27_12-01-08.218318811.log'
    )
    # 2D 
    EKF_PARAMS = {'Qpos': 1.163e+01, 'Qvel': 6.430e+01, 'QorientXY': 7.6699e-02, 'QorientZ': 1.2276e-02, 'Qacc': 9.4026e-02, 'QgyrXY': 1.5175e-04, 'QgyrZ': 8.9939e-03, 'Rpos': 70.38, 'Rvel': 7.234e+00, 'beta_acc': -3.224e-08, 'beta_gyr': -5.504e-02, 'P_pos_std': 0.13, 'P_vel_std': 0.16, 'P_orient_std': 0.091, 'P_acc_std': 1.883e-03, 'P_gyr_std': 1.616e-03, }
    # 3D
    # EKF_PARAMS = {'Qpos': 7.911e+01, 'Qvel': 6.693e-01, 'QorientXY': 7.3964e-05, 'QorientZ': 6.3778e-02, 'Qacc': 9.7516e-02, 'QgyrXY': 2.0384e-06, 'QgyrZ': 3.6157e-02, 'Rpos': 3.40, 'Rvel': 5.000e-01, 'beta_acc': -3.810e-07, 'beta_gyr': -1.623e-01, 'P_pos_std': 2.73, 'P_vel_std': 0.52, 'P_orient_std': 0.029, 'P_acc_std': 2.852e-03, 'P_gyr_std': 5.651e-03}
