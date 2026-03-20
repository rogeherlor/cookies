# -*- coding: utf-8 -*-
"""
iSAM2 Online Smoother
=====================
Paper : Kaess et al., "iSAM2: Incremental Smoothing and Mapping Using the
        Bayes Tree", IJRR 2012.  https://doi.org/10.1177/0278364911430419
Library: GTSAM 4.x  (install with: conda install -c conda-forge gtsam)

NOTE — numpy compatibility
--------------------------
The pip wheel ``gtsam==4.2`` was compiled against numpy 1.x and will segfault
on numpy 2.x.  Install the conda-forge build instead:

    pip uninstall gtsam
    conda install -c conda-forge gtsam

Differences from the original paper
-------------------------------------
1. Coordinate frame: body = FLU (Forward-Left-Up), world = ENU (East-North-Up).
   Paper is frame-agnostic.  nav.accel_flu / nav.gyro_flu are passed directly to
   PreintegratedCombinedMeasurements; gravity is set to [0, 0, -9.81] in ENU via
   PreintegrationCombinedParams.MakeSharedU(9.81).

2. Sensor fusion: GPS position added as a GPSFactor at 1 Hz.  The original paper
   describes iSAM2 as a general-purpose framework; GPS integration is added here.

3. Causality enforced — online/causal smoother: only past measurements are used.
   Each iSAM2 update ingests one GPS interval's worth of IMU + one GPS fix and
   immediately produces a smoothed estimate.  The graph is never re-solved with
   future data, so the estimator is strictly causal and suitable for fair
   comparison against the forward-only Kalman filters.

4. Update rate: the iSAM2 factor graph is updated at 1 Hz (GPS rate).  Between
   GPS updates the optimised NavState is propagated forward at 100 Hz using
   pim.predict() to fill the per-sample output arrays — the same strategy used
   by the EKF-family filters.

5. use_3d_rotation flag: ignored.  GTSAM Pose3 operates in full SE(3) at all
   times; roll and pitch are never constrained, even in "2D mode".

6. Outage simulation: uses the same outage_config mechanism as every other
   filter.  GPS factors are simply omitted during the outage window; IMU
   preintegration continues and the Bayes tree stays connected via IMU
   constraints alone.

7. Marginal covariance: std_pos / std_vel / std_orient are extracted via
   isam.marginalCovariance() at 1 Hz and held constant until the next GPS
   update.  The EKF-family filters propagate the covariance matrix at every
   IMU step; iSAM2 does not maintain a per-step covariance natively.
"""

import sys
import numpy as np
import pymap3d as pm
from pathlib import Path
from math import sin, cos, atan2, asin

_SCRIPTS = Path(__file__).parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# ── Lazy GTSAM import (clear error if not available / wrong numpy) ─────────────

def _import_gtsam():
    try:
        import gtsam
        from gtsam.symbol_shorthand import X, V, B
        return gtsam, X, V, B
    except ImportError as e:
        raise ImportError(
            "GTSAM is required for the iSAM2 smoother.\n"
            "Install with:  conda install -c conda-forge gtsam\n"
            f"Original error: {e}"
        ) from e


# ── Default parameters ────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    # IMU noise (continuous-time, pre-integration)
    'acc_noise_sigma':   0.1,     # [m/s²/√s]  accelerometer white noise density
    'gyr_noise_sigma':   1e-3,    # [rad/s/√s] gyroscope white noise density
    'acc_bias_sigma':    1e-3,    # [m/s²/√Hz] accelerometer bias random-walk
    'gyr_bias_sigma':    1e-5,    # [rad/s/√Hz] gyroscope bias random-walk
    # GPS measurement noise
    'Rpos':              4.0,     # position std dev [m]; covariance = Rpos²·I₃
    # iSAM2 re-linearisation
    'isam2_relinearize_threshold': 0.1,
    'isam2_relinearize_skip':      1,
    # Initial state uncertainty (std dev)
    'P_pos_std':    1.0,     # [m]
    'P_vel_std':    0.3,     # [m/s]
    'P_orient_std': 0.1,     # [rad]
    'P_acc_std':    1e-2,    # [m/s²]  initial bias uncertainty
    'P_gyr_std':    1e-3,    # [rad/s]
}


# ── Rotation utilities ────────────────────────────────────────────────────────

def _euler_to_Rbn(rpy):
    """ZYX Euler angles (roll, pitch, yaw) → body-to-nav rotation matrix."""
    r, p, y = rpy
    cr, sr = cos(r), sin(r)
    cp, sp = cos(p), sin(p)
    cy, sy = cos(y), sin(y)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr           ],
    ])


def _rbn_to_rpy(Rbn):
    """Rotation matrix (body-to-nav) → ZYX Euler angles [roll, pitch, yaw]."""
    pitch = asin(-Rbn[2, 0])
    if abs(cos(pitch)) > 1e-8:
        roll = atan2(Rbn[2, 1], Rbn[2, 2])
        yaw  = atan2(Rbn[1, 0], Rbn[0, 0])
    else:
        roll = 0.0
        yaw  = atan2(-Rbn[0, 1], Rbn[1, 1])
    return np.array([roll, pitch, yaw])


def _in_outage(i, sample_rate, outage_cfg):
    """Return True if sample i is inside the GPS outage window."""
    if outage_cfg is None:
        return False
    t = i / sample_rate
    return outage_cfg['start'] <= t < outage_cfg['start'] + outage_cfg['duration']


def _mat_to_rot3(gtsam, R):
    """Create gtsam.Rot3 from a 3×3 numpy rotation matrix (element-by-element)."""
    return gtsam.Rot3(
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2],
        R[2, 0], R[2, 1], R[2, 2],
    )


# ── Public interface ──────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run iSAM2 online smoother and return navigation estimates.

    iSAM2 updates the factor graph at 1 Hz (every GPS measurement).  Between
    GPS updates, the current NavState is propagated forward with raw IMU at
    100 Hz to fill the output arrays.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s} for GPS
                         blackout.  GPS factors are omitted in this window.
        use_3d_rotation: Accepted for interface compatibility; always ignored —
                         iSAM2 always operates in full SE(3).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
    """
    gtsam, X, V, B = _import_gtsam()

    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    accel_flu = nav_data.accel_flu      # (N, 3) body-frame IMU [m/s²]
    gyro_flu  = nav_data.gyro_flu       # (N, 3) body-frame gyro [rad/s]
    orient    = nav_data.orient         # (N, 3) Euler [roll, pitch, yaw] [rad]
    vel_enu   = nav_data.vel_enu        # (N, 3) ENU velocity [m/s]
    lla       = nav_data.lla            # (N, 3) geodetic [lat, lon, alt]
    lla0      = nav_data.lla0
    N         = accel_flu.shape[0]
    Ts        = 1.0 / nav_data.sample_rate

    # GPS positions in ENU
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0], lla0[1], lla0[2],
    )
    p_gps = np.column_stack([e, n, u])     # (N, 3) ENU positions from GPS

    # ── Output arrays ─────────────────────────────────────────────────────────
    p_out        = np.zeros((N, 3))
    v_out        = np.zeros((N, 3))
    r_out        = np.zeros((N, 3))
    b_acc_out    = np.zeros((N, 3))
    b_gyr_out    = np.zeros((N, 3))
    std_pos      = np.zeros((N, 3))
    std_vel      = np.zeros((N, 3))
    std_orient   = np.zeros((N, 3))
    std_b_acc    = np.zeros((N, 3))
    std_b_gyr    = np.zeros((N, 3))

    # ── iSAM2 setup ───────────────────────────────────────────────────────────
    isam2_p = gtsam.ISAM2Params()
    isam2_p.setRelinearizeThreshold(p_cfg['isam2_relinearize_threshold'])
    isam2_p.relinearizeSkip = int(p_cfg['isam2_relinearize_skip'])
    isam = gtsam.ISAM2(isam2_p)

    # ── IMU preintegration parameters (ENU Z-up frame) ────────────────────────
    # MakeSharedU(g) sets n_gravity = [0, 0, -g] — correct for ENU frame.
    # nav.accel_flu is the raw body-frame accelerometer reading:
    #   a_measured = R_bn @ (a_true_ENU − g_ENU) + b_a
    # which is exactly what GTSAM's preintegrator expects.
    pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    pim_params.setAccelerometerCovariance(
        np.eye(3) * p_cfg['acc_noise_sigma'] ** 2)
    pim_params.setGyroscopeCovariance(
        np.eye(3) * p_cfg['gyr_noise_sigma'] ** 2)
    pim_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    pim_params.setBiasAccCovariance(np.eye(3) * p_cfg['acc_bias_sigma'] ** 2)
    pim_params.setBiasOmegaCovariance(np.eye(3) * p_cfg['gyr_bias_sigma'] ** 2)
    pim_params.setBiasAccOmegaInit(np.eye(6) * 1e-8)

    # Initial IMU bias (zero — will be estimated online)
    bias_prev = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedCombinedMeasurements(pim_params, bias_prev)

    # ── Initial state ─────────────────────────────────────────────────────────
    Rbn0  = _euler_to_Rbn(orient[0])
    rot0  = _mat_to_rot3(gtsam, Rbn0)
    pose_prev = gtsam.Pose3(rot0, p_gps[0])
    vel_prev  = vel_enu[0].copy()

    p_out[0]  = p_gps[0]
    v_out[0]  = vel_prev
    r_out[0]  = orient[0].copy()

    # ── Insert initial priors into factor graph ────────────────────────────────
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    values.insert(X(0), pose_prev)
    values.insert(V(0), vel_prev)
    values.insert(B(0), bias_prev)

    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'],
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
    ]))
    vel_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_vel_std'], p_cfg['P_vel_std'], p_cfg['P_vel_std'],
    ]))
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_acc_std'], p_cfg['P_acc_std'], p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'], p_cfg['P_gyr_std'], p_cfg['P_gyr_std'],
    ]))

    graph.push_back(gtsam.PriorFactorPose3(X(0), pose_prev, pose_noise))
    graph.push_back(gtsam.PriorFactorVector(V(0), vel_prev, vel_noise))
    graph.push_back(gtsam.PriorFactorConstantBias(B(0), bias_prev, bias_noise))

    isam.update(graph, values)
    result = isam.calculateEstimate()

    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # GPS noise (position only — iSAM2 GPSFactor constrains translation of Pose3)
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['Rpos'], p_cfg['Rpos'], p_cfg['Rpos'],
    ]))

    nav_state_prev = gtsam.NavState(pose_prev, vel_prev)
    k = 0   # variable key index (incremented at each GPS step)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for i in range(N - 1):
        # Integrate one IMU sample (body-frame accel & gyro)
        pim.integrateMeasurement(accel_flu[i], gyro_flu[i], Ts)

        # Propagate current NavState forward at IMU rate for output fill
        nav_prop = pim.predict(nav_state_prev, bias_prev)
        p_out[i + 1] = nav_prop.pose().translation()
        v_out[i + 1] = nav_prop.velocity()
        R_prop       = nav_prop.pose().rotation().matrix()
        r_out[i + 1] = _rbn_to_rpy(R_prop)
        # std_* filled at last GPS update (held constant between updates)
        std_pos[i + 1]    = std_pos[i]
        std_vel[i + 1]    = std_vel[i]
        std_orient[i + 1] = std_orient[i]
        std_b_acc[i + 1]  = std_b_acc[i]
        std_b_gyr[i + 1]  = std_b_gyr[i]
        b_acc_out[i + 1]  = b_acc_out[i]
        b_gyr_out[i + 1]  = b_gyr_out[i]

        # ── GPS update ────────────────────────────────────────────────────────
        if nav_data.gps_available[i + 1] and not _in_outage(i + 1, nav_data.sample_rate, outage_config):
            k += 1

            # Predicted NavState as initial estimate for the new variables
            nav_pred = pim.predict(nav_state_prev, bias_prev)

            values.insert(X(k), nav_pred.pose())
            values.insert(V(k), nav_pred.velocity())
            values.insert(B(k), bias_prev)

            # IMU factor: connects (X(k-1), V(k-1), B(k-1)) → (X(k), V(k), B(k))
            imu_factor = gtsam.CombinedImuFactor(
                X(k - 1), V(k - 1),
                X(k),     V(k),
                B(k - 1), B(k),
                pim,
            )
            graph.push_back(imu_factor)

            # GPS factor: constrains the translation of Pose3 at X(k)
            graph.push_back(gtsam.GPSFactor(
                X(k), p_gps[i + 1], gps_noise,
            ))

            # iSAM2 incremental update — only past measurements, strictly causal
            isam.update(graph, values)
            # Extra update iterations help convergence in curved trajectories
            isam.update()
            isam.update()
            result = isam.calculateEstimate()

            graph  = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()

            # Extract optimised state at key k
            pose_k = result.atPose3(X(k))
            vel_k  = result.atVector(V(k))
            bias_k = result.atConstantBias(B(k))

            # Update running state for next IMU propagation interval
            pose_prev      = pose_k
            vel_prev       = vel_k
            bias_prev      = bias_k
            nav_state_prev = gtsam.NavState(pose_k, vel_k)

            # Reset IMU preintegration for the next GPS interval
            pim.resetIntegrationAndSetBias(bias_k)

            # Overwrite propagated output with corrected (smoothed) values
            p_out[i + 1] = pose_k.translation()
            v_out[i + 1] = vel_k
            R_k           = pose_k.rotation().matrix()
            r_out[i + 1] = _rbn_to_rpy(R_k)
            b_a = bias_k.accelerometer()
            b_g = bias_k.gyroscope()
            b_acc_out[i + 1] = b_a
            b_gyr_out[i + 1] = b_g

            # Marginal covariance (called at GPS rate — ~1 Hz)
            # Pose3 covariance in GTSAM tangent space: [rot(3) | trans(3)]
            try:
                cov_pose = isam.marginalCovariance(X(k))  # 6×6
                cov_vel  = isam.marginalCovariance(V(k))  # 3×3
                cov_bias = isam.marginalCovariance(B(k))  # 6×6
                std_pos[i + 1]    = np.sqrt(np.maximum(np.diag(cov_pose[3:6, 3:6]), 0))
                std_vel[i + 1]    = np.sqrt(np.maximum(np.diag(cov_vel),            0))
                std_orient[i + 1] = np.sqrt(np.maximum(np.diag(cov_pose[0:3, 0:3]), 0))
                std_b_acc[i + 1]  = np.sqrt(np.maximum(np.diag(cov_bias[0:3, 0:3]), 0))
                std_b_gyr[i + 1]  = np.sqrt(np.maximum(np.diag(cov_bias[3:6, 3:6]), 0))
            except Exception:
                pass    # marginal covariance may fail on first steps; keep zeros

    return {
        'p':           p_out,
        'v':           v_out,
        'r':           r_out,
        'bias_acc':    b_acc_out,
        'bias_gyr':    b_gyr_out,
        'std_pos':     std_pos,
        'std_vel':     std_vel,
        'std_orient':  std_orient,
        'std_bias_acc': std_b_acc,
        'std_bias_gyr': std_b_gyr,
    }
