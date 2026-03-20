# -*- coding: utf-8 -*-
"""
FGO-Batch — Batch Factor Graph Optimisation Ground Truth
=========================================================
Builds the same GTSAM factor graph as the iSAM2 online smoother
(PriorFactors + CombinedImuFactor + GPSFactor) but solves it **offline**
over the entire dataset in a single Levenberg-Marquardt pass.

Key differences vs iSAM2 (isam2_runner.py):
  - Non-causal: every GPS measurement is in the graph before solving starts,
    so the solution sees "the future" and is therefore unsuitable as a filter
    but provides the best possible non-causal reference trajectory.
  - outage_config is intentionally **ignored** — GPS factors are never removed.
    The batch solution is the ground truth, not a tested filter.
  - Covariances come from gtsam.Marginals (batch) instead of
    isam.marginalCovariance (incremental).

Output schema identical to isam2_runner.run() so it plugs into ins_compare.py
without further changes.
"""

import sys
import numpy as np
import pymap3d as pm
from pathlib import Path
from math import sin, cos, atan2, asin

_SCRIPTS = Path(__file__).parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ── Lazy GTSAM import ─────────────────────────────────────────────────────────

def _import_gtsam():
    try:
        import gtsam
        from gtsam.symbol_shorthand import X, V, B
        return gtsam, X, V, B
    except ImportError as e:
        raise ImportError(
            "GTSAM is required for the FGO-Batch smoother.\n"
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
    # Initial state uncertainty (std dev)
    'P_pos_std':    1.0,     # [m]
    'P_vel_std':    0.3,     # [m/s]
    'P_orient_std': 0.1,     # [rad]
    'P_acc_std':    1e-2,    # [m/s²]  initial bias uncertainty
    'P_gyr_std':    1e-3,    # [rad/s]
}


# ── Rotation utilities (identical to isam2_runner) ────────────────────────────

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


def _mat_to_rot3(gtsam, R):
    """Create gtsam.Rot3 from a 3×3 numpy rotation matrix."""
    return gtsam.Rot3(
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2],
        R[2, 0], R[2, 1], R[2, 2],
    )


# ── Public interface ──────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run FGO-Batch smoother and return full-rate navigation estimates.

    Builds a GTSAM factor graph with all GPS measurements (outage_config is
    intentionally ignored — this is the ground truth, not a filter) and
    solves it in one Levenberg-Marquardt pass.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Accepted for interface compatibility; always ignored.
        use_3d_rotation: Accepted for interface compatibility; always ignored —
                         GTSAM Pose3 always operates in full SE(3).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
    """
    gtsam, X, V, B = _import_gtsam()

    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    accel_flu = nav_data.accel_flu      # (N, 3)
    gyro_flu  = nav_data.gyro_flu       # (N, 3)
    orient    = nav_data.orient         # (N, 3) Euler [roll, pitch, yaw]
    vel_enu   = nav_data.vel_enu        # (N, 3)
    lla       = nav_data.lla            # (N, 3)
    lla0      = nav_data.lla0
    N         = accel_flu.shape[0]
    Ts        = 1.0 / nav_data.sample_rate

    # GPS positions in ENU
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0], lla0[1], lla0[2],
    )
    p_gps = np.column_stack([e, n, u])     # (N, 3)

    # ── Output arrays ─────────────────────────────────────────────────────────
    p_out      = np.zeros((N, 3))
    v_out      = np.zeros((N, 3))
    r_out      = np.zeros((N, 3))
    b_acc_out  = np.zeros((N, 3))
    b_gyr_out  = np.zeros((N, 3))
    std_pos    = np.zeros((N, 3))
    std_vel    = np.zeros((N, 3))
    std_orient = np.zeros((N, 3))
    std_b_acc  = np.zeros((N, 3))
    std_b_gyr  = np.zeros((N, 3))

    # ── IMU preintegration parameters ─────────────────────────────────────────
    pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    pim_params.setAccelerometerCovariance(np.eye(3) * p_cfg['acc_noise_sigma'] ** 2)
    pim_params.setGyroscopeCovariance(np.eye(3) * p_cfg['gyr_noise_sigma'] ** 2)
    pim_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    pim_params.setBiasAccCovariance(np.eye(3) * p_cfg['acc_bias_sigma'] ** 2)
    pim_params.setBiasOmegaCovariance(np.eye(3) * p_cfg['gyr_bias_sigma'] ** 2)
    pim_params.setBiasAccOmegaInit(np.eye(6) * 1e-8)

    bias_zero = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedCombinedMeasurements(pim_params, bias_zero)

    # ── Noise models ──────────────────────────────────────────────────────────
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([
        p_cfg['P_orient_std']] * 3 + [p_cfg['P_pos_std']] * 3))
    vel_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([p_cfg['P_vel_std']] * 3))
    bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
        [p_cfg['P_acc_std']] * 3 + [p_cfg['P_gyr_std']] * 3))
    gps_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([p_cfg['Rpos']] * 3))

    # ── Initial state ─────────────────────────────────────────────────────────
    Rbn0      = _euler_to_Rbn(orient[0])
    rot0      = _mat_to_rot3(gtsam, Rbn0)
    pose_init = gtsam.Pose3(rot0, p_gps[0])
    vel_init  = vel_enu[0].copy()

    # ── Phase 1: build full factor graph ──────────────────────────────────────
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # Priors at k=0
    values.insert(X(0), pose_init)
    values.insert(V(0), vel_init)
    values.insert(B(0), bias_zero)
    graph.push_back(gtsam.PriorFactorPose3(X(0), pose_init, pose_noise))
    graph.push_back(gtsam.PriorFactorVector(V(0), vel_init, vel_noise))
    graph.push_back(gtsam.PriorFactorConstantBias(B(0), bias_zero, bias_noise))

    # Forward pass: accumulate IMU, add a new node at each GPS epoch
    # epoch_samples[k] = sample index where variable X(k) lives
    epoch_samples = [0]   # X(0) lives at sample 0

    nav_state_prev = gtsam.NavState(pose_init, vel_init)
    bias_prev      = bias_zero
    k = 0

    for i in range(N - 1):
        pim.integrateMeasurement(accel_flu[i], gyro_flu[i], Ts)

        # GPS available at sample i+1? (no outage filtering — this is GT)
        if nav_data.gps_available[i + 1]:
            k += 1
            nav_pred = pim.predict(nav_state_prev, bias_prev)

            values.insert(X(k), nav_pred.pose())
            values.insert(V(k), nav_pred.velocity())
            values.insert(B(k), bias_prev)

            graph.push_back(gtsam.CombinedImuFactor(
                X(k - 1), V(k - 1),
                X(k),     V(k),
                B(k - 1), B(k),
                pim,
            ))
            graph.push_back(gtsam.GPSFactor(X(k), p_gps[i + 1], gps_noise))

            epoch_samples.append(i + 1)
            nav_state_prev = nav_pred
            pim.resetIntegrationAndSetBias(bias_prev)

    K = k   # total number of GPS variables (0..K)

    # ── Phase 2: batch solve ──────────────────────────────────────────────────
    lm_params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    result    = optimizer.optimize()

    # ── Phase 3: reconstruct full-rate trajectory ─────────────────────────────
    # Replay IMU preintegration between consecutive GPS epochs using the
    # optimised states, filling p_out/v_out/r_out at every IMU sample.

    # First sample from the initial (k=0) estimate
    pose0 = result.atPose3(X(0))
    p_out[0] = pose0.translation()
    v_out[0] = result.atVector(V(0))
    r_out[0] = _rbn_to_rpy(pose0.rotation().matrix())
    b0 = result.atConstantBias(B(0))
    b_acc_out[0] = b0.accelerometer()
    b_gyr_out[0] = b0.gyroscope()

    for ki in range(K):
        i_start = epoch_samples[ki]
        i_end   = epoch_samples[ki + 1]

        pose_s  = result.atPose3(X(ki))
        vel_s   = result.atVector(V(ki))
        bias_ki = result.atConstantBias(B(ki))

        pim.resetIntegrationAndSetBias(bias_ki)
        nav_s = gtsam.NavState(pose_s, vel_s)

        for i in range(i_start, i_end):
            pim.integrateMeasurement(accel_flu[i], gyro_flu[i], Ts)
            nav_prop = pim.predict(nav_s, bias_ki)
            p_out[i + 1] = nav_prop.pose().translation()
            v_out[i + 1] = nav_prop.velocity()
            r_out[i + 1] = _rbn_to_rpy(nav_prop.pose().rotation().matrix())
            b_acc_out[i + 1] = bias_ki.accelerometer()
            b_gyr_out[i + 1] = bias_ki.gyroscope()

        # Overwrite GPS epoch sample with exact optimised state
        pose_e = result.atPose3(X(ki + 1))
        vel_e  = result.atVector(V(ki + 1))
        bias_e = result.atConstantBias(B(ki + 1))
        p_out[i_end] = pose_e.translation()
        v_out[i_end] = vel_e
        r_out[i_end] = _rbn_to_rpy(pose_e.rotation().matrix())
        b_acc_out[i_end] = bias_e.accelerometer()
        b_gyr_out[i_end] = bias_e.gyroscope()

    # Fill any trailing samples after the last GPS epoch
    last_sample = epoch_samples[K]
    if last_sample < N - 1:
        pose_last = result.atPose3(X(K))
        vel_last  = result.atVector(V(K))
        bias_last = result.atConstantBias(B(K))
        pim.resetIntegrationAndSetBias(bias_last)
        nav_last = gtsam.NavState(pose_last, vel_last)
        for i in range(last_sample, N - 1):
            pim.integrateMeasurement(accel_flu[i], gyro_flu[i], Ts)
            nav_prop = pim.predict(nav_last, bias_last)
            p_out[i + 1] = nav_prop.pose().translation()
            v_out[i + 1] = nav_prop.velocity()
            r_out[i + 1] = _rbn_to_rpy(nav_prop.pose().rotation().matrix())
            b_acc_out[i + 1] = bias_last.accelerometer()
            b_gyr_out[i + 1] = bias_last.gyroscope()

    # ── Covariance from batch marginals ───────────────────────────────────────
    try:
        marginals = gtsam.Marginals(graph, result)
        for ki in range(K + 1):
            i_ep = epoch_samples[ki]
            try:
                cov_pose = marginals.marginalCovariance(X(ki))   # 6×6 [rot|trans]
                cov_vel  = marginals.marginalCovariance(V(ki))   # 3×3
                cov_bias = marginals.marginalCovariance(B(ki))   # 6×6
                sp  = np.sqrt(np.maximum(np.diag(cov_pose[3:6, 3:6]), 0))
                sv  = np.sqrt(np.maximum(np.diag(cov_vel),            0))
                so  = np.sqrt(np.maximum(np.diag(cov_pose[0:3, 0:3]), 0))
                sba = np.sqrt(np.maximum(np.diag(cov_bias[0:3, 0:3]), 0))
                sbg = np.sqrt(np.maximum(np.diag(cov_bias[3:6, 3:6]), 0))
            except Exception:
                continue

            # Hold constant until next GPS epoch
            i_next = epoch_samples[ki + 1] if ki < K else N
            std_pos[i_ep:i_next]    = sp
            std_vel[i_ep:i_next]    = sv
            std_orient[i_ep:i_next] = so
            std_b_acc[i_ep:i_next]  = sba
            std_b_gyr[i_ep:i_next]  = sbg
    except Exception:
        pass   # covariance is optional; zeros are safe defaults

    return {
        'p':            p_out,
        'v':            v_out,
        'r':            r_out,
        'bias_acc':     b_acc_out,
        'bias_gyr':     b_gyr_out,
        'std_pos':      std_pos,
        'std_vel':      std_vel,
        'std_orient':   std_orient,
        'std_bias_acc': std_b_acc,
        'std_bias_gyr': std_b_gyr,
    }
