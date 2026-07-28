# -*- coding: utf-8 -*-
"""
iSAM2 Fixed-Lag (Marginalised) Online Smoother
==============================================
A bounded-window counterpart to the full-history incremental smoother in
``isam2_runner.py``.  It uses GTSAM's ``IncrementalFixedLagSmoother`` (from
``gtsam_unstable``) to keep only a sliding window of the most recent
``smoother_lag`` seconds (default 120 s = 2 min); nodes older than the lag are
**marginalised out** with the Schur complement, so state size, memory and
per-update cost stay bounded regardless of sequence length.

Everything else is identical to ``isam2_runner.py`` — same variables
(pose X, velocity V, bias B per GPS epoch), same factors (prior, CombinedImu,
GPS), same outage handling (no intermediate nodes; one preintegration factor
bridges the gap), same output schema.  This makes it a fair, apples-to-apples
comparison against the incremental smoother: only the back-end differs.

Why 2 min?  The lag must comfortably exceed the longest GNSS outage (≤60 s here)
so the pre-outage anchor and the accumulated IMU-bias / velocity information
remain inside the window across the blackout; 120 s leaves ample margin.

Instrumentation
---------------
For the incremental-vs-fixed-lag timing/compute study the output dict carries
two extra per-epoch arrays (ignored by ins_compare/ins_runner, which read a
fixed set of keys):
    update_ms : wall-time [ms] of each update()+calculateEstimate() block.
    n_vars    : number of variables in the smoother estimate (bounded here).

NOTE — install the conda-forge gtsam build (the pip wheel segfaults on numpy 2.x):
    conda install -c conda-forge gtsam
"""

import sys
import time
import numpy as np
import pymap3d as pm
from pathlib import Path
from math import sin, cos, atan2, asin

_SCRIPTS = Path(__file__).parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ── Lazy GTSAM import (fixed-lag lives in gtsam_unstable) ──────────────────────

def _import_gtsam():
    try:
        import gtsam
        from gtsam.symbol_shorthand import X, V, B
        from gtsam_unstable import (IncrementalFixedLagSmoother,
                                    FixedLagSmootherKeyTimestampMap)
        return (gtsam, X, V, B,
                IncrementalFixedLagSmoother, FixedLagSmootherKeyTimestampMap)
    except ImportError as e:
        raise ImportError(
            "GTSAM (with gtsam_unstable) is required for the fixed-lag smoother.\n"
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
    # Fixed-lag window
    'smoother_lag':      120.0,   # [s] sliding-window length (nodes older are marginalised)
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


def _timestamps(TSMap, pairs):
    """Build a FixedLagSmootherKeyTimestampMap from (key, time) pairs.

    The pybind binding takes a std::pair via insert((key, value)); a plain Python
    dict is not auto-converted.
    """
    ts = TSMap()
    for key, t in pairs:
        ts.insert((key, float(t)))
    return ts


# ── Public interface ──────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the 2-minute fixed-lag iSAM2 smoother and return navigation estimates.

    Identical to isam2_runner.run() except that the factor graph is solved by an
    IncrementalFixedLagSmoother that marginalises nodes older than
    ``smoother_lag`` seconds, bounding state size and per-update cost.

    Transparently dispatches to whichever environment actually has GTSAM —
    native in-process if importable here, otherwise bridged through the
    isolated /opt/conda-gtsam interpreter as a subprocess (see
    fgo_batch_runner.run()'s docstring for the full rationale; same pattern).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr,
                        and (extra, for the timing study) update_ms, n_vars.
    """
    try:
        import gtsam  # noqa: F401
    except ImportError:
        from ._conda_gtsam_bridge import run_via_conda_subprocess
        return run_via_conda_subprocess("isam2_fixedlag", nav_data, params, outage_config, use_3d_rotation)
    return _run_native(nav_data, params, outage_config, use_3d_rotation)


def _run_native(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """The original, in-process GTSAM implementation — see run()'s docstring."""
    (gtsam, X, V, B,
     IncrementalFixedLagSmoother, TSMap) = _import_gtsam()

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
    update_ms    = np.zeros(N)     # per-epoch update+estimate wall-time [ms]
    n_vars       = np.zeros(N, dtype=int)

    # ── Fixed-lag smoother setup ──────────────────────────────────────────────
    isam2_p = gtsam.ISAM2Params()
    isam2_p.setRelinearizeThreshold(p_cfg['isam2_relinearize_threshold'])
    isam2_p.relinearizeSkip = int(p_cfg['isam2_relinearize_skip'])
    smoother = IncrementalFixedLagSmoother(float(p_cfg['smoother_lag']), isam2_p)

    # ── IMU preintegration parameters (ENU Z-up frame) ────────────────────────
    pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    pim_params.setAccelerometerCovariance(
        np.eye(3) * p_cfg['acc_noise_sigma'] ** 2)
    pim_params.setGyroscopeCovariance(
        np.eye(3) * p_cfg['gyr_noise_sigma'] ** 2)
    pim_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    pim_params.setBiasAccCovariance(np.eye(3) * p_cfg['acc_bias_sigma'] ** 2)
    pim_params.setBiasOmegaCovariance(np.eye(3) * p_cfg['gyr_bias_sigma'] ** 2)
    pim_params.setBiasAccOmegaInit(np.eye(6) * 1e-8)

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

    # Timestamps for the fixed-lag window: the k=0 node lives at t=0.
    smoother.update(graph, values,
                    _timestamps(TSMap, [(X(0), 0.0), (V(0), 0.0), (B(0), 0.0)]))
    result = smoother.calculateEstimate()

    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    # GPS noise (position only — GPSFactor constrains translation of Pose3)
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
            t_k = (i + 1) / nav_data.sample_rate    # epoch time for the fixed-lag window

            # Predicted NavState as initial estimate for the new variables
            nav_pred = pim.predict(nav_state_prev, bias_prev)

            values.insert(X(k), nav_pred.pose())
            values.insert(V(k), nav_pred.velocity())
            values.insert(B(k), bias_prev)

            # IMU factor: connects (X(k-1), V(k-1), B(k-1)) → (X(k), V(k), B(k))
            graph.push_back(gtsam.CombinedImuFactor(
                X(k - 1), V(k - 1),
                X(k),     V(k),
                B(k - 1), B(k),
                pim,
            ))

            # GPS factor: constrains the translation of Pose3 at X(k)
            graph.push_back(gtsam.GPSFactor(X(k), p_gps[i + 1], gps_noise))

            # Fixed-lag incremental update.  Nodes with timestamp < t_k - lag are
            # marginalised out.  (Only a single update() per epoch — the extra
            # no-arg iterations used by plain ISAM2 are not available here.)
            ts = _timestamps(TSMap, [(X(k), t_k), (V(k), t_k), (B(k), t_k)])
            t0 = time.perf_counter()
            smoother.update(graph, values, ts)
            result = smoother.calculateEstimate()
            update_ms[i + 1] = (time.perf_counter() - t0) * 1e3
            n_vars[i + 1]    = result.size()

            graph  = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()

            # Extract optimised state at key k
            pose_k = result.atPose3(X(k))
            vel_k  = result.atVector(V(k))
            bias_k = result.atConstantBias(B(k))

            pose_prev      = pose_k
            vel_prev       = vel_k
            bias_prev      = bias_k
            nav_state_prev = gtsam.NavState(pose_k, vel_k)

            pim.resetIntegrationAndSetBias(bias_k)

            # Overwrite propagated output with corrected (smoothed) values
            p_out[i + 1] = pose_k.translation()
            v_out[i + 1] = vel_k
            R_k           = pose_k.rotation().matrix()
            r_out[i + 1] = _rbn_to_rpy(R_k)
            b_acc_out[i + 1] = bias_k.accelerometer()
            b_gyr_out[i + 1] = bias_k.gyroscope()

            # Marginal covariance — via the underlying ISAM2 (fixed-lag has no
            # direct marginalCovariance).  X(k) is the newest key, always in-window.
            try:
                isam = smoother.getISAM2()
                cov_pose = isam.marginalCovariance(X(k))  # 6×6 [rot | trans]
                cov_vel  = isam.marginalCovariance(V(k))  # 3×3
                cov_bias = isam.marginalCovariance(B(k))  # 6×6
                std_pos[i + 1]    = np.sqrt(np.maximum(np.diag(cov_pose[3:6, 3:6]), 0))
                std_vel[i + 1]    = np.sqrt(np.maximum(np.diag(cov_vel),            0))
                std_orient[i + 1] = np.sqrt(np.maximum(np.diag(cov_pose[0:3, 0:3]), 0))
                std_b_acc[i + 1]  = np.sqrt(np.maximum(np.diag(cov_bias[0:3, 0:3]), 0))
                std_b_gyr[i + 1]  = np.sqrt(np.maximum(np.diag(cov_bias[3:6, 3:6]), 0))
            except RuntimeError:
                # Indeterminate marginal (GTSAM RuntimeError), legitimate on the
                # first GPS steps — keep previous std_* there (reported uncertainty
                # only, trajectory unaffected). Narrowed from `except Exception` so
                # a real bug in the std extraction above is no longer masked.
                pass

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
        # Extra (timing study)
        'update_ms':   update_ms,
        'n_vars':      n_vars,
    }
