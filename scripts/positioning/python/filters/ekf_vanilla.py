# -*- coding: utf-8 -*-
"""
EKF Vanilla — Euler-angle Error-State Extended Kalman Filter.
GPS position update only.

Reference:
    Groves, P.D., "Principles of GNSS, Inertial, and Multisensor Integrated
    Navigation Systems", 2nd ed., Artech House, 2013. Ch. 14.

State vector (15 elements — error state):
    dx[0:3]   — position error  δp  in ENU  [m]
    dx[3:6]   — velocity error  δv  in ENU  [m/s]
    dx[6:9]   — attitude error  δφ  in navigation frame  [rad]
    dx[9:12]  — accelerometer bias  b_a  in FLU body frame  [m/s²]
    dx[12:15] — gyroscope bias  b_g  in FLU body frame  [rad/s]

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Euler angles: ZYX convention, stored as [roll, pitch, yaw]  [rad]
    Attitude error δφ: defined in the navigation frame (Groves convention)
"""
import numpy as np
import pymap3d as pm
from math import sin, cos, tan

# ── Physical constants ─────────────────────────────────────────────────────────
GRAVITY = np.array([0.0, 0.0, -9.81])   # ENU [m/s²]

# ── Default parameters ─────────────────────────────────────────────────────────
DEFAULT_PARAMS = {
    # Process noise Q
    'Qpos':      5.312e-06,
    'Qvel':      4.702e-06,
    'QorientXY': 0.0002,
    'QorientZ':  0.2,
    'Qacc':      0.1,
    'QgyrXY':    0.0001,
    'QgyrZ':     0.1,

    # Measurement noise R (GPS position)
    'Rpos': 67.79,

    # Gauss-Markov decay coefficients (must be negative for stability)
    'beta_acc': -1.910e-06,
    'beta_gyr': -7.077e-02,

    # Initial covariance diagonal (1-sigma values)
    'P_pos_std':    0.23,
    'P_vel_std':    0.17,
    'P_orient_std': 0.239,
    'P_acc_std':    0.01,
    'P_gyr_std':    0.001,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _skew(v):
    """3×3 skew-symmetric matrix of vector v."""
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


def _euler_to_Rbn(rpy):
    """
    Rotation matrix R_bn (body→navigation) from ZYX Euler angles [roll, pitch, yaw].
    Maps FLU body vectors to ENU navigation vectors.
    """
    roll, pitch, yaw = rpy
    cr, sr = cos(roll),  sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw),   sin(yaw)
    return np.array([
        [cy*cp,   cy*sp*sr - sy*cr,   cy*sp*cr + sy*sr],
        [sy*cp,   sy*sp*sr + cy*cr,   sy*sp*cr - cy*sr],
        [-sp,     cp*sr,              cp*cr            ],
    ])


def _euler_rate_matrix(rpy):
    """
    Euler-rate matrix T such that ṙpy = T @ ω_body.
    Gyro measures at the same time, but euler angles are applied sequentially (ZYX).
    ZYX convention: [roll_dot, pitch_dot, yaw_dot] = T @ [ω_x, ω_y, ω_z].
    """
    roll, pitch, _ = rpy
    sr, cr = sin(roll), cos(roll)
    cp = cos(pitch)
    tp = sin(pitch) / cp if abs(cp) > 1e-6 else 0.0
    rcp = 1.0 / cp if abs(cp) > 1e-6 else 0.0
    return np.array([
        [1.0,  sr * tp,   cr * tp],
        [0.0,  cr,        -sr    ],
        [0.0,  sr * rcp,  cr * rcp],
    ])


# ── Main filter ────────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the vanilla Euler-angle EKF (GPS position update only).

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s} for GPS blackout.
        use_3d_rotation: True → full roll/pitch/yaw; False → yaw-only (2D flat-earth).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
        All arrays are Nx3, ENU/FLU where applicable.
    """
    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    accel_flu = nav_data.accel_flu
    gyro_flu  = nav_data.gyro_flu
    lla       = nav_data.lla
    orient    = nav_data.orient
    vel_enu   = nav_data.vel_enu
    frecIMU   = nav_data.sample_rate
    lla0      = nav_data.lla0

    g  = GRAVITY
    Ts = 1.0 / frecIMU
    NN = lla.shape[0]

    # GPS outage window [samples]
    if outage_config is None:
        A, B = 0, 0
    else:
        A = int(outage_config['start'] * frecIMU)
        B = int((outage_config['start'] + outage_config['duration']) * frecIMU)

    # ── Output arrays ──────────────────────────────────────────────────────────
    pos        = np.zeros((NN, 3))
    vel        = np.zeros((NN, 3))
    rpy_out    = np.zeros((NN, 3))
    b_acc_out  = np.zeros((NN, 3))
    b_gyr_out  = np.zeros((NN, 3))
    std_pos    = np.zeros((NN, 3))
    std_vel    = np.zeros((NN, 3))
    std_orient = np.zeros((NN, 3))
    std_b_acc  = np.zeros((NN, 3))
    std_b_gyr  = np.zeros((NN, 3))

    # ── Initial state ──────────────────────────────────────────────────────────
    pos[0, :] = pm.geodetic2enu(lla[0,0], lla[0,1], lla[0,2], lla0[0], lla0[1], lla0[2])
    vel[0, :] = vel_enu[0, :]
    rpy_out[0, :] = orient[0, :]

    pIMU = pos[0, :].copy()
    vIMU = vel[0, :].copy()
    rpy  = orient[0, :].copy()   # [roll, pitch, yaw]
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)          # error state (stays ~0 between updates)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    # ── Process noise ──────────────────────────────────────────────────────────
    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.eye(3) * (p_cfg['Qpos'] * Ts**2)
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.diag([p_cfg['QorientXY'], p_cfg['QorientXY'], p_cfg['QorientZ']])
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc'] * Ts)
    Q[12:15, 12:15] = np.diag([p_cfg['QgyrXY'], p_cfg['QgyrXY'], p_cfg['QgyrZ']]) * Ts

    # ── Initial covariance ─────────────────────────────────────────────────────
    P = np.diag([
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'] * 2,
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    # ── Measurement model (GPS position) ───────────────────────────────────────
    R_pos = np.eye(3) * p_cfg['Rpos']
    H = np.zeros((3, 15))
    H[0:3, 0:3] = np.eye(3)    # observe δp directly

    # ── Main loop ──────────────────────────────────────────────────────────────
    for i in range(NN - 1):

        # 1. Bias-corrected IMU measurements
        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        # 2. Nominal-state propagation (Euler angle mechanization)
        if use_3d_rotation:
            Rbn = _euler_to_Rbn(rpy)
            T   = _euler_rate_matrix(rpy)
            rpy = rpy + Ts * (T @ omega_b)
        else:
            # 2D mode: yaw-only rotation, flat-earth
            yaw = rpy[2]
            cy, sy = cos(yaw), sin(yaw)
            Rbn = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            rpy[2] = rpy[2] + Ts * omega_b[2]

        # Wrap yaw to (−π, π]
        rpy[2] = (rpy[2] + np.pi) % (2.0 * np.pi) - np.pi

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # 3. Error-state transition matrix F  (15-state, flat-Earth ENU — "Groves/ENU" formulation)
        #
        #    This is the φ-angle (nav-frame) error model from Groves (2013) Ch. 12,
        #    simplified for short-range navigation: Earth-rate, transport-rate and
        #    gravity-gradient terms are dropped, and position error is kept in
        #    Cartesian ENU metres instead of geodetic (lat/lon/h).
        #    Sensor biases follow a Gauss-Markov model (Groves eq. 3.3).
        #
        #    δṗ = δv                          position error integrates velocity error
        #    δv̇ = −[f^n ×] δφ + Rbn @ b_a   velocity error from attitude tilt and accel bias
        #    δφ̇ = −Rbn @ b_g                 attitude error from gyro bias
        #    ḃ_a = −β_acc · b_a              accel bias decays with time constant 1/β_acc
        #    ḃ_g = −β_gyr · b_g              gyro  bias decays with time constant 1/β_gyr
        F = np.zeros((15, 15))
        F[0:3,   3:6]   = np.eye(3)               # position  ← velocity error
        F[3:6,   6:9]   = -_skew(accENU)          # velocity  ← attitude error (via skew of nav-frame specific force)
        F[3:6,   9:12]  = Rbn                      # velocity  ← accel bias (body→nav)
        F[6:9,   12:15] = -Rbn                     # attitude  ← gyro bias (body→nav)
        F[9:12,  9:12]  = beta_acc * np.eye(3)    # accel bias Gauss-Markov decay (β already negative)
        F[12:15, 12:15] = beta_gyr * np.eye(3)    # gyro  bias Gauss-Markov decay (β already negative)

        Fd = np.eye(15) + F * Ts

        # Prediction step
        P  = Fd @ P @ Fd.T + Q
        dx = Fd @ dx

        # 4. GPS position update (sparse: 3 observations)
        gps_ok    = nav_data.gps_available[i]
        not_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_ok and not_outage:
            p_gps = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            z     = p_gps - pIMU
            innov = z - H @ dx

            S     = H @ P @ H.T + R_pos
            S_reg = S + 1e-9 * np.eye(3)
            K     = np.linalg.solve(S_reg, H @ P).T   # stable solve, 15×3

            dx = dx + K @ innov
            IKH = np.eye(15) - K @ H
            P   = IKH @ P @ IKH.T + K @ R_pos @ K.T  # Joseph form
            P   = 0.5 * (P + P.T)

            # Error injection (nav-frame attitude error added directly to rpy)
            pIMU   += dx[0:3]
            vIMU   += dx[3:6]
            rpy    += dx[6:9]
            b_a    += dx[9:12]
            b_g    += dx[12:15]
            rpy[2]  = (rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
            dx[:]   = 0.0

        # 5. Store outputs
        pos[i+1, :]       = pIMU
        vel[i+1, :]       = vIMU
        rpy_out[i+1, :]   = rpy
        b_acc_out[i+1, :] = b_a
        b_gyr_out[i+1, :] = b_g
        std_pos[i+1, :]      = np.sqrt(np.maximum(np.diag(P[0:3,   0:3]),   0.0))
        std_vel[i+1, :]      = np.sqrt(np.maximum(np.diag(P[3:6,   3:6]),   0.0))
        std_orient[i+1, :]   = np.sqrt(np.maximum(np.diag(P[6:9,   6:9]),   0.0))
        std_b_acc[i+1, :]    = np.sqrt(np.maximum(np.diag(P[9:12,  9:12]),  0.0))
        std_b_gyr[i+1, :]    = np.sqrt(np.maximum(np.diag(P[12:15, 12:15]), 0.0))

    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_acc_out, 'bias_gyr': b_gyr_out,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_b_acc, 'std_bias_gyr': std_b_gyr,
    }
