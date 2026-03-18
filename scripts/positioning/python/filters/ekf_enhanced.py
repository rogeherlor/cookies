# -*- coding: utf-8 -*-
"""
EKF Enhanced — Euler-angle EKF with Non-Holonomic Constraints (NHC) and
Zero-Velocity Updates (ZUPT).

GPS position update + NHC + ZUPT.

References:
    Groves, P.D., "Principles of GNSS, Inertial, and Multisensor Integrated
    Navigation Systems", 2nd ed., Artech House, 2013. Ch. 14.

    Dissanayake, G. et al., "The aiding of a low-cost strapdown inertial
    measurement unit using vehicle model constraints for land vehicle
    applications", IEEE Transactions on Vehicular Technology, 2001.
    DOI: 10.1109/25.892572

    Foxlin, E., "Pedestrian Tracking with Shoe-Mounted Inertial Sensors",
    IEEE Computer Graphics & Applications, vol. 25, no. 6, 2005.
    DOI: 10.1109/MCG.2005.140

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
"""
import numpy as np
import pymap3d as pm
from math import sin, cos

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

    # NHC — Non-Holonomic Constraints (Dissanayake 2001)
    # Vehicle cannot slide sideways or fly: v_lateral ≈ v_vertical ≈ 0 in body frame.
    'Rnhc': 0.1,      # measurement noise [(m/s)²]

    # ZUPT — Zero-Velocity Update (Foxlin 2005)
    # When the vehicle is stationary the ENU velocity should be ~0.
    'Rzupt':                0.01,    # measurement noise [(m/s)²]
    'zupt_accel_threshold': 0.3,     # |‖acc_b‖ − 9.81| threshold [m/s²]
    'zupt_gyro_threshold':  0.05,    # ‖ω_b‖ threshold [rad/s]
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _skew(v):
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


def _euler_to_Rbn(rpy):
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
    roll, pitch, _ = rpy
    sr, cr = sin(roll), cos(roll)
    cp = cos(pitch)
    tp  = sin(pitch) / cp if abs(cp) > 1e-6 else 0.0
    rcp = 1.0 / cp         if abs(cp) > 1e-6 else 0.0
    return np.array([
        [1.0,  sr * tp,   cr * tp ],
        [0.0,  cr,        -sr     ],
        [0.0,  sr * rcp,  cr * rcp],
    ])


# ── Main filter ────────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run EKF with NHC + ZUPT enhancements.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s} for GPS blackout.
        use_3d_rotation: True → full roll/pitch/yaw; False → yaw-only (2D).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
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
    rpy  = orient[0, :].copy()
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.eye(3) * (p_cfg['Qpos'] * Ts**2)
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.diag([p_cfg['QorientXY'], p_cfg['QorientXY'], p_cfg['QorientZ']])
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc'] * Ts)
    Q[12:15, 12:15] = np.diag([p_cfg['QgyrXY'], p_cfg['QgyrXY'], p_cfg['QgyrZ']]) * Ts

    P = np.diag([
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'] * 2,
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_pos  = np.eye(3) * p_cfg['Rpos']
    R_nhc  = np.eye(2) * p_cfg['Rnhc']
    R_zupt = np.eye(3) * p_cfg['Rzupt']

    H_pos = np.zeros((3, 15))
    H_pos[0:3, 0:3] = np.eye(3)

    # ── Main loop ──────────────────────────────────────────────────────────────
    for i in range(NN - 1):

        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        # Nominal-state propagation
        if use_3d_rotation:
            Rbn = _euler_to_Rbn(rpy)
            T   = _euler_rate_matrix(rpy)
            rpy = rpy + Ts * (T @ omega_b)
        else:
            yaw = rpy[2]
            cy, sy = cos(yaw), sin(yaw)
            Rbn = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            rpy[2] = rpy[2] + Ts * omega_b[2]

        rpy[2] = (rpy[2] + np.pi) % (2.0 * np.pi) - np.pi

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # Error-state transition matrix F (nav-frame attitude error, Groves Ch. 14)
        F = np.zeros((15, 15))
        F[0:3,   3:6]   = np.eye(3)
        F[3:6,   6:9]   = -_skew(accENU)
        F[3:6,   9:12]  = Rbn
        F[6:9,   12:15] = -Rbn
        F[9:12,  9:12]  = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + F * Ts
        P  = Fd @ P @ Fd.T + Q
        dx = Fd @ dx

        update_occurred = False

        # ── A. GPS Position Update ─────────────────────────────────────────────
        gps_ok     = nav_data.gps_available[i]
        not_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_ok and not_outage:
            p_gps = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            z     = p_gps - pIMU
            innov = z - H_pos @ dx

            S     = H_pos @ P @ H_pos.T + R_pos
            S_reg = S + 1e-9 * np.eye(3)
            K     = np.linalg.solve(S_reg, H_pos @ P).T   # 15×3, stable solve
            dx    = dx + K @ innov
            IKH   = np.eye(15) - K @ H_pos
            P     = IKH @ P @ IKH.T + K @ R_pos @ K.T    # Joseph form
            P     = 0.5 * (P + P.T)
            update_occurred = True

        # ── B. Non-Holonomic Constraints (NHC) ────────────────────────────────
        # Vehicle cannot slide sideways or fly vertically.
        # Measurement: v_lateral = v_vertical = 0  in body frame.
        # H maps [δv(nav), δφ(nav)] to [δv_lateral, δv_vertical] in body frame.
        # (Dissanayake 2001; H derivation: δv_body = Rnb @ δv + skew(v_body) @ δφ)
        Rnb    = Rbn.T
        v_body = Rnb @ vIMU
        z_nhc  = -v_body[1:3]   # target: lateral and vertical body speed = 0

        H_v_nhc     = Rnb[1:3, :]            # 2×3 velocity block
        H_theta_nhc = _skew(v_body)[1:3, :]  # 2×3 attitude block
        H_nhc       = np.hstack((H_v_nhc, H_theta_nhc))  # 2×6

        innov_nhc = z_nhc - H_nhc @ dx[3:9]
        P_36      = P[3:9, 3:9]
        S_nhc     = H_nhc @ P_36 @ H_nhc.T + R_nhc
        S_nhc_reg = S_nhc + 1e-9 * np.eye(2)
        K_nhc     = np.linalg.solve(S_nhc_reg, H_nhc @ P[3:9, :]).T  # 15×2

        dx = dx + K_nhc @ innov_nhc
        H_nhc_full = np.zeros((2, 15)); H_nhc_full[:, 3:9] = H_nhc
        IKH_nhc    = np.eye(15) - K_nhc @ H_nhc_full
        P  = IKH_nhc @ P @ IKH_nhc.T + K_nhc @ R_nhc @ K_nhc.T      # Joseph form
        P  = 0.5 * (P + P.T)
        update_occurred = True

        # ── C. Zero-Velocity Update (ZUPT) ────────────────────────────────────
        # Trigger: near-zero specific force deviation and near-zero gyro.
        accel_dev = abs(np.linalg.norm(acc_b) - 9.81)
        gyro_mag  = np.linalg.norm(omega_b)
        speed     = np.linalg.norm(vIMU)

        if (accel_dev < p_cfg['zupt_accel_threshold'] and
                gyro_mag  < p_cfg['zupt_gyro_threshold'] and
                speed < 1.0):
            z_zupt = -vIMU
            innov_zupt = z_zupt - dx[3:6]

            S_zupt     = P[3:6, 3:6] + R_zupt
            S_zupt_reg = S_zupt + 1e-9 * np.eye(3)
            K_zupt     = np.linalg.solve(S_zupt_reg, P[3:6, :]).T     # 15×3
            dx = dx + K_zupt @ innov_zupt
            H_zupt = np.zeros((3, 15)); H_zupt[:, 3:6] = np.eye(3)
            IKH_zupt = np.eye(15) - K_zupt @ H_zupt
            P  = IKH_zupt @ P @ IKH_zupt.T + K_zupt @ R_zupt @ K_zupt.T  # Joseph form
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # ── Error injection and reset ──────────────────────────────────────────
        if update_occurred:
            pIMU   += dx[0:3]
            vIMU   += dx[3:6]
            rpy    += dx[6:9]
            b_a    += dx[9:12]
            b_g    += dx[12:15]
            rpy[2]  = (rpy[2] + np.pi) % (2.0 * np.pi) - np.pi
            dx[:]   = 0.0

        # Store outputs
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
