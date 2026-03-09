# -*- coding: utf-8 -*-
"""
ESKF Enhanced — Quaternion ESKF with Non-Holonomic Constraints (NHC) and
Zero-Velocity Updates (ZUPT).

GPS position update + NHC + ZUPT using sparse Kalman gain updates.

References:
    Solà, J., "Quaternion kinematics for the error-state Kalman filter",
    arXiv:1711.02508, 2017.  https://arxiv.org/abs/1711.02508

    Dissanayake, G. et al., "The aiding of a low-cost strapdown inertial
    measurement unit using vehicle model constraints for land vehicle
    applications", IEEE Transactions on Vehicular Technology, 2001.
    DOI: 10.1109/25.892572

    Foxlin, E., "Pedestrian Tracking with Shoe-Mounted Inertial Sensors",
    IEEE Computer Graphics & Applications, vol. 25, no. 6, 2005.
    DOI: 10.1109/MCG.2005.140

State vector (15 elements — error state, body-frame attitude):
    dx[0:3]   — position error  δp  in ENU  [m]
    dx[3:6]   — velocity error  δv  in ENU  [m/s]
    dx[6:9]   — attitude error  δφ  in FLU body frame  [rad]
    dx[9:12]  — accelerometer bias  b_a  in FLU body frame  [m/s²]
    dx[12:15] — gyroscope bias  b_g  in FLU body frame  [rad/s]

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Quaternion: Hamilton convention  q = [w, x, y, z]
"""
import numpy as np
import pymap3d as pm

GRAVITY = np.array([0.0, 0.0, -9.81])

DEFAULT_PARAMS = {
    'Qpos':      5.312e-06,
    'Qvel':      4.702e-06,
    'QorientXY': 0.0002,
    'QorientZ':  0.2,
    'Qacc':      0.1,
    'QgyrXY':    0.0001,
    'QgyrZ':     0.1,
    'Rpos':      67.79,
    'beta_acc':  -1.910e-06,
    'beta_gyr':  -7.077e-02,
    'P_pos_std':    0.23,
    'P_vel_std':    0.17,
    'P_orient_std': 0.239,
    'P_acc_std':    0.01,
    'P_gyr_std':    0.001,
    # NHC (Dissanayake 2001)
    'Rnhc': 0.1,
    # ZUPT (Foxlin 2005)
    'Rzupt':                0.01,
    'zupt_accel_threshold': 0.3,
    'zupt_gyro_threshold':  0.05,
}


# ── Quaternion utilities ───────────────────────────────────────────────────────

def _skew(v):
    return np.array([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],   v[0],  0.0 ],
    ])


def _qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])


def _qmul(q1, q2):
    w1, x1, y1, z1 = q1;  w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qfrom_axis_angle(dtheta):
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return _qnorm(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
    axis = dtheta / angle;  s = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0]*s, axis[1]*s, axis[2]*s])


def _qfrom_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    return _qnorm(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]))


def _qto_rpy(q):
    w, x, y, z = q
    roll  = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def _qto_Rbn(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)    ],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)    ],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


# ── Main filter ────────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the ESKF with NHC + ZUPT enhancements.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s}.
        use_3d_rotation: True → full 3D; False → yaw-only (2D).

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

    pos        = np.zeros((NN, 3));  vel       = np.zeros((NN, 3))
    rpy_out    = np.zeros((NN, 3));  b_acc_out = np.zeros((NN, 3))
    b_gyr_out  = np.zeros((NN, 3))
    std_pos    = np.zeros((NN, 3));  std_vel   = np.zeros((NN, 3))
    std_orient = np.zeros((NN, 3))
    std_b_acc  = np.zeros((NN, 3));  std_b_gyr = np.zeros((NN, 3))

    pos[0, :]     = pm.geodetic2enu(lla[0,0], lla[0,1], lla[0,2], lla0[0], lla0[1], lla0[2])
    vel[0, :]     = vel_enu[0, :]
    rpy_out[0, :] = orient[0, :]

    pIMU = pos[0, :].copy()
    vIMU = vel[0, :].copy()
    q    = _qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2])
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    Q = np.zeros((15, 15))
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

    for i in range(NN - 1):

        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0.0, 0.0, omega_b[2] * Ts])

        q   = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn = _qto_Rbn(q)

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        F = np.zeros((15, 15))
        F[0:3,   3:6]   = np.eye(3)
        F[3:6,   6:9]   = -Rbn @ _skew(acc_b)
        F[3:6,   9:12]  = -Rbn
        F[6:9,   6:9]   = -_skew(omega_b)
        F[6:9,   12:15] = -np.eye(3)
        F[9:12,  9:12]  = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd           = np.eye(15) + F * Ts
        Fd[6:9, 6:9] = _qto_Rbn(_qfrom_axis_angle(omega_b * Ts)).T

        P  = Fd @ P @ Fd.T + Q
        update_occurred = False

        # ── A. GPS Position Update ─────────────────────────────────────────────
        gps_ok     = nav_data.gps_available[i]
        not_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_ok and not_outage:
            p_gps = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            z_pos = p_gps - pIMU
            innov = z_pos - dx[0:3]
            S     = P[0:3, 0:3] + R_pos
            K     = P[:, 0:3] @ np.linalg.inv(S)
            dx    = dx + K @ innov
            P     = P - K @ S @ K.T
            P     = 0.5 * (P + P.T)
            update_occurred = True

        # ── B. Non-Holonomic Constraints (NHC) ────────────────────────────────
        # H derivation: δv_body = Rnb @ δv + skew(v_body) @ δφ_body
        Rnb    = Rbn.T
        v_body = Rnb @ vIMU
        z_nhc  = -v_body[1:3]

        H_v_nhc     = Rnb[1:3, :]
        H_theta_nhc = _skew(v_body)[1:3, :]
        H_nhc       = np.hstack((H_v_nhc, H_theta_nhc))  # 2×6

        innov_nhc = z_nhc - H_nhc @ dx[3:9]
        S_nhc     = H_nhc @ P[3:9, 3:9] @ H_nhc.T + R_nhc
        K_nhc     = P[:, 3:9] @ H_nhc.T @ np.linalg.inv(S_nhc)
        dx = dx + K_nhc @ innov_nhc
        P  = P - K_nhc @ S_nhc @ K_nhc.T
        P  = 0.5 * (P + P.T)
        update_occurred = True

        # ── C. Zero-Velocity Update (ZUPT) ────────────────────────────────────
        accel_dev = abs(np.linalg.norm(acc_b) - 9.81)
        gyro_mag  = np.linalg.norm(omega_b)
        speed     = np.linalg.norm(vIMU)

        if (accel_dev < p_cfg['zupt_accel_threshold'] and
                gyro_mag  < p_cfg['zupt_gyro_threshold'] and
                speed < 1.0):
            z_zupt     = -vIMU
            innov_zupt = z_zupt - dx[3:6]
            S_zupt     = P[3:6, 3:6] + R_zupt
            K_zupt     = P[:, 3:6] @ np.linalg.inv(S_zupt)
            dx = dx + K_zupt @ innov_zupt
            P  = P - K_zupt @ S_zupt @ K_zupt.T
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # ── Error injection & covariance reset (Solà §7.3) ────────────────────
        if update_occurred:
            pIMU += dx[0:3]
            vIMU += dx[3:6]
            b_a  += dx[9:12]
            b_g  += dx[12:15]

            delta_theta = dx[6:9]
            q = _qnorm(_qmul(q, _qfrom_axis_angle(delta_theta)))

            G           = np.eye(15)
            G[6:9, 6:9] = np.eye(3) - 0.5 * _skew(delta_theta)
            P           = G @ P @ G.T
            dx[:]       = 0.0

        pos[i+1, :]       = pIMU
        vel[i+1, :]       = vIMU
        rpy_out[i+1, :]   = _qto_rpy(q)
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
