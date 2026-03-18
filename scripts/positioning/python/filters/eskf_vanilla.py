# -*- coding: utf-8 -*-
"""
ESKF Vanilla — Quaternion Error-State Kalman Filter.
GPS position update only.

Reference:
    Solà, J., "Quaternion kinematics for the error-state Kalman filter",
    arXiv:1711.02508, 2017.  https://arxiv.org/abs/1711.02508

State vector (15 elements — error state):
    dx[0:3]   — position error  δp  in ENU  [m]
    dx[3:6]   — velocity error  δv  in ENU  [m/s]
    dx[6:9]   — attitude error  δφ  in FLU body frame  [rad]   (Solà convention)
    dx[9:12]  — accelerometer bias  b_a  in FLU body frame  [m/s²]
    dx[12:15] — gyroscope bias  b_g  in FLU body frame  [rad/s]

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Quaternion: Hamilton convention  q = [w, x, y, z]
    q_NB      : body-to-navigation rotation (Solà notation q^n_b)
"""
import numpy as np
import pymap3d as pm

# ── Physical constants ─────────────────────────────────────────────────────────
GRAVITY = np.array([0.0, 0.0, -9.81])   # ENU [m/s²]

# ── Default parameters ─────────────────────────────────────────────────────────
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
}


# ── Quaternion utilities (Hamilton convention, q = [w, x, y, z]) ───────────────

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
    """Hamilton product q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _qfrom_axis_angle(dtheta):
    """Exact small-angle quaternion from rotation vector dθ."""
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return _qnorm(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
    axis  = dtheta / angle
    s     = np.sin(0.5 * angle)
    return np.array([np.cos(0.5 * angle), axis[0]*s, axis[1]*s, axis[2]*s])


def _qfrom_euler(roll, pitch, yaw):
    """Build q_NB from ZYX Euler angles."""
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
    """Convert q_NB to [roll, pitch, yaw] (ZYX, ENU frame)."""
    w, x, y, z = q
    roll  = np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))
    pitch = np.arcsin(np.clip(2.0*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def _qto_Rbn(q):
    """Rotation matrix R_bn (body→navigation) from quaternion q_NB."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)    ],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)    ],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


# ── Main filter ────────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the vanilla ESKF (GPS position update only).

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Optional {'start': t1_s, 'duration': d_s} for GPS blackout.
        use_3d_rotation: True → full 3D; False → yaw-only (2D flat-earth).

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
    pos[0, :]   = pm.geodetic2enu(lla[0,0], lla[0,1], lla[0,2], lla0[0], lla0[1], lla0[2])
    vel[0, :]   = vel_enu[0, :]
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

    R_pos = np.eye(3) * p_cfg['Rpos']

    # ── Main loop ──────────────────────────────────────────────────────────────
    for i in range(NN - 1):

        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        # Quaternion attitude propagation (exact integration)
        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0.0, 0.0, omega_b[2] * Ts])

        q   = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn = _qto_Rbn(q)

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # Error-state transition matrix F (Solà §7, body-frame attitude error)
        F = np.zeros((15, 15))
        F[0:3,   3:6]   = np.eye(3)
        F[3:6,   6:9]   = -Rbn @ _skew(acc_b)
        F[3:6,   9:12]  = -Rbn
        F[6:9,   6:9]   = -_skew(omega_b)
        F[6:9,   12:15] = -np.eye(3)
        F[9:12,  9:12]  = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd            = np.eye(15) + F * Ts
        Fd[6:9, 6:9]  = _qto_Rbn(_qfrom_axis_angle(omega_b * Ts)).T  # exact attitude propagation

        P  = Fd @ P @ Fd.T + Q

        update_occurred = False

        # GPS Position Update (sparse)
        gps_ok     = nav_data.gps_available[i]
        not_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_ok and not_outage:
            p_gps = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            z_pos = p_gps - pIMU

            # Sparse Kalman gain (avoids full H @ P @ H.T with dense H)
            innov = z_pos - dx[0:3]
            S     = P[0:3, 0:3] + R_pos
            S_reg = S + 1e-9 * np.eye(3)
            K     = np.linalg.solve(S_reg, P[0:3, :]).T   # 15×3, stable solve

            dx = dx + K @ innov
            H_pos = np.zeros((3, 15)); H_pos[:, 0:3] = np.eye(3)
            IKH   = np.eye(15) - K @ H_pos
            P     = IKH @ P @ IKH.T + K @ R_pos @ K.T    # Joseph form
            P     = 0.5 * (P + P.T)
            update_occurred = True

        # Error injection and covariance reset (Solà §7.3)
        if update_occurred:
            pIMU += dx[0:3]
            vIMU += dx[3:6]
            b_a  += dx[9:12]
            b_g  += dx[12:15]

            delta_theta = dx[6:9]
            q = _qnorm(_qmul(q, _qfrom_axis_angle(delta_theta)))

            # Covariance reset via G-matrix (Solà Eq. 288)
            G           = np.eye(15)
            G[6:9, 6:9] = np.eye(3) - 0.5 * _skew(delta_theta)
            P           = G @ P @ G.T

            dx[:] = 0.0

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
