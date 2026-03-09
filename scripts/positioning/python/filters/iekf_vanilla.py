# -*- coding: utf-8 -*-
"""
IEKF Vanilla — Left-Invariant Extended Kalman Filter on SE_2(3).
GPS position update only.

Reference:
    Barrau, A. & Bonnabel, S., "The Invariant Extended Kalman Filter as a
    Stable Observer", IEEE Transactions on Automatic Control, vol. 62, no. 4,
    pp. 1797-1812, April 2017.  DOI: 10.1109/TAC.2016.2594085

Implementation notes:
    This is the "imperfect IEKF" formulation (Barrau 2017, Sec. V) where
    the IMU biases are treated as Euclidean states rather than group elements.
    The group state X = (R, v, p) ∈ SE_2(3) follows exact group propagation
    while biases use the standard additive Gauss-Markov model.

    Error definition: left-invariant error  η = X̂⁻¹ · X.
    At identity the Lie algebra coordinates give:
        ξ[0:3]  — attitude error  φ      in FLU body frame  [rad]
        ξ[3:6]  — velocity error  ξ_v    in FLU body frame  [m/s]
        ξ[6:9]  — position error  ξ_p    in FLU body frame  [m]
        ξ[9:12] — accelerometer bias error  δb_a  [m/s²]
        ξ[12:15]— gyroscope bias error      δb_g  [rad/s]

    Key property: the transition Jacobian for (φ, ξ_v, ξ_p) is
    state-independent (depends only on slowly-varying bias estimates),
    improving linearization robustness vs. the standard EKF/ESKF.

    GPS update: the position measurement p_GPS (nav frame) maps to a
    linear function of ξ_p (body frame):
        H = [0, 0, R̂ᵀ, 0, 0]   (3×15)
        z = R̂ᵀ @ (p_GPS − p̂)

    Error injection after update:
        p  += R̂ @ ξ[6:9]     (body → nav)
        v  += R̂ @ ξ[3:6]     (body → nav)
        q   = q ⊗ exp(ξ[0:3])
        b_a += ξ[9:12]
        b_g += ξ[12:15]
    Covariance reset via G-matrix (Solà Eq. 288, attitude block only).

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Quaternion: Hamilton convention  q = [w, x, y, z]  (q_NB = body→nav)
"""
import numpy as np
import pymap3d as pm

GRAVITY = np.array([0.0, 0.0, -9.81])   # ENU [m/s²]

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


# ── Quaternion utilities (Hamilton, q = [w, x, y, z]) ─────────────────────────

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
    Run the vanilla left-invariant EKF (GPS position update only).

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
    xi   = np.zeros(15)     # left-invariant error state (body frame)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    # ── Process noise (same structure as ESKF; Barrau §VI uses standard Q) ─────
    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.diag([p_cfg['QorientXY'], p_cfg['QorientXY'], p_cfg['QorientZ']])
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.eye(3) * (p_cfg['Qpos'] * Ts**2)
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc'] * Ts)
    Q[12:15, 12:15] = np.diag([p_cfg['QgyrXY'], p_cfg['QgyrXY'], p_cfg['QgyrZ']]) * Ts

    # ── Initial covariance (same ordering as error-state: φ, ξ_v, ξ_p, b_a, b_g) ──
    P = np.diag([
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'] * 2,
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_pos = np.eye(3) * p_cfg['Rpos']

    for i in range(NN - 1):

        acc_b   = accel_flu[i, :] - b_a
        omega_b = gyro_flu[i, :]  - b_g

        # ── Nominal-state propagation (same quaternion mechanization as ESKF) ──
        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0.0, 0.0, omega_b[2] * Ts])

        q   = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn = _qto_Rbn(q)   # body → nav

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # ── Left-invariant Jacobian (continuous time, Barrau 2017 Eq. 26) ────
        # Error state ordering: [φ(3), ξ_v(3), ξ_p(3), δb_a(3), δb_g(3)]
        # Key property: Ajac[0:9, 0:9] depends only on b̂_a, b̂_g, not on R/v/p.
        Ajac = np.zeros((15, 15))
        # Attitude dynamics
        Ajac[0:3,  0:3 ] = -_skew(b_g)     # φ̇ ~ −b̂_g × φ
        Ajac[0:3,  12:15] = -np.eye(3)     # φ̇ ~ −δb_g
        # Velocity dynamics (body frame)
        Ajac[3:6,  0:3 ] = -_skew(b_a)    # ξ̇_v ~ −b̂_a × φ
        Ajac[3:6,  3:6 ] = -_skew(b_g)    # ξ̇_v ~ −b̂_g × ξ_v
        Ajac[3:6,  9:12] = -np.eye(3)     # ξ̇_v ~ −δb_a
        # Position dynamics (body frame)
        Ajac[6:9,  3:6 ] = np.eye(3)      # ξ̇_p ~ ξ_v
        Ajac[6:9,  6:9 ] = -_skew(b_g)   # ξ̇_p ~ −b̂_g × ξ_p
        # Bias dynamics (Gauss-Markov)
        Ajac[9:12,  9:12]  = beta_acc * np.eye(3)
        Ajac[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + Ajac * Ts

        # Prediction (covariance only; xi stays ~0 between updates)
        P = Fd @ P @ Fd.T + Q

        update_occurred = False

        # ── GPS Position Update ────────────────────────────────────────────────
        # h(X) = p  →  in left-invariant error: z = Rbnᵀ @ (p_GPS − p̂)
        # H = [0, 0, I, 0, 0] but applied in body frame (Rbnᵀ maps error to meas)
        gps_ok     = nav_data.gps_available[i]
        not_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_ok and not_outage:
            p_gps   = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            # Measurement in body frame (left-invariant residual)
            z_body  = Rbn.T @ (p_gps - pIMU)
            innov   = z_body - xi[6:9]   # account for accumulated xi

            # Observation: H selects ξ_p (columns 6:9), pre-multiplied by I
            S = P[6:9, 6:9] + R_pos
            K = P[:, 6:9] @ np.linalg.inv(S)

            xi = xi + K @ innov
            P  = P - K @ S @ K.T
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # ── Error injection (body → nav conversion for p and v) ───────────────
        if update_occurred:
            # Position and velocity corrections: from body frame back to nav frame
            pIMU += Rbn @ xi[6:9]
            vIMU += Rbn @ xi[3:6]
            b_a  += xi[9:12]
            b_g  += xi[12:15]

            # Attitude update via quaternion multiplication (same as ESKF)
            delta_theta = xi[0:3]
            q = _qnorm(_qmul(q, _qfrom_axis_angle(delta_theta)))
            Rbn = _qto_Rbn(q)   # update Rbn after attitude correction

            # Covariance reset (Solà Eq. 288; only attitude block is nonlinear)
            G           = np.eye(15)
            G[0:3, 0:3] = np.eye(3) - 0.5 * _skew(delta_theta)
            P           = G @ P @ G.T

            xi[:] = 0.0

        pos[i+1, :]       = pIMU
        vel[i+1, :]       = vIMU
        rpy_out[i+1, :]   = _qto_rpy(q)
        b_acc_out[i+1, :] = b_a
        b_gyr_out[i+1, :] = b_g
        # Map IEKF covariance back to ENU for output (φ, ξ_v, ξ_p → p, v, orient)
        # For display purposes, multiply body-frame std by Rbn to get nav-frame std
        cov_p_nav   = Rbn @ P[6:9, 6:9] @ Rbn.T
        cov_v_nav   = Rbn @ P[3:6, 3:6] @ Rbn.T
        std_pos[i+1, :]      = np.sqrt(np.maximum(np.diag(cov_p_nav),           0.0))
        std_vel[i+1, :]      = np.sqrt(np.maximum(np.diag(cov_v_nav),           0.0))
        std_orient[i+1, :]   = np.sqrt(np.maximum(np.diag(P[0:3, 0:3]),         0.0))
        std_b_acc[i+1, :]    = np.sqrt(np.maximum(np.diag(P[9:12,  9:12]),      0.0))
        std_b_gyr[i+1, :]    = np.sqrt(np.maximum(np.diag(P[12:15, 12:15]),     0.0))

    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_acc_out, 'bias_gyr': b_gyr_out,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_b_acc, 'std_bias_gyr': std_b_gyr,
    }
