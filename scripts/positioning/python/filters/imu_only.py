# -*- coding: utf-8 -*-
"""
IMU-Only — IEKF Dead-Reckoning following AI-IMU (Brossard et al. 2020).

Implements the deterministic core of the AI-IMU algorithm: an Invariant
Extended Kalman Filter on SE_2(3) with biases appended (15-state imperfect
IEKF, same architecture as iekf_vanilla.py) running Non-Holonomic
Constraints (NHC) as the only aiding source.  No GNSS, no ZUPT, no
exteroceptive sensor of any kind — just the IMU plus the wheeled-vehicle
side-information that lateral and vertical car-frame velocities are
approximately zero.

Reference (algorithm replicated here):
    Brossard, M., Barrau, A. & Bonnabel, S., "AI-IMU Dead-Reckoning",
    IEEE Transactions on Intelligent Vehicles 5(4):585-595, December 2020.
    DOI: 10.1109/TIV.2020.2980758

    The state-space model is from §III.B eq. (8); the kinematics from
    §III.A eq. (3)-(7); the NHC pseudo-measurement from §IV.B eq. (15);
    the IEKF on SE_2(3) framework from §IV.C and the appendix (this
    file uses the left-invariant dual of AI-IMU's right-invariant F
    matrix — see "Differences" below).

Differences from the full AI-IMU paper:
    * Neural-network covariance adapter (§V.A) is REPLACED by a fixed
      Rnhc tuning parameter.  This is the "AI-IMU minus the AI" baseline:
      same filter, same pseudo-measurement, no learning on the noise.
    * Car-frame misalignment states (R^c_n, p^c_n — eq. 12-13, plus the
      lever-arm correction in eq. 14) are DROPPED.  IMU and car frames
      are assumed coincident, so v_body = R̂ᵀ·v_nav directly with no
      lever arm.  This reduces the state vector from AI-IMU's 21
      dimensions to 15, matching the rest of this project.
    * Error convention: AI-IMU §IV-A uses the *right-invariant* error
      χ = exp(ξ)·χ̂ (eq. 22), whose F matrix has gravity [g]_× in
      ξ̇_v ← ξ_R and rotation matrices on the bias columns.  This file
      uses the *left-invariant* dual η = X̂⁻¹·X (parity with
      iekf_vanilla.py), whose F matrix has −[â]_× and −I instead.  Both
      forms are valid IEKF parametrisations of the same filter; the
      nominal-state equations (eq. 5-7) and the NHC observation
      structure are identical in either convention.

Companion references for the underlying machinery:
    * Barrau, A. & Bonnabel, S., "Invariant Kalman Filtering",
      Annu. Rev. Control Robot. Auton. Syst. 1:237-257, 2018.
      (The imperfect IEKF with appended Euclidean biases is §3 Remark 3.)
    * Dissanayake, G. et al., "The aiding of a low-cost strapdown IMU
      using vehicle model constraints for land vehicle applications",
      IEEE Trans. Vehicular Technology, 2001.  (Original NHC formulation.)

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Quaternion: Hamilton convention  q = [w, x, y, z]  (q_NB = body → nav)
    Error definition: left-invariant  η = X̂⁻¹·X  (matches iekf_vanilla.py)

State vector ξ ∈ ℝ¹⁵ (left-invariant error, body-frame coords for ξ_R/v/p):
    ξ[0:3]   — attitude error  ξ_R    in FLU body frame  [rad]
    ξ[3:6]   — velocity error  ξ_v    in FLU body frame  [m/s]
    ξ[6:9]   — position error  ξ_p    in FLU body frame  [m]
    ξ[9:12]  — accelerometer bias error  δb_a            [m/s²]
    ξ[12:15] — gyroscope     bias error  δb_g            [rad/s]
"""
import time

import numpy as np
import pymap3d as pm

GRAVITY = np.array([0.0, 0.0, -9.81])   # ENU [m/s²], constant per AI-IMU §III.A

DEFAULT_PARAMS = {
    # Process noise (same scaling convention as iekf_vanilla.py)
    'Qpos':      5.312e-06,
    'Qvel':      4.702e-06,
    'QorientXY': 0.0002,
    'QorientZ':  0.2,
    'Qacc':      0.1,
    'QgyrXY':    0.0001,
    'QgyrZ':     0.1,

    # NHC pseudo-measurement noise (AI-IMU eq. 15; replaces NN-learned N_n)
    # Larger values = looser constraint (more lateral/vertical slip allowed).
    'Rnhc': 0.1,                 # [(m/s)²] per axis on (v_lat, v_vert)

    # Gauss-Markov bias decay (extends AI-IMU eq. 3-4 random walks; β must be < 0)
    'beta_acc':  -1.910e-06,
    'beta_gyr':  -7.077e-02,

    # Initial covariance (1-σ values; same as iekf_vanilla.py)
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
    AI-IMU-style dead reckoning: IEKF on SE_2(3) + NHC, no GNSS.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS.
        outage_config  : Accepted for interface compatibility but ignored —
                         this filter never reads GNSS regardless.
        use_3d_rotation: True → full 3D quaternion update; False → yaw-only.

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
    xi   = np.zeros(15)     # left-invariant error (resets to 0 after each update)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    # Process noise (matches iekf_vanilla.py block ordering)
    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.diag([p_cfg['QorientXY'], p_cfg['QorientXY'], p_cfg['QorientZ']])
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.eye(3) * (p_cfg['Qpos'] * Ts**2)
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc'] * Ts)
    Q[12:15, 12:15] = np.diag([p_cfg['QgyrXY'], p_cfg['QgyrXY'], p_cfg['QgyrZ']]) * Ts

    P = np.diag([
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'] * 2,
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_nhc = np.eye(2) * p_cfg['Rnhc']

    # Per-event wall-clock instrumentation: one entry per IMU sample on which
    # a measurement update actually fired, timing the correction itself (gain
    # solve, Joseph covariance update, error injection) in isolation from the
    # propagation that runs on every sample. Same convention as the DL
    # runners' `net_latency_s`, so the real-time tables can treat a GPS
    # correction and a network call as the same kind of Event.
    event_latency_s = []
    event_cpu_s = []

    for i in range(NN - 1):

        acc_b   = accel_flu[i, :] - b_a    # â = a_meas − b̂_a   (AI-IMU eq. 2)
        omega_b = gyro_flu[i, :]  - b_g    # ω̂ = ω_meas − b̂_g   (AI-IMU eq. 1)

        # ── Nominal-state propagation (AI-IMU eq. 5-7) ─────────────────────────
        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0.0, 0.0, omega_b[2] * Ts])

        q   = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))   # eq. 5
        Rbn = _qto_Rbn(q)
        Rnb = Rbn.T

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)   # eq. 7
        vIMU   = vIMU + Ts * (accENU + g)                         # eq. 6

        # ── Left-invariant Jacobian (iekf1 §3.2; same as iekf_vanilla.py) ─────
        # Depends on bias-corrected inputs (ω̂, â), not on R/v/p — IEKF autonomy.
        Ajac = np.zeros((15, 15))
        Ajac[0:3,  0:3 ]   = -_skew(omega_b)   # ξ̇_R = −[ω̂]_× ξ_R
        Ajac[0:3,  12:15]  = -np.eye(3)        # ξ̇_R = −δb_g
        Ajac[3:6,  0:3 ]   = -_skew(acc_b)     # ξ̇_v = −[â]_× ξ_R
        Ajac[3:6,  3:6 ]   = -_skew(omega_b)   # ξ̇_v = −[ω̂]_× ξ_v
        Ajac[3:6,  9:12]   = -np.eye(3)        # ξ̇_v = −δb_a
        Ajac[6:9,  3:6 ]   = np.eye(3)         # ξ̇_p = ξ_v
        Ajac[6:9,  6:9 ]   = -_skew(omega_b)   # ξ̇_p = −[ω̂]_× ξ_p
        Ajac[9:12,  9:12]  = beta_acc * np.eye(3)
        Ajac[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + Ajac * Ts
        P  = Fd @ P @ Fd.T + Q

        _ev_t0 = time.perf_counter()

        _ev_c0 = time.thread_time()
        # ── Non-Holonomic Constraints (AI-IMU §IV.B eq. 15) ────────────────────
        # Pseudo-measurement: lateral and vertical car-frame velocity = 0.
        # With car frame = IMU body frame (R^c=I, p^c=0 — see docstring),
        # v_body = Rnb · v_nav, and we observe v_body[1:3] ≈ 0.
        # In left-invariant error, δv_body = ξ_v + skew(v_body)·ξ_R, so:
        #   H_nhc maps [ξ_R, ξ_v] → 2-vector [v_lat, v_vert].
        v_body = Rnb @ vIMU
        z_nhc  = -v_body[1:3]                    # innovation = (target − measured) = 0 − v_body[1:3]

        H_xiR = _skew(v_body)[1:3, :]            # 2×3 attitude block
        H_v   = np.eye(3)[1:3, :]                # 2×3 velocity block (selects rows 1, 2)
        H_nhc = np.hstack((H_xiR, H_v))          # 2×6 over [ξ_R(0:3), ξ_v(3:6)]

        innov     = z_nhc - H_nhc @ xi[0:6]
        S         = H_nhc @ P[0:6, 0:6] @ H_nhc.T + R_nhc
        S_reg     = S + 1e-9 * np.eye(2)
        K         = np.linalg.solve(S_reg, H_nhc @ P[0:6, :]).T   # 15×2

        xi = xi + K @ innov

        H_full = np.zeros((2, 15));  H_full[:, 0:6] = H_nhc
        IKH    = np.eye(15) - K @ H_full
        P      = IKH @ P @ IKH.T + K @ R_nhc @ K.T              # Joseph form
        P      = 0.5 * (P + P.T)

        # ── Error injection (left-invariant: nav-frame quantities += R̂·ξ) ────
        pIMU += Rbn @ xi[6:9]
        vIMU += Rbn @ xi[3:6]
        b_a  += xi[9:12]
        b_g  += xi[12:15]

        xi_R = xi[0:3]
        q    = _qnorm(_qmul(q, _qfrom_axis_angle(xi_R)))
        Rbn  = _qto_Rbn(q)

        # Covariance reset (Solà eq. 288; only attitude block is non-trivial)
        G           = np.eye(15)
        G[0:3, 0:3] = np.eye(3) - 0.5 * _skew(xi_R)
        P           = G @ P @ G.T

        xi[:] = 0.0
        event_cpu_s.append(time.thread_time() - _ev_c0)
        event_latency_s.append(time.perf_counter() - _ev_t0)

        pos[i+1, :]       = pIMU
        vel[i+1, :]       = vIMU
        rpy_out[i+1, :]   = _qto_rpy(q)
        b_acc_out[i+1, :] = b_a
        b_gyr_out[i+1, :] = b_g

        # Output 1-σ envelopes (body→nav rotation for position/velocity)
        cov_p_nav = Rbn @ P[6:9, 6:9] @ Rbn.T
        cov_v_nav = Rbn @ P[3:6, 3:6] @ Rbn.T
        std_pos[i+1, :]      = np.sqrt(np.maximum(np.diag(cov_p_nav),       0.0))
        std_vel[i+1, :]      = np.sqrt(np.maximum(np.diag(cov_v_nav),       0.0))
        std_orient[i+1, :]   = np.sqrt(np.maximum(np.diag(P[0:3, 0:3]),     0.0))
        std_b_acc[i+1, :]    = np.sqrt(np.maximum(np.diag(P[9:12,  9:12]),  0.0))
        std_b_gyr[i+1, :]    = np.sqrt(np.maximum(np.diag(P[12:15, 12:15]), 0.0))

    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_acc_out, 'bias_gyr': b_gyr_out,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_b_acc, 'std_bias_gyr': std_b_gyr,
        'event_latency_s': np.asarray(event_latency_s),
        'event_cpu_s': np.asarray(event_cpu_s),
    }
