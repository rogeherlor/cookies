# -*- coding: utf-8 -*-
"""
IEKF Enhanced — `iekf_vanilla` augmented with two pseudo-measurements
common to ground-vehicle navigation:

    - Non-Holonomic Constraints (NHC) — lateral/vertical body-frame
      velocity treated as ~0; applied at every IMU step.
    - Zero-Velocity Update (ZUPT) — full body-frame velocity treated
      as 0 when an IMU-only stationarity detector fires.

The base mechanisation, F structure, conventions, sign convention, and
discretisation are identical to `iekf_vanilla.py` — only the measurement
blocks differ. Refer to `iekf.md` for the line-by-line justification of
the shared parts; the NHC/ZUPT additions are derived in §B–§C below.

References:
    [iekf1] Barrau, A. & Bonnabel, S., "Invariant Kalman Filtering",
        Annual Review of Control, Robotics, and Autonomous Systems
        1:237-257, 2018.  (Tutorial. The "imperfect IEKF" with appended
        Euclidean biases is described in §3 Remark 3.)
    [iekf2] Barrau, A. & Bonnabel, S., "The Invariant Extended Kalman
        Filter as a Stable Observer", IEEE Transactions on Automatic
        Control 62(4):1797-1812, April 2017.
        DOI: 10.1109/TAC.2016.2594085  (Theoretical foundation —
        log-linear property and stability proof; the SE_2(3) flat-Earth
        navigation example is in §V, without biases.)
    Dissanayake, G., Sukkarieh, S., Nebot, E., Durrant-Whyte, H., "The
        aiding of a low-cost strapdown inertial measurement unit using
        vehicle model constraints for land vehicle applications",
        IEEE Trans. Robotics and Automation 17 (5), 2001 — original NHC
        formulation.
    (Neither iekf1 nor iekf2 covers NHC or ZUPT. The ZUPT constraint
    and detector follow the textbook treatment in Groves §15.4.1,
    applied here within the left-invariant IEKF framework.)

Implementation notes:
    This is the "imperfect IEKF" formulation (iekf1 §3 Remark 3) where
    the IMU biases are appended as Euclidean states rather than group
    elements. The group state X = (R, v, p) ∈ SE_2(3) follows exact
    group propagation; biases use additive Gauss-Markov dynamics, which
    extends iekf1's pure random-walk treatment.

    Error definition: left-invariant error  η = X̂⁻¹ · X
        (this is the inverse of iekf1 footnote 2's η = χ⁻¹·χ̂; both are
        "left-invariant" — the choice fixes sign conventions in the F
        matrix and injection rules. The X̂⁻¹·X convention used here gives
        the body-frame injection p ← p + R̂·ξ_p shown below.)
    At identity the Lie algebra coordinates give:
        ξ[0:3]  — attitude error  ξ_R    in FLU body frame  [rad]   (δφ in some derivations)
        ξ[3:6]  — velocity error  ξ_v    in FLU body frame  [m/s]
        ξ[6:9]  — position error  ξ_p    in FLU body frame  [m]
        ξ[9:12] — accelerometer bias error  δb_a  [m/s²]
        ξ[12:15]— gyroscope bias error      δb_g  [rad/s]

    Key property: the transition Jacobian for (ξ_R, ξ_v, ξ_p) is
    state-independent — it depends only on the bias-corrected sensor
    inputs ω̂ = ω_meas − b̂_g and â = a_meas − b̂_a, not on R, v, or p.
    This is the log-linear property of invariant systems (iekf2 §III)
    and is what improves linearization robustness vs. EKF/ESKF.

    GPS update: the body-frame residual z_body = R̂ᵀ(p_GPS − p̂) is a
    linear function of ξ_p (body frame), so:
        H = [0, 0, I, 0, 0]   (3×15)         # selects ξ_p directly
        z_body = R̂ᵀ @ (p_GPS − p̂)            # rotate residual into body frame

    NHC update (§B): the body-frame velocity is `v_body = R̂ᵀ v_nav`
    and its left-invariant linearisation is δv_body = ξ_v + [v_body]_× ξ_R,
    so H_NHC selects rows [1:3] of [[v^b]_× | I] over the (ξ_R, ξ_v)
    columns. Lateral and vertical body-frame velocities ≈ 0.

    ZUPT update (§C): when the vehicle is stationary, all three body-
    frame velocity components are constrained to 0. The full linearisation
    is δv_body = ξ_v + [v_body]_× ξ_R, but the stationarity trigger
    ensures ‖v_body‖ ≪ 1 m/s, so the [v_body]_× term is negligible and
    H_ZUPT = [0, I, 0, 0, 0] is the standard simplification.

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

    # NHC — Non-Holonomic Constraints (Dissanayake 2001)
    'Rnhc': 0.1,
    # ZUPT — Zero-Velocity Update (Groves §15.4.1)
    # These four are textbook defaults, used only as a fallback when no
    # per-fold tuned override exists in filter_params.json (see
    # tune_nhc_zupt_loo.py — proper nested-LOO tuning writes fold-specific
    # values directly into filter_params.json, not here).
    'Rzupt':                0.01,
    'zupt_accel_threshold': 0.3,
    'zupt_gyro_threshold':  0.05,
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
    Run the IEKF Enhanced filter (GPS + NHC + ZUPT).

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

    # ── Initial covariance (state ordering: ξ_R, ξ_v, ξ_p, b_a, b_g) ───────────
    P = np.diag([
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'] * 2,
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_pos  = np.eye(3) * p_cfg['Rpos']
    R_nhc  = np.eye(2) * p_cfg['Rnhc']
    R_zupt = np.eye(3) * p_cfg['Rzupt']

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
        Rnb = Rbn.T

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # ── Left-invariant Jacobian (continuous time, iekf1 §3.1 Theorem 1) ───
        # Error state ordering: [ξ_R(3), ξ_v(3), ξ_p(3), δb_a(3), δb_g(3)]
        # Key property: Ajac[0:9, 0:9] depends only on the bias-corrected
        # inputs (ω̂, â), not on R/v/p — this is the IEKF state-autonomy.
        # See iekf.md §5 "Convention note" for the relationship to the
        # right-invariant dual that appears in Brossard 2020 §IV / Hartley 2020.
        Ajac = np.zeros((15, 15))
        # Attitude dynamics
        Ajac[0:3,  0:3 ] = -_skew(omega_b)  # ξ̇_R = −[ω̂]_× ξ_R
        Ajac[0:3,  12:15] = -np.eye(3)      # ξ̇_R = −δb_g
        # Velocity dynamics (body frame)
        Ajac[3:6,  0:3 ] = -_skew(acc_b)    # ξ̇_v = −[â]_× ξ_R
        Ajac[3:6,  3:6 ] = -_skew(omega_b)  # ξ̇_v = −[ω̂]_× ξ_v
        Ajac[3:6,  9:12] = -np.eye(3)       # ξ̇_v = −δb_a
        # Position dynamics (body frame)
        Ajac[6:9,  3:6 ] = np.eye(3)        # ξ̇_p = ξ_v
        Ajac[6:9,  6:9 ] = -_skew(omega_b)  # ξ̇_p = −[ω̂]_× ξ_p
        # Bias dynamics (Gauss-Markov)
        Ajac[9:12,  9:12]  = beta_acc * np.eye(3)
        Ajac[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + Ajac * Ts

        # Prediction
        P = Fd @ P @ Fd.T + Q
        update_occurred = False

        # ── A. GPS Position Update ─────────────────────────────────────────────
        # h(X) = p  →  in left-invariant error: z = R̂ᵀ @ (p_GPS − p̂)
        # H = [0, 0, I, 0, 0] (selects ξ_p in body frame)
        gps_ok     = nav_data.gps_available[i]
        not_outage = ((i + 1) < A) or ((i + 1) > B)

        if gps_ok and not_outage:
            p_gps  = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            z_body = Rnb @ (p_gps - pIMU)           # body-frame residual
            innov  = z_body - xi[6:9]

            S     = P[6:9, 6:9] + R_pos
            S_reg = S + 1e-9 * np.eye(3)
            K     = np.linalg.solve(S_reg, P[6:9, :]).T   # 15×3, stable solve

            xi    = xi + K @ innov
            H_pos = np.zeros((3, 15)); H_pos[:, 6:9] = np.eye(3)
            IKH   = np.eye(15) - K @ H_pos
            P     = IKH @ P @ IKH.T + K @ R_pos @ K.T    # Joseph form
            P     = 0.5 * (P + P.T)
            update_occurred = True

        # ── B. Non-Holonomic Constraints (NHC) — applied every step ───────────
        # Body-frame lateral (Y) and vertical (Z) velocity ≈ 0.
        # Linearisation in left-invariant convention:
        #   v = v̂ + R̂·ξ_v,   R = R̂·Exp(ξ_R)
        #   ⇒  δv_body = ξ_v + [v_body]_× ξ_R
        # H selects rows [1:3] of [[v^b]_× | I] over the (ξ_R, ξ_v) columns.
        # Matches the IEKF row at 3.tex:545-547.
        v_body = Rnb @ vIMU
        z_nhc  = -v_body[1:3]

        H_xiR_nhc = _skew(v_body)[1:3, :]       # 2×3 attitude block (cols 0:3)
        H_v_nhc   = np.eye(3)[1:3, :]           # 2×3 velocity block (cols 3:6)
        H_nhc     = np.hstack((H_xiR_nhc, H_v_nhc))   # 2×6 over ξ_R, ξ_v

        innov_nhc = z_nhc - H_nhc @ xi[0:6]
        S_nhc     = H_nhc @ P[0:6, 0:6] @ H_nhc.T + R_nhc
        S_nhc_reg = S_nhc + 1e-9 * np.eye(2)
        K_nhc     = np.linalg.solve(S_nhc_reg, H_nhc @ P[0:6, :]).T   # 15×2

        xi = xi + K_nhc @ innov_nhc
        H_nhc_full = np.zeros((2, 15)); H_nhc_full[:, 0:6] = H_nhc
        IKH_nhc    = np.eye(15) - K_nhc @ H_nhc_full
        P  = IKH_nhc @ P @ IKH_nhc.T + K_nhc @ R_nhc @ K_nhc.T        # Joseph form
        P  = 0.5 * (P + P.T)
        update_occurred = True

        # ── C. Zero-Velocity Update (ZUPT) ────────────────────────────────────
        # When the vehicle is stationary, all 3 body-frame velocity components
        # are ≈ 0. The full linearisation is δv_body = ξ_v + [v_body]_× ξ_R
        # but the trigger ensures ‖v_body‖ ≪ 1 m/s, so the attitude coupling
        # is negligible and H_ZUPT = [0, I, 0, 0, 0] is the standard form.
        # Detector: |‖f^b‖−g| AND ‖ω^b‖ small (Groves §15.4.1) plus a nav-speed
        # guard against false positives during smooth cruise.
        accel_dev = abs(np.linalg.norm(acc_b) - 9.81)
        gyro_mag  = np.linalg.norm(omega_b)
        speed     = np.linalg.norm(vIMU)

        if (accel_dev < p_cfg['zupt_accel_threshold'] and
                gyro_mag  < p_cfg['zupt_gyro_threshold'] and
                speed     < 1.0):
            z_zupt     = -v_body                # all 3 body-frame velocity components
            innov_zupt = z_zupt - xi[3:6]

            S_zupt     = P[3:6, 3:6] + R_zupt
            S_zupt_reg = S_zupt + 1e-9 * np.eye(3)
            K_zupt     = np.linalg.solve(S_zupt_reg, P[3:6, :]).T     # 15×3

            xi = xi + K_zupt @ innov_zupt
            H_zupt = np.zeros((3, 15)); H_zupt[:, 3:6] = np.eye(3)
            IKH_zupt = np.eye(15) - K_zupt @ H_zupt
            P  = IKH_zupt @ P @ IKH_zupt.T + K_zupt @ R_zupt @ K_zupt.T  # Joseph form
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # ── Error injection (body → nav conversion for p and v) ───────────────
        if update_occurred:
            # Position and velocity corrections: body frame back to nav frame
            pIMU += Rbn @ xi[6:9]
            vIMU += Rbn @ xi[3:6]
            b_a  += xi[9:12]
            b_g  += xi[12:15]

            # Attitude update via quaternion multiplication (same as ESKF)
            xi_R = xi[0:3]
            q   = _qnorm(_qmul(q, _qfrom_axis_angle(xi_R)))
            Rbn = _qto_Rbn(q)   # update Rbn after attitude correction

            # Covariance reset (Solà Eq. 288; only attitude block is nonlinear)
            G           = np.eye(15)
            G[0:3, 0:3] = np.eye(3) - 0.5 * _skew(xi_R)
            P           = G @ P @ G.T

            xi[:] = 0.0

        pos[i+1, :]       = pIMU
        vel[i+1, :]       = vIMU
        rpy_out[i+1, :]   = _qto_rpy(q)
        b_acc_out[i+1, :] = b_a
        b_gyr_out[i+1, :] = b_g
        # Map IEKF covariance back to ENU for output (body-frame → nav-frame std)
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
    }
