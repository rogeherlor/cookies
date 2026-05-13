# -*- coding: utf-8 -*-
"""
ES-EKF Groves (vanilla) — Phi-angle Error-State EKF in local Cartesian ENU
following Groves (2013) Ch. 14, with full Earth-rate and transport-rate
dynamics. GPS position update only.

Lineage / deviations from textbook Groves:
    - Mechanisation in local Cartesian ENU at a fixed origin lla0. Groves
      Ch. 14.2.4 main text uses curvilinear lat/lon/h; the Cartesian-position
      variant used here is the one developed in Groves Appendix I.2. 
      Tangent-plane geometric error is sub-cm under ~30 km, sub-metre under ~100 km.
    - Gravity-gradient term ∂g/∂p in F is dropped (negligible for short
      range; reduces F to a constant once Earth-rate is fixed).
    - Attitude parametrised as Euler ZYX angles Θ = [φ, θ, ψ] integrated
      via the Euler-rate matrix T(Θ); δε is the Euler-frame attitude error.
      Singularity at θ = ±90° is acceptable for ground vehicles.

References:
    Groves, P.D., "Principles of GNSS, Inertial, and Multisensor Integrated
        Navigation Systems", 2nd ed., Artech House, 2013. Ch. 5, Ch. 14,
        and Appendix I.2 (companion CD).

State vector (15 elements — error state δx):
    δx[0:3]   — position error  δp  in ENU  [m]
    δx[3:6]   — velocity error  δv  in ENU  [m/s]
    δx[6:9]   — attitude error  δε  in navigation frame (phi-angle)  [rad]
    δx[9:12]  — accelerometer bias  b_a  in FLU body frame  [m/s²]
    δx[12:15] — gyroscope bias  b_g  in FLU body frame  [rad/s]

Conventions:
    IMU       : FLU frame (Forward, Left, Up)
    Navigation: ENU frame (East, North, Up)
    Euler angles: ZYX, stored as [roll, pitch, yaw]  [rad]
    Attitude error δε: defined in the navigation frame (Groves phi-angle)
"""
import numpy as np
import pymap3d as pm
from math import sin, cos, sqrt

# ── Physical constants ─────────────────────────────────────────────────────────
WGS84_OMEGA_E = 7.2921151467e-5     # Earth rotation rate [rad/s]
WGS84_A       = 6378137.0           # semi-major axis [m]
WGS84_E2      = 6.69437999014e-3    # first eccentricity squared

# Kept for backward-compatibility with callers that imported it directly.
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


def _wgs84_gravity(lat_rad, h=0.0):
    """
    WGS84 normal gravity (Somigliana) at geodetic latitude `lat_rad` and
    height `h` [m]. Returns the magnitude (positive scalar) [m/s²].
    """
    s2 = sin(lat_rad) ** 2
    g0 = 9.7803253359 * (1.0 + 0.00193185265241 * s2) / sqrt(1.0 - 0.00669437999013 * s2)
    return g0 - 3.086e-6 * h   # free-air correction, linear in h near surface


def _earth_rate_enu(lat_rad):
    """ω_ie^n in ENU at latitude `lat_rad` [rad/s]."""
    return WGS84_OMEGA_E * np.array([0.0, cos(lat_rad), sin(lat_rad)])


def _wgs84_radii(lat_rad):
    """
    Meridian (R_M) and prime-vertical (R_N) radii of curvature at latitude `lat_rad` [m].
    Note: Groves writes R_N for meridian and R_E for transverse; this code uses the
    common geodesy convention (R_M = meridian, R_N = "normal"/transverse). Formulas
    correspond to Groves eq. 2.105 (meridian) and eq. 2.106 (transverse).
    """
    s2 = sin(lat_rad) ** 2
    denom = 1.0 - WGS84_E2 * s2
    R_N = WGS84_A / sqrt(denom)
    R_M = WGS84_A * (1.0 - WGS84_E2) / (denom * sqrt(denom))
    return R_M, R_N


def _transport_rate_enu(v_enu, lat_rad, h, R_M, R_N):
    """
    Transport rate ω_en^n in ENU. As the vehicle moves on the local tangent
    plane, the navigation frame rotates relative to ECEF; the components
    below are Groves (2013) eq. (5.44) re-expressed in ENU axes.
    """
    v_E, v_N, _ = v_enu
    return np.array([
        -v_N / (R_M + h),                    # E
         v_E / (R_N + h),                    # N
         v_E * np.tan(lat_rad) / (R_N + h),  # U
    ])


# ── Main filter ────────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True,
        disable_earth_terms=False):
    """
    Run the ES-EKF Groves vanilla filter (GPS position update only).

    Args:
        nav_data           : NavigationData dataclass (data_loader.py).
        params             : Optional dict overriding DEFAULT_PARAMS.
        outage_config      : Optional {'start': t1_s, 'duration': d_s} for GPS blackout.
        use_3d_rotation    : True → full roll/pitch/yaw; False → yaw-only (2D flat-earth).
        disable_earth_terms: When True, zero out ω_ie / ω_en and use the constant
                             −9.81 gravity vector — reduces the filter to its
                             pre-Groves-alignment behaviour for regression testing.

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

    Ts = 1.0 / frecIMU
    NN = lla.shape[0]

    # ── Earth-frame setup at the tangent-plane origin ──────────────────────────
    lat0_rad = np.deg2rad(lla0[0])
    h0       = lla0[2]
    if disable_earth_terms:
        omega_ie_n = np.zeros(3)
        g_n        = GRAVITY
    else:
        omega_ie_n = _earth_rate_enu(lat0_rad)
        g_n        = np.array([0.0, 0.0, -_wgs84_gravity(lat0_rad, h0)])
    R_M0, R_N0 = _wgs84_radii(lat0_rad)

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
        f_b     = accel_flu[i, :] - b_a   # specific force in body frame [m/s²]
        omega_b = gyro_flu[i, :]  - b_g   # angular rate (inertial→body) in body frame [rad/s]

        # 2. Transport rate (uses current velocity; small for short-range ground vehicles)
        if disable_earth_terms:
            omega_en_n = np.zeros(3)
        else:
            omega_en_n = _transport_rate_enu(vIMU, lat0_rad, h0, R_M0, R_N0)
        omega_in_n = omega_ie_n + omega_en_n   # nav-frame inertial rate

        # 3. Nominal-state propagation (Euler mechanisation with Earth-rate compensation)
        if use_3d_rotation:
            Rbn = _euler_to_Rbn(rpy)
            T   = _euler_rate_matrix(rpy)
            # ω_nb^b = ω_ib^b - R_n^b · ω_in^n  — body-relative rotation rate
            omega_nb_b = omega_b - Rbn.T @ omega_in_n
            rpy = rpy + Ts * (T @ omega_nb_b)
        else:
            yaw = rpy[2]
            cy, sy = cos(yaw), sin(yaw)
            Rbn = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            rpy[2] = rpy[2] + Ts * (omega_b[2] - omega_in_n[2])

        # Wrap yaw to (−π, π]
        rpy[2] = (rpy[2] + np.pi) % (2.0 * np.pi) - np.pi

        f_n     = Rbn @ f_b
        cor_acc = -np.cross(2.0 * omega_ie_n + omega_en_n, vIMU)   # Coriolis on velocity
        accENU  = f_n + cor_acc + g_n
        pIMU    = pIMU + Ts * vIMU + 0.5 * Ts**2 * accENU
        vIMU    = vIMU + Ts * accENU

        # 4. Error-state transition matrix F  (15-state, ENU phi-angle, Earth dynamics)
        #
        #    δṗ = δv
        #    δv̇ = −[2ω_ie^n + ω_en^n]× δv  −  [f^n]× δε  −  R_b^n δb_a
        #    δε̇ = −[ω_ie^n + ω_en^n]× δε  −  R_b^n δb_g
        #    ḃ_a = β_acc · b_a              (Gauss-Markov, β already negative)
        #    ḃ_g = β_gyr · b_g
        F = np.zeros((15, 15))
        F[0:3,   3:6]   = np.eye(3)                              # position ← velocity
        F[3:6,   3:6]   = -_skew(2.0 * omega_ie_n + omega_en_n)  # Coriolis on velocity error
        F[3:6,   6:9]   = -_skew(f_n)                            # velocity ← attitude (nav-frame f)
        F[3:6,   9:12]  = -Rbn                                   # velocity ← accel bias (truth−estimate convention; Groves uses +Ĉ_b^n with estimate−truth)
        F[6:9,   6:9]   = -_skew(omega_ie_n + omega_en_n)        # transport on attitude error
        F[6:9,   12:15] = -Rbn                                   # attitude ← gyro bias
        F[9:12,  9:12]  = beta_acc * np.eye(3)                   # accel bias Gauss-Markov decay
        F[12:15, 12:15] = beta_gyr * np.eye(3)                   # gyro  bias Gauss-Markov decay

        # First-order discretisation Φ = I + F·Ts (Groves eq. 14.72).
        # Sufficient at IMU rates: ‖F·Ts‖ is dominated by [f^n]× ≈ g, giving a
        # second-order correction ½(F·Ts)² ≲ 1e-3 at 100 Hz — well below Q.
        Fd = np.eye(15) + F * Ts

        # Prediction step
        P  = Fd @ P @ Fd.T + Q
        dx = Fd @ dx

        # 5. GPS position update (sparse: 3 observations)
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
            P   = IKH @ P @ IKH.T + K @ R_pos @ K.T   # Joseph form
            P   = 0.5 * (P + P.T)

            # Error injection (nav-frame attitude error added directly to rpy)
            pIMU   += dx[0:3]
            vIMU   += dx[3:6]
            rpy    += dx[6:9]
            b_a    += dx[9:12]
            b_g    += dx[12:15]
            rpy[2]  = (rpy[2] + np.pi) % (2.0 * np.pi) - np.pi

            # Reset error state (no covariance reset Jacobian: Groves §3.2.6 / §14.2.5
            # treats the closed-loop reset as identity. Solà's G_α = I − [½ δα]_× is
            # derived for body-frame right-multiplicative axis-angle errors and does
            # not apply to the nav-frame phi-angle convention used here.)
            dx[:] = 0.0

        # 6. Store outputs
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
