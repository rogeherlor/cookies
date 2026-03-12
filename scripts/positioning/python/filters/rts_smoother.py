# -*- coding: utf-8 -*-
"""
RTS Smoother — Rauch-Tung-Striebel backward smoother built on top of
EKF Enhanced (Euler-angle EKF + NHC + ZUPT).

Two-pass algorithm:
    Forward : identical to ekf_enhanced, but stores the per-step state,
              posterior covariance, prior (predicted) covariance and
              discrete transition matrix needed for the backward sweep.
    Backward: standard RTS equations propagate information backward from
              the final epoch to give the optimal (MMSE) smoothed estimate
              at every time step.

Intended use: run once with no GNSS outage to obtain the "best achievable"
trajectory from the available IMU+GPS data, and use it as a reference when
comparing the performance of causal forward filters.

Note: outage_config is accepted for interface compatibility but is silently
      ignored — the smoother always runs with full GPS access to maximise
      the quality of the reference trajectory.

References:
    Rauch, H.E., Tung, F., Striebel, C.T., "Maximum likelihood estimates of
    linear dynamic systems", AIAA J., 1965.

    Sarkka, S., "Bayesian Filtering and Smoothing", Cambridge, 2013. Ch. 8.
"""
import numpy as np
import pymap3d as pm
from math import sin, cos

from filters.ekf_enhanced import (
    DEFAULT_PARAMS,
    GRAVITY,
    _skew,
    _euler_to_Rbn,
    _euler_rate_matrix,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ── Forward pass with storage ─────────────────────────────────────────────────

def _forward_pass(nav_data, params, use_3d_rotation):
    """
    Run the EKF Enhanced forward pass and collect the arrays required for
    the RTS backward sweep.

    Returns a dict with the standard output keys plus private keys:
        '_xs'       — list of NN packed nominal states  [pos(3)|vel(3)|rpy(3)|ba(3)|bg(3)]
        '_Ps_post'  — list of NN posterior covariances  (15×15)
        '_Ps_prior' — list of NN prior covariances, index 0 is the initial P
                      (identical to Ps_post[0]); indices 1..NN-1 are the
                      predicted covariances before updates at each step
        '_Fds'      — list of NN-1 discrete transition matrices (15×15)
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

    # ── RTS storage — pre-allocate with initial values ─────────────────────────
    # xs_post[k]  : nominal state at time k  (after error injection, dx=0)
    # xs_prior[k] : nominal state at time k  AFTER IMU propagation, BEFORE updates
    #               (xs_prior[0] unused — set to initial state as placeholder)
    # Ps_post[k]  : posterior covariance at time k (after all updates)
    # Ps_prior[k] : prior (predicted) covariance at time k (before updates)
    #               Ps_prior[0] unused — set to initial P as placeholder
    # Fds[k]      : error-state Fd used for transition k→k+1  (length NN-1)
    _xs_post  = [np.concatenate([pIMU, vIMU, rpy, b_a, b_g])]
    _xs_prior = [np.concatenate([pIMU, vIMU, rpy, b_a, b_g])]  # placeholder
    _Ps_post  = [P.copy()]
    _Ps_prior = [P.copy()]   # placeholder for index 0
    _Fds      = []

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

        rpy[2] = _wrap_angle(rpy[2])

        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + g)
        vIMU   = vIMU + Ts * (accENU + g)

        # Error-state transition matrix F
        F = np.zeros((15, 15))
        F[0:3,   3:6]   = np.eye(3)
        F[3:6,   6:9]   = -_skew(accENU)
        F[3:6,   9:12]  = Rbn
        F[6:9,   12:15] = -Rbn
        F[9:12,  9:12]  = beta_acc * np.eye(3)
        F[12:15, 12:15] = beta_gyr * np.eye(3)

        Fd = np.eye(15) + F * Ts
        _Fds.append(Fd.copy())          # Fd[i]: error-state transition from i to i+1

        P  = Fd @ P @ Fd.T + Q
        _Ps_prior.append(P.copy())      # prior (error-state) P at time i+1
        dx = Fd @ dx

        # Store the nominal predicted state at i+1 (BEFORE any measurement updates).
        # This is what the RTS backward pass uses as x_{k+1}^- — NOT Fd @ x_k^+.
        _xs_prior.append(np.concatenate([pIMU, vIMU, rpy, b_a, b_g]))

        update_occurred = False

        # ── A. GPS Position Update (always on — no outage) ─────────────────────
        if nav_data.gps_available[i]:
            p_gps = np.array(pm.geodetic2enu(
                lla[i, 0], lla[i, 1], lla[i, 2], lla0[0], lla0[1], lla0[2]))
            z     = p_gps - pIMU
            innov = z - H_pos @ dx

            S = H_pos @ P @ H_pos.T + R_pos
            K = P @ H_pos.T @ np.linalg.inv(S)
            dx = dx + K @ innov
            P  = P - K @ S @ K.T
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # ── B. Non-Holonomic Constraints (NHC) ────────────────────────────────
        Rnb    = Rbn.T
        v_body = Rnb @ vIMU
        z_nhc  = -v_body[1:3]

        H_v_nhc     = Rnb[1:3, :]
        H_theta_nhc = _skew(v_body)[1:3, :]
        H_nhc       = np.hstack((H_v_nhc, H_theta_nhc))

        innov_nhc = z_nhc - H_nhc @ dx[3:9]
        P_36 = P[3:9, 3:9]
        S_nhc = H_nhc @ P_36 @ H_nhc.T + R_nhc
        K_nhc = P[:, 3:9] @ H_nhc.T @ np.linalg.inv(S_nhc)

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
            z_zupt = -vIMU
            innov_zupt = z_zupt - dx[3:6]

            S_zupt = P[3:6, 3:6] + R_zupt
            K_zupt = P[:, 3:6] @ np.linalg.inv(S_zupt)
            dx = dx + K_zupt @ innov_zupt
            P  = P - K_zupt @ S_zupt @ K_zupt.T
            P  = 0.5 * (P + P.T)
            update_occurred = True

        # ── Error injection and reset ──────────────────────────────────────────
        if update_occurred:
            pIMU   += dx[0:3]
            vIMU   += dx[3:6]
            rpy    += dx[6:9]
            b_a    += dx[9:12]
            b_g    += dx[12:15]
            rpy[2]  = _wrap_angle(rpy[2])
            dx[:]   = 0.0

        # Store forward outputs and RTS arrays
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

        _xs_post.append(np.concatenate([pIMU, vIMU, rpy, b_a, b_g]))  # posterior state at i+1
        _Ps_post.append(P.copy())                                     # posterior P at i+1

    fwd = {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_acc_out, 'bias_gyr': b_gyr_out,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_b_acc, 'std_bias_gyr': std_b_gyr,
        '_xs_post': _xs_post, '_xs_prior': _xs_prior,
        '_Ps_post': _Ps_post, '_Ps_prior': _Ps_prior, '_Fds': _Fds,
    }
    return fwd


# ── RTS backward sweep ────────────────────────────────────────────────────────

def _rts_backward(xs_post, xs_prior, Ps_post, Ps_prior, Fds):
    """
    RTS backward sweep for an error-state EKF.

    The key distinction from a standard linear KF: Fd is the error-state
    transition matrix, not a full-state transition.  Therefore the predicted
    nominal state at k+1 is xs_prior[k+1] (stored from IMU mechanization),
    NOT Fd[k] @ xs_post[k].

    RTS equations (Sarkka 2013, Ch. 8, adapted for error-state EKF):
        G_k        = P_post[k] @ Fd[k].T @ inv(P_prior[k+1])
        xs_smooth[k] = xs_post[k] + G_k @ (xs_smooth[k+1] - xs_prior[k+1])
        Ps_smooth[k] = P_post[k] + G_k @ (Ps_smooth[k+1] - P_prior[k+1]) @ G_k.T

    Args:
        xs_post  : list of NN posterior nominal states (15D), indices 0..NN-1
        xs_prior : list of NN prior nominal states (after IMU propag., before updates)
                   index 0 is unused; valid for indices 1..NN-1
        Ps_post  : list of NN posterior covariances (15×15), indices 0..NN-1
        Ps_prior : list of NN prior covariances (15×15), index 0 unused, 1..NN-1 valid
        Fds      : list of NN-1 error-state transition matrices, Fds[k]: k→k+1

    Returns:
        xs_smooth : list of NN smoothed nominal states
        Ps_smooth : list of NN smoothed covariances
    """
    N = len(xs_post)
    xs_smooth = [None] * N
    Ps_smooth = [None] * N

    xs_smooth[-1] = xs_post[-1].copy()
    Ps_smooth[-1] = Ps_post[-1].copy()

    for k in range(N - 2, -1, -1):
        # RTS gain: G = P_post[k] @ Fd[k].T @ inv(P_prior[k+1])
        # Solve P_prior[k+1] @ G.T = Fd[k] @ P_post[k]  for numerical stability.
        rhs = Fds[k] @ Ps_post[k]                              # (15, 15)
        G = np.linalg.solve(Ps_prior[k + 1], rhs).T           # (15, 15)

        # Smoothed state: use xs_prior[k+1] as the predicted nominal state,
        # NOT Fd[k] @ xs_post[k] (Fd is an error-state matrix, not full-state).
        delta = G @ (xs_smooth[k + 1] - xs_prior[k + 1])
        xs_smooth[k] = xs_post[k] + delta
        xs_smooth[k][8] = _wrap_angle(xs_smooth[k][8])        # wrap yaw (index 8)

        # Smoothed covariance
        dP = Ps_smooth[k + 1] - Ps_prior[k + 1]
        Ps_smooth[k] = Ps_post[k] + G @ dP @ G.T
        Ps_smooth[k] = 0.5 * (Ps_smooth[k] + Ps_smooth[k].T)  # symmetrise

    return xs_smooth, Ps_smooth


# ── Public interface ──────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the RTS smoother and return smoothed navigation estimates.

    The smoother always runs with full GPS access (outage_config is ignored)
    to produce the best possible reference trajectory.

    Args:
        nav_data       : NavigationData dataclass (data_loader.py).
        params         : Optional dict overriding DEFAULT_PARAMS (ekf_enhanced).
        outage_config  : Ignored. Accepted for interface compatibility only.
        use_3d_rotation: True → full roll/pitch/yaw; False → yaw-only (2D).

    Returns:
        dict with keys: p, v, r, bias_acc, bias_gyr,
                        std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
        std_* arrays reflect the smoothed (reduced) uncertainty.
    """
    # Forward pass
    fwd = _forward_pass(nav_data, params, use_3d_rotation)

    # Backward sweep
    xs_smooth, Ps_smooth = _rts_backward(
        fwd['_xs_post'], fwd['_xs_prior'],
        fwd['_Ps_post'], fwd['_Ps_prior'], fwd['_Fds'],
    )

    # Unpack smoothed states into output arrays
    N = len(xs_smooth)
    pos     = np.array([x[0:3]   for x in xs_smooth])
    vel     = np.array([x[3:6]   for x in xs_smooth])
    rpy_out = np.array([x[6:9]   for x in xs_smooth])
    b_a_out = np.array([x[9:12]  for x in xs_smooth])
    b_g_out = np.array([x[12:15] for x in xs_smooth])

    std_pos    = np.array([np.sqrt(np.maximum(np.diag(P[0:3,   0:3]),   0)) for P in Ps_smooth])
    std_vel    = np.array([np.sqrt(np.maximum(np.diag(P[3:6,   3:6]),   0)) for P in Ps_smooth])
    std_orient = np.array([np.sqrt(np.maximum(np.diag(P[6:9,   6:9]),   0)) for P in Ps_smooth])
    std_b_acc  = np.array([np.sqrt(np.maximum(np.diag(P[9:12,  9:12]),  0)) for P in Ps_smooth])
    std_b_gyr  = np.array([np.sqrt(np.maximum(np.diag(P[12:15, 12:15]), 0)) for P in Ps_smooth])

    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_a_out, 'bias_gyr': b_g_out,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_b_acc, 'std_bias_gyr': std_b_gyr,
    }
