# -*- coding: utf-8 -*-
"""
Deep Kalman Filter: Simultaneous Multi-Sensor Integration and Modelling
========================================================================
Paper : Hosseinyalamdary, "Deep Kalman Filter: Simultaneous Multi-Sensor
        Integration and Modelling; A GNSS/IMU Case Study",
        MDPI Sensors 18(5):1316, 2018
        https://doi.org/10.3390/s18051316
Code  : https://github.com/siavashha/DeepKF — incomplete Jupyter notebook,
        no neural network code.  Full implementation from scratch in this file.

Architecture (Eqs. 18-21, Figure 3)
------------------------------------
The LSTM receives the posterior state x_{t-1}^+ (15D) and predicts the next
state x_t^{+-} (15D) via residual prediction:

    x_t^{+-} = decoder(LSTM(x_{t-1}^+)) + x_{t-1}^+

This prediction replaces the strapdown-propagated prior x_t^- when GPS is
not available.  When GPS IS available, the standard ESKF update produces
x_t^+ which is more trustworthy than the DNN prediction.

Differences from the original paper
------------------------------------
1. F matrix: Solà 2017 ESKF linearized F matrix (same as esekfs_enhanced.py)
   is used for error-state covariance propagation.  The original paper ALSO
   uses an analytic F (a Noureldin / Ref. [3] nav-frame error model, Appendix
   Eq. A2) — it does NOT learn the covariance.  W_xx / W_h in the paper are the
   network's coefficient (weight) matrices of the modelling step (Eqs. 20-21),
   not P.  We reuse Solà's F for consistency with the other filters; it is more
   numerically stable and leverages the established kinematic model.

2. State space: the network input/target is the absolute 15-state
   [p, v, rpy(Euler), b_a, b_g]; the error-state propagated for the covariance
   is [δp, δv, δα, δb_a, δb_g], identical to esekfs_enhanced.py.  The original
   paper's KF state is the explicit error-state [δp, δv, δθ, b_acc, b_gyro]
   (Appendix Eq. A1) and its 3000-node latent vector h is separate; we model
   the absolute posterior because the error posterior resets to ~0 each step.

3. GPS update: standard ESKF position update (H = [I_3 | 0_3×12]) applied
   when gps_available[i] is True.  Paper also uses GPS; update structure same.

4. Training: LOO CV on KITTI clean sequences (01,04,06,07,08,09,10).
   Original paper: single urban driving dataset.

5. Coordinate frames: FLU body / ENU navigation.  The original paper uses a
   local geographic navigation frame (curvilinear lat/lon/h; see the R_M, R_N,
   tan φ terms of Appendix Eq. A2), not ECEF.

6. Outage simulation and DR_MODE: added for project compatibility, not in paper.

Weights search order:
  1. DEEP_KF_WEIGHTS env var
  2. artifacts/deep_kf/fold_<seq_id>.pt  (LOO fold)
  3. artifacts/deep_kf/deep_kf.pt        (all-sequences checkpoint)
If no weights found, run() raises RuntimeError.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent
_ARTIFACTS = _REPO_ROOT / 'artifacts/deep_kf'

DEFAULT_PARAMS = {
    # GPS measurement noise (σ² [m²])
    'Rpos': 4.0,
    # LSTM architecture
    'latent_dim': 128,
    'num_layers':  2,
    # Process noise — same structure as esekfs_enhanced.py
    'Qpos':      1e-4,
    'Qvel':      1e-3,
    'QorientXY': 1e-5,
    'QorientZ':  1e-4,
    'Qacc':      1e-6,
    'QgyrXY':    1e-7,
    'QgyrZ':     1e-6,
    # Initial covariance (1-σ std, squared inside run())
    'P_pos_std':    1.0,
    'P_vel_std':    0.5,
    'P_orient_std': 0.1,
    'P_acc_std':    0.05,
    'P_gyr_std':    0.01,
    # Classical Gauss-Markov decay (for covariance propagation)
    'beta_acc': -1e-6,
    'beta_gyr': -1e-4,
}

GRAVITY = np.array([0., 0., -9.81])


# ── Weight loading ─────────────────────────────────────────────────────────────

def _resolve_seq_id(seq_id):
    """Convert full KITTI drive name to short seq ID if needed."""
    if seq_id is None:
        return None
    if len(seq_id) <= 2:
        return seq_id
    from data_loader import KITTI_SEQ_TO_DRIVE
    _drive_to_seq = {v: k for k, v in KITTI_SEQ_TO_DRIVE.items()}
    return _drive_to_seq.get(seq_id, seq_id)


def _find_weights(seq_id: str = None) -> Path:
    """
    Locate Deep KF weights.  Search order:
      1. DEEP_KF_WEIGHTS env var (explicit override / escape hatch)
      2. artifacts/deep_kf/fold_<seq_id>.pt  (LOO fold — REQUIRED when seq_id is given)
      3. artifacts/deep_kf/deep_kf.pt        (all-sequences, only when seq_id is None)

    A missing exact fold raises instead of silently borrowing another fold or the
    all-sequences checkpoint: for a held-out LOO sequence either of those would
    train/test-leak (they were trained *including* the test sequence). For an
    intentional non-LOO test, set DEEP_KF_WEIGHTS to the checkpoint explicitly.
    """
    env = os.environ.get('DEEP_KF_WEIGHTS')
    if env and Path(env).exists():
        return Path(env)

    short_id = _resolve_seq_id(seq_id)
    if short_id is not None:
        fold = _ARTIFACTS / f'fold_{short_id}.pt'
        if fold.exists():
            return fold
        raise RuntimeError(
            f"Deep KF fold_{short_id}.pt not found in {_ARTIFACTS}. Refusing to fall "
            f"back to another fold or the all-sequences checkpoint (both would "
            f"train/test-leak for held-out sequence {short_id}).\n"
            f"  Train it:  python ins_train.py deep_kf --seqs {short_id}\n"
            f"  Intentional non-LOO test: set DEEP_KF_WEIGHTS to a checkpoint path."
        )

    default = _ARTIFACTS / 'deep_kf.pt'
    if default.exists():
        return default

    raise RuntimeError(
        "Deep KF weights not found.  Train the model first:\n"
        f"  python ins_train.py deep_kf\n"
        "or set DEEP_KF_WEIGHTS environment variable."
    )


def _load_model(weights_path: Path, latent_dim: int, num_layers: int,
                device: str = 'cpu'):
    import torch
    _dkf_dir = str(_HERE)
    if _dkf_dir not in sys.path:
        sys.path.insert(0, _dkf_dir)
    from model import DeepKFNet

    model = DeepKFNet(nav_state_dim=15,
                      hidden_dim=latent_dim, num_layers=num_layers)
    ckpt = torch.load(weights_path, map_location=device)
    norm_mean = norm_std = None
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        norm_mean = ckpt.get('norm_mean')
        norm_std  = ckpt.get('norm_std')
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    if norm_mean is not None and norm_std is not None:
        # ckpt was loaded with map_location=device, so on a GPU box these are
        # CUDA tensors; np.asarray() can't convert them directly. Move to CPU.
        if hasattr(norm_mean, 'detach'):
            norm_mean = norm_mean.detach().cpu()
        if hasattr(norm_std, 'detach'):
            norm_std = norm_std.detach().cpu()
        norm_mean = np.asarray(norm_mean, dtype=float).reshape(15)
        norm_std  = np.asarray(norm_std,  dtype=float).reshape(15)
    return model, norm_mean, norm_std


# ── Quaternion utilities (copied from esekfs_enhanced.py) ───────────────────────

def _skew(v):
    return np.array([
        [ 0.,    -v[2],  v[1]],
        [ v[2],   0.,   -v[0]],
        [-v[1],   v[0],  0.  ],
    ])


def _qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 0. else np.array([1., 0., 0., 0.])


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
        return _qnorm(np.array([1., 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
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
    roll  = np.arctan2(2.*(w*x + y*z), 1. - 2.*(x*x + y*y))
    pitch = np.arcsin(np.clip(2.*(w*y - z*x), -1., 1.))
    yaw   = np.arctan2(2.*(w*z + x*y), 1. - 2.*(y*y + z*z))
    return np.array([roll, pitch, yaw])


def _qto_Rbn(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),     2*(x*z+y*w)  ],
        [2*(x*y+z*w),     1-2*(x*x+z*z),   2*(y*z-x*w)  ],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x*x+y*y)],
    ])


def _wrap(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (a + np.pi) % (2. * np.pi) - np.pi


def _free_dead_reckoning(accel_flu, gyro_flu, p0, v0, q0, Ts, N,
                         use_3d_rotation=True):
    """
    Open-loop strapdown mechanisation — the free-running (uncorrected) IMU
    trajectory the paper's feedforward filter (pyins.FeedforwardFilter) estimates
    the error against.  No GPS, no bias correction, no error feedback.

    Returns absolute position, velocity (N,3) and roll-pitch-yaw (N,3).
    """
    p_dr  = np.zeros((N, 3))
    v_dr  = np.zeros((N, 3))
    th_dr = np.zeros((N, 3))
    p = np.asarray(p0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    q = q0.copy()
    p_dr[0], v_dr[0], th_dr[0] = p, v, _qto_rpy(q)
    for i in range(N - 1):
        omega_b = gyro_flu[i]
        acc_b   = accel_flu[i]
        dtheta  = omega_b * Ts if use_3d_rotation else np.array([0., 0., omega_b[2] * Ts])
        q = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        a_enu = _qto_Rbn(q) @ acc_b
        p = p + Ts * v + 0.5 * Ts**2 * (a_enu + GRAVITY)
        v = v + Ts * (a_enu + GRAVITY)
        p_dr[i+1], v_dr[i+1], th_dr[i+1] = p, v, _qto_rpy(q)
    return p_dr, v_dr, th_dr


# ── Main filter ───────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True,
        backend='cpu', hailo_net=None):
    """
    Run the Deep KF filter on nav_data.

    The LSTM predicts the full 15D state x_t^{+-} at each step, replacing the
    strapdown-propagated prior x_t^-.  Covariance propagation still uses the
    linearized Solà F matrix.

    Parameters
    ----------
    nav_data       : NavigationData (data_loader.py)
    params         : Optional dict overriding DEFAULT_PARAMS.
    outage_config  : Optional {'start': t1_s, 'duration': d_s}.
    use_3d_rotation: True -> full 3D strapdown; False -> yaw-only (2D).
    backend        : 'cpu' (default, unchanged behaviour — error-state EKF
                     with the LSTM run on torch) or 'hailo' (see
                     _run_hailo below — a different, simplified algorithm;
                     the Hailo-compiled network is NOT calibration-
                     compatible with the error-state input the CPU path
                     uses, see hailo_backend.HailoDeepKF's docstring).
    hailo_net      : Required when backend='hailo' — an already-activated
                     hailo_backend.HailoDeepKF instance (caller owns its
                     lifecycle so device open/close isn't timed).

    Returns
    -------
    dict with keys: p, v, r, bias_acc, bias_gyr,
                    std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr,
                    net_latency_s (per-network-call wall time, seconds).
    All arrays shape (N, 3), dtype float64 (net_latency_s is 1-D).
    """
    if backend == 'hailo':
        return _run_hailo(nav_data, params, outage_config, hailo_net)

    import torch

    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────
    accel_flu   = nav_data.accel_flu
    gyro_flu    = nav_data.gyro_flu
    orient      = nav_data.orient
    lla         = nav_data.lla
    vel_enu     = nav_data.vel_enu
    sample_rate = nav_data.sample_rate
    lla0        = nav_data.lla0

    import pymap3d as pm
    N  = accel_flu.shape[0]
    Ts = 1.0 / sample_rate

    # ── Load model ────────────────────────────────────────────────────────
    seq_id = getattr(nav_data, 'dataset_name', None)
    weights_path = _find_weights(seq_id)
    latent_dim = int(p_cfg['latent_dim'])
    num_layers  = int(p_cfg['num_layers'])
    model, norm_mean, norm_std = _load_model(weights_path, latent_dim, num_layers, device)
    print(f"Deep KF: loaded weights from {weights_path} (device={device})")

    # ── GPS outage mask ────────────────────────────────────────────────────
    # Narrow to ImportError (don't mask a bug inside ins_config) and surface the
    # fallback — dr_mode toggles GPS aiding, so a silent wrong default drifts.
    try:
        import ins_config as _ic
        dr_mode = getattr(_ic, 'DR_MODE', False)
    except ImportError as _e:
        print(f"Deep KF: WARNING — ins_config not importable ({_e}); "
              "defaulting DR_MODE=False (GPS aiding ON).")
        dr_mode = False

    gps_avail = nav_data.gps_available.copy()
    if outage_config is not None:
        t1 = outage_config.get('start', 0.)
        d  = outage_config.get('duration', 0.)
        A  = int(t1 * sample_rate)
        B  = int((t1 + d) * sample_rate)
        gps_avail[A:B] = False
    if dr_mode:
        gps_avail[:] = False

    # ── GPS positions in ENU ───────────────────────────────────────────────
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0], lla0[1], lla0[2])
    p_gps_enu = np.column_stack([e, n, u])

    # ── Output arrays ──────────────────────────────────────────────────────
    pos        = np.zeros((N, 3))
    vel        = np.zeros((N, 3))
    rpy_out    = np.zeros((N, 3))
    b_acc_out  = np.zeros((N, 3))
    b_gyr_out  = np.zeros((N, 3))
    std_pos    = np.zeros((N, 3))
    std_vel    = np.zeros((N, 3))
    std_orient = np.zeros((N, 3))
    std_b_acc  = np.zeros((N, 3))
    std_b_gyr  = np.zeros((N, 3))

    # ── Feedforward initialisation ─────────────────────────────────────────
    pos[0]     = p_gps_enu[0]
    vel[0]     = vel_enu[0]
    rpy_out[0] = orient[0]

    if norm_mean is None or norm_std is None:
        raise RuntimeError(
            "Deep KF weights lack error-normalisation stats (norm_mean/norm_std). "
            "This checkpoint predates the feedforward error-state model — retrain "
            "with:  python ins_train.py deep_kf")

    # Free-running (open-loop) IMU trajectory — the paper estimates the error
    # against this uncorrected mechanisation (pyins.FeedforwardFilter), not
    # against a GPS-corrected nominal.
    q0 = _qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2])
    p_dr, v_dr, th_dr = _free_dead_reckoning(
        accel_flu, gyro_flu, p_gps_enu[0], vel_enu[0], q0, Ts, N, use_3d_rotation)

    # Feedforward error state e = [δp, δv, δθ, b_a, b_g]; accumulates (no reset).
    e = np.zeros(15)

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.eye(3) * p_cfg['Qpos']
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.diag([p_cfg['QorientXY'], p_cfg['QorientXY'], p_cfg['QorientZ']])
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc']  * Ts)
    Q[12:15, 12:15] = np.diag([p_cfg['QgyrXY'], p_cfg['QgyrXY'], p_cfg['QgyrZ']]) * Ts

    P = np.diag([
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'],
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_pos = np.eye(3) * p_cfg['Rpos']

    # ── LSTM (feedforward error model) state ────────────────────────────────
    hidden = model.init_hidden(batch_size=1, device=device)
    net_latency_s = np.zeros(N - 1)

    print(f"Deep KF: running feedforward error filter on {N} samples ...")

    for i in range(N - 1):

        # ── A. Covariance propagation (Solà F around the free-IMU nominal) ──
        acc_b   = accel_flu[i]          # raw specific force (open-loop DR)
        omega_b = gyro_flu[i]           # raw angular rate
        Rbn = _qto_Rbn(_qfrom_euler(th_dr[i, 0], th_dr[i, 1], th_dr[i, 2]))

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
        P = Fd @ P @ Fd.T + Q

        # ── B. LSTM forward on the previous error (normalised) ──────────────
        # Faithful to §3, the network replaces the prior only "in the absence of
        # GNSS": under aiding the ordinary feedforward EKF runs and the LSTM is
        # only kept warm (output ignored); under outage the LSTM prediction is
        # the prior.  Either way the network sees the previous error once/step.
        e_norm_in = (e - norm_mean) / norm_std
        _t0 = time.perf_counter()
        with torch.no_grad():
            e_t = torch.from_numpy(e_norm_in).float().unsqueeze(0).to(device)
            e_pred_t, hidden = model(e_t, hidden)
            e_lstm = e_pred_t[0].cpu().numpy() * norm_std + norm_mean
        net_latency_s[i] = time.perf_counter() - _t0

        if gps_avail[i + 1]:
            # ── C. AIDING: analytic feedforward EKF prior + GPS position update
            e = Fd @ e                                            # Solà-F error prior
            corrected_p = p_dr[i + 1] + e[0:3]
            innov     = p_gps_enu[i + 1] - corrected_p
            S_gps     = P[0:3, 0:3] + R_pos
            S_gps_reg = S_gps + 1e-9 * np.eye(3)
            K_gps     = np.linalg.solve(S_gps_reg, P[0:3, :]).T   # 15×3
            e         = e + K_gps @ innov
            H_gps     = np.zeros((3, 15)); H_gps[:, 0:3] = np.eye(3)
            IKH_gps   = np.eye(15) - K_gps @ H_gps
            P         = IKH_gps @ P @ IKH_gps.T + K_gps @ R_pos @ K_gps.T  # Joseph
            P         = 0.5 * (P + P.T)
        else:
            # ── C. OUTAGE: LSTM prediction replaces the analytic prior ──────
            e = e_lstm

        # ── D. Output = free IMU + estimated error (feedforward add-back) ────
        pos[i+1]       = p_dr[i+1] + e[0:3]
        vel[i+1]       = v_dr[i+1] + e[3:6]
        rpy_out[i+1]   = _wrap(th_dr[i+1] + e[6:9])
        b_acc_out[i+1] = e[9:12]
        b_gyr_out[i+1] = e[12:15]
        std_pos[i+1]      = np.sqrt(np.maximum(np.diag(P[0:3,   0:3]),   0.))
        std_vel[i+1]      = np.sqrt(np.maximum(np.diag(P[3:6,   3:6]),   0.))
        std_orient[i+1]   = np.sqrt(np.maximum(np.diag(P[6:9,   6:9]),   0.))
        std_b_acc[i+1]    = np.sqrt(np.maximum(np.diag(P[9:12,  9:12]),  0.))
        std_b_gyr[i+1]    = np.sqrt(np.maximum(np.diag(P[12:15, 12:15]), 0.))

    return {
        'p':            pos,
        'v':            vel,
        'r':            rpy_out,
        'bias_acc':     b_acc_out,
        'bias_gyr':     b_gyr_out,
        'std_pos':      std_pos,
        'std_vel':      std_vel,
        'std_orient':   std_orient,
        'std_bias_acc': std_b_acc,
        'std_bias_gyr': std_b_gyr,
        'net_latency_s': net_latency_s,
    }


def _run_hailo(nav_data, params, outage_config, hailo_net):
    """
    backend='hailo' path — a standalone autoregressive full-state predictor,
    NOT the error-state EKF the CPU path runs (see hailo_backend.HailoDeepKF's
    docstring for why: the compiled HEF was quantisation-calibrated on real
    full nav states, not on the small normalised error-state residuals the
    CPU EKF feeds its LSTM — feeding it error-state values would be invalid,
    out-of-distribution input for its INT8 quantisation range).

    Same trained LSTM+decoder weights, applied causally tick-by-tick via
    HailoRT's persistent recurrent state across calls (one continuous
    `activate()` for the whole run — see hailo_backend._BaseHailoNet).  GPS,
    when available, hard-resets the position component (no covariance-based
    gain — this path has no P/std to report); the network's own prediction
    carries velocity/orientation/biases forward at all times, and *is* the
    whole state during outage.  This is the only role the HEF is actually
    compatible with, so it's what gets measured.
    """
    p_cfg = dict(DEFAULT_PARAMS)
    if params:
        p_cfg.update(params)

    accel_flu = nav_data.accel_flu
    orient    = nav_data.orient
    lla       = nav_data.lla
    vel_enu   = nav_data.vel_enu
    lla0      = nav_data.lla0
    sample_rate = nav_data.sample_rate

    import pymap3d as pm
    N = accel_flu.shape[0]

    # Narrow to ImportError + surface the fallback (see the CPU-path note above);
    # this is the Hailo backend path, same dr_mode semantics.
    try:
        import ins_config as _ic
        dr_mode = getattr(_ic, 'DR_MODE', False)
    except ImportError as _e:
        print(f"Deep KF (hailo): WARNING — ins_config not importable ({_e}); "
              "defaulting DR_MODE=False (GPS aiding ON).")
        dr_mode = False

    gps_avail = nav_data.gps_available.copy()
    if outage_config is not None:
        t1 = outage_config.get('start', 0.)
        d  = outage_config.get('duration', 0.)
        A  = int(t1 * sample_rate)
        B  = int((t1 + d) * sample_rate)
        gps_avail[A:B] = False
    if dr_mode:
        gps_avail[:] = False

    e, n, u = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], lla0[0], lla0[1], lla0[2])
    p_gps_enu = np.column_stack([e, n, u])

    pos       = np.zeros((N, 3))
    vel       = np.zeros((N, 3))
    rpy_out   = np.zeros((N, 3))
    b_acc_out = np.zeros((N, 3))
    b_gyr_out = np.zeros((N, 3))
    zeros3    = np.zeros((N, 3))
    net_latency_s = np.zeros(N - 1)

    pos[0]     = p_gps_enu[0]
    vel[0]     = vel_enu[0]
    rpy_out[0] = orient[0]

    x_full = np.concatenate([pos[0], vel[0], rpy_out[0], np.zeros(3), np.zeros(3)])

    print(f"Deep KF (Hailo, standalone full-state predictor): running on {N} samples ...")

    for i in range(N - 1):
        x_pred, dt = hailo_net.step(x_full)
        net_latency_s[i] = dt

        if gps_avail[i + 1]:
            x_full = x_pred.copy()
            x_full[0:3] = p_gps_enu[i + 1]
        else:
            x_full = x_pred

        pos[i+1]       = x_full[0:3]
        vel[i+1]       = x_full[3:6]
        rpy_out[i+1]   = _wrap(x_full[6:9])
        b_acc_out[i+1] = x_full[9:12]
        b_gyr_out[i+1] = x_full[12:15]

    return {
        'p': pos, 'v': vel, 'r': rpy_out,
        'bias_acc': b_acc_out, 'bias_gyr': b_gyr_out,
        'std_pos': zeros3, 'std_vel': zeros3, 'std_orient': zeros3,
        'std_bias_acc': zeros3, 'std_bias_gyr': zeros3,
        'net_latency_s': net_latency_s,
    }
