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

Differences from the original paper:
--------------------------------------
1. F matrix: Solà 2017 ESKF linearized F matrix (same as eskf_enhanced.py)
   is used for error-state covariance propagation.  The original paper learns
   a generic W_xx matrix via gradient descent (Eqs. 24–32).  Solà's F is
   more numerically stable and leverages the established kinematic model.

2. State space: 15-state ESKF [δp, δv, δφ, δb_a, δb_g] in navigation space,
   identical to eskf_enhanced.py.  The original paper uses a generic latent
   vector state.

3. LSTM role: learns [δb_acc(3), δb_gyr(3)] additive IMU bias corrections in
   the FLU body frame.  These replace/augment the Gauss-Markov bias model used
   in classical ESKF filters.  The original paper has the LSTM learn the full
   system model.

4. GPS update: standard ESKF position update (H = [I_3 | 0_3×12]) applied
   when gps_available[i] is True.  Paper also uses GPS; update structure same.

5. Training: LOO CV on KITTI clean sequences (01,04,06,07,08,09,10).
   Original paper: single urban driving dataset.

6. Coordinate frames: FLU body / ENU navigation.  Original paper unspecified
   (likely ECEF-derived).

7. Outage simulation and DR_MODE: added for project compatibility, not in paper.

Weights search order:
  1. DEEP_KF_WEIGHTS env var
  2. artifacts/deep_kf/fold_<seq_id>.pt  (LOO fold)
  3. artifacts/deep_kf/deep_kf.pt        (all-sequences checkpoint)
If no weights found, run() raises RuntimeError.
"""

import os
import sys
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
    'imu_window_seconds': 1.0,   # rate-agnostic (= 100 samples at 100 Hz)
    # Process noise — same structure as eskf_enhanced.py
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
    # Classical Gauss-Markov decay (kept for bias fallback when LSTM not available)
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
      1. DEEP_KF_WEIGHTS env var
      2. artifacts/deep_kf/fold_<seq_id>.pt  (LOO fold)
      3. artifacts/deep_kf/deep_kf.pt        (all-sequences checkpoint)
      4. Any available fold_*.pt             (fallback with warning)
    """
    env = os.environ.get('DEEP_KF_WEIGHTS')
    if env and Path(env).exists():
        return Path(env)

    short_id = _resolve_seq_id(seq_id)
    if short_id is not None:
        fold = _ARTIFACTS / f'fold_{short_id}.pt'
        if fold.exists():
            return fold

    default = _ARTIFACTS / 'deep_kf.pt'
    if default.exists():
        return default

    # Fallback: use any available fold
    available = sorted(_ARTIFACTS.glob('fold_*.pt'))
    available = [p for p in available if '_ckpt' not in p.name]
    if available:
        print(f"WARNING: Deep KF fold_{short_id}.pt not found, falling back to {available[0].name}. "
              f"Train the proper fold with: python ins_train.py deep_kf --seqs {short_id}")
        return available[0]

    raise RuntimeError(
        "Deep KF weights not found.  Train the model first:\n"
        f"  python ins_train.py deep_kf\n"
        "or set DEEP_KF_WEIGHTS environment variable."
    )


def _load_model(weights_path: Path, latent_dim: int, num_layers: int,
                device: str = 'cpu'):
    import torch
    # Local import to avoid circular dependency
    _dkf_dir = str(_HERE)
    if _dkf_dir not in sys.path:
        sys.path.insert(0, _dkf_dir)
    from model import DeepKFNet

    model = DeepKFNet(nav_state_dim=15, imu_dim=6,
                      hidden_dim=latent_dim, num_layers=num_layers)
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


# ── Quaternion utilities (copied from eskf_enhanced.py) ───────────────────────

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


# ── Main filter ───────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the Deep KF filter on nav_data.

    Parameters
    ----------
    nav_data       : NavigationData (data_loader.py)
    params         : Optional dict overriding DEFAULT_PARAMS.
    outage_config  : Optional {'start': t1_s, 'duration': d_s}.
    use_3d_rotation: True → full 3D strapdown; False → yaw-only (2D).

    Returns
    -------
    dict with keys: p, v, r, bias_acc, bias_gyr,
                    std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr.
    All arrays shape (N, 3), dtype float64.
    """
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

    # IMU window size for LSTM (one step per window_seconds)
    T_win = int(p_cfg['imu_window_seconds'] * sample_rate)   # 100 @ 100 Hz

    # ── Load model ────────────────────────────────────────────────────────
    seq_id = getattr(nav_data, 'dataset_name', None)
    weights_path = _find_weights(seq_id)
    latent_dim = int(p_cfg['latent_dim'])
    num_layers  = int(p_cfg['num_layers'])
    model = _load_model(weights_path, latent_dim, num_layers, device)
    print(f"Deep KF: loaded weights from {weights_path} (device={device})")

    # ── GPS outage mask ────────────────────────────────────────────────────
    try:
        import ins_config as _ic
        dr_mode = getattr(_ic, 'DR_MODE', False)
    except Exception:
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

    # ── ESKF initialisation ────────────────────────────────────────────────
    pos[0]     = p_gps_enu[0]
    vel[0]     = vel_enu[0]
    rpy_out[0] = orient[0]

    pIMU = pos[0].copy()
    vIMU = vel[0].copy()
    q    = _qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2])
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)

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

    # ── LSTM state ─────────────────────────────────────────────────────────
    hidden = model.init_hidden(batch_size=1, device=device)
    # LSTM bias corrections (FLU body frame)
    db_acc = np.zeros(3)
    db_gyr = np.zeros(3)

    # IMU window buffer for LSTM modelling step
    imu_buf_acc = np.zeros((T_win, 3))  # rolling buffer
    imu_buf_gyr = np.zeros((T_win, 3))

    print(f"Deep KF: running filter on {N} samples ...")

    for i in range(N - 1):

        # ── A. Correct IMU with LSTM bias estimate ─────────────────────
        acc_b_raw = accel_flu[i]
        gyr_b_raw = gyro_flu[i]
        acc_b   = acc_b_raw - db_acc      # corrected acceleration (FLU)
        omega_b = gyr_b_raw - db_gyr      # corrected angular rate (FLU)

        # Update rolling IMU buffer
        buf_idx = i % T_win
        imu_buf_acc[buf_idx] = acc_b_raw
        imu_buf_gyr[buf_idx] = gyr_b_raw

        # ── B. Strapdown propagation (Solà 2017) ──────────────────────
        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0., 0., omega_b[2] * Ts])

        q    = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn  = _qto_Rbn(q)
        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + GRAVITY)
        vIMU   = vIMU + Ts * (accENU + GRAVITY)

        # ── C. Covariance propagation (Solà F matrix) ─────────────────
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

        # ── D. GPS position update ─────────────────────────────────────
        if gps_avail[i + 1]:
            z_pos = p_gps_enu[i + 1] - pIMU
            innov = z_pos - dx[0:3]
            S_gps     = P[0:3, 0:3] + R_pos
            S_gps_reg = S_gps + 1e-9 * np.eye(3)
            K_gps     = np.linalg.solve(S_gps_reg, P[0:3, :]).T   # 15×3
            dx        = dx + K_gps @ innov
            H_gps     = np.zeros((3, 15)); H_gps[:, 0:3] = np.eye(3)
            IKH_gps   = np.eye(15) - K_gps @ H_gps
            P         = IKH_gps @ P @ IKH_gps.T + K_gps @ R_pos @ K_gps.T  # Joseph form
            P         = 0.5 * (P + P.T)
            update_occurred = True

        # ── E. Error injection ──────────────────────────────────────────
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
            dx[:]       = 0.

        # ── F. LSTM modelling step (paper Figure 3 bottom block) ───────
        # Run at each IMU step after the classical update.
        # Build nav-state vector (post-update nominal state packed as dx-like)
        # We use the nominal state values directly (not error state).
        rpy_now = _qto_rpy(q)
        x_post_np = np.concatenate([pIMU, vIMU, rpy_now, b_a, b_g])  # (15,)

        # IMU window mean (mean of the rolling buffer)
        imu_mean_np = np.concatenate([
            imu_buf_acc.mean(axis=0),   # (3,)
            imu_buf_gyr.mean(axis=0),   # (3,)
        ])  # (6,)

        with torch.no_grad():
            x_post_t  = torch.from_numpy(x_post_np).float().unsqueeze(0).to(device)
            imu_mean_t = torch.from_numpy(imu_mean_np).float().unsqueeze(0).to(device)
            bias_t, hidden = model(x_post_t, imu_mean_t, hidden)
            bias_np = bias_t[0].cpu().numpy()

        db_acc = bias_np[0:3]   # δb_acc in FLU [m/s²]
        db_gyr = bias_np[3:6]   # δb_gyr in FLU [rad/s]

        # ── G. Store outputs ───────────────────────────────────────────
        pos[i+1]       = pIMU
        vel[i+1]       = vIMU
        rpy_out[i+1]   = _qto_rpy(q)
        b_acc_out[i+1] = db_acc       # LSTM-decoded bias (interpretable)
        b_gyr_out[i+1] = db_gyr
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
    }
