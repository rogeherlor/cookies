# -*- coding: utf-8 -*-
"""
TLIO: Tight Learned Inertial Odometry
======================================
Paper : Liu et al., "TLIO: Tight Learned Inertial Odometry", IEEE RA-L 2020
        https://doi.org/10.1109/LRA.2020.3007421  |  arXiv:2007.01867
Code  : https://github.com/CathIAS/TLIO  (cloned to external/tlio/)

Differences from the original paper / repository:
---------------------------------------------------
1. IMU rate: 100 Hz (this project) vs 200 Hz (original paper).
   Window W = int(2.0 * nav_data.sample_rate) = 200 samples @ 100 Hz.
   net_config["in_dim"] = W // 32 + 1 = 7 @ 100 Hz (matches imu_tracker.py line 118).

2. Platform: wheeled vehicle (KITTI) vs pedestrian headset (original).
   Network is retrained from scratch on KITTI; pedestrian pretrained weights
   are not used.

3. GPS integration: GPS position updates added to the 15-state ESKF when
   gps_available[i] is True.  The original paper is IMU-only (no GPS).

4. EKF state: 15-state ESKF [δp, δv, δφ, δb_a, δb_g] in navigation space,
   reusing the Solà 2017 F matrix (same as eskf_enhanced.py).  The original
   paper uses a full stochastic-cloning MSCKF with augmented past-pose states.

5. Displacement update: simplified vs the original stochastic-cloning EKF.
   We treat the predicted displacement as a position innovation w.r.t. the
   anchor position saved at each stride boundary
   (H = [I_3 | 0_3×12], standard position-error measurement).

6. Training protocol: Leave-One-Out CV on KITTI clean sequences
   (01, 04, 06, 07, 08, 09, 10) — see train_tlio.py.
   Original paper uses a custom pedestrian split.

7. Coordinate frames: FLU body / ENU navigation (this project).
   Original paper uses a world frame anchored to the first pose.
   Gravity-aligned preprocessing is identical: R_ga = Ry(pitch) @ Rx(roll).

8. Outage simulation and DR_MODE: added for project compatibility, not in paper.

Weights search order:
  1. TLIO_WEIGHTS env var
  2. artifacts/tlio/tlio_resnet.pt
  3. artifacts/tlio/fold_<seq>.pt  (LOO fold matching current sequence)
If no weights are found, run() raises RuntimeError with instructions.
"""

import os
import sys
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent   # cookies/
_TLIO_SRC  = _REPO_ROOT / 'external/tlio/src'
_ARTIFACTS = _REPO_ROOT / 'artifacts/tlio'

DEFAULT_PARAMS = {
    # GPS measurement noise (σ² [m²])
    'Rpos': 4.0,
    # Window and stride in seconds (rate-agnostic)
    'window_seconds': 2.0,
    'stride_seconds': 0.5,
    # Process noise
    'Qpos':     1e-4,
    'Qvel':     1e-3,
    'Qorient':  1e-5,
    'Qacc':     1e-6,
    'Qgyr':     1e-7,
    # Initial covariance (1-σ std, squared inside run())
    'P_pos_std':    1.0,
    'P_vel_std':    0.5,
    'P_orient_std': 0.1,
    'P_acc_std':    0.05,
    'P_gyr_std':    0.01,
    # Gauss-Markov bias decay (set to 0 → random-walk)
    'beta_acc': 0.0,
    'beta_gyr': 0.0,
}


# ── Model loading ─────────────────────────────────────────────────────────────

def _add_tlio_src():
    src = str(_TLIO_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _find_weights(seq_id: str = None) -> Path:
    """
    Locate TLIO weights file.  Search order:
      1. TLIO_WEIGHTS env var
      2. artifacts/tlio/fold_<seq_id>.pt  (LOO fold)
      3. artifacts/tlio/tlio_resnet.pt    (all-sequences checkpoint)
    Raises RuntimeError if nothing found.
    """
    env = os.environ.get('TLIO_WEIGHTS')
    if env and Path(env).exists():
        return Path(env)

    if seq_id is not None:
        fold = _ARTIFACTS / f'fold_{seq_id}.pt'
        if fold.exists():
            return fold

    default = _ARTIFACTS / 'tlio_resnet.pt'
    if default.exists():
        return default

    raise RuntimeError(
        "TLIO weights not found.  Train the model first:\n"
        f"  python dl_filters/tlio/train_tlio.py --mode all --output {_ARTIFACTS}\n"
        "or set the TLIO_WEIGHTS environment variable to an existing .pt file."
    )


def _load_model(weights_path: Path, window_size: int, device: str = 'cpu'):
    """Load TLIO ResNet1D model and return it in eval mode."""
    import torch
    _add_tlio_src()
    from network.model_factory import get_model

    # in_dim formula from imu_tracker.py line 118
    in_dim = window_size // 32 + 1
    net_config = {'in_dim': in_dim}

    # output_dim=3: ResNet1D has two FcBlock heads (mean + logstd), each dim=3
    model = get_model('resnet', net_config, input_dim=6, output_dim=3)
    state = torch.load(weights_path, map_location=device)
    # Support both raw state_dict and checkpoint dicts
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Quaternion / rotation utilities ─────────────────────────────────────────
# (copied from eskf_enhanced.py to keep this module self-contained)

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


def _Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0., sa], [0., 1., 0.], [-sa, 0., ca]])


def _Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., ca, -sa], [0., sa, ca]])


def _Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])


GRAVITY = np.array([0., 0., -9.81])


# ── Network inference helper ──────────────────────────────────────────────────

def _predict_displacement(model, imu_window: np.ndarray, device: str = 'cpu'):
    """
    Run one forward pass and decode displacement + covariance.

    Parameters
    ----------
    imu_window : (6, W) float32 — [gyro_ga | accel_ga_motion]

    Returns
    -------
    dp_ga    : (3,) float64 — predicted displacement in gravity-aligned frame
    Sigma_ga : (3, 3) float64 — diagonal predicted covariance in GA frame
    """
    import torch
    x = torch.from_numpy(imu_window[None]).float().to(device)  # (1, 6, W)
    with torch.no_grad():
        mean, logstd = model(x)   # each (1, 3)
    dp_ga    = mean[0].cpu().numpy().astype(np.float64)
    log_std  = logstd[0].cpu().numpy().astype(np.float64)
    # Eq. 3 of the paper: Σ = diag(exp(2·û_i))
    std2     = np.exp(2. * log_std)
    Sigma_ga = np.diag(std2)
    return dp_ga, Sigma_ga


# ── Main filter ───────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the TLIO filter on nav_data.

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

    # ── Determine torch device ─────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Data ──────────────────────────────────────────────────────────────
    accel_flu = nav_data.accel_flu    # (N, 3)
    gyro_flu  = nav_data.gyro_flu     # (N, 3)
    orient    = nav_data.orient       # (N, 3) [roll, pitch, yaw]
    lla       = nav_data.lla
    vel_enu   = nav_data.vel_enu
    sample_rate = nav_data.sample_rate
    lla0      = nav_data.lla0

    import pymap3d as pm
    lla0 = nav_data.lla0
    N = accel_flu.shape[0]
    Ts = 1.0 / sample_rate

    # ── Window parameters ─────────────────────────────────────────────────
    W = int(p_cfg['window_seconds'] * sample_rate)   # 200 @ 100 Hz
    S = int(p_cfg['stride_seconds'] * sample_rate)   # 50  @ 100 Hz

    # ── Weights ───────────────────────────────────────────────────────────
    seq_id = getattr(nav_data, 'dataset_name', None)
    weights_path = _find_weights(seq_id)
    model = _load_model(weights_path, W, device)
    print(f"TLIO: loaded weights from {weights_path}  (W={W}, S={S}, device={device})")

    # ── GPS outage mask ───────────────────────────────────────────────────
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

    # ── GPS positions in ENU ──────────────────────────────────────────────
    e, n, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0], lla0[1], lla0[2])
    p_gps_enu = np.column_stack([e, n, u])      # (N, 3)

    # ── Output arrays ─────────────────────────────────────────────────────
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

    # ── ESKF initialisation ───────────────────────────────────────────────
    pos[0]     = p_gps_enu[0]
    vel[0]     = vel_enu[0]
    rpy_out[0] = orient[0]

    pIMU = pos[0].copy()
    vIMU = vel[0].copy()
    q    = _qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2])
    b_a  = np.zeros(3)
    b_g  = np.zeros(3)
    dx   = np.zeros(15)

    Q = np.zeros((15, 15))
    Q[0:3,   0:3]   = np.eye(3) * p_cfg['Qpos']
    Q[3:6,   3:6]   = np.eye(3) * (p_cfg['Qvel'] * Ts**2)
    Q[6:9,   6:9]   = np.eye(3) * p_cfg['Qorient']
    Q[9:12,  9:12]  = np.eye(3) * (p_cfg['Qacc']  * Ts)
    Q[12:15, 12:15] = np.eye(3) * (p_cfg['Qgyr']  * Ts)

    P = np.diag([
        p_cfg['P_pos_std'],    p_cfg['P_pos_std'],    p_cfg['P_pos_std'],
        p_cfg['P_vel_std'],    p_cfg['P_vel_std'],    p_cfg['P_vel_std'],
        p_cfg['P_orient_std'], p_cfg['P_orient_std'], p_cfg['P_orient_std'],
        p_cfg['P_acc_std'],    p_cfg['P_acc_std'],    p_cfg['P_acc_std'],
        p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],    p_cfg['P_gyr_std'],
    ]) ** 2

    R_pos  = np.eye(3) * p_cfg['Rpos']

    beta_acc = p_cfg['beta_acc']
    beta_gyr = p_cfg['beta_gyr']

    # ── Displacement update book-keeping ──────────────────────────────────
    # anchor_pos : ENU position saved at each stride boundary
    # anchor_idx : IMU index of the last anchor
    anchor_pos = pIMU.copy()
    anchor_idx = 0
    next_update_idx = W   # first network update happens when the window is full

    # Pre-build gravity-aligned windows for every stride boundary
    # We do this on-the-fly inside the loop for memory efficiency, but we
    # pre-cache the IMU array views for speed.

    print(f"TLIO: running filter on {N} samples ...")

    for i in range(N - 1):
        # ── A. IMU strapdown propagation ───────────────────────────────
        acc_b   = accel_flu[i] - b_a
        omega_b = gyro_flu[i]  - b_g

        if use_3d_rotation:
            dtheta = omega_b * Ts
        else:
            dtheta = np.array([0., 0., omega_b[2] * Ts])

        q    = _qnorm(_qmul(q, _qfrom_axis_angle(dtheta)))
        Rbn  = _qto_Rbn(q)
        accENU = Rbn @ acc_b
        pIMU   = pIMU + Ts * vIMU + 0.5 * Ts**2 * (accENU + GRAVITY)
        vIMU   = vIMU + Ts * (accENU + GRAVITY)

        # ── B. Covariance propagation (Solà 2017 F matrix) ────────────
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

        # ── C. TLIO displacement update (at stride boundaries) ─────────
        if (i + 1) == next_update_idx and (i + 1) >= W:
            # Build gravity-aligned window [i+1-W : i+1]
            ws  = i + 1 - W
            roll_ws  = orient[ws, 0]
            pitch_ws = orient[ws, 1]
            yaw_ws   = orient[ws, 2]

            R_ga = _Ry(pitch_ws) @ _Rx(roll_ws)
            accel_w  = accel_flu[ws:ws + W]     # (W, 3)
            gyro_w   = gyro_flu[ws:ws + W]      # (W, 3)
            accel_ga = R_ga @ accel_w.T          # (3, W)
            gyro_ga  = R_ga @ gyro_w.T           # (3, W)
            accel_ga[2, :] -= 9.81               # remove gravity

            imu_win = np.vstack([gyro_ga, accel_ga]).astype(np.float32)  # (6, W)

            dp_ga, Sigma_ga = _predict_displacement(model, imu_win, device)

            # Rotate prediction to ENU
            Rz = _Rz(yaw_ws)
            dp_enu    = Rz @ dp_ga
            Sigma_enu = Rz @ Sigma_ga @ Rz.T

            # Innovation: measured displacement - estimated displacement
            dp_estimated = pIMU - anchor_pos    # estimated ENU displacement since anchor
            z = dp_enu - dp_estimated

            # H selects position error state (indices 0:3)
            S = P[0:3, 0:3] + Sigma_enu
            K = P[:, 0:3] @ np.linalg.inv(S)
            dx    = dx + K @ (z - dx[0:3])
            P     = P - K @ S @ K.T
            P     = 0.5 * (P + P.T)
            update_occurred = True

            # Advance anchor and next update
            anchor_pos = pIMU.copy()
            anchor_idx = i + 1
            next_update_idx = i + 1 + S

        # ── D. GPS position update ─────────────────────────────────────
        if gps_avail[i + 1]:
            z_pos = p_gps_enu[i + 1] - pIMU
            innov = z_pos - dx[0:3]
            S_gps = P[0:3, 0:3] + R_pos
            K_gps = P[:, 0:3] @ np.linalg.inv(S_gps)
            dx    = dx + K_gps @ innov
            P     = P - K_gps @ S_gps @ K_gps.T
            P     = 0.5 * (P + P.T)
            update_occurred = True

        # ── E. Error injection (Solà §7.3) ────────────────────────────
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

        # ── F. Store outputs ──────────────────────────────────────────
        pos[i+1]       = pIMU
        vel[i+1]       = vIMU
        rpy_out[i+1]   = _qto_rpy(q)
        b_acc_out[i+1] = b_a
        b_gyr_out[i+1] = b_g
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
