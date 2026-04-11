# -*- coding: utf-8 -*-
"""
TLIO: Tight Learned Inertial Odometry
======================================
Paper : Liu et al., "TLIO: Tight Learned Inertial Odometry", IEEE RA-L 2020
        https://doi.org/10.1109/LRA.2020.3007421  |  arXiv:2007.01867
Code  : https://github.com/CathIAS/TLIO  (cloned to external/tlio/)

Filter backend: ImuMSCKF (Stochastic Cloning EKF) from
external/tlio/src/tracker/scekf.py.  This replaces a prior simplified 15-state
ESKF that diverged during GPS outages because the displacement measurement
anchor was not in the state vector.  In the SCEKF, both begin and end window
positions are augmented clones, so H@Sigma@H.T stays bounded during outages.

Key design decisions
---------------------
* Clone augmentation: every _UPDATE_STRIDE=5 raw IMU steps → 20 Hz.
* Window: TLIO update fires when a clone at (t_now − 1 s) exists in the state.
  First fire at step 105 (1.05 s); thereafter every 5 steps (50 ms stride).
* TLIO measurement: displacement in gravity-aligned frame, shape (3,1).
  SCEKF.update() rotates internally via Rz(yaw_at_begin).
* GPS: position-only update on the 15-element evolving state; cross-covariance
  propagates the fix into all clones.
* Mahalanobis gate: built into SCEKF.update(), active after 10 s.

Differences from the original paper
-------------------------------------
1. IMU rate: 100 Hz (KITTI) vs 200 Hz (paper).  Window W_imu=100 samples is
   upsampled to W_net=200 before network inference.
2. GPS integration added (original is IMU-only).
3. LOO training on KITTI; original uses a pedestrian split.
4. FLU body / ENU navigation frames.  g = [0,0,−9.81] (ENU Z-up).

Weights search order:
  1. TLIO_WEIGHTS env var
  2. artifacts/tlio/fold_<seq>.pt   (LOO fold)
  3. artifacts/tlio/tlio_resnet.pt  (all-sequences)
  4. Any available fold_*.pt        (fallback with warning)
"""

import os
import sys
import numpy as np
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent.parent   # cookies/
_TLIO_SRC  = _REPO_ROOT / 'external/tlio/src'
_ARTIFACTS = _REPO_ROOT / 'artifacts/tlio'

# Network input: always W=200 samples (1 s at 200 Hz — original TLIO paper).
_TLIO_NET_WINDOW = 200
_TLIO_NET_HZ     = 200.0

# Clone schedule (matches imu_tracker.py).
_IMU_HZ        = 100          # raw KITTI IMU rate [Hz]
_UPDATE_HZ     = 20           # clone augmentation rate [Hz]
_UPDATE_STRIDE = _IMU_HZ // _UPDATE_HZ   # = 5 IMU steps between clones

DEFAULT_PARAMS = {
    # GPS measurement noise (σ² [m²])
    'Rpos':              4.0,
    # Window length in seconds
    'window_seconds':    1.0,
    # SCEKF IMU noise parameters
    'sigma_na':          np.sqrt(1e-3),   # accelerometer noise [m/s² / √Hz]
    'sigma_ng':          np.sqrt(1e-4),   # gyroscope noise [rad/s / √Hz]
    'ita_ba':            1e-4,            # accel bias random walk [m/s² / √s]
    'ita_bg':            1e-6,            # gyro bias random walk [rad/s / √s]
    # SCEKF initial uncertainty (1-σ std)
    'init_attitude_sigma': 10.0 / 180.0 * np.pi,
    'init_yaw_sigma':       0.1 / 180.0 * np.pi,
    'init_vel_sigma':       1.0,
    'init_pos_sigma':       0.001,
    'init_bg_sigma':        0.0001,
    'init_ba_sigma':        0.2,
    # Network covariance scale (meascov_scale=10 matches original TLIO repo)
    'meascov_scale':     10.0,
}


# ── ImuMSCKF config object ────────────────────────────────────────────────────

class _TLIOConfig:
    """Thin wrapper that exposes DEFAULT_PARAMS as attributes for ImuMSCKF."""
    def __init__(self, p_cfg):
        self.sigma_na            = p_cfg['sigma_na']
        self.sigma_ng            = p_cfg['sigma_ng']
        self.ita_ba              = p_cfg['ita_ba']
        self.ita_bg              = p_cfg['ita_bg']
        self.init_attitude_sigma = p_cfg['init_attitude_sigma']
        self.init_yaw_sigma      = p_cfg['init_yaw_sigma']
        self.init_vel_sigma      = p_cfg['init_vel_sigma']
        self.init_pos_sigma      = p_cfg['init_pos_sigma']
        self.init_bg_sigma       = p_cfg['init_bg_sigma']
        self.init_ba_sigma       = p_cfg['init_ba_sigma']
        self.meascov_scale       = p_cfg['meascov_scale']
        self.mahalanobis_fail_scale = 0   # drop outliers; do not inflate R
        self.g_norm              = 9.81
        self.use_const_cov       = False
        self.add_sim_meas_noise  = False


# ── Path helpers ──────────────────────────────────────────────────────────────

def _add_tlio_src():
    # Add external/tlio/src as a package root so that `from utils.from_scipy`
    # and `from network.model_factory` resolve correctly.
    # Do NOT add subdirectories (tracker/, utils/) as flat path entries — a
    # flat `utils/` entry shadows the utils package with 'utils is not a package'.
    for p in [str(_TLIO_SRC / 'tracker'), str(_TLIO_SRC)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # external/ai-imu-dr/src/utils.py is a flat file (not a package).  If it
    # was imported earlier (e.g. by iekf_ai_imu runner), evict it from the
    # module cache so that `import utils` re-resolves to the TLIO utils package.
    if 'utils' in sys.modules:
        import importlib
        utils_mod = sys.modules['utils']
        utils_file = getattr(utils_mod, '__file__', '') or ''
        if 'ai-imu-dr' in utils_file or not getattr(utils_mod, '__path__', None):
            del sys.modules['utils']
            # Also evict any utils.* submodules that may have been cached wrong
            for key in list(sys.modules):
                if key.startswith('utils.'):
                    del sys.modules[key]


def _resolve_seq_id(seq_id):
    """Convert full KITTI drive name to short seq ID (e.g. '01') if needed."""
    if seq_id is None:
        return None
    if len(seq_id) <= 2:
        return seq_id
    from data_loader import KITTI_SEQ_TO_DRIVE
    _drive_to_seq = {v: k for k, v in KITTI_SEQ_TO_DRIVE.items()}
    return _drive_to_seq.get(seq_id, seq_id)


def _find_weights(seq_id: str = None) -> Path:
    """
    Locate TLIO weights.  Search order:
      1. TLIO_WEIGHTS env var
      2. artifacts/tlio/fold_<seq_id>.pt  (LOO fold)
      3. artifacts/tlio/tlio_resnet.pt    (all-sequences)
      4. Any available fold_*.pt          (fallback with warning)
    """
    env = os.environ.get('TLIO_WEIGHTS')
    if env and Path(env).exists():
        return Path(env)

    short_id = _resolve_seq_id(seq_id)
    if short_id is not None:
        fold = _ARTIFACTS / f'fold_{short_id}.pt'
        if fold.exists():
            return fold

    default = _ARTIFACTS / 'tlio_resnet.pt'
    if default.exists():
        return default

    available = sorted(_ARTIFACTS.glob('fold_*.pt'))
    available = [p for p in available if '_ckpt' not in p.name]
    if available:
        print(f"WARNING: TLIO fold_{short_id}.pt not found, "
              f"falling back to {available[0].name}. "
              f"Train the proper fold: python ins_train.py tlio --seqs {short_id}")
        return available[0]

    raise RuntimeError(
        "TLIO weights not found.  Train the model first:\n"
        "  python ins_train.py tlio\n"
        "or set the TLIO_WEIGHTS environment variable to an existing .pt file."
    )


def _load_model(weights_path: Path, window_size: int, device: str = 'cpu'):
    """Load TLIO ResNet1D and return it in eval mode."""
    import torch
    import importlib.util

    _mf_path = _HERE / 'network/model_factory.py'
    _spec = importlib.util.spec_from_file_location('_tlio_model_factory', _mf_path)
    _mf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mf)
    get_model = _mf.get_model

    in_dim = window_size // 32 + 1          # imu_tracker.py line 118
    net_config = {'in_dim': in_dim}
    model = get_model('resnet', net_config, input_dim=6, output_dim=3)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── Rotation helpers ──────────────────────────────────────────────────────────

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


def _qto_Rbn(q):
    """Hamilton quaternion [w,x,y,z] → 3×3 body→nav rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w)  ],
        [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w)  ],
        [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y)],
    ])


def _Ry(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0., sa], [0., 1., 0.], [-sa, 0., ca]])


def _Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1., 0., 0.], [0., ca, -sa], [0., sa, ca]])


def _R_to_rpy(R):
    """
    Extract [roll, pitch, yaw] from body→nav rotation R_nb = Rz(yaw)@Ry(pitch)@Rx(roll).
    ZYX Tait-Bryan / aerospace convention.
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw  = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


# ── Network inference helper ──────────────────────────────────────────────────

def _predict_displacement(model, imu_window: np.ndarray, device: str = 'cpu'):
    """
    Run one TLIO forward pass and decode displacement + covariance.

    Parameters
    ----------
    imu_window : (6, W) float32 — [gyro_ga(3) | accel_ga_motion(3)]

    Returns
    -------
    dp_ga    : (3,) float64 — predicted displacement in gravity-aligned frame
    Sigma_ga : (3, 3) float64 — diagonal covariance
    """
    import torch
    x = torch.from_numpy(imu_window[None]).float().to(device)   # (1, 6, W)
    with torch.no_grad():
        mean, logstd = model(x)    # each (1, 3)
    dp_ga   = mean[0].cpu().numpy().astype(np.float64)
    log_std = logstd[0].cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(dp_ga)):
        return np.zeros(3), np.eye(3) * 100.0
    # Eq. 3: Σ = diag(exp(2·logstd)).  Floor: 10 cm or 5% of predicted displacement.
    log_std  = np.clip(log_std, -10.0, 10.0)
    min_std  = np.maximum(0.05 * np.abs(dp_ga), 0.1)
    std2     = np.maximum(np.exp(2.0 * log_std), min_std ** 2)
    if not np.all(np.isfinite(std2)):
        std2 = np.full(3, 1.0)
    return dp_ga, np.diag(std2)


# ── GPS update (position-only on evolving state) ──────────────────────────────

def _gps_update(filt, gps_pos_enu: np.ndarray, gps_noise_var: float):
    """
    GPS position-only Kalman update on the ImuMSCKF evolving state.

    Evolving-state error ordering (last 15 elements of the full state):
      [dθ(0:3), dv(3:6), dp(6:9), dbg(9:12), dba(12:15)]

    Position is at offset +6 within the evolving block → global column 6*N+6.
    Cross-covariance propagates the GPS fix into all clone positions.

    Parameters
    ----------
    filt          : ImuMSCKF instance
    gps_pos_enu   : (3,) array — GPS position in ENU [m]
    gps_noise_var : float — GPS position variance [m²]
    """
    N = filt.state.N
    n = 15 + 6 * N
    H = np.zeros((3, n))
    H[:, 6*N + 6 : 6*N + 9] = np.eye(3)

    R_gps = np.eye(3) * gps_noise_var
    S     = H @ filt.Sigma @ H.T + R_gps
    K     = filt.Sigma @ H.T @ np.linalg.inv(S)   # (n, 3)

    innov = gps_pos_enu.reshape(3, 1) - filt.state.s_p   # (3, 1)
    dX    = K @ innov                                      # (n, 1)
    filt.state.apply_correction(dX)

    I_KH       = np.eye(n) - K @ H
    filt.Sigma = I_KH @ filt.Sigma @ I_KH.T + K @ R_gps @ K.T   # Joseph form
    filt.Sigma = 0.5 * (filt.Sigma + filt.Sigma.T)
    filt.Sigma15 = filt.Sigma[-15:, -15:]


# ── Main filter ───────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run TLIO on nav_data using the original ImuMSCKF (stochastic cloning EKF).

    Parameters
    ----------
    nav_data       : NavigationData (data_loader.py)
    params         : Optional dict overriding DEFAULT_PARAMS.
    outage_config  : Optional {'start': t1_s, 'duration': d_s}.
    use_3d_rotation: Accepted for API compatibility; SCEKF always uses full 3D.

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

    # ── Imports ───────────────────────────────────────────────────────────
    _add_tlio_src()
    from scekf import ImuMSCKF   # external/tlio/src/tracker/scekf.py

    # ── Data ──────────────────────────────────────────────────────────────
    accel_flu   = nav_data.accel_flu    # (N, 3) body FLU frame
    gyro_flu    = nav_data.gyro_flu     # (N, 3) body FLU frame
    orient      = nav_data.orient       # (N, 3) [roll, pitch, yaw] radians
    lla         = nav_data.lla
    vel_enu     = nav_data.vel_enu
    sample_rate = nav_data.sample_rate
    lla0        = nav_data.lla0

    import pymap3d as pm
    N   = accel_flu.shape[0]

    # IMU window length in raw samples (1 s at sample_rate Hz)
    W_imu = int(p_cfg['window_seconds'] * sample_rate)   # 100 at 100 Hz
    # Window length in µs — used to find the begin clone
    _window_us = int(W_imu * 1e6 / sample_rate)          # 1 000 000 µs

    # ── Weights and network ───────────────────────────────────────────────
    seq_id       = getattr(nav_data, 'dataset_name', None)
    weights_path = _find_weights(seq_id)
    model        = _load_model(weights_path, _TLIO_NET_WINDOW, device)
    print(f"TLIO: loaded weights from {weights_path}  "
          f"(W_imu={W_imu}→W_net={_TLIO_NET_WINDOW}, "
          f"clone every {_UPDATE_STRIDE} steps, device={device})")

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
    e, n_arr, u = pm.geodetic2enu(
        lla[:, 0], lla[:, 1], lla[:, 2],
        lla0[0], lla0[1], lla0[2])
    p_gps_enu = np.column_stack([e, n_arr, u])   # (N, 3)

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

    # ── ImuMSCKF initialisation ───────────────────────────────────────────
    config  = _TLIOConfig(p_cfg)
    filt    = ImuMSCKF(config)
    R_init  = _qto_Rbn(_qfrom_euler(orient[0, 0], orient[0, 1], orient[0, 2]))
    v_init  = vel_enu[0].reshape(3, 1)
    p_init  = p_gps_enu[0].reshape(3, 1)
    ba_init = np.zeros((3, 1))
    bg_init = np.zeros((3, 1))
    filt.initialize_with_state(0, R_init, v_init, p_init, ba_init, bg_init)

    pos[0]     = p_init.flatten()
    vel[0]     = v_init.flatten()
    rpy_out[0] = orient[0]

    # ── Upsample helpers (pre-built grids) ────────────────────────────────
    from scipy.interpolate import interp1d as _interp1d
    _t_src = np.linspace(0., 1., W_imu)
    _t_up  = np.linspace(0., 1., _TLIO_NET_WINDOW)
    _kw    = dict(axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')

    # ── Counters ──────────────────────────────────────────────────────────
    _tlio_n_attempts = 0
    _tlio_n_accepted = 0
    _gps_updates     = 0

    print(f"TLIO (ImuMSCKF): running filter on {N} samples ...")

    for i in range(N - 1):
        # Target time at end of this integration step [µs]
        t_us = int((i + 1) * 1e6 / sample_rate)

        # Clone augmentation: every _UPDATE_STRIDE steps (20 Hz)
        augment  = ((i + 1) % _UPDATE_STRIDE == 0)
        t_aug_us = t_us if augment else None

        # ── A. Propagate (IMU strapdown + covariance; optional clone) ─────
        acc = accel_flu[i].reshape(3, 1)
        gyr = gyro_flu[i].reshape(3, 1)
        filt.propagate(acc, gyr, t_us, t_augmentation_us=t_aug_us)

        # ── B. TLIO displacement update ───────────────────────────────────
        # Fire when a clone at exactly (t_now − 1 s) exists.
        # First possible fire: t_begin=50000 µs (step 5) when t_now=1050000 µs (step 105).
        # Thereafter: every _UPDATE_STRIDE steps = 50 ms stride.
        if augment:
            t_begin_us = t_us - _window_us
            ts_list    = filt.state.si_timestamps_us   # list of augmented clone times

            if (t_begin_us > 0
                    and t_begin_us in ts_list
                    and t_us in ts_list):

                _tlio_n_attempts += 1

                # Build gravity-aligned network input from raw IMU samples [ws:we]
                ws       = int(t_begin_us * sample_rate / 1e6)   # start index
                we       = ws + W_imu                             # end index
                accel_w  = accel_flu[ws:we]    # (W_imu, 3)
                gyro_w   = gyro_flu[ws:we]     # (W_imu, 3)

                roll_ws  = orient[ws, 0]
                pitch_ws = orient[ws, 1]
                R_ga     = _Ry(pitch_ws) @ _Rx(roll_ws)

                accel_up  = _interp1d(_t_src, accel_w, **_kw)(_t_up)   # (W_net, 3)
                gyro_up   = _interp1d(_t_src, gyro_w,  **_kw)(_t_up)   # (W_net, 3)
                accel_ga  = R_ga @ accel_up.T                           # (3, W_net)
                gyro_ga   = R_ga @ gyro_up.T                            # (3, W_net)
                accel_ga[2, :] -= 9.81                                  # remove gravity

                imu_win = np.vstack([gyro_ga, accel_ga]).astype(np.float32)  # (6, W_net)

                dp_ga, Sigma_ga = _predict_displacement(model, imu_win, device)

                # SCEKF update: meas = dp in GA frame; filter computes pred = Rz.T@(p_end−p_begin)
                # Mahalanobis gate is built-in (active after 10 s).
                _inno_before = filt.innovation.copy()
                filt.update(dp_ga.reshape(3, 1), Sigma_ga, t_begin_us, t_us)
                if not np.array_equal(filt.innovation, _inno_before):
                    _tlio_n_accepted += 1

                # Marginalize the begin clone (and any older ones) unconditionally
                begin_idx = ts_list.index(t_begin_us)
                filt.marginalize(begin_idx)

        # ── C. GPS position update ─────────────────────────────────────────
        if gps_avail[i + 1]:
            _gps_update(filt, p_gps_enu[i + 1], p_cfg['Rpos'])
            _gps_updates += 1

        # ── D. Store outputs ───────────────────────────────────────────────
        R, v, p, ba, bg = filt.get_evolving_state()
        pos[i+1]       = p.flatten()
        vel[i+1]       = v.flatten()
        rpy_out[i+1]   = _R_to_rpy(R)
        b_acc_out[i+1] = ba.flatten()
        b_gyr_out[i+1] = bg.flatten()

        S15 = filt.Sigma15
        std_pos[i+1]    = np.sqrt(np.maximum(np.diag(S15[6:9,   6:9]),   0.))
        std_vel[i+1]    = np.sqrt(np.maximum(np.diag(S15[3:6,   3:6]),   0.))
        std_orient[i+1] = np.sqrt(np.maximum(np.diag(S15[0:3,   0:3]),   0.))
        std_b_gyr[i+1]  = np.sqrt(np.maximum(np.diag(S15[9:12,  9:12]),  0.))
        std_b_acc[i+1]  = np.sqrt(np.maximum(np.diag(S15[12:15, 12:15]), 0.))

    rate = 100 * _tlio_n_accepted / max(_tlio_n_attempts, 1)
    print(f"TLIO: {_tlio_n_accepted}/{_tlio_n_attempts} TLIO updates accepted "
          f"({rate:.1f}%), {_gps_updates} GPS updates")

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
