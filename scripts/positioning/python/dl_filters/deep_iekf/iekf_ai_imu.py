"""
iekf_ai_imu.py — AI-IMU Dead-Reckoning filter, benchmark-compatible wrapper.

Implements the standard filter interface:
    run(nav_data, params=None, outage_config=None, use_3d_rotation=True) -> dict

Reference
---------
Brossard, Barrau, Bonnabel — "AI-IMU Dead-Reckoning", IEEE TIV 2020
DOI: 10.1109/TIV.2020.2980758
Code: https://github.com/mbrossar/ai-imu-dr

Credit / License
----------------
The IEKF filter core (NUMPYIEKF, TORCHIEKF, MesNet CNN) and KITTIParameters are the
work of Martin Brossard, Axel Barrau, and Silvère Bonnabel, distributed under the MIT
License (see external/ai-imu-dr/LICENSE).  This file is a thin wrapper that adapts
their public API to our NavigationData interface; no original algorithmic code is
duplicated here.

Architecture
------------
Two-block system (Figure 4 of the paper):
  1. IEKF (Invariant Extended Kalman Filter) — 21-state SE_2(3) filter with
     NHC pseudo-measurements (lateral + vertical body velocity ≈ 0).
  2. CNN adapter (MesNet) — takes a window of raw IMU samples and outputs the
     2x2 diagonal measurement noise covariance N_n used by the IEKF update step.

The IEKF state (Appendix A, eq. 21) is [φ(0:3), ξ_v(3:6), ξ_p(6:9),
δb_ω(9:12), δb_a(12:15), ξ_Rc(15:18), ξ_pc(18:21)] — 21D.
The car-to-IMU calibration (Rc, pc) is estimated online, starting at identity/zero.

GPS / GNSS
----------
When nav_data.gps_available is True at a given timestep, the filter applies a
standard position measurement update (H selects ξ_p at state indices 6:9).
The CNN adapter continues to regulate NHC covariances independently.
During GPS outages (gps_available=False), the filter runs pure CNN dead-reckoning.
Set DR_MODE=True in ins_config.py to disable GPS for all filters (paper comparison).

10 Hz vs 100 Hz
---------------
The paper trains and tests at 100 Hz (raw KITTI OXTS).  Our nav_data is at 10 Hz
(synchronized KITTI).  The CNN window (N=15 samples) thus covers 1.5 s instead of
0.15 s.  For best performance, train at 100 Hz and run at 100 Hz (see train_ai_imu.py).
Running at 10 Hz with weights trained at 10 Hz still works; paper results (1.10% t_rel)
require 100 Hz data.

Weights
-------
Pre-trained weights are looked up in this order:
  1. Path in env var  AI_IMU_WEIGHTS
  2. <repo>/artifacts/deep_iekf/iekfnets.p  (default training output)
  3. <repo>/external/ai-imu-dr/src/iekfnets.p  (AI-IMU default save location)
If no weights are found the filter falls back to the IEKF with fixed covariances
(KITTI-tuned defaults from KITTIParameters, no CNN adapter).
"""

import sys
import os
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_REPO_ROOT   = _HERE.parent.parent.parent.parent.parent   # c:/Github/cookies
_AI_IMU_SRC  = _REPO_ROOT / 'external/ai-imu-dr/src'
_ARTIFACTS   = _REPO_ROOT / 'artifacts/deep_iekf'

DEFAULT_PARAMS = {}   # no tunable parameters (learned weights, not genetic params)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _add_aimu_path():
    src = str(_AI_IMU_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _find_onnx():
    """Return path to iekfnets.onnx, or None if not found."""
    env_path = os.environ.get('AI_IMU_ONNX')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    artifacts = _ARTIFACTS / 'iekfnets.onnx'
    if artifacts.exists():
        return artifacts
    repo = _AI_IMU_SRC / 'iekfnets.onnx'
    if repo.exists():
        return repo
    return None


def _find_weights():
    """Return path to iekfnets.p, or None if not found."""
    # 1. Environment variable
    env_path = os.environ.get('AI_IMU_WEIGHTS')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    # 2. Repo artifacts folder (default training output)
    artifacts = _ARTIFACTS / 'iekfnets.p'
    if artifacts.exists():
        return artifacts
    # 3. AI-IMU repo default location
    repo = _AI_IMU_SRC / 'iekfnets.p'
    if repo.exists():
        return repo
    return None


def _find_norm_factors(weights_path):
    """
    Look for saved normalization factors (u_loc, u_std) next to the weights file.
    Tries, in order:
      1. sibling <stem>_norm.p   (e.g. fold_01.p → fold_01_norm.p)
      2. legacy iekfnets_norm.p name in the same directory
      3. any other *_norm.p sibling in the same directory
    Returns dict or None if nothing valid is found.
    """
    if weights_path is None:
        return None
    wp = Path(weights_path)
    candidates = [
        wp.with_name(wp.stem + '_norm.p'),
        wp.parent / 'iekfnets_norm.p',
    ]
    candidates.extend(sorted(wp.parent.glob('*_norm.p')))
    seen = set()
    for norm_path in candidates:
        if norm_path in seen or not norm_path.exists():
            seen.add(norm_path)
            continue
        seen.add(norm_path)
        try:
            import torch
            return torch.load(norm_path)
        except Exception:
            continue
    return None


def _skew(v):
    """Skew-symmetric matrix for cross product."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _so3_exp(phi):
    """SO(3) exponential map (Rodrigues formula)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + _skew(phi)
    K = _skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _gps_update_step(Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, P,
                     z_gps, R_gps):
    """
    GPS position measurement update for the right-invariant IEKF.

    z_gps = p_gps - p_est  (3D ENU innovation)
    H     = [0₃, 0₃, I₃, 0₁₂]  — selects ξ_p (state indices 6:9)
    R_gps = 3×3 position measurement covariance [m²]

    Returns updated state variables and covariance P.
    """
    H = np.zeros((3, 21))
    H[:, 6:9] = np.eye(3)

    S = H @ P @ H.T + R_gps                              # (3,3)
    K = P @ H.T @ np.linalg.solve(S.T, np.eye(3)).T      # (21,3)
    delta = K @ z_gps                                     # (21,) error state

    # Apply correction to each state component
    p_new       = p       + delta[6:9]
    v_new       = v       + delta[3:6]
    b_omega_new = b_omega + delta[9:12]
    b_acc_new   = b_acc   + delta[12:15]
    t_c_i_new   = t_c_i  + delta[18:21]
    Rot_new     = _so3_exp(delta[0:3])   @ Rot
    Rot_c_i_new = _so3_exp(delta[15:18]) @ Rot_c_i

    P_new = (np.eye(21) - K @ H) @ P
    return Rot_new, v_new, p_new, b_omega_new, b_acc_new, Rot_c_i_new, t_c_i_new, P_new


def _build_rotation_matrix(roll, pitch, yaw):
    """ZYX Euler angles → 3×3 rotation matrix (body→nav)."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])


def _run_filter_loop(iekf, t, u, measurements_covs_np, v0, ang0,
                     p_gps=None, gps_available=None, R_gps=None):
    """
    Run NUMPYIEKF propagate/update loop, tracking covariance P at each step.

    Parameters
    ----------
    iekf                 : NUMPYIEKF instance (with learned or default covariances)
    t                    : (N,) timestamps [s]
    u                    : (N,6) IMU data [gyro_flu(3), accel_flu(3)]
    measurements_covs_np : (N,2) measurement covariances [cov_lat, cov_up]
    v0                   : (3,) initial ENU velocity
    ang0                 : (3,) initial [roll, pitch, yaw] [rad]
    p_gps                : (N,3) GPS ENU positions [m], origin-relative, or None
    gps_available        : (N,) bool mask — True where GPS update should be applied
    R_gps                : (3,3) GPS position measurement covariance [m²], or None

    Returns
    -------
    p, v, r, b_omega, b_acc, std_pos, std_vel, std_orient, std_bias_gyr, std_bias_acc
    all (N, 3) numpy arrays.
    """
    _add_aimu_path()
    from utils_numpy_filter import NUMPYIEKF

    N = t.shape[0]
    dt = t[1:] - t[:-1]

    use_gps = (p_gps is not None and gps_available is not None
               and gps_available.any())
    if R_gps is None:
        R_gps = np.eye(3) * 4.0   # default: 2 m position std

    # Allocate state arrays
    Rot     = np.zeros((N, 3, 3))
    v       = np.zeros((N, 3))
    p       = np.zeros((N, 3))
    b_omega = np.zeros((N, 3))
    b_acc   = np.zeros((N, 3))
    Rot_c_i = np.zeros((N, 3, 3))
    t_c_i   = np.zeros((N, 3))

    # Initialise
    Rot[0]     = _build_rotation_matrix(ang0[0], ang0[1], ang0[2])
    v[0]       = v0
    Rot_c_i[0] = np.eye(3)
    P = iekf.init_covariance()

    # Diagonal of P at each step — avoids storing (N,21,21)
    P_diag = np.zeros((N, 21))
    P_diag[0] = np.diag(P)

    for i in range(1, N):
        Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
            iekf.propagate(Rot[i-1], v[i-1], p[i-1], b_omega[i-1], b_acc[i-1],
                           Rot_c_i[i-1], t_c_i[i-1], P, u[i], dt[i-1])

        # NHC pseudo-measurement update (CNN-adapted covariance)
        Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i], t_c_i[i], P = \
            iekf.update(Rot[i], v[i], p[i], b_omega[i], b_acc[i], Rot_c_i[i],
                        t_c_i[i], P, u[i], i, measurements_covs_np[i])

        # GPS position update (independent of NHC; runs when GPS is available)
        if use_gps and gps_available[i]:
            z_gps = p_gps[i] - p[i]
            (Rot[i], v[i], p[i], b_omega[i], b_acc[i],
             Rot_c_i[i], t_c_i[i], P) = _gps_update_step(
                Rot[i], v[i], p[i], b_omega[i], b_acc[i],
                Rot_c_i[i], t_c_i[i], P, z_gps, R_gps)

        # Periodic SO(3) normalisation to prevent numerical drift
        if i % iekf.n_normalize_rot == 0:
            Rot[i] = NUMPYIEKF.normalize_rot(Rot[i])
        if i % iekf.n_normalize_rot_c_i == 0:
            Rot_c_i[i] = NUMPYIEKF.normalize_rot(Rot_c_i[i])

        P_diag[i] = np.diag(P)

    # Extract standard deviations from P diagonal
    # P ordering: [φ(0:3), ξ_v(3:6), ξ_p(6:9), δb_ω(9:12), δb_a(12:15), ...]
    std_orient   = np.sqrt(np.maximum(P_diag[:, 0:3],  0.0))
    std_vel      = np.sqrt(np.maximum(P_diag[:, 3:6],  0.0))
    std_pos      = np.sqrt(np.maximum(P_diag[:, 6:9],  0.0))
    std_bias_gyr = np.sqrt(np.maximum(P_diag[:, 9:12], 0.0))
    std_bias_acc = np.sqrt(np.maximum(P_diag[:, 12:15], 0.0))

    # Convert rotation matrices → Euler angles [roll, pitch, yaw]
    r_out = np.array([NUMPYIEKF.to_rpy(R) for R in Rot])  # (N, 3)

    return p, v, r_out, b_omega, b_acc, std_pos, std_vel, std_orient, std_bias_gyr, std_bias_acc


# ── Public API ─────────────────────────────────────────────────────────────────

def run(nav_data, params=None, outage_config=None, use_3d_rotation=True):
    """
    Run the AI-IMU IEKF filter with optional GPS-aided updates.

    Parameters
    ----------
    nav_data        : NavigationData — IMU + GNSS data
    params          : optional dict — supports 'Rpos' (GPS noise std [m], default 2.0)
    outage_config   : optional {'start': float, 'duration': float} — GPS blackout window
    use_3d_rotation : ignored (filter always runs in full 3D)

    GPS is used whenever nav_data.gps_available is True AND the current timestep
    is not inside the outage window.  Set ins_config.DR_MODE=True to disable GPS
    globally (produces pure dead-reckoning comparable to Brossard et al. 2020).

    Returns
    -------
    dict with keys: p, v, r, bias_acc, bias_gyr,
                    std_pos, std_vel, std_orient, std_bias_acc, std_bias_gyr
    Each value is an (N, 3) numpy array.
    """
    _add_aimu_path()

    N  = len(nav_data.accel_flu)
    sr = nav_data.sample_rate

    # Build timestamps
    if nav_data.time is not None:
        t = nav_data.time.astype(np.float64)
    else:
        t = np.arange(N, dtype=np.float64) / sr

    # IMU input: u = [gyro_flu(3), accel_flu(3)]
    u_np = np.hstack([nav_data.gyro_flu, nav_data.accel_flu]).astype(np.float64)

    ang0 = nav_data.orient[0].astype(np.float64)   # [roll, pitch, yaw]
    v0   = nav_data.vel_enu[0].astype(np.float64)   # initial ENU velocity

    # ── Load IEKF with KITTI-tuned parameters ──────────────────────────────────
    from utils_numpy_filter import NUMPYIEKF

    # Import KITTIParameters from AI-IMU
    try:
        sys.path.insert(0, str(_AI_IMU_SRC))
        from main_kitti import KITTIParameters
        iekf = NUMPYIEKF(KITTIParameters)
    except Exception:
        iekf = NUMPYIEKF()   # fallback: base Parameters

    # Make gravity a numpy array (in case KITTIParameters stores it as list)
    if not isinstance(iekf.g, np.ndarray):
        iekf.g = np.array(iekf.g)

    # ── Try ONNX inference (fastest, no PyTorch needed at runtime) ────────────
    onnx_path = _find_onnx()
    measurements_covs_np = None

    if onnx_path is not None:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path),
                                        providers=['CPUExecutionProvider'])
            measurements_covs_np = sess.run(None, {'u': u_np})[0]  # (N, 2)
            print(f"AI-IMU: ONNX inference from {onnx_path}.")
        except Exception as e:
            print(f"Warning: ONNX inference failed ({e}). Trying PyTorch weights.")
            measurements_covs_np = None

    # ── Fall back to PyTorch .p weights ───────────────────────────────────────
    if measurements_covs_np is None:
        weights_path = _find_weights()
        if weights_path is not None:
            try:
                import torch
                from utils_torch_filter import TORCHIEKF

                try:
                    from main_kitti import KITTIParameters as _KP
                    torch_iekf = TORCHIEKF(_KP)
                except Exception:
                    torch_iekf = TORCHIEKF()
                if torch_iekf.cov0_measurement is None:
                    # Fallback baseline: KITTIParameters defaults
                    torch_iekf.cov0_measurement = torch.tensor([0.2, 300.0]).double()
                if isinstance(torch_iekf.g, np.ndarray):
                    torch_iekf.g = torch.from_numpy(torch_iekf.g).double()

                mondict = torch.load(weights_path, map_location='cpu')
                torch_iekf.load_state_dict(mondict)
                torch_iekf.eval()

                norm = _find_norm_factors(weights_path)
                if norm is not None:
                    torch_iekf.u_loc = norm['u_loc'].double()
                    torch_iekf.u_std = norm['u_std'].double()
                else:
                    u_loc = torch.from_numpy(np.mean(u_np, axis=0)).double()
                    u_std = torch.from_numpy(np.std(u_np, axis=0)).double()
                    u_std[u_std < 1e-6] = 1.0
                    torch_iekf.u_loc = u_loc
                    torch_iekf.u_std = u_std
                    print("Warning: normalization factors not found — computed from inference data. "
                          "For best results, run train_ai_imu.py which saves iekfnets_norm.p.")

                # Transfer learned Q and initial covariance to numpy IEKF
                iekf.set_learned_covariance(torch_iekf)

                with torch.no_grad():
                    u_tensor = torch.from_numpy(u_np).double()
                    measurements_covs_torch = torch_iekf.forward_nets(u_tensor)
                    measurements_covs_np = measurements_covs_torch.cpu().numpy()  # (N, 2)

                print(f"AI-IMU: loaded PyTorch weights from {weights_path}.")

            except Exception as e:
                print(f"Warning: could not load AI-IMU weights ({e}). "
                      f"Falling back to fixed covariances.")
                measurements_covs_np = None

    # ── Fallback: fixed measurement covariances ────────────────────────────────
    if measurements_covs_np is None:
        cov_lat = iekf.cov_lat
        cov_up  = iekf.cov_up
        measurements_covs_np = np.tile([cov_lat, cov_up], (N, 1))
        print(f"AI-IMU: using fixed covariances [cov_lat={cov_lat}, cov_up={cov_up}] "
              f"(no CNN adapter).")

    # ── GPS data preparation ───────────────────────────────────────────────────
    # Compute ENU positions from geodetic (round-trip exact for same lla0).
    # Apply outage mask: GPS blackout window from outage_config.
    p_gps_for_loop   = None
    gps_avail_masked = None
    R_gps            = None

    try:
        import ins_config as _ic
        dr_mode = getattr(_ic, 'DR_MODE', False)
    except Exception:
        dr_mode = False

    if not dr_mode and hasattr(nav_data, 'gps_available') and nav_data.gps_available.any():
        import pymap3d as pm
        e, n, u_enu = pm.geodetic2enu(
            nav_data.lla[:, 0], nav_data.lla[:, 1], nav_data.lla[:, 2],
            nav_data.lla0[0], nav_data.lla0[1], nav_data.lla0[2])
        p_gps_for_loop = np.column_stack([e, n, u_enu])

        gps_avail_masked = nav_data.gps_available.copy()
        if outage_config is not None:
            t1 = outage_config.get('start', 0.0)
            d  = outage_config.get('duration', 0.0)
            A  = int(t1 * nav_data.sample_rate)
            B  = int((t1 + d) * nav_data.sample_rate)
            gps_avail_masked[A:B] = False

        rpos = float((params or {}).get('Rpos', 4.0))
        R_gps = np.eye(3) * rpos

        n_gps = int(gps_avail_masked.sum())
        print(f"AI-IMU: GPS-aided mode — {n_gps} GPS updates "
              f"({'outage applied' if outage_config else 'no outage'}).")
    elif dr_mode:
        print("AI-IMU: DR_MODE=True — pure dead-reckoning (no GPS).")

    # ── Run filter loop ────────────────────────────────────────────────────────
    p, v, r, b_omega, b_acc, std_pos, std_vel, std_orient, std_bias_gyr, std_bias_acc = \
        _run_filter_loop(iekf, t, u_np, measurements_covs_np, v0, ang0,
                         p_gps=p_gps_for_loop, gps_available=gps_avail_masked,
                         R_gps=R_gps)

    return {
        'p':           p,
        'v':           v,
        'r':           r,
        'bias_acc':    b_acc,
        'bias_gyr':    b_omega,
        'std_pos':     std_pos,
        'std_vel':     std_vel,
        'std_orient':  std_orient,
        'std_bias_acc': std_bias_acc,
        'std_bias_gyr': std_bias_gyr,
    }
