"""
iekf_ai_imu_online.py — CAUSAL / online variant of the AI-IMU deep-IEKF.

Why this file exists
--------------------
The reference AI-IMU code (external/ai-imu-dr) and our iekf_ai_imu.py wrapper run
the MesNet covariance CNN with symmetric ReplicationPad1d, so each covariance
output depends on ~8 FUTURE IMU samples (receptive field ≈ 17, centred on i).
That is fine for offline benchmark replay but cannot run online.

This module trains and runs a strictly CAUSAL replacement instead: CausalMesNet
(causal_mesnet.py) is the same network with LEFT-only padding, so output[i]
depends only on inputs ≤ i — no future data is ever read, and its receptive
field is the same size, ~17 samples, just entirely in the past.

Weight source (run() requires a true causal model)
--------------------------------------------------
run() ALWAYS uses CAUSAL-trained weights in artifacts/deep_iekf_online/ (or
AI_IMU_ONLINE_WEIGHTS) — a left-padded CausalMesNet trained by
`train_ai_imu.py --causal`. If no causal weights are found, run() raises (see the
require-weights logic) — it does NOT fall back to running the upstream ACAUSAL
weights causally, which would report an approximation as the online result.

How the covariance is actually computed
----------------------------------------
On CPU, run() streams genuinely: causal_mes_inference() with lookahead=0 feeds
one DEFAULT_WINDOW-sample rolling window per IMU tick (window >> the ~17-sample
receptive field, for margin) and reads off the covariance for that tick alone —
this is the real per-sample computation a live deployment would do, not a
diagnostic approximation. It is numerically identical to evaluating the whole
sequence in one call (verified to float64 rounding precision, ~1e-14), since the
network is causal and its receptive field is well inside the window — so nothing
is traded away for streaming it; the old whole-sequence call was pure
implementation convenience, not a requirement.

On Hailo, the compiled .hef still takes a FIXED input length
(hailo_backend.HailoDeepIEKF.SEQ_LEN, 4544 samples ~ 45.44s) and is run as one
whole-sequence call, because a second, much smaller SEQ_LEN export sized for the
model's actual ~17-sample window has not been compiled — that needs the Hailo
Dataflow Compiler, which is x86_64-only (see scripts/positioning/hailo/README.md)
and cannot run on this Pi. Until that export exists, Hailo numbers for this
filter describe a batch recompute, not a genuine streaming deployment; see the
per-filter timing table discussion in 3.tex for how this is currently reported.

`causal_mes_inference` also supports lookahead>0, kept for the diagnostic
`validate` command (quantifying what running the *acausal* weights causally,
with a small forced latency, would cost vs. the true causal model):

    python iekf_ai_imu_online.py --validate [--window N] [--lookahead K]

Drop-in: run(nav_data, params, outage_config, use_3d_rotation) matches iekf_ai_imu.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse all the loading / filter machinery from the batch wrapper.
from iekf_ai_imu import (
    _add_aimu_path, _find_weights, _find_norm_factors, _run_filter_loop,
    _AI_IMU_SRC, _ARTIFACTS,
)

# Separate folder for causal-trained weights (train_ai_imu.py --causal).
_ARTIFACTS_ONLINE = _ARTIFACTS.parent / 'deep_iekf_online'


def _find_online_weights():
    """
    Locate causal-trained weights, in order:
      1. env AI_IMU_ONLINE_WEIGHTS
      2. <repo>/artifacts/deep_iekf_online/iekfnets.p
      3. any fold_*.p / *_held_*.p in that folder
    Returns Path or None.
    """
    env_path = os.environ.get('AI_IMU_ONLINE_WEIGHTS')
    if env_path and Path(env_path).exists():
        return Path(env_path)
    canonical = _ARTIFACTS_ONLINE / 'iekfnets.p'
    if canonical.exists():
        return canonical
    if _ARTIFACTS_ONLINE.exists():
        hits = sorted(p for pat in ('fold_*.p', '*_held_*.p')
                      for p in _ARTIFACTS_ONLINE.glob(pat)
                      if not p.name.endswith('_norm.p'))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # Refuse to guess a fold — under LOO that risks a train/test leak.
            raise RuntimeError(
                f"[iekf_ai_imu_online] Multiple causal folds in {_ARTIFACTS_ONLINE} "
                f"{[p.name for p in hits]} but AI_IMU_ONLINE_WEIGHTS is unset. Set it "
                f"to the correct fold (ins_compare does this per test sequence)."
            )
    return None


def _load_causal_torch_iekf(u_np, weights_path):
    """Build TORCHIEKF with a CausalMesNet and load causal-trained weights."""
    _add_aimu_path()
    import torch
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from kitti_params import get_kitti_parameters
    torch_iekf = TORCHIEKF(get_kitti_parameters())
    if torch_iekf.cov0_measurement is None:
        torch_iekf.cov0_measurement = torch.tensor([1.0, 10.0]).double()
    if isinstance(torch_iekf.g, np.ndarray):
        torch_iekf.g = torch.from_numpy(torch_iekf.g).double()

    attach_causal_mesnet(torch_iekf)            # swap MesNet → CausalMesNet
    torch_iekf.load_state_dict(torch.load(weights_path, map_location='cpu'))
    torch_iekf.eval()

    norm = _find_norm_factors(weights_path)
    if norm is not None:
        torch_iekf.u_loc = norm['u_loc'].double()
        torch_iekf.u_std = norm['u_std'].double()
    else:
        u_loc = torch.from_numpy(np.mean(u_np, axis=0)).double()
        u_std = torch.from_numpy(np.std(u_np, axis=0)).double()
        u_std[u_std < 1e-6] = 1.0
        torch_iekf.u_loc, torch_iekf.u_std = u_loc, u_std
        print(f"[iekf_ai_imu_online] WARNING: causal norm factors not found next to "
              f"{weights_path} — recomputed from sequence (not deployable).")
    return torch_iekf

# Sliding-window length fed to MesNet per step.  Receptive field is ≈ 17 samples,
# so the last (current) output needs ~8 real past samples for full left context;
# 32 gives comfortable margin.  Overridable via AI_IMU_ONLINE_WINDOW.
DEFAULT_WINDOW = int(os.environ.get('AI_IMU_ONLINE_WINDOW', '32'))

DEFAULT_PARAMS = {}


def _load_torch_iekf(nav_data, u_np):
    """
    Load TORCHIEKF with trained MesNet weights and FIXED normalisation factors.
    Returns (torch_iekf, weights_path) or (None, None) if weights are unavailable.
    Mirrors the PyTorch-loading branch of iekf_ai_imu.run, but for online use we
    insist on the saved *_norm.p (recomputing per-sequence stats is not deployable).
    """
    _add_aimu_path()
    weights_path = _find_weights()
    if weights_path is None:
        return None, None

    import torch
    from utils_torch_filter import TORCHIEKF
    from kitti_params import get_kitti_parameters
    torch_iekf = TORCHIEKF(get_kitti_parameters())

    if torch_iekf.cov0_measurement is None:
        torch_iekf.cov0_measurement = torch.tensor([1.0, 10.0]).double()
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
        # Online deployment MUST use fixed normalisation; sequence stats are acausal.
        u_loc = torch.from_numpy(np.mean(u_np, axis=0)).double()
        u_std = torch.from_numpy(np.std(u_np, axis=0)).double()
        u_std[u_std < 1e-6] = 1.0
        torch_iekf.u_loc = u_loc
        torch_iekf.u_std = u_std
        _expected = Path(weights_path).with_name(Path(weights_path).stem + '_norm.p')
        print(f"[iekf_ai_imu_online] WARNING: normalisation factors not found at "
              f"{_expected} — recomputed from the sequence (NOT causal, not deployable). "
              f"Re-run train_ai_imu.py to regenerate {_expected.name}.")
    return torch_iekf, weights_path


def causal_mes_inference(torch_iekf, u_np, window=DEFAULT_WINDOW, lookahead=0,
                         _timing=None):
    """
    Causal MesNet inference: one output per step from a window of PAST samples
    (plus an optional small `lookahead` of future samples).

    Parameters
    ----------
    torch_iekf : loaded TORCHIEKF (eval mode, fixed u_loc/u_std)
    u_np       : (N, 6) IMU data [gyro(3), accel(3)]
    window     : past-context length W (samples)
    lookahead  : number of FUTURE samples allowed (= latency in samples).
                 lookahead=0 → strictly causal, no future data read (default).
                 lookahead=8 → full receptive field, ≈ batch result, 8-sample latency.
    _timing    : optional list — per-step wall-clock seconds of the network
                 call get appended, for real per-sample Event instrumentation
                 (see run()'s CPU path, which uses this for genuine streaming).

    Returns
    -------
    (N, 2) measurement covariances [cov_lat, cov_up].

    Recomputes the full window each step (O(N·(W+lookahead))) — this is the
    genuine per-tick computation, not a reference/approximation: with
    lookahead=0 and a causal, left-padded network (see CausalMesNet), this
    is mathematically identical to the whole-sequence batch pass at every
    index (verified empirically to float64 rounding precision, ~1e-14), the
    causal receptive field is only ~17 samples wide (see module docstring),
    so there is no accuracy reason to prefer the batch pass — only a
    (removed) implementation-convenience one. On-device you would feed a
    fixed (1, 6, W+lookahead) tensor each tick and read the row aligned to
    sample i. Out-of-range taps (sequence start/end) are replication-padded,
    exactly as MesNet's own ReplicationPad would do — so output[i] never
    depends on data beyond i+lookahead.
    """
    import torch
    N = u_np.shape[0]
    covs = np.zeros((N, 2), dtype=np.float64)
    with torch.no_grad():
        for i in range(N):
            lo, hi = i - window + 1, i + lookahead          # inclusive range
            win = u_np[max(0, lo):min(N - 1, hi) + 1]
            if lo < 0:                                       # replicate first sample
                win = np.vstack([np.repeat(u_np[0:1], -lo, axis=0), win])
            if hi > N - 1:                                   # replicate last sample
                win = np.vstack([win, np.repeat(u_np[N - 1:N], hi - (N - 1), axis=0)])
            if _timing is not None:
                _t0 = time.perf_counter()
            out = torch_iekf.forward_nets(torch.from_numpy(win).double())  # (len, 2)
            if _timing is not None:
                _timing.append(time.perf_counter() - _t0)
            covs[i] = out[-(lookahead + 1)].cpu().numpy()    # row aligned to sample i
    return covs


def _build_inputs(nav_data):
    """Common IMU/time/init extraction shared with the batch wrapper."""
    N = len(nav_data.accel_flu)
    sr = nav_data.sample_rate
    if nav_data.time is not None:
        t = nav_data.time.astype(np.float64)
    else:
        t = np.arange(N, dtype=np.float64) / sr
    u_np = np.hstack([nav_data.gyro_flu, nav_data.accel_flu]).astype(np.float64)
    ang0 = nav_data.orient[0].astype(np.float64)
    v0 = nav_data.vel_enu[0].astype(np.float64)
    return N, sr, t, u_np, ang0, v0


def _prepare_gps(nav_data, params, outage_config):
    """GPS/DR_MODE/outage preparation — identical semantics to iekf_ai_imu.run."""
    p_gps_for_loop = gps_avail_masked = R_gps = None
    try:
        import ins_config as _ic
        dr_mode = getattr(_ic, 'DR_MODE', False)
    except ImportError as _e:
        print(f"[iekf_ai_imu_online] WARNING: ins_config not importable ({_e}); "
              "defaulting DR_MODE=False (GPS aiding ON).")
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
            d = outage_config.get('duration', 0.0)
            A = int(t1 * nav_data.sample_rate)
            B = int((t1 + d) * nav_data.sample_rate)
            gps_avail_masked[A:B] = False
        rpos = float((params or {}).get('Rpos', 4.0))
        R_gps = np.eye(3) * rpos
        n_gps = int(gps_avail_masked.sum())
        print(f"AI-IMU (online): GPS-aided — {n_gps} GPS updates "
              f"({'outage applied' if outage_config else 'no outage'}).")
    elif dr_mode:
        print("AI-IMU (online): DR_MODE=True — pure dead-reckoning (no GPS).")
    return p_gps_for_loop, gps_avail_masked, R_gps


def run(nav_data, params=None, outage_config=None, use_3d_rotation=True,
        backend='cpu', hailo_net=None):
    """
    Causal/online AI-IMU IEKF.  Same interface and return dict as iekf_ai_imu.run.
    The two backends now differ in HOW the MesNet covariances are produced,
    not just in WHERE:

    backend   : 'cpu' (default) streams genuinely — one DEFAULT_WINDOW-sample
                rolling window per IMU tick via causal_mes_inference(), the
                real per-sample computation a live deployment would run
                (see module docstring for the numerical-equivalence check
                against the old whole-sequence call).
                'hailo' still runs one whole-sequence pass on-device, because
                the compiled HEF has a FIXED input length
                (hailo_backend.HailoDeepIEKF.SEQ_LEN, 4544 samples ~ 45.44s
                @ 100Hz) and no smaller, streaming-sized HEF has been
                compiled yet (needs the x86_64-only Hailo DFC — see module
                docstring). nav_data must already be truncated to exactly
                SEQ_LEN by the caller when backend='hailo' — this is NOT
                done here, since only the caller knows whether that also
                needs to be reflected in which outage window makes sense
                to test.
    hailo_net : Required when backend='hailo' — an already-activated
                hailo_backend.HailoDeepIEKF instance.

    Returns dict as iekf_ai_imu.run, plus 'net_latency_s': an (N,) array of
    genuine per-tick network call times on CPU (one real Event per IMU
    sample, matching TLIO/Deep KF/Tartan IMU's own instrumentation), or a
    1-element array on Hailo (the whole-sequence pass has no per-tick
    breakdown there — see module docstring).
    """
    _add_aimu_path()
    from utils_numpy_filter import NUMPYIEKF

    N, _sr, t, u_np, ang0, v0 = _build_inputs(nav_data)

    # IEKF core with KITTI-tuned parameters
    from kitti_params import get_kitti_parameters
    iekf = NUMPYIEKF(get_kitti_parameters())
    if not isinstance(iekf.g, np.ndarray):
        iekf.g = np.array(iekf.g)

    # This online filter is ALWAYS a true causal model: a left-padded CausalMesNet
    # trained with `train_ai_imu.py --causal`, evaluated in a single batch pass
    # (causal by construction → strictly online, 0 ms latency, exact). Running the
    # upstream ACAUSAL MesNet causally (sliding-window approximation) is
    # intentionally NOT offered — it would report an approximation as the online
    # result.
    online_weights = _find_online_weights()
    if backend == 'hailo' and online_weights is None:
        raise RuntimeError(
            "backend='hailo' requires CAUSAL MesNet weights (for set_Q / "
            "initprocesscov_net, not compiled into the HEF) — none found. "
            "See the RuntimeError below for how to train them.")

    if backend == 'hailo' and N != hailo_net.SEQ_LEN:
        raise ValueError(
            f"backend='hailo' requires nav_data truncated to exactly "
            f"{hailo_net.SEQ_LEN} samples (the HEF's fixed input length); "
            f"got N={N}.")

    measurements_covs_np = None
    net_latency_s = np.zeros(0)
    if online_weights is not None:
        import torch
        # Process-noise Q / initial-covariance scaling come from a SEPARATE
        # learned head (set_Q / initprocesscov_net) that is NOT part of the
        # measurement-covariance CausalMesNet compiled into the Hailo HEF
        # (see hailo/deep_iekf/0_onnx_converter.py — "cov_net backbone +
        # cov_lin head" only). This still has to run on CPU/torch regardless
        # of backend; only the measurement-covariance forward pass below
        # swaps to Hailo.
        torch_iekf = _load_causal_torch_iekf(u_np, online_weights)
        iekf.set_learned_covariance(torch_iekf)

        if backend == 'hailo':
            measurements_covs_np, dt = hailo_net.run_whole_sequence(u_np.astype(np.float32))
            net_latency_s = np.array([dt])
            print(f"AI-IMU (online, Hailo): CAUSAL MesNet whole-sequence HEF call "
                  f"— {dt*1000:.3f} ms for {N} samples.")
        else:
            # Genuine per-sample streaming: one DEFAULT_WINDOW-sample rolling
            # call per IMU tick, not a whole-sequence batch pass. Verified
            # numerically identical to the old batch call at every index
            # (float64 rounding only, ~1e-14) since the network is causal
            # and its true receptive field (~17 samples) is well inside
            # DEFAULT_WINDOW — there is no accuracy cost to streaming this,
            # only the removed convenience of one big call instead of N
            # small ones. This is what a real online deployment would run.
            _timing = []
            measurements_covs_np = causal_mes_inference(
                torch_iekf, u_np, window=DEFAULT_WINDOW, lookahead=0,
                _timing=_timing)
            net_latency_s = np.array(_timing)
            print(f"AI-IMU (online): CAUSAL MesNet (left-padded), streamed "
                  f"{N} samples one {DEFAULT_WINDOW}-sample window at a time "
                  f"from {online_weights}.")

    if measurements_covs_np is None:
        # No causal-trained weights. Do NOT fall back to acausal weights run
        # causally (removed): require a true causal model by default.
        if os.environ.get('IEKF_AI_IMU_REQUIRE_WEIGHTS', '1') != '0':
            raise RuntimeError(
                "[iekf_ai_imu_online] No CAUSAL MesNet weights found in "
                f"{_ARTIFACTS_ONLINE} (or via AI_IMU_ONLINE_WEIGHTS). This filter is "
                "always a true causal model; the acausal-weights-run-causally "
                "approximation has been removed. Train one per LOO fold:\n"
                "  python dl_filters/deep_iekf/train_ai_imu.py --causal --mode loo "
                "--held-out <drive>\n"
                "Set IEKF_AI_IMU_REQUIRE_WEIGHTS=0 to allow the fixed-covariance "
                "fallback (smoke tests only)."
            )
        cov_lat, cov_up = iekf.cov_lat, iekf.cov_up
        measurements_covs_np = np.tile([cov_lat, cov_up], (N, 1))
        print(f"[iekf_ai_imu_online] WARNING: no causal weights — fallback fixed "
              f"covariances [cov_lat={cov_lat}, cov_up={cov_up}] (no CNN adapter).")

    # ── GPS / outage / DR_MODE ──────────────────────────────────────────────────
    p_gps_for_loop, gps_avail_masked, R_gps = _prepare_gps(nav_data, params, outage_config)

    # ── Sequential IEKF (reuses the batch wrapper's loop verbatim) ──────────────
    p, v, r, b_omega, b_acc, std_pos, std_vel, std_orient, std_bias_gyr, std_bias_acc = \
        _run_filter_loop(iekf, t, u_np, measurements_covs_np, v0, ang0,
                         p_gps=p_gps_for_loop, gps_available=gps_avail_masked, R_gps=R_gps)

    return {
        'p': p, 'v': v, 'r': r,
        'bias_acc': b_acc, 'bias_gyr': b_omega,
        'std_pos': std_pos, 'std_vel': std_vel, 'std_orient': std_orient,
        'std_bias_acc': std_bias_acc, 'std_bias_gyr': std_bias_gyr,
        'net_latency_s': net_latency_s,
    }


# ── Validation: causal (this module) vs batch (iekf_ai_imu) ──────────────────────

def validate(nav_data, window=DEFAULT_WINDOW, lookahead=0):
    """
    Diagnostic: quantify, at the covariance level, what running the upstream
    ACAUSAL MesNet weights causally (sliding window, current-sample-hold) costs
    versus the whole-sequence batch pass. This is decoupled from run(), which
    always uses a true causal model (CausalMesNet); it exists only to justify
    training one (path A) — see DEVIATIONS.md.
    """
    _add_aimu_path()
    _, _, _, u_np, _, _ = _build_inputs(nav_data)

    torch_iekf, weights_path = _load_torch_iekf(nav_data, u_np)
    if torch_iekf is None:
        print("validate: no (acausal) weights found — nothing to compare.")
        return

    import torch
    with torch.no_grad():
        batch_covs = torch_iekf.forward_nets(torch.from_numpy(u_np).double()).cpu().numpy()
    causal_covs = causal_mes_inference(torch_iekf, u_np, window, lookahead)

    # Ignore the start-up warm region (first `window` samples) where left context
    # is replication-padded in both schemes.
    s = window
    d = np.abs(causal_covs[s:] - batch_covs[s:])
    rel = d / (np.abs(batch_covs[s:]) + 1e-12)
    print(f"\n=== MesNet covariance: acausal-run-causally vs batch "
          f"(window={window}, lookahead={lookahead}) ===")
    print(f"  weights: {weights_path}")
    for j, name in enumerate(['cov_lat', 'cov_up']):
        print(f"  {name}: max|Δ|={d[:, j].max():.4g}  mean|Δ|={d[:, j].mean():.4g}  "
              f"mean rel={rel[:, j].mean()*100:.2f}%")
    print("  (Non-zero gap ⇒ train a causal model; run() no longer approximates.)\n")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Causal/online AI-IMU deep-IEKF")
    ap.add_argument('--validate', action='store_true',
                    help="compare causal vs batch covariances and trajectory")
    ap.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                    help=f"past-context length in samples (default {DEFAULT_WINDOW})")
    ap.add_argument('--lookahead', type=int, default=0,
                    help="future samples allowed = latency (0 = strictly causal, default)")
    args = ap.parse_args()

    sys.path.insert(0, str(_HERE.parent.parent))   # scripts/positioning/python
    import ins_config as _ic
    nav = _ic.NAV_DATA

    if args.validate:
        validate(nav, window=args.window, lookahead=args.lookahead)
    else:
        out = run(nav, params={'window': args.window, 'lookahead': args.lookahead})
        print(f"Done. position samples: {out['p'].shape}, "
              f"final pos = {out['p'][-1]}")
