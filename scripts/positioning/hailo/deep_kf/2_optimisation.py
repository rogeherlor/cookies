"""
DeepKFNet optimisation and cross-backend comparison
====================================================
Runs the same synthetic inputs through five backends and prints a comparison:

  1. PyTorch      — original .pt checkpoint (ground truth)
  2. ONNX         — onnxruntime on deep_kf.onnx
  3. SDK_NATIVE   — Hailo emulator, no changes (full precision)
  4. SDK_FP_OPT   — after optimize_full_precision()
  5. SDK_QUANTIZED— after quantization with calibration data

Note on output
--------------
The model predicts the full 15D navigation state x_t^{+-}:
    [p(3) | v(3) | rpy(3) | b_acc(3) | b_gyr(3)]

This is a residual prediction: x_t^{+-} = decoder(LSTM(x_{t-1}^+)) + x_{t-1}^+

Usage
-----
    python 2_optimisation.py [--artifact artifacts/deep_kf/fold_01.pt]
                             [--hidden-dim 128] [--num-layers 2]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from hailo_sdk_client import ClientRunner, InferenceContext

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_kf"

if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from model import DeepKFNet  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
ONNX_PATH         = _HERE / "deep_kf.onnx"
HAR_PATH          = _HERE / "deep_kf_hailo_model.har"
QUANTIZED_HAR_PATH= _HERE / "deep_kf_quantized_model.har"

NAV_DIM    = 15
STATE_DIM  = 15
N_CALIB    = 1024   # calibration samples (Hailo recommends >= 1024)
N_INFER    = 8      # samples used for comparison printout


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("deep_kf_optimisation")

log = init_logging()   # module-level so helpers can use it


# ── PyTorch wrapper — imported, not duplicated ────────────────────────────────
# 0_onxx_converter.py owns the definition of the exported graph. This file used
# to keep its own copy, which is how the two drifted: the copy here still
# carried BIAS_HH_EPS and constant h0 after the exporter moved to a stateful
# cell, so the "MAE vs PyTorch" line would have compared the quantised stateful
# graph against a memoryless reference and reported nonsense.

def _load_converter():
    import importlib.util
    _p = _HERE / "0_onxx_converter.py"
    _spec = importlib.util.spec_from_file_location("_dkf_conv", str(_p))
    _m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_m)
    return _m

DeepKFNetONNX = _load_converter().DeepKFNetONNX


# ── Inference helpers ─────────────────────────────────────────────────────────
# Every backend is fed the SAME captured (x, h, c) tuples, so each call is an
# independent single step with a real recurrent context. That is both what the
# quantiser needs to see and what makes the three backends comparable per step.

def _make_x(nav_np):
    """Add seq dim -> [N, 1, D]."""
    return nav_np[:, np.newaxis, :]


def infer_pytorch(model, nav_np, states):
    """Run the PyTorch cell on captured states, return state [N, 15]."""
    h0, c0, h1, c1 = states
    with torch.no_grad():
        delta, *_ = model(torch.from_numpy(_make_x(nav_np)).float(),
                          torch.from_numpy(_make_x(h0)).float(),
                          torch.from_numpy(_make_x(c0)).float(),
                          torch.from_numpy(_make_x(h1)).float(),
                          torch.from_numpy(_make_x(c1)).float())
    return delta.numpy() + nav_np      # x_t^{+-} = delta + x_{t-1}^+


def infer_onnx(session, nav_np, states):
    """Run onnxruntime session on captured states, return state [N, 15]."""
    h0, c0, h1, c1 = states
    x = _make_x(nav_np)
    xh0, xc0, xh1, xc1 = _make_x(h0), _make_x(c0), _make_x(h1), _make_x(c1)
    results = []
    for i in range(len(x)):
        out = session.run(None, {"x": x[i:i+1], "h_l0": xh0[i:i+1], "c_l0": xc0[i:i+1],
                                 "h_l1": xh1[i:i+1], "c_l1": xc1[i:i+1]})
        results.append(out[0])   # delta [1, 15]
    return np.concatenate(results, axis=0) + nav_np


def _hailo_input_names(runner):
    """Return ordered list of actual Hailo input layer names from the HN."""
    layers = runner._hn.get_input_layers()
    return [l.name for l in layers]


def _to_nhwc(arr3):
    """[N, 1, D] -> [N, 1, 1, D] NHWC, the 4-D layout Hailo stores inputs in."""
    return arr3[:, :, np.newaxis, :]


def infer_hailo(runner, ctx, nav_np, states):
    """Run Hailo emulator inference on captured states, return state [N, 15].

    Five inputs now (x plus both layers' h/c) and five outputs. The input names
    are taken in parse order, which 1_parsing.py fixes as
    x, h_l0, c_l0, h_l1, c_l1.
    """
    h0, c0, h1, c1 = states
    names = _hailo_input_names(runner)
    if len(names) != 5:
        raise RuntimeError(
            f"Expected 5 Hailo input layers (x, h_l0, c_l0, h_l1, c_l1), "
            f"got {len(names)}: {names}. Re-run 1_parsing.py — the graph must "
            f"be the stateful cell, not the old memoryless export.")
    feed = dict(zip(names, (_to_nhwc(_make_x(nav_np)),
                            _to_nhwc(_make_x(h0)), _to_nhwc(_make_x(c0)),
                            _to_nhwc(_make_x(h1)), _to_nhwc(_make_x(c1)))))
    results = runner.infer(ctx, feed)
    if not isinstance(results, (list, tuple)):
        results = [results]
    # Pick the delta head by width rather than position: the compiler is free to
    # reorder outputs, and silently taking results[0] would return a hidden
    # state (128 wide) reshaped into a state vector.
    delta = None
    for r in results:
        a = np.asarray(r)
        if a.reshape(a.shape[0], -1).shape[1] == nav_np.shape[1]:
            delta = a.reshape(a.shape[0], -1)
            break
    if delta is None:
        raise RuntimeError(
            f"No Hailo output of width {nav_np.shape[1]} (the delta head); "
            f"got widths {[np.asarray(r).reshape(np.asarray(r).shape[0], -1).shape[1] for r in results]}")
    return delta + nav_np  # residual


# ── Comparison printout ───────────────────────────────────────────────────────

def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


STATE_LABELS = "p_e  p_n  p_u  v_e  v_n  v_u  roll  pitch  yaw  ba_x  ba_y  ba_z  bg_x  bg_y  bg_z"


def print_comparison(backends: dict[str, np.ndarray], reference: str = "PyTorch"):
    ref = backends[reference]
    print("\n" + "=" * 80)
    print(f"STATE PREDICTIONS  [{STATE_LABELS}]")
    print("=" * 80)
    for i in range(ref.shape[0]):
        print(f"\nSample {i}:")
        for name, arr in backends.items():
            print(f"  {name:<14} {_fmt(arr[i])}")

    print("\n" + "-" * 80)
    print(f"MAE vs {reference}:")
    for name, arr in backends.items():
        if name == reference:
            continue
        mae = float(np.mean(np.abs(arr - ref)))
        print(f"  {name:<14} {mae:.6e}")
    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact",   type=Path,
                        default=_REPO_ROOT / "artifacts/deep_kf/fold_01.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nav-dim",    type=int, default=NAV_DIM)
    parser.add_argument("--calib-seqs", nargs="+", default=["00"],
                        help="KITTI sequences to capture calibration states "
                             "from. MUST exclude this fold's held-out sequence. "
                             "build_per_fold_hefs.py passes the fold's five "
                             "training sequences; the default '00' is a "
                             "leak-free but narrow fallback.")
    args = parser.parse_args()

    # ── Calibration data — REAL e_norm_in from a live CPU run ─────────────────
    # The HEF is fed the normalised error state e_norm_in = (e-norm_mean)/
    # norm_std (see deep_kf_runner.py's run()/​_run_hailo(), and
    # hailo_backend.HailoDeepKF's docstring) — small, ~O(1) values, NOT raw
    # absolute nav states (position in metres, velocity in m/s, with 3 bias
    # channels that are always exactly zero at t=0 and never move far from
    # it). Calibrating on the wrong distribution made the quantiser assign
    # a badly wrong dynamic range: measured mean abs error ~0.19 on the real
    # signal (the actual signal itself has mean abs magnitude ~0.09 — the
    # error was LARGER than the quantity being predicted), denormalising to
    # several metres of spurious position error. Capturing the real signal
    # from an actual run (aided + a genuine outage window, so the calibration
    # set spans the network's whole real operating range) via a forward hook
    # — no filter logic duplicated, so this is the true production
    # distribution, not an approximation of it.
    _scripts = _REPO_ROOT / "scripts/positioning/python"
    try:
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        _dkf_dir = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_kf"
        if str(_dkf_dir) not in sys.path:
            sys.path.insert(0, str(_dkf_dir))
        import os as _os
        import data_loader as _dl
        import deep_kf_runner as _dkr
        from model import DeepKFNet as _DeepKFNet

        # Capture over SEVERAL sequences, not one. Calibration sets the INT8
        # ranges, and during an outage this model runs autoregressively: its own
        # prediction becomes the next input, so the state wanders well outside
        # the range any single drive visits. Calibrating on seq 00 alone left
        # the quantiser mis-ranged out there, and the resulting error was not
        # zero-mean noise but a systematic per-channel BIAS (~0.095 against a
        # signal of magnitude ~1.08, measured along the real seq-10 trajectory).
        # Integrated over ~6000 outage steps that bias is what turned a 69 m
        # CPU result into 1041 m on device. Injecting zero-mean noise of the
        # same size moved ATE by only ~13%, which is how the bias was isolated.
        _captured = []
        _orig_forward = _DeepKFNet.forward

        def _hooked(self, nav_state, hidden=None):
            # Capture the INPUT recurrent context alongside the input state.
            # The graph now takes h/c as real inputs, so the quantiser has to
            # see their true distribution; calibrating them at zero was what
            # collapsed their range and forced the old constant-h0 workaround.
            _hd = self.lstm.lstm.hidden_size
            if hidden is None:
                _b = nav_state.shape[0]
                _h = np.zeros((2, _b, _hd), np.float32)
                _c = np.zeros((2, _b, _hd), np.float32)
            else:
                _h = hidden[0].detach().cpu().numpy().copy()
                _c = hidden[1].detach().cpu().numpy().copy()
            _captured.append((nav_state.detach().cpu().numpy().copy(), _h, _c))
            return _orig_forward(self, nav_state, hidden)

        _prev_w = _os.environ.get('DEEP_KF_WEIGHTS')
        _os.environ['DEEP_KF_WEIGHTS'] = str(args.artifact)  # calibrate on THESE weights
        _DeepKFNet.forward = _hooked
        try:
            # 40s/60s matches the project's standard outage scenario, so the
            # calibration set spans both aided (kept-warm) and genuine-outage
            # (actually-used) e_norm_in values.
            for _cs in args.calib_seqs:
                _nav_c = _dl.get_kitti_dataset(_cs)
                _dkr.run(_nav_c, backend='cpu',
                         outage_config={'start': 40., 'duration': 60.})
        finally:
            _DeepKFNet.forward = _orig_forward
            if _prev_w is None:
                _os.environ.pop('DEEP_KF_WEIGHTS', None)
            else:
                _os.environ['DEEP_KF_WEIGHTS'] = _prev_w

        _x = np.concatenate([a for a, _, _ in _captured], axis=0)
        _x = (_x[:, 0, :] if _x.ndim == 3 else _x).astype(np.float32)
        _hh = np.stack([h for _, h, _ in _captured], axis=0)   # (T, 2, B, H)
        _cc = np.stack([c for _, _, c in _captured], axis=0)
        _hh = _hh[:, :, 0, :].astype(np.float32)               # (T, 2, H)
        _cc = _cc[:, :, 0, :].astype(np.float32)
        idx = np.linspace(0, len(_x) - 1, N_CALIB, dtype=int)
        calib_nav    = _x[idx]
        calib_states = (_hh[idx, 0], _cc[idx, 0], _hh[idx, 1], _cc[idx, 1])
        log.info("Calibration: using %d real (e_norm_in, h, c) tuples captured "
                 "from live CPU runs on seqs %s (40s/60s outage each)",
                 len(calib_nav), ",".join(args.calib_seqs))
    except Exception as _e_cal:
        log.warning("Real capture unavailable (%s) — using synthetic N(0,1) "
                    "data (matches the ~O(1) normalised scale).", _e_cal)
        rng = np.random.default_rng(42)
        calib_nav = rng.standard_normal((N_CALIB, args.nav_dim)).astype(np.float32)
        _hd = args.hidden_dim
        calib_states = tuple(
            rng.standard_normal((N_CALIB, _hd)).astype(np.float32) for _ in range(4))
    infer_nav    = calib_nav[:N_INFER]
    infer_states = tuple(a[:N_INFER] for a in calib_states)

    backends = {}

    # ── 1. PyTorch ────────────────────────────────────────────────────────────
    log.info("Loading PyTorch checkpoint: %s", args.artifact)
    ckpt = torch.load(args.artifact, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = cfg.get("latent_dim", args.hidden_dim)
    num_layers  = cfg.get("num_layers",  args.num_layers)

    pt_model = DeepKFNet(
        nav_state_dim=args.nav_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
    )
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    wrapped = DeepKFNetONNX(pt_model)
    wrapped.eval()

    log.info("PyTorch inference ...")
    backends["PyTorch"] = infer_pytorch(wrapped, infer_nav, infer_states)

    # ── 2. ONNX ───────────────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        import onnxruntime as ort
        log.info("ONNX inference: %s", ONNX_PATH)
        session = ort.InferenceSession(str(ONNX_PATH))
        backends["ONNX"] = infer_onnx(session, infer_nav, infer_states)
    else:
        log.warning("ONNX not found (%s) — skipping.", ONNX_PATH)

    # ── 3-5. Hailo HAR ────────────────────────────────────────────────────────
    if not HAR_PATH.exists():
        log.warning("HAR not found (%s) — skipping Hailo stages.", HAR_PATH)
    else:
        runner = ClientRunner(har=str(HAR_PATH))
        hailo_names = _hailo_input_names(runner)
        log.info("Hailo input layer name(s): %s", hailo_names)
        # All five inputs are calibrated on their REAL captured distributions.
        _ch0, _cc0, _ch1, _cc1 = calib_states
        calib_dataset = dict(zip(hailo_names, (
            _to_nhwc(_make_x(calib_nav)),
            _to_nhwc(_make_x(_ch0)), _to_nhwc(_make_x(_cc0)),
            _to_nhwc(_make_x(_ch1)), _to_nhwc(_make_x(_cc1)))))

        # 3. SDK_NATIVE — no modifications
        log.info("Hailo SDK_NATIVE inference ...")
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            backends["SDK_NATIVE"] = infer_hailo(runner, ctx, infer_nav, infer_states)

        # 4. Full-precision optimization -> SDK_FP_OPTIMIZED
        log.info("Running optimize_full_precision() ...")
        runner.optimize_full_precision(calib_dataset)

        log.info("Hailo SDK_FP_OPTIMIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            backends["SDK_FP_OPT"] = infer_hailo(runner, ctx, infer_nav, infer_states)

        # 5. Quantization -> SDK_QUANTIZED
        # Two Hailo SDK bugs encountered and worked around:
        #
        # Bug 1 — dead_layers_removal IndexError:
        #   Normalization layers wrapping LSTM hidden-state inputs have near-zero
        #   weights and are flagged as dead.  After removal the output node has no
        #   predecessor → IndexError in model_flow.py.
        #   Fix: disable dead_layers_removal via model script (below).
        #
        # Bug 2 — EW_Add zero-scale crash (ValueError in int_smallnum_factorize):
        #   With h0=0 the LSTM recurrent branch (U·h + b_hh) is all-zeros when
        #   b_hh≈0, giving scale=0 → desired_factor=inf → np.arange crash.
        #   Fix: BIAS_HH_EPS=1e-2 added to bias_hh in DeepKFNetONNX.__init__ so
        #   the recurrent branch is never all-zeros even at the first calibration
        #   step.  This is applied in the ONNX wrapper only (trained weights unchanged).
        #
        runner.load_model_script(
            "pre_quantization_optimization(dead_layers_removal, policy=disabled)\n"
        )
        log.info("Running optimize() (quantization) ...")
        try:
            runner.optimize(calib_dataset)
            runner.save_har(str(QUANTIZED_HAR_PATH))
            log.info("Quantized HAR saved: %s", QUANTIZED_HAR_PATH)

            log.info("Hailo SDK_QUANTIZED inference ...")
            with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
                backends["SDK_QUANTIZED"] = infer_hailo(runner, ctx, infer_nav, infer_states)
        except Exception as e:
            # Fatal on purpose — see the same guard in tlio/2_optimisation.py.
            # 3_compilation.py compiles whatever quantized HAR is on disk, so a
            # swallowed failure here silently ships the previous fold's weights.
            log.error(
                "Quantization FAILED (%s: %s). "
                "Root cause: normalization2/7 (wrapping W_hh @ h in the LSTM recurrent "
                "branch) receive near-zero activations during calibration because Hailo "
                "runs each calibration sample independently with constant h₀. "
                "The SDK over-compresses them to ≤2-bit, producing NaN kernels. "
                "Fix: re-export ONNX with H_INIT=1.0 constant h₀ (0_onxx_converter.py), "
                "re-parse (1_parsing.py), then re-run this script. "
                "Not writing the quantized HAR — the stale one on disk belongs to a "
                "different export, and no HEF may be built from this run.",
                type(e).__name__, e,
            )
            print_comparison(backends)
            raise

    # ── Print results ─────────────────────────────────────────────────────────
    print_comparison(backends)


if __name__ == "__main__":
    main()
