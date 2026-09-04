"""
Streaming (small-window) Deep-IEKF CausalMesNet -> Hailo HEF
============================================================
Companion to ../deep_iekf/ (the whole-sequence SEQ_LEN=4544 build).  This one
compiles the SAME causal network at a SMALL fixed window (STREAM_W samples) so
it can be driven online: measurements arrive from the sensor and are processed
in fixed STREAM_W-wide windows, either

  * per-tick  : slide the window by 1 each new IMU sample, keep the last output
                (genuine online, 1-sample latency), or
  * block-K   : slide by K = STREAM_W-16, keep the last K outputs per call
                (amortised; K=16 with STREAM_W=32).

Both are bit-faithful to the whole-sequence pass in steady state because the
network is a causal TCN with receptive field 17: an output keeps only its own
16 samples of real left context, so any window that carries >=16 real samples
before the kept output(s) reproduces the full-sequence result exactly (proven
in scratchpad/verify_chunk_math.py to <=2e-6).  Only the first ~16 samples of
the WHOLE stream differ (ZeroPad warmup — identical to the 4544 model's own).

Runs entirely inside the hailo_ai_sw_suite DFC container (x86-only), same as
../deep_iekf/0..3.  Produces, next to this script:
    deep_iekf_stream.onnx              (1,6,1,STREAM_W)
    deep_iekf_stream_hailo_model.har
    deep_iekf_stream_quantized_model.har
    deep_iekf_stream.hef
    deep_iekf_postproc.npz             (u_loc,u_std — window-size independent)

Usage (inside container):
    python3 build_stream.py [--weights .../fold_01.p] [--window 32] [--calib-seq 00]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths (derive from this file so container & host both work) ────────────────
_HERE      = Path(__file__).resolve().parent
_REPO      = _HERE.parent.parent.parent.parent
_IEKF_DIR  = _REPO / "scripts/positioning/python/dl_filters/deep_iekf"
_AI_IMU    = _REPO / "external/ai-imu-dr/src"
_SCRIPTS   = _REPO / "scripts/positioning/python"
_DEEP_IEKF = _REPO / "scripts/positioning/hailo/deep_iekf"   # reuse MesNetFullHailo
_ONLINE    = _REPO / "artifacts/deep_iekf_online"
for _p in [str(_IEKF_DIR), str(_AI_IMU), str(_SCRIPTS), str(_DEEP_IEKF)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

IMU_CH  = 6
COV_DIM = 2


def build_wrapper(weights: Path, window: int):
    """Return (z-output wrapper at eval, u_loc, u_std, beta(2,), cov0(2,)).

    The on-device network outputs the BOUNDED z = tanh(cov_lin(cov_net(u))) in
    [-1,1] — NOT the scaled covariance cov0*10**(beta*z).  Baking the exp scaling
    on-device (MesNetFullHailo) makes the HEF emit a heavy-tailed 0..~10000 range
    that INT8 quantises coarsely — catastrophic for folds whose covariances spike
    (e.g. fold_06: cov_up up to ~8800 -> 23% error -> 150 m drift).  A bounded z is
    uniformly well-conditioned for every fold; the cheap scalar scaling
    cov = cov0*10**(beta*z) is done in Python (deep_iekf_stream_postproc.npz).
    """
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    from iekf_ai_imu_online import _find_online_weights
    from kitti_params import get_kitti_parameters
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "onnx_conv", str(_DEEP_IEKF / "0_onnx_converter.py"))
    onnx_conv = importlib.util.module_from_spec(spec); spec.loader.exec_module(onnx_conv)

    class MesNetZHailo(onnx_conv.MesNetFullHailo):
        """Same backbone+head as MesNetFullHailo but returns the bounded z (no exp)."""
        def forward(self, u_norm_conv):
            feat = self.cov_net(u_norm_conv)                 # (1,32,1,N)
            return self.tanh(self.cov_lin_conv(feat))        # (1, 2,1,N) in [-1,1]

    if not weights.exists():
        # Do NOT fall back to _find_online_weights() here. Its priority order
        # reaches a generic artifacts/deep_iekf_online/iekfnets.p — whatever
        # training run wrote last — before any per-fold file, so a mistyped or
        # not-yet-trained --weights would quietly compile a HEF tagged
        # deep_iekf_stream_fold_<seq> that holds some other fold's weights.
        # That is the exact leak the per-fold build exists to remove, and it
        # would be invisible: the file name would still say fold_<seq>.
        raise FileNotFoundError(
            f"CAUSAL weights not found: {weights}\n"
            f"Refusing to substitute another checkpoint — the output is named per "
            f"fold and a substitution would be undetectable downstream.\n"
            f"Train the fold first, or pass an explicit --weights path.")
    ti = TORCHIEKF(get_kitti_parameters())
    if ti.cov0_measurement is None:
        ti.cov0_measurement = torch.tensor([1.0, 10.0]).double()
    attach_causal_mesnet(ti)
    print(f"Loading CAUSAL weights: {weights}")
    ti.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False))
    ti.eval()
    norm = _find_norm_factors(weights)
    if norm is None:
        raise FileNotFoundError(f"norm factors (<stem>_norm.p) not found next to {weights}")
    ti.u_loc = norm["u_loc"].double(); ti.u_std = norm["u_std"].double()
    wrapped = MesNetZHailo(ti).eval()
    beta = ti.mes_net.beta_measurement.detach().numpy().astype(np.float32)      # (2,)
    cov0 = ti.cov0_measurement.detach().numpy().astype(np.float32)              # (2,)
    return (wrapped, ti.u_loc.numpy().astype(np.float32),
            ti.u_std.numpy().astype(np.float32), beta, cov0)


def calib_windows(u_loc, u_std, window, seq="00", stride=16, max_win=800):
    """Real KITTI IMU sliced into (M,1,window,6) NHWC normalized windows for PTQ."""
    import data_loader
    nav = data_loader.get_kitti_dataset(seq)
    imu = np.concatenate([nav.gyro_flu, nav.accel_flu], axis=1).astype(np.float32)
    u_norm = (imu - u_loc) / u_std                       # (N,6)
    starts = list(range(0, len(u_norm) - window, stride))[:max_win]
    wins = np.stack([u_norm[s:s + window] for s in starts])          # (M,window,6)
    return wins[:, None, :, :]                                        # (M,1,window,6) NHWC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=_ONLINE / "fold_01.p")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--calib-seq", default="00")
    ap.add_argument("--opset", type=int, default=11)
    ap.add_argument("--tag", default="",
                    help="Output suffix, e.g. 'fold_01' -> deep_iekf_stream_fold_01.{onnx,hef,...}")
    args = ap.parse_args()
    W = args.window
    stem = "deep_iekf_stream" + (f"_{args.tag}" if args.tag else "")

    wrapped, u_loc, u_std, beta, cov0 = build_wrapper(args.weights, W)

    # ── postproc: u_loc/u_std (per-fold) + beta/cov0 for the Python scaling ────
    #   HEF emits z in [-1,1]; host computes  cov = cov0 * 10**(beta * z).
    np.savez(str(_HERE / f"{stem}_postproc.npz"),
             u_loc=u_loc, u_std=u_std, beta=beta, cov0=cov0)

    # ── step 0: export ONNX at (1,6,1,W) ──────────────────────────────────────
    onnx_path = _HERE / f"{stem}.onnx"
    dummy = torch.zeros(1, IMU_CH, 1, W, dtype=torch.float32)
    print(f"Exporting ONNX (opset {args.opset}, window={W}) -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(wrapped, (dummy,), str(onnx_path),
                          input_names=["u_norm_conv"], output_names=["measurement_covs"],
                          opset_version=args.opset, do_constant_folding=True, dynamo=False)
    # optional simplify+verify vs torch
    try:
        import onnx, onnxruntime as ort, onnxsim
        m0 = onnx.load(str(onnx_path))
        rng = np.random.default_rng(0)
        xs = [rng.standard_normal([1, IMU_CH, 1, W]).astype(np.float32) for _ in range(8)]
        with torch.no_grad():
            refs = [wrapped(torch.from_numpy(x)).detach().numpy().astype(np.float64) for x in xs]
        ms, ok = onnxsim.simplify(m0)
        if ok:
            sess = ort.InferenceSession(ms.SerializeToString())
            e = max(float(np.max(np.abs(sess.run(None, {"u_norm_conv": x})[0] - r)))
                    for x, r in zip(xs, refs))
            if e < 1e-3:
                onnx.save(ms, str(onnx_path))
                print(f"ONNX simplified+verified vs PyTorch (max|Δ|={e:.2e}, "
                      f"{len(ms.graph.node)} nodes)")
    except Exception as e:
        print(f"[info] onnx simplify/verify skipped ({type(e).__name__}: {e})")

    # ── step 1: parse -> HAR ──────────────────────────────────────────────────
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(hw_arch="hailo8l")
    runner.translate_onnx_model(
        str(onnx_path), stem,
        start_node_names=["u_norm_conv"], end_node_names=["measurement_covs"],
        net_input_shapes={"u_norm_conv": [1, IMU_CH, 1, W]})
    har = _HERE / f"{stem}_hailo_model.har"
    runner.save_har(str(har)); print(f"HAR: {har}")

    # ── step 2: optimize (PTQ) with multi-window real KITTI calibration ───────
    names = [l.name for l in runner._hn.get_input_layers()]
    calib = calib_windows(u_loc, u_std, W, seq=args.calib_seq)
    print(f"Calibration windows: {calib.shape}  (input layer {names[0]})")
    calib_ds = {names[0]: calib}
    runner.optimize_full_precision(calib_ds)
    runner.load_model_script(
        "pre_quantization_optimization(dead_layers_removal, policy=disabled)\n")
    runner.optimize(calib_ds)
    qhar = _HERE / f"{stem}_quantized_model.har"
    runner.save_har(str(qhar)); print(f"Quantized HAR: {qhar}")

    # ── emulator sanity: SDK_QUANTIZED vs PyTorch, in RECONSTRUCTED cov space ─
    #   Both z-outputs are scaled by the host formula cov = cov0*10**(beta*z)
    #   before comparing, so the MAE reflects the real covariance error the
    #   filter sees (the whole point of moving the exp scaling to the host).
    def to_cov(z):                     # z: (...,2) -> cov: (...,2)
        return cov0 * np.power(10.0, beta * z)
    try:
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            res = runner.infer(ctx, {names[0]: calib[:64]})
        qz = np.asarray(res).reshape(-1, W, COV_DIM)[:, -1, :]   # kept z per window (64,2)
        with torch.no_grad():
            pz = wrapped(torch.from_numpy(
                calib[:64].transpose(0, 3, 1, 2).astype(np.float32))).detach().numpy()[:, :, 0, -1]
        mae_z   = float(np.mean(np.abs(qz - pz)))
        mae_cov = float(np.mean(np.abs(to_cov(qz) - to_cov(pz))))
        print(f"[emulator] SDK_QUANTIZED vs PyTorch over 64 windows: "
              f"z-MAE={mae_z:.4e}  reconstructed-cov-MAE={mae_cov:.4e}")
    except Exception as e:
        print(f"[emulator] sanity skipped ({type(e).__name__}: {e})")

    # ── step 3: compile -> HEF ────────────────────────────────────────────────
    hef = runner.compile()
    hef_path = _HERE / f"{stem}.hef"
    with open(hef_path, "wb") as f:
        f.write(hef)
    print(f"HEF: {hef_path}  ({hef_path.stat().st_size} bytes)")
    # compiled-model report (has the DFC static latency/FPS estimate)
    try:
        runner.save_har(str(_HERE / f"{stem}_compiled_model.har"))
    except Exception:
        pass
    print("DONE stream build.")


if __name__ == "__main__":
    main()
