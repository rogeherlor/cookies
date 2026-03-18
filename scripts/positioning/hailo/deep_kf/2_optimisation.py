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
The paper (DKF, Fig. 3/4, Eq. 21) outputs the full 15D error state x_t^{+-}.
This implementation is a deliberate simplification (see model.py "Differences
from the original paper", item 3): the decoder outputs only 6D IMU bias
corrections  [δb_acc(3) | δb_gyr(3)]  in the FLU body frame.

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
IMU_DIM    = 6
BIAS_DIM   = 6
N_CALIB    = 1024   # calibration samples (Hailo recommends ≥ 1024)
N_INFER    = 8      # samples used for comparison printout


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("deep_kf_optimisation")

log = init_logging()   # module-level so helpers can use it


# ── PyTorch wrapper (mirrors 0_onxx_converter.py, no hidden state I/O) ───────

class DeepKFNetONNX(torch.nn.Module):
    """Matches the exported ONNX: two unrolled single-layer LSTMs, no h0/c0.

    Accepts a single 3-D input x [batch, 1, 21] = pre-concatenated
    [nav_state | imu_mean].  The concatenation is done by the caller so
    Hailo's emulator sees a clean 3-D LSTM-style input.
    """

    def __init__(self, model: DeepKFNet):
        super().__init__()
        orig = model.lstm.lstm
        d_in  = orig.input_size   # 21
        d_hid = orig.hidden_size  # 128

        self.lstm_l0 = torch.nn.LSTM(d_in,  d_hid, num_layers=1, batch_first=True)
        self.lstm_l1 = torch.nn.LSTM(d_hid, d_hid, num_layers=1, batch_first=True)

        self.lstm_l0.weight_ih_l0.data.copy_(orig.weight_ih_l0)
        self.lstm_l0.weight_hh_l0.data.copy_(orig.weight_hh_l0)
        self.lstm_l0.bias_ih_l0.data.copy_(orig.bias_ih_l0)
        self.lstm_l0.bias_hh_l0.data.copy_(orig.bias_hh_l0)

        self.lstm_l1.weight_ih_l0.data.copy_(orig.weight_ih_l1)
        self.lstm_l1.weight_hh_l0.data.copy_(orig.weight_hh_l1)
        self.lstm_l1.bias_ih_l0.data.copy_(orig.bias_ih_l1)
        self.lstm_l1.bias_hh_l0.data.copy_(orig.bias_hh_l1)

        self.decoder = model.decoder

    def forward(self, x):   # x: [batch, 1, 21]
        out, _ = self.lstm_l0(x)
        out, _ = self.lstm_l1(out)
        return self.decoder(out[:, 0, :])


# ── Inference helpers ─────────────────────────────────────────────────────────

def _make_x(nav_np, imu_np):
    """Concatenate nav_state + imu_mean and add seq dim → [N, 1, 21]."""
    return np.concatenate([nav_np, imu_np], axis=-1)[:, np.newaxis, :]


def infer_pytorch(model, nav_np, imu_np):
    """Run PyTorch model on numpy arrays, return numpy bias [N, 6]."""
    x = _make_x(nav_np, imu_np)
    with torch.no_grad():
        return model(torch.from_numpy(x).float()).numpy()


def infer_onnx(session, nav_np, imu_np):
    """Run onnxruntime session, return numpy bias [N, 6]."""
    x = _make_x(nav_np, imu_np)
    results = []
    for i in range(len(x)):
        out = session.run(None, {"x": x[i : i + 1]})
        results.append(out[0])   # bias [1, 6]
    return np.concatenate(results, axis=0)


def _hailo_input_names(runner):
    """Return ordered list of actual Hailo input layer names from the HN.

    Hailo renames ONNX input nodes internally.  We query the HN directly so
    that our dataset dict uses the names Hailo actually assigned.
    """
    layers = runner._hn.get_input_layers()
    return [l.name for l in layers]


def infer_hailo(runner, ctx, nav_np, imu_np):
    """Run Hailo emulator inference, return numpy bias [N, 6].

    Hailo stores inputs in NHWC 4-D format internally.  Parsing with shape
    [1, 1, 21] causes Hailo to add a channel dim → [1, 1, 21, 1].  We must
    provide the same 4-D shape at inference time.
    """
    names = _hailo_input_names(runner)
    log.debug("Hailo input layer names: %s", names)
    if len(names) != 1:
        raise RuntimeError(
            f"Expected 1 Hailo input layer, got {len(names)}: {names}"
        )
    x = _make_x(nav_np, imu_np)[:, :, np.newaxis, :]  # [N, 1, 1, 21] NHWC
    results = runner.infer(ctx, {names[0]: x})
    return np.concatenate(results, axis=0)


# ── Comparison printout ───────────────────────────────────────────────────────

def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


def print_comparison(backends: dict[str, np.ndarray], reference: str = "PyTorch"):
    ref = backends[reference]
    print("\n" + "=" * 70)
    print("BIAS CORRECTIONS  [δb_acc_x  δb_acc_y  δb_acc_z  δb_gyr_x  δb_gyr_y  δb_gyr_z]")
    print("=" * 70)
    for i in range(ref.shape[0]):
        print(f"\nSample {i}:")
        for name, arr in backends.items():
            print(f"  {name:<14} {_fmt(arr[i])}")

    print("\n" + "-" * 70)
    print(f"MAE vs {reference}:")
    for name, arr in backends.items():
        if name == reference:
            continue
        mae = float(np.mean(np.abs(arr - ref)))
        print(f"  {name:<14} {mae:.6e}")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact",   type=Path,
                        default=_REPO_ROOT / "artifacts/deep_kf/fold_01.pt")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nav-dim",    type=int, default=NAV_DIM)
    parser.add_argument("--imu-dim",    type=int, default=IMU_DIM)
    args = parser.parse_args()

    # ── Synthetic data ────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    calib_nav = rng.standard_normal((N_CALIB, args.nav_dim)).astype(np.float32)
    calib_imu = rng.standard_normal((N_CALIB, args.imu_dim)).astype(np.float32)
    infer_nav = calib_nav[:N_INFER]
    infer_imu = calib_imu[:N_INFER]

    backends = {}

    # ── 1. PyTorch ────────────────────────────────────────────────────────────
    log.info("Loading PyTorch checkpoint: %s", args.artifact)
    ckpt = torch.load(args.artifact, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = cfg.get("latent_dim", args.hidden_dim)
    num_layers  = cfg.get("num_layers",  args.num_layers)

    pt_model = DeepKFNet(
        nav_state_dim=args.nav_dim, imu_dim=args.imu_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
    )
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    wrapped = DeepKFNetONNX(pt_model)
    wrapped.eval()

    log.info("PyTorch inference ...")
    backends["PyTorch"] = infer_pytorch(wrapped, infer_nav, infer_imu)

    # ── 2. ONNX ───────────────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        import onnxruntime as ort
        log.info("ONNX inference: %s", ONNX_PATH)
        session = ort.InferenceSession(str(ONNX_PATH))
        backends["ONNX"] = infer_onnx(session, infer_nav, infer_imu)
    else:
        log.warning("ONNX not found (%s) — skipping.", ONNX_PATH)

    # ── 3-5. Hailo HAR ────────────────────────────────────────────────────────
    if not HAR_PATH.exists():
        log.warning("HAR not found (%s) — skipping Hailo stages.", HAR_PATH)
    else:
        runner = ClientRunner(har=str(HAR_PATH))
        hailo_names = _hailo_input_names(runner)
        log.info("Hailo input layer name(s): %s", hailo_names)
        calib_x = _make_x(calib_nav, calib_imu)[:, :, np.newaxis, :]  # [N_CALIB, 1, 1, 21] NHWC
        calib_dataset = {hailo_names[0]: calib_x}

        # 3. SDK_NATIVE — no modifications
        log.info("Hailo SDK_NATIVE inference ...")
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            backends["SDK_NATIVE"] = infer_hailo(runner, ctx, infer_nav, infer_imu)

        # 4. Full-precision optimization → SDK_FP_OPTIMIZED
        log.info("Running optimize_full_precision() ...")
        runner.optimize_full_precision(calib_dataset)

        log.info("Hailo SDK_FP_OPTIMIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            backends["SDK_FP_OPT"] = infer_hailo(runner, ctx, infer_nav, infer_imu)

        # 5. Quantization → SDK_QUANTIZED
        log.info("Running optimize() (quantization) ...")
        runner.optimize(calib_dataset)
        runner.save_har(str(QUANTIZED_HAR_PATH))
        log.info("Quantized HAR saved: %s", QUANTIZED_HAR_PATH)

        log.info("Hailo SDK_QUANTIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
            backends["SDK_QUANTIZED"] = infer_hailo(runner, ctx, infer_nav, infer_imu)

    # ── Print results ─────────────────────────────────────────────────────────
    print_comparison(backends)


if __name__ == "__main__":
    main()
