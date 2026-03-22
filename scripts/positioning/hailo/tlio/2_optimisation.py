"""
TLIO HAR optimisation and cross-backend comparison
====================================================
Runs the same synthetic inputs through three backends and prints a comparison:

  1. PyTorch      — original .pt checkpoint (ground truth)
  2. SDK_NATIVE   — Hailo emulator, no changes (full precision)
  3. SDK_FP_OPT   — after optimize_full_precision()
  4. SDK_QUANTIZED— after quantization with calibration data

Note on calibration data
------------------------
Calibration uses real KITTI gravity-aligned IMU windows from tlio_dataset.py.
Synthetic random data gives unrepresentative activation ranges through the
BatchNorm layers, causing the Hailo SDK to over-compress them.  Real windows
have the correct dynamic range so the quantizer assigns sensible bit-widths.

Note on output
--------------
disp_logstd[:, :3] = mean displacement in gravity-aligned frame [m]
disp_logstd[:, 3:] = log-std [log m]

Usage
-----
    python 2_optimisation.py [--artifact artifacts/tlio/tlio_resnet.pt]
                             [--window   200]
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
_MODEL_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/tlio/network"
_SCRIPTS   = _REPO_ROOT / "scripts/positioning/python"

for _p in [str(_MODEL_DIR), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_resnet import ResNet1D, BasicBlock1D  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
ONNX_PATH          = _HERE / "tlio.onnx"
HAR_PATH           = _HERE / "tlio_hailo_model.har"
QUANTIZED_HAR_PATH = _HERE / "tlio_quantized_model.har"

WINDOW_SIZE  = 200
IMU_CHANNELS = 6
OUT_DIM      = 3
GROUP_SIZES  = [2, 2, 2, 2]
N_CALIB      = 1024   # Hailo recommends >= 1024 calibration samples
N_INFER      = 8      # samples used for comparison printout


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("tlio_optimisation")

log = init_logging()


# ── PyTorch wrapper (mirrors 0_onnx_converter.py) ─────────────────────────────

class TLIOWrapperHailo(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):  # (1, 6, W)
        mean, logstd = self.net(x)
        return torch.cat([mean, logstd], dim=-1)  # (1, 6)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_pytorch(model, windows_np):
    """Run PyTorch model on (N, 6, W) numpy array, return (N, 6)."""
    results = []
    with torch.no_grad():
        for i in range(len(windows_np)):
            x = torch.from_numpy(windows_np[i : i + 1]).float()
            results.append(model(x).numpy())
    return np.concatenate(results, axis=0)


def _hailo_input_names(runner):
    layers = runner._hn.get_input_layers()
    return [l.name for l in layers]


def infer_hailo(runner, ctx, windows_np):
    """Run Hailo emulator inference on (N, 6, W) array, return (N, 6)."""
    names = _hailo_input_names(runner)
    if len(names) != 1:
        raise RuntimeError(
            f"Expected 1 Hailo input layer, got {len(names)}: {names}"
        )
    # Hailo stores inputs in NHWC.  (1, 6, W) → add channel dim → (1, 6, W, 1)
    x = windows_np[:, :, :, np.newaxis]  # (N, 6, W, 1) NHWC
    results = runner.infer(ctx, {names[0]: x})
    return np.concatenate(results, axis=0)


# ── Comparison printout ───────────────────────────────────────────────────────

def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


OUTPUT_LABELS = "dp_x  dp_y  dp_z  logstd_x  logstd_y  logstd_z"


def print_comparison(backends: dict, reference: str = "PyTorch"):
    ref = backends[reference]
    print("\n" + "=" * 80)
    print(f"TLIO OUTPUTS  [{OUTPUT_LABELS}]")
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
    parser.add_argument("--artifact", type=Path,
                        default=_REPO_ROOT / "artifacts/tlio/tlio_resnet.pt")
    parser.add_argument("--window",   type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    # ── Calibration data — real KITTI gravity-aligned IMU windows ─────────────
    # Synthetic random data gives unrepresentative activation ranges through the
    # BatchNorm layers.  Real KITTI gravity-aligned windows (gyro_ga | accel_ga)
    # produce the correct dynamic range so the quantizer assigns proper bit-widths.
    try:
        import data_loader as _dl
        from tlio_dataset import build_windows

        _nav = _dl.get_kitti_dataset("00")
        _imu_windows, _, _, _ = build_windows(_nav, window_size=args.window, stride=50)
        # _imu_windows: (M, 6, W) float32
        n_avail = len(_imu_windows)
        idx = np.linspace(0, n_avail - 1, min(N_CALIB, n_avail), dtype=int)
        calib_windows = _imu_windows[idx]   # (N_CALIB, 6, W) float32
        log.info("Calibration: using %d real KITTI gravity-aligned windows", len(calib_windows))
    except Exception as _e_cal:
        log.warning(
            "KITTI calibration data unavailable (%s) — using synthetic data.", _e_cal
        )
        rng = np.random.default_rng(42)
        calib_windows = rng.standard_normal(
            (N_CALIB, IMU_CHANNELS, args.window)
        ).astype(np.float32)

    infer_windows = calib_windows[:N_INFER]

    backends = {}

    # ── 1. PyTorch ────────────────────────────────────────────────────────────
    log.info("Loading PyTorch checkpoint: %s", args.artifact)
    ckpt = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    inter_dim = args.window // 32 + 1
    net = ResNet1D(
        block_type=BasicBlock1D,
        in_dim=IMU_CHANNELS,
        out_dim=OUT_DIM,
        group_sizes=GROUP_SIZES,
        inter_dim=inter_dim,
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()

    wrapped = TLIOWrapperHailo(net)
    wrapped.eval()

    log.info("PyTorch inference ...")
    backends["PyTorch"] = infer_pytorch(wrapped, infer_windows)

    # ── 2. ONNX (optional) ────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        import onnxruntime as ort
        log.info("ONNX inference: %s", ONNX_PATH)
        session = ort.InferenceSession(str(ONNX_PATH))
        results = []
        for i in range(len(infer_windows)):
            out = session.run(None, {"imu_window": infer_windows[i : i + 1]})
            results.append(out[0])
        backends["ONNX"] = np.concatenate(results, axis=0)
    else:
        log.warning("ONNX not found (%s) — skipping.", ONNX_PATH)

    # ── 3-5. Hailo HAR ────────────────────────────────────────────────────────
    if not HAR_PATH.exists():
        log.warning("HAR not found (%s) — skipping Hailo stages.", HAR_PATH)
    else:
        runner = ClientRunner(har=str(HAR_PATH))
        hailo_names = _hailo_input_names(runner)
        log.info("Hailo input layer name(s): %s", hailo_names)
        # NHWC format: (N, 6, W, 1)
        calib_x = calib_windows[:, :, :, np.newaxis]
        calib_dataset = {hailo_names[0]: calib_x}

        # 3. SDK_NATIVE
        log.info("Hailo SDK_NATIVE inference ...")
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            backends["SDK_NATIVE"] = infer_hailo(runner, ctx, infer_windows)

        # 4. Full-precision optimization
        log.info("Running optimize_full_precision() ...")
        runner.optimize_full_precision(calib_dataset)

        log.info("Hailo SDK_FP_OPTIMIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            backends["SDK_FP_OPT"] = infer_hailo(runner, ctx, infer_windows)

        # 5. Quantization
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
                backends["SDK_QUANTIZED"] = infer_hailo(runner, ctx, infer_windows)
        except Exception as e:
            log.warning(
                "Quantization failed (%s: %s). "
                "SDK_NATIVE and SDK_FP_OPTIMIZED results are still valid.",
                type(e).__name__, e,
            )

    # ── Print results ─────────────────────────────────────────────────────────
    print_comparison(backends)


if __name__ == "__main__":
    main()
