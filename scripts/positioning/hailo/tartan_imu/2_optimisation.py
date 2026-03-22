"""
Tartan IMU HAR optimisation and cross-backend comparison
=========================================================
Runs the same KITTI IMU windows through three backends and prints a comparison:

  1. PyTorch      — original checkpoint (ground truth)
  2. SDK_NATIVE   — Hailo emulator, no changes (full precision)
  3. SDK_FP_OPT   — after optimize_full_precision()
  4. SDK_QUANTIZED— after quantization with calibration data

Note on calibration data
------------------------
Calibration uses real KITTI IMU data upsampled to 200 Hz and formatted as
10-step LSTM windows via tartan_dataset.build_lstm_input().  The Tartan
model is an LSTM that processes temporal velocity correlations — synthetic
random data has no temporal structure and gives unrepresentative activation
ranges through the LSTM gates.

Note on output
--------------
vel_logstd[:, :3] = predicted body-frame velocity [m/s]
vel_logstd[:, 3:] = log-std of velocity prediction

Usage
-----
    python 2_optimisation.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from hailo_sdk_client import ClientRunner, InferenceContext

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_REPO_ROOT  = _HERE.parent.parent.parent.parent
_TARTAN_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/tartan_imu"
_SCRIPTS    = _REPO_ROOT / "scripts/positioning/python"

for _p in [str(_TARTAN_DIR), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
ONNX_PATH          = _HERE / "tartan_imu.onnx"
HAR_PATH           = _HERE / "tartan_imu_hailo_model.har"
QUANTIZED_HAR_PATH = _HERE / "tartan_imu_quantized_model.har"

LSTM_STEPS   = 10
STEP_SAMPLES = 200
IMU_CHANNELS = 6
VEL_DIM      = 3
OUT_DIM      = VEL_DIM * 2   # 6 (vel + log_std)
N_CALIB      = 64    # windows; 64 × (10 × 200 × 6) ≈ 7.7 MB
N_INFER      = 4     # windows used for comparison printout


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("tartan_imu_optimisation")

log = init_logging()


# ── PyTorch wrapper ───────────────────────────────────────────────────────────

class TartanWrapperHailo(torch.nn.Module):
    def __init__(self, model, robot_type='car'):
        super().__init__()
        self.model = model
        self.robot_type = robot_type

    def forward(self, x):  # (1, 10, 200, 6)
        v, log_std = self.model(x, robot_type=self.robot_type)
        return torch.cat([v, log_std], dim=-1)  # (1, 6)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_pytorch(model, windows_np):
    """Run model on (N, 1, 10, 200, 6) or (N, 10, 200, 6) array, return (N, 6)."""
    results = []
    with torch.no_grad():
        for i in range(len(windows_np)):
            x = torch.from_numpy(windows_np[i : i + 1]).float()  # (1, 10, 200, 6)
            results.append(model(x).numpy())
    return np.concatenate(results, axis=0)


def _hailo_input_names(runner):
    layers = runner._hn.get_input_layers()
    return [l.name for l in layers]


def infer_hailo(runner, ctx, windows_np):
    """Run Hailo emulator inference on (N, 10, 200, 6) array, return (N, 6)."""
    names = _hailo_input_names(runner)
    if len(names) != 1:
        raise RuntimeError(f"Expected 1 Hailo input, got {len(names)}: {names}")
    # Input is already 4-D (1, 10, 200, 6) — matches NHWC format
    results = runner.infer(ctx, {names[0]: windows_np})
    return np.concatenate(results, axis=0)


# ── Comparison printout ───────────────────────────────────────────────────────

def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


OUTPUT_LABELS = "vx  vy  vz  logstd_x  logstd_y  logstd_z"


def print_comparison(backends: dict, reference: str = "PyTorch"):
    ref = backends[reference]
    print("\n" + "=" * 80)
    print(f"TARTAN IMU OUTPUTS  [{OUTPUT_LABELS}]")
    print("=" * 80)
    for i in range(ref.shape[0]):
        print(f"\nWindow {i}:")
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
    # ── Calibration data — real KITTI LSTM windows at 200 Hz ─────────────────
    # The Tartan LSTM processes temporal velocity correlations across 10 seconds
    # of IMU data.  Synthetic random inputs have no temporal structure → the
    # LSTM gates record arbitrary activation ranges → quantizer assigns wrong bit-
    # widths.  Real KITTI windows at 200 Hz give the correct dynamic range.
    try:
        import data_loader as _dl
        from tartan_dataset import (upsample_imu, build_lstm_input,
                                    LSTM_STEPS as _LS, STEP_SAMPLES as _SS,
                                    TARGET_HZ)

        _nav  = _dl.get_kitti_dataset("00")
        _accel_up, _gyro_up, _t_up = upsample_imu(
            _nav.accel_flu, _nav.gyro_flu,
            src_rate=float(_nav.sample_rate), tgt_rate=TARGET_HZ,
        )
        _t_src = np.arange(len(_nav.accel_flu)) / float(_nav.sample_rate)

        calib_windows = []
        stride_up = _SS  # 1 window per second
        min_idx = _LS * _SS
        indices = list(range(min_idx, len(_accel_up), stride_up))
        np.random.default_rng(42).shuffle(indices)
        for idx in indices[:N_CALIB]:
            win = build_lstm_input(
                _accel_up, _gyro_up, _nav.orient, _t_up, _t_src, idx
            )
            if win is not None:
                calib_windows.append(win[np.newaxis])  # (1, 10, 200, 6)
            if len(calib_windows) >= N_CALIB:
                break

        if calib_windows:
            calib_np = np.concatenate(calib_windows, axis=0)  # (N, 10, 200, 6)
            log.info("Calibration: using %d real KITTI LSTM windows", len(calib_np))
        else:
            raise ValueError("No calibration windows built from KITTI data.")
    except Exception as _e_cal:
        log.warning("KITTI calibration data unavailable (%s) — using synthetic data.", _e_cal)
        rng = np.random.default_rng(42)
        calib_np = rng.standard_normal(
            (N_CALIB, LSTM_STEPS, STEP_SAMPLES, IMU_CHANNELS)
        ).astype(np.float32)

    infer_np = calib_np[:N_INFER]

    backends = {}

    # ── 1. PyTorch ────────────────────────────────────────────────────────────
    from tartan_runner import _find_tartan_weights, _find_lora_adapter, _load_tartan_model, _TartanImuStub
    try:
        weights_path = _find_tartan_weights()
        lora_path = _find_lora_adapter()
        model, _ = _load_tartan_model(weights_path, lora_path, lora_rank=8)
    except RuntimeError as e:
        log.warning("Base model unavailable (%s). Using _TartanImuStub.", e)
        model = _TartanImuStub()
        model.eval()

    wrapped = TartanWrapperHailo(model)
    wrapped.eval()

    log.info("PyTorch inference ...")
    backends["PyTorch"] = infer_pytorch(wrapped, infer_np)

    # ── 2. ONNX (optional) ────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        import onnxruntime as ort
        log.info("ONNX inference: %s", ONNX_PATH)
        session = ort.InferenceSession(str(ONNX_PATH))
        results = []
        for i in range(len(infer_np)):
            out = session.run(None, {"imu_lstm": infer_np[i : i + 1]})
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
        # Input is (N, 10, 200, 6) — already 4-D NHWC
        calib_dataset = {hailo_names[0]: calib_np.astype(np.float32)}

        # 3. SDK_NATIVE
        log.info("Hailo SDK_NATIVE inference ...")
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            backends["SDK_NATIVE"] = infer_hailo(runner, ctx, infer_np)

        # 4. Full-precision optimization
        log.info("Running optimize_full_precision() ...")
        runner.optimize_full_precision(calib_dataset)

        log.info("Hailo SDK_FP_OPTIMIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            backends["SDK_FP_OPT"] = infer_hailo(runner, ctx, infer_np)

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
                backends["SDK_QUANTIZED"] = infer_hailo(runner, ctx, infer_np)
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
