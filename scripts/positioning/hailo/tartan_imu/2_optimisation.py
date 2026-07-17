"""
Tartan IMU HAR optimisation and cross-backend comparison
=========================================================
Runs the same KITTI IMU windows through three backends and prints a comparison:

  1. PyTorch      — original checkpoint (ground truth)
  2. SDK_NATIVE   — Hailo emulator, no changes (full precision)
  3. SDK_FP_OPT   — after optimize_full_precision()
  4. SDK_QUANTIZED— after quantization with calibration data

Architecture split
-------------------
Hailo only runs the CNN backbone, one LSTM step at a time (see
0_onnx_converter.py). The LSTM + IMU_Trunk (transformer) + robot head run on
the host from the weights saved to tartan_imu_postproc.pt. So each backend
here does: for each of the 10 steps in a window, run the CNN (PyTorch/ONNX/
Hailo) -> (1, 1664) feature; stack into (1, 10, 1664); run the host-side
LSTM/Trunk/head on that sequence -> (v_body, log_std).

Note on calibration data
------------------------
Calibration uses real KITTI IMU data upsampled to 200 Hz and formatted as
10-step LSTM windows via tartan_dataset.build_lstm_input(), then flattened to
individual CNN steps. The Tartan model is an LSTM that processes temporal
velocity correlations — synthetic random data has no temporal structure and
gives unrepresentative activation ranges through the LSTM gates.

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
HAR_PATH            = _HERE / "tartan_imu_hailo_model.har"
QUANTIZED_HAR_PATH  = _HERE / "tartan_imu_quantized_model.har"
POSTPROC_PATH       = _HERE / "tartan_imu_postproc.pt"

LSTM_STEPS   = 10
STEP_SAMPLES = 200
IMU_CHANNELS = 6
CNN_OUT_CH   = 128
CNN_OUT_T    = 13
CNN_FEAT_DIM = CNN_OUT_CH * CNN_OUT_T  # 1664
VEL_DIM      = 3
OUT_DIM      = VEL_DIM * 2   # 6 (vel + log_std)
N_CALIB      = 64    # windows; 64 x 10 steps = 640 CNN calibration samples
N_INFER      = 4     # windows used for comparison printout
ROBOT_TYPE   = "car"


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("tartan_imu_optimisation")

log = init_logging()


# ── PyTorch wrapper (whole model, for the ground-truth reference only) ────────

class TartanWrapperHailo(torch.nn.Module):
    def __init__(self, model, robot_type='car'):
        super().__init__()
        self.model = model
        self.robot_type = robot_type

    def forward(self, x):  # (1, 10, 200, 6)
        v, log_std = self.model(x, robot_type=self.robot_type)
        return torch.cat([v, log_std], dim=-1)  # (1, 6)


# ── Host-side postproc: LSTM + IMU_Trunk + robot head ─────────────────────────

class TartanPostproc(torch.nn.Module):
    """Runs after Hailo's per-step CNN backbone: LSTM -> IMU_Trunk -> robot head."""

    def __init__(self, postproc_path: Path, robot_type: str = ROBOT_TYPE):
        super().__init__()
        from tartan_runner import _IMUTrunk, _RobotHead

        pp = torch.load(postproc_path, map_location="cpu")
        self.lstm = torch.nn.LSTM(input_size=CNN_FEAT_DIM, hidden_size=64, batch_first=True)
        self.lstm.load_state_dict(pp["lstm_state"])
        self.trunk = _IMUTrunk()
        self.trunk.load_state_dict(pp["trunk_state"])
        heads = torch.nn.ModuleDict({r: _RobotHead() for r in ("car", "dog", "human", "drone")})
        heads.load_state_dict(pp["heads_state"])
        self.head = heads[robot_type]
        self.eval()

    def forward(self, feats_seq: torch.Tensor):
        """feats_seq: (B, T, 1664) per-step CNN features -> (v_body(B,3), log_std(B,3))."""
        lstm_out, _ = self.lstm(feats_seq)
        trunk_out   = self.trunk(lstm_out)
        feat        = trunk_out[:, -1, :]
        return self.head(feat)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_pytorch(model, windows_np):
    """Run the full PyTorch model on (N, 10, 200, 6) numpy array, return (N, 6)."""
    results = []
    with torch.no_grad():
        for i in range(len(windows_np)):
            x = torch.from_numpy(windows_np[i : i + 1]).float()
            results.append(model(x).numpy())
    return np.concatenate(results, axis=0)


def _step_to_nchw(step_np):
    """(T, 200, 6) -> (T, 6, 1, 200) NCHW, matching the ONNX graph's declared input."""
    return step_np.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (T, 6, 1, 200)


def _step_to_nhwc(step_np):
    """(T, 200, 6) -> (T, 1, 200, 6) NHWC, one Hailo input per LSTM step."""
    return _step_to_nchw(step_np).transpose(0, 2, 3, 1)  # (T, 1, 200, 6)


def _hailo_input_names(runner):
    layers = runner._hn.get_input_layers()
    return [l.name for l in layers]


def infer_onnx(session, windows_np, postproc: TartanPostproc):
    """windows_np: (N, 10, 200, 6) -> (N, 6) via per-step ONNX CNN + host LSTM/Trunk/head."""
    results = []
    for i in range(len(windows_np)):
        step_nchw = _step_to_nchw(windows_np[i])  # (10, 6, 1, 200)
        feats = []
        for t in range(LSTM_STEPS):
            out = session.run(None, {"imu_step": step_nchw[t : t + 1]})[0]  # (1,128,1,13)
            feats.append(out.reshape(1, -1))
        feats_seq = np.concatenate(feats, axis=0)[np.newaxis]  # (1, 10, 1664)
        with torch.no_grad():
            v, log_std = postproc(torch.from_numpy(feats_seq).float())
        results.append(torch.cat([v, log_std], dim=-1).numpy())
    return np.concatenate(results, axis=0)


def infer_hailo(runner, ctx, windows_np, postproc: TartanPostproc):
    """windows_np: (N, 10, 200, 6) -> (N, 6) via per-step Hailo CNN + host LSTM/Trunk/head."""
    names = _hailo_input_names(runner)
    if len(names) != 1:
        raise RuntimeError(f"Expected 1 Hailo input, got {len(names)}: {names}")
    results = []
    for i in range(len(windows_np)):
        step_nhwc = _step_to_nhwc(windows_np[i])  # (10, 1, 200, 6) — batch of 10 steps
        out = runner.infer(ctx, {names[0]: step_nhwc})  # (10, 1, 13, 128) NHWC, single output, batched
        out = np.asarray(out[0]) if isinstance(out, (list, tuple)) else np.asarray(out)
        # NHWC (T,1,13,128) -> NCW (T,128,13) to match forward_cnn's channel-then-width flatten
        out = out.squeeze(1).transpose(0, 2, 1)
        feats_seq = out.reshape(1, LSTM_STEPS, -1)          # (1, 10, 1664)
        with torch.no_grad():
            v, log_std = postproc(torch.from_numpy(feats_seq.astype(np.float32)))
        results.append(torch.cat([v, log_std], dim=-1).numpy())
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

    wrapped = TartanWrapperHailo(model, robot_type=ROBOT_TYPE)
    wrapped.eval()

    log.info("PyTorch inference ...")
    backends["PyTorch"] = infer_pytorch(wrapped, infer_np)

    if not POSTPROC_PATH.exists():
        log.warning("Postproc weights not found (%s) — skipping ONNX/Hailo stages.", POSTPROC_PATH)
        print_comparison(backends)
        return

    postproc = TartanPostproc(POSTPROC_PATH, robot_type=ROBOT_TYPE)

    # ── 2. ONNX (optional) ────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        import onnxruntime as ort
        log.info("ONNX inference: %s", ONNX_PATH)
        session = ort.InferenceSession(str(ONNX_PATH))
        backends["ONNX"] = infer_onnx(session, infer_np, postproc)
    else:
        log.warning("ONNX not found (%s) — skipping.", ONNX_PATH)

    # ── 3-5. Hailo HAR ────────────────────────────────────────────────────────
    if not HAR_PATH.exists():
        log.warning("HAR not found (%s) — skipping Hailo stages.", HAR_PATH)
    else:
        runner = ClientRunner(har=str(HAR_PATH))
        hailo_names = _hailo_input_names(runner)
        log.info("Hailo input layer name(s): %s", hailo_names)
        # Calibration: flatten (N,10,200,6) windows into individual CNN steps.
        calib_steps = calib_np.reshape(-1, STEP_SAMPLES, IMU_CHANNELS)  # (N*10, 200, 6)
        calib_dataset = {hailo_names[0]: _step_to_nhwc(calib_steps)}

        # 3. SDK_NATIVE
        log.info("Hailo SDK_NATIVE inference ...")
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            backends["SDK_NATIVE"] = infer_hailo(runner, ctx, infer_np, postproc)

        # 4. Full-precision optimization
        log.info("Running optimize_full_precision() ...")
        runner.optimize_full_precision(calib_dataset)

        log.info("Hailo SDK_FP_OPTIMIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            backends["SDK_FP_OPT"] = infer_hailo(runner, ctx, infer_np, postproc)

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
                backends["SDK_QUANTIZED"] = infer_hailo(runner, ctx, infer_np, postproc)
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
