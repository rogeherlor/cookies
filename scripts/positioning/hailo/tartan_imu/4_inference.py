"""
Tartan IMU on-device HEF inference (hailort)
==============================================
Runs the compiled .hef on a physical Hailo-8 device via the hailort Python
bindings and compares the result against PyTorch ground-truth.

Usage
-----
    python 4_inference.py [--hef scripts/positioning/hailo/tartan_imu/tartan_imu.hef]
                          [--n-samples 4]

Input interface (matches 0_onnx_converter.py)
---------------------------------------------
    imu_lstm   [1, 10, 200, 6]   10 LSTM steps × 200 samples × [accel_gf|gyro]

Output
------
    vel_logstd [1, 6]   cat([v_body(3), log_std(3)])

Profiler with runtime data
--------------------------
    hailortcli run2 -m raw measure-fw-actions \\
        --output-path runtime_data_tartan_imu.json \\
        set-net tartan_imu.hef

    hailo profiler tartan_imu_compiled_model.har \\
        --runtime-data runtime_data_tartan_imu.json \\
        --out-path runtime_profiler.html
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_REPO_ROOT  = _HERE.parent.parent.parent.parent
_TARTAN_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/tartan_imu"
_SCRIPTS    = _REPO_ROOT / "scripts/positioning/python"

for _p in [str(_TARTAN_DIR), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
LSTM_STEPS   = 10
STEP_SAMPLES = 200
IMU_CHANNELS = 6
OUT_DIM      = 6   # vel(3) + log_std(3)


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("tartan_imu_inference")

log = init_logging()


# ── PyTorch wrapper ───────────────────────────────────────────────────────────

class TartanWrapperHailo(torch.nn.Module):
    def __init__(self, model, robot_type='car'):
        super().__init__()
        self.model = model
        self.robot_type = robot_type

    def forward(self, x):
        v, log_std = self.model(x, robot_type=self.robot_type)
        return torch.cat([v, log_std], dim=-1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_pytorch(model, windows_np):
    """Run model on (N, 10, 200, 6) array, return (N, 6)."""
    results = []
    with torch.no_grad():
        for i in range(len(windows_np)):
            x = torch.from_numpy(windows_np[i : i + 1]).float()
            results.append(model(x).numpy())
    return np.concatenate(results, axis=0)


def infer_hef(hef_path: Path, windows_np: np.ndarray) -> np.ndarray:
    """Run inference on a physical Hailo-8 device.

    Parameters
    ----------
    hef_path   : Path to the compiled .hef file
    windows_np : [N, 10, 200, 6] float32

    Returns
    -------
    out : [N, 6] float32
    """
    try:
        import hailo_platform as hp
    except ImportError:
        raise ImportError(
            "hailo_platform not found.  Install the HailoRT Python wheel "
            "(hailort-*.whl) that matches your firmware version."
        )

    n_samples = windows_np.shape[0]
    results = []

    params = hp.VDevice.create_params()
    with hp.VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(str(hef_path))
        infer_model.set_batch_size(1)

        with infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings()
            input_name  = infer_model.input().name
            output_name = infer_model.output().name

            for i in range(n_samples):
                sample = windows_np[i].astype(np.float32)  # [10, 200, 6]
                bindings.input(input_name).set_buffer(sample)

                out_buf = np.empty((1, OUT_DIM), dtype=np.float32)
                bindings.output(output_name).set_buffer(out_buf)

                configured_model.run([bindings], timeout_ms=1000)
                results.append(out_buf.copy())

    return np.concatenate(results, axis=0)  # [N, 6]


def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


OUTPUT_LABELS = "vx  vy  vz  logstd_x  logstd_y  logstd_z"


def print_comparison(pt_out: np.ndarray, hef_out: np.ndarray):
    print("\n" + "=" * 80)
    print(f"TARTAN IMU OUTPUTS  [{OUTPUT_LABELS}]")
    print("=" * 80)
    for i in range(pt_out.shape[0]):
        print(f"\nWindow {i}:")
        print(f"  PyTorch  {_fmt(pt_out[i])}")
        print(f"  HEF      {_fmt(hef_out[i])}")

    mae = float(np.mean(np.abs(hef_out - pt_out)))
    print("\n" + "-" * 80)
    print(f"MAE (HEF vs PyTorch): {mae:.6e}")
    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hef",       type=Path,
                        default=_HERE / "tartan_imu.hef")
    parser.add_argument("--n-samples", type=int, default=4)
    args = parser.parse_args()

    # ── Test data — real KITTI LSTM windows at 200 Hz ────────────────────────
    try:
        import data_loader as _dl
        from tartan_dataset import (upsample_imu, build_lstm_input,
                                    LSTM_STEPS as _LS, STEP_SAMPLES as _SS,
                                    TARGET_HZ)

        _nav = _dl.get_kitti_dataset("00")
        _accel_up, _gyro_up, _t_up = upsample_imu(
            _nav.accel_flu, _nav.gyro_flu,
            src_rate=float(_nav.sample_rate), tgt_rate=TARGET_HZ,
        )
        _t_src = np.arange(len(_nav.accel_flu)) / float(_nav.sample_rate)

        test_windows = []
        stride_up = _SS
        min_idx = _LS * _SS
        indices = list(range(min_idx, len(_accel_up), stride_up))
        for idx in indices[:args.n_samples]:
            win = build_lstm_input(_accel_up, _gyro_up, _nav.orient, _t_up, _t_src, idx)
            if win is not None:
                test_windows.append(win[np.newaxis])

        test_np = np.concatenate(test_windows, axis=0).astype(np.float32)
        log.info("Test data: %d real KITTI LSTM windows", len(test_np))
    except Exception as _e:
        log.warning("KITTI data unavailable (%s) — using synthetic data.", _e)
        rng = np.random.default_rng(42)
        test_np = rng.standard_normal(
            (args.n_samples, LSTM_STEPS, STEP_SAMPLES, IMU_CHANNELS)
        ).astype(np.float32)

    # ── PyTorch ground truth ──────────────────────────────────────────────────
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
    pt_out = infer_pytorch(wrapped, test_np)

    # ── HEF on-device inference ───────────────────────────────────────────────
    if not args.hef.exists():
        log.error("HEF not found: %s — run 3_compilation.py first.", args.hef)
        return

    log.info("HEF inference on Hailo-8: %s", args.hef)
    hef_out = infer_hef(args.hef, test_np)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(pt_out, hef_out)


if __name__ == "__main__":
    main()
