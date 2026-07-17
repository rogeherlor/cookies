"""
Tartan IMU on-device HEF inference (hailort)
==============================================
Runs the compiled .hef on a physical Hailo-8 device via the hailort Python
bindings and compares the result against PyTorch ground-truth.

Usage
-----
    python 4_inference.py [--hef scripts/positioning/hailo/tartan_imu/tartan_imu.hef]
                          [--n-samples 4]

Architecture split
------------------
Hailo only runs the CNN backbone, one LSTM step at a time; the LSTM +
IMU_Trunk (transformer) + robot head run on the host from the weights saved
to tartan_imu_postproc.pt (see 0_onnx_converter.py / 2_optimisation.py).

Input interface (matches 0_onnx_converter.py)
---------------------------------------------
    imu_step   [1, 6, 1, 200]   one LSTM step (10 calls per window)

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
CNN_OUT_CH   = 128
CNN_OUT_T    = 13
CNN_FEAT_DIM = CNN_OUT_CH * CNN_OUT_T  # 1664
OUT_DIM      = 6   # vel(3) + log_std(3)
POSTPROC_PATH = _HERE / "tartan_imu_postproc.pt"
ROBOT_TYPE   = "car"


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_pytorch(model, windows_np):
    """Run model on (N, 10, 200, 6) array, return (N, 6)."""
    results = []
    with torch.no_grad():
        for i in range(len(windows_np)):
            x = torch.from_numpy(windows_np[i : i + 1]).float()
            results.append(model(x).numpy())
    return np.concatenate(results, axis=0)


def _step_to_nhwc(step_np):
    """(T, 200, 6) -> (T, 1, 200, 6) NHWC, one Hailo input per LSTM step."""
    step_nchw = step_np.transpose(0, 2, 1)[:, :, np.newaxis, :]  # (T, 6, 1, 200)
    return np.ascontiguousarray(step_nchw.transpose(0, 2, 3, 1))  # (T, 1, 200, 6)


def infer_hef(hef_path: Path, windows_np: np.ndarray, postproc: TartanPostproc) -> np.ndarray:
    """Run inference on a physical Hailo-8 device.

    Hailo only runs the CNN backbone, one LSTM step at a time; the
    LSTM + IMU_Trunk + robot head run on the host (see TartanPostproc).

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
        infer_model.input().set_format_type(hp.FormatType.FLOAT32)
        infer_model.output().set_format_type(hp.FormatType.FLOAT32)

        with infer_model.configure() as configured_model:
            configured_model.activate()
            input_name  = infer_model.input().name
            output_name = infer_model.output().name
            out_shape   = infer_model.output().shape  # (1, 13, 128) per step

            for i in range(n_samples):
                step_nhwc = _step_to_nhwc(windows_np[i])  # (10, 1, 200, 6)

                feats = []
                for t in range(LSTM_STEPS):
                    bindings = configured_model.create_bindings()
                    bindings.input(input_name).set_buffer(step_nhwc[t])

                    out_buf = np.empty(out_shape, dtype=np.float32)
                    bindings.output(output_name).set_buffer(out_buf)

                    configured_model.run([bindings], 1000)
                    # NHWC (1,13,128) -> (128,13) to match forward_cnn's channel-then-width flatten
                    feats.append(out_buf.squeeze(0).T.reshape(1, -1))

                feats_seq = np.concatenate(feats, axis=0)[np.newaxis]  # (1, 10, 1664)
                with torch.no_grad():
                    v, log_std = postproc(torch.from_numpy(feats_seq.astype(np.float32)))
                results.append(torch.cat([v, log_std], dim=-1).numpy())
            configured_model.deactivate()

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

    if not POSTPROC_PATH.exists():
        log.error("Postproc weights not found: %s — run 0_onnx_converter.py first.", POSTPROC_PATH)
        return
    postproc = TartanPostproc(POSTPROC_PATH, robot_type=ROBOT_TYPE)

    log.info("HEF inference on Hailo-8: %s", args.hef)
    hef_out = infer_hef(args.hef, test_np, postproc)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(pt_out, hef_out)


if __name__ == "__main__":
    main()
