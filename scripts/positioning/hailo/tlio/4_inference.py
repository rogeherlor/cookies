"""
TLIO on-device HEF inference (hailort)
========================================
Runs the compiled .hef on a physical Hailo-8 device via the hailort Python
bindings and compares the result against PyTorch ground-truth.

Usage
-----
    python 4_inference.py [--artifact artifacts/tlio/tlio_resnet.pt]
                          [--hef      scripts/positioning/hailo/tlio/tlio.hef]
                          [--n-samples 8]
                          [--window    200]

Input interface (matches 0_onnx_converter.py)
---------------------------------------------
    imu_window  [1, 6, 200]   gravity-aligned IMU window

Output
------
    disp_logstd [1, 6]   cat([mean_disp(3), logstd(3)])

Profiler with runtime data
--------------------------
    hailortcli run2 -m raw measure-fw-actions \\
        --output-path runtime_data_tlio.json \\
        set-net tlio.hef

    hailo profiler tlio_compiled_model.har \\
        --runtime-data runtime_data_tlio.json \\
        --out-path runtime_profiler.html
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

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
WINDOW_SIZE  = 200
IMU_CHANNELS = 6
OUT_DIM      = 3
GROUP_SIZES  = [2, 2, 2, 2]
OUT_COMBINED = OUT_DIM * 2  # 6


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("tlio_inference")

log = init_logging()


# ── PyTorch wrapper (same as 2_optimisation.py) ───────────────────────────────

class TLIOWrapperHailo(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):  # (1, 6, W)
        mean, logstd = self.net(x)
        return torch.cat([mean, logstd], dim=-1)  # (1, 6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_pytorch(model, windows_np):
    """Run PyTorch model on (N, 6, W) array, return (N, 6)."""
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
    windows_np : [N, 6, W] float32

    Returns
    -------
    out : [N, 6] float32  (cat([mean_disp, logstd]))
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
                sample = windows_np[i].astype(np.float32)  # [6, W]
                bindings.input(input_name).set_buffer(sample)

                out_buf = np.empty((1, OUT_COMBINED), dtype=np.float32)
                bindings.output(output_name).set_buffer(out_buf)

                configured_model.run([bindings], timeout_ms=1000)
                results.append(out_buf.copy())

    return np.concatenate(results, axis=0)  # [N, 6]


def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


OUTPUT_LABELS = "dp_x  dp_y  dp_z  logstd_x  logstd_y  logstd_z"


def print_comparison(pt_out: np.ndarray, hef_out: np.ndarray):
    print("\n" + "=" * 80)
    print(f"TLIO OUTPUTS  [{OUTPUT_LABELS}]")
    print("=" * 80)
    for i in range(pt_out.shape[0]):
        print(f"\nSample {i}:")
        print(f"  PyTorch  {_fmt(pt_out[i])}")
        print(f"  HEF      {_fmt(hef_out[i])}")

    mae = float(np.mean(np.abs(hef_out - pt_out)))
    print("\n" + "-" * 80)
    print(f"MAE (HEF vs PyTorch): {mae:.6e}")
    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact",  type=Path,
                        default=_REPO_ROOT / "artifacts/tlio/tlio_resnet.pt")
    parser.add_argument("--hef",       type=Path,
                        default=_HERE / "tlio.hef")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--window",    type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    # ── Test data — real KITTI gravity-aligned windows ────────────────────────
    try:
        import data_loader as _dl
        from tlio_dataset import build_windows

        _nav = _dl.get_kitti_dataset("00")
        _imu_windows, _, _, _ = build_windows(_nav, window_size=args.window, stride=50)
        idx = np.linspace(0, len(_imu_windows) - 1, args.n_samples, dtype=int)
        test_windows = _imu_windows[idx].astype(np.float32)
        log.info("Test data: %d real KITTI gravity-aligned windows", len(test_windows))
    except Exception as _e:
        log.warning("KITTI data unavailable (%s) — using synthetic data.", _e)
        rng = np.random.default_rng(42)
        test_windows = rng.standard_normal(
            (args.n_samples, IMU_CHANNELS, args.window)
        ).astype(np.float32)

    # ── PyTorch ground truth ──────────────────────────────────────────────────
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
    pt_out = infer_pytorch(wrapped, test_windows)

    # ── HEF on-device inference ───────────────────────────────────────────────
    if not args.hef.exists():
        log.error("HEF not found: %s — run 3_compilation.py first.", args.hef)
        return

    log.info("HEF inference on Hailo-8: %s", args.hef)
    hef_out = infer_hef(args.hef, test_windows)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(pt_out, hef_out)


if __name__ == "__main__":
    main()
