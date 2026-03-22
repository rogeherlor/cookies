"""
AI-IMU MesNet on-device HEF inference (hailort)
=================================================
Runs the compiled .hef on a physical Hailo-8 device and compares output
against PyTorch ground-truth.

What runs on Hailo vs Python
-----------------------------
Hailo executes only the cov_net Conv1d backbone:
    Input : u_norm_conv (1, 6, N) — normalized IMU in Conv1d format
    Output: cov_features (1, 32, N) — raw CNN features

Python handles pre/postprocessing (params in deep_iekf_postproc.npz + deep_iekf_cov_lin.pt):
    Pre : u_norm = (u - u_loc) / u_std → (1, 6, 1, N) NCHW → NHWC for HEF
    Post: features = cov_features[0, :, 0, :].T    # (N, 32)  from NCHW
          z = cov_lin(features)                     # (N, 2) Sequential(Linear, Tanh)
          covs = cov0 * (10.0 ** (beta * z))        # (N, 2)

Usage
-----
    python 4_inference.py [--weights   artifacts/deep_iekf/iekfnets.p]
                          [--norm      artifacts/deep_iekf/iekfnets_norm.p]
                          [--hef       scripts/positioning/hailo/deep_iekf/deep_iekf.hef]
                          [--n-samples 8]

Profiler with runtime data
--------------------------
    hailortcli run2 -m raw measure-fw-actions \\
        --output-path runtime_data_deep_iekf.json \\
        set-net deep_iekf.hef

    hailo profiler deep_iekf_compiled_model.har \\
        --runtime-data runtime_data_deep_iekf.json \\
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
_IEKF_DIR   = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_iekf"
_AI_IMU_SRC = _REPO_ROOT / "external/ai-imu-dr/src"
_SCRIPTS    = _REPO_ROOT / "scripts/positioning/python"
_ARTIFACTS  = _REPO_ROOT / "artifacts/deep_iekf"

for _p in [str(_IEKF_DIR), str(_AI_IMU_SRC), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
POSTPROC_PATH = _HERE / "deep_iekf_postproc.npz"
COV_LIN_PATH  = _HERE / "deep_iekf_cov_lin.pt"
SEQ_LEN       = 4544
IMU_CHANNELS  = 6
COV_DIM       = 2


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("deep_iekf_inference")

log = init_logging()


# ── Pre/postprocessing (shared with 2_optimisation.py) ───────────────────────

def load_postproc(npz_path: Path, cov_lin_path: Path) -> dict:
    """Load pre/postprocessing parameters saved by 0_onnx_converter.py."""
    for p in [npz_path, cov_lin_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Postprocessing file not found: {p}\n"
                "Run 0_onnx_converter.py first."
            )
    d = np.load(str(npz_path))
    pp = {k: d[k] for k in d.files}
    pp["cov_lin"] = torch.load(str(cov_lin_path), map_location="cpu",
                               weights_only=False).float()
    pp["cov_lin"].eval()
    return pp


def preprocess(imu_np: np.ndarray, pp: dict) -> np.ndarray:
    """(N, 6) float32 → (1, 6, 1, N) float32 NCHW."""
    u_norm = (imu_np.astype(np.float32) - pp["u_loc"]) / pp["u_std"]
    return u_norm.T[np.newaxis, :, np.newaxis, :]  # (1, 6, 1, N)


def postprocess(cov_features: np.ndarray, pp: dict) -> np.ndarray:
    """Hailo/ONNX output → (N, 2) measurement covariances.

    Accepts NCHW (1, 32, 1, N) from onnxruntime or
    NHWC (1, 1, N, 32)/(1, N, 32) from HailoRT.
    """
    if cov_features.ndim == 4 and cov_features.shape[1] == 32:
        features = cov_features[0, :, 0, :].T   # NCHW → (N, 32)
    else:
        features = cov_features.reshape(-1, 32)  # NHWC → (N, 32)
    with torch.no_grad():
        z_cov = pp["cov_lin"](torch.from_numpy(features)).numpy()    # (N, 2)
    z_scaled = pp["beta_measurement"] * z_cov
    return pp["cov0_measurement"] * (10.0 ** z_scaled)               # (N, 2)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_pytorch(torch_iekf, imu_np):
    """Run float64 MesNet on (N, 6), return (N, 2) covs.
    MesNet.cov_net is Conv1d and expects (1, 6, N) NCL format."""
    u = torch.from_numpy(imu_np.astype(np.float64)).T.unsqueeze(0)  # (1, 6, N)
    with torch.no_grad():
        return torch_iekf.mes_net(u, torch_iekf).numpy()  # (N, 2)


def infer_hef(hef_path: Path, imu_np: np.ndarray, pp: dict) -> np.ndarray:
    """Run on Hailo-8 device, return (N, 2) covariances.

    The HEF processes the cov_net backbone only.  Pre/postprocessing is done
    in Python via the pp dict loaded from deep_iekf_postproc.npz.
    """
    try:
        import hailo_platform as hp
    except ImportError:
        raise ImportError(
            "hailo_platform not found.  Install the HailoRT Python wheel."
        )

    n = imu_np.shape[0]
    u_nchw = preprocess(imu_np, pp)               # (1, 6, 1, N) NCHW
    u_nhwc = u_nchw.transpose(0, 2, 3, 1)        # (1, 1, N, 6) NHWC for HEF

    params = hp.VDevice.create_params()
    with hp.VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(str(hef_path))
        infer_model.set_batch_size(1)

        with infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings()
            input_name  = infer_model.input().name
            output_name = infer_model.output().name

            out_info = infer_model.output()
            out_shape = out_info.shape             # Hailo NHWC shape
            out_buf = np.empty(out_shape, dtype=np.float32)
            bindings.input(input_name).set_buffer(u_nhwc)
            bindings.output(output_name).set_buffer(out_buf)
            configured_model.run([bindings], timeout_ms=5000)

    return postprocess(out_buf, pp)               # postprocess handles any NHWC shape


def _fmt(arr):
    return np.array2string(arr, precision=6, suppress_small=True, separator=", ")


def print_comparison(pt_covs, hef_covs, n_show=8):
    print("\n" + "=" * 72)
    print("MEASUREMENT COVARIANCES  [cov_lat, cov_up]")
    print("=" * 72)
    for i in range(n_show):
        print(f"\nTimestep {i}:")
        print(f"  PyTorch (f64)  {_fmt(pt_covs[i])}")
        print(f"  HEF            {_fmt(hef_covs[i])}")

    mae = float(np.mean(np.abs(
        hef_covs[:n_show] - pt_covs[:n_show].astype(np.float32)
    )))
    print("\n" + "-" * 72)
    print(f"MAE (HEF vs PyTorch): {mae:.6e}")
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   type=Path, default=_ARTIFACTS / "iekfnets.p")
    parser.add_argument("--norm",      type=Path, default=_ARTIFACTS / "iekfnets_norm.p")
    parser.add_argument("--hef",       type=Path, default=_HERE / "deep_iekf.hef")
    parser.add_argument("--n-samples", type=int,  default=8)
    args = parser.parse_args()

    # ── Load postprocessing params ────────────────────────────────────────────
    pp = load_postproc(POSTPROC_PATH, COV_LIN_PATH)

    # ── Test data — real KITTI IMU ────────────────────────────────────────────
    try:
        import data_loader as _dl
        _nav   = _dl.get_kitti_dataset("00")
        _gyro  = _nav.gyro_flu.astype(np.float32)
        _accel = _nav.accel_flu.astype(np.float32)
        test_imu = np.concatenate([_gyro, _accel], axis=1)[:SEQ_LEN]
        log.info("Test data: %d real KITTI IMU samples", len(test_imu))
    except Exception as _e:
        log.warning("KITTI unavailable (%s) — using synthetic data.", _e)
        rng = np.random.default_rng(42)
        test_imu = rng.standard_normal((SEQ_LEN, IMU_CHANNELS)).astype(np.float32)

    # ── PyTorch ground truth ──────────────────────────────────────────────────
    from utils_torch_filter import TORCHIEKF
    try:
        from main_kitti import KITTIParameters
        torch_iekf = TORCHIEKF(KITTIParameters)
    except Exception:
        torch_iekf = TORCHIEKF()

    mondict = torch.load(args.weights, map_location="cpu", weights_only=False)
    torch_iekf.load_state_dict(mondict)
    torch_iekf.eval()

    norm = torch.load(args.norm, map_location="cpu", weights_only=False)
    torch_iekf.u_loc = norm["u_loc"].double()
    torch_iekf.u_std = norm["u_std"].double()
    if torch_iekf.cov0_measurement is None:
        torch_iekf.set_param_attr()

    log.info("PyTorch inference ...")
    pt_covs = infer_pytorch(torch_iekf, test_imu)

    # ── HEF on-device inference ───────────────────────────────────────────────
    if not args.hef.exists():
        log.error("HEF not found: %s — run 3_compilation.py first.", args.hef)
        return

    log.info("HEF inference on Hailo-8: %s", args.hef)
    hef_covs = infer_hef(args.hef, test_imu, pp)

    print_comparison(pt_covs, hef_covs, n_show=args.n_samples)


if __name__ == "__main__":
    main()
