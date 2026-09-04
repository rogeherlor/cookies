"""
AI-IMU CausalMesNet on-device HEF inference (hailort)
=====================================================
Runs the compiled .hef on a physical Hailo-8 device and compares output
against the causal PyTorch ground-truth (artifacts/deep_iekf_online/).

What runs on Hailo vs Python
-----------------------------
Hailo executes the WHOLE CausalMesNet (cov_net + cov_lin head + scaling):
    Input : u_norm_conv (1, 6, 1, N) NCHW — normalized IMU (H=1 dummy dim)
    Output: measurement_covs (1, 2, 1, N) NCHW — final [cov_lat, cov_up]

Python handles only input normalisation (params in deep_iekf_postproc.npz):
    Pre : u_norm = (u - u_loc) / u_std → (1, 6, 1, N) NCHW → NHWC for HEF
    Post: covs = measurement_covs[0, :, 0, :].T    # (N, 2) — reshape only

Usage
-----
    python 4_inference.py [--weights   artifacts/deep_iekf_online/fold_01.p]
                          [--hef       scripts/positioning/hailo/deep_iekf/deep_iekf.hef]
                          [--n-samples 8]

Normalisation factors (u_loc, u_std) are auto-discovered from the sibling
`<stem>_norm.p` next to the weights.

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
_ARTIFACTS        = _REPO_ROOT / "artifacts/deep_iekf"          # acausal (diagnostic)
_ARTIFACTS_ONLINE = _REPO_ROOT / "artifacts/deep_iekf_online"   # causal (definitive)

for _p in [str(_IEKF_DIR), str(_AI_IMU_SRC), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
POSTPROC_PATH = _HERE / "deep_iekf_postproc.npz"
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

def load_postproc(npz_path: Path) -> dict:
    """Load the input-normalisation parameters saved by 0_onnx_converter.py."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Normalisation file not found: {npz_path}\nRun 0_onnx_converter.py first."
        )
    d = np.load(str(npz_path))
    return {k: d[k] for k in d.files}   # u_loc, u_std


def preprocess(imu_np: np.ndarray, pp: dict) -> np.ndarray:
    """(N, 6) float32 → (1, 6, 1, N) float32 NCHW."""
    u_norm = (imu_np.astype(np.float32) - pp["u_loc"]) / pp["u_std"]
    return u_norm.T[np.newaxis, :, np.newaxis, :]  # (1, 6, 1, N)


def postprocess(covs_dev: np.ndarray, pp: dict) -> np.ndarray:
    """Device output → (N, 2) measurement covariances (reshape only).

    The cov_lin head and output scaling run on-device, so the model emits the
    final [cov_lat, cov_up].  Accepts NCHW (1, 2, 1, N) from onnxruntime or
    NHWC (1, 1, N, 2)/(1, N, 2) from HailoRT.
    """
    if covs_dev.ndim == 4 and covs_dev.shape[1] == COV_DIM:
        return covs_dev[0, :, 0, :].T        # NCHW → (N, 2)
    return covs_dev.reshape(-1, COV_DIM)     # NHWC (any shape) → (N, 2)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_pytorch(torch_iekf, imu_np, pp):
    """Run float64 MesNet on (N, 6), return (N, 2) covs.
    MesNet.cov_net is Conv1d and expects (1, 6, N) NCL format. mes_net has no
    internal normalisation — production code normalises before calling it
    (TORCHIEKF.forward_nets() in utils_torch_filter.py); must match that here
    with the same u_loc/u_std saved to deep_iekf_postproc.npz."""
    u_loc = pp["u_loc"].astype(np.float64)
    u_std = pp["u_std"].astype(np.float64)
    u_norm = (imu_np.astype(np.float64) - u_loc) / u_std
    u = torch.from_numpy(u_norm).T.unsqueeze(0)  # (1, 6, N)
    with torch.no_grad():
        return torch_iekf.mes_net(u, torch_iekf).numpy()  # (N, 2)


def infer_hef(hef_path: Path, imu_np: np.ndarray, pp: dict) -> np.ndarray:
    """Run on Hailo-8 device, return (N, 2) covariances.

    The HEF processes the whole CausalMesNet (backbone + head + scaling); only
    input normalisation is done in Python via the pp dict (deep_iekf_postproc.npz).
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
        infer_model.input().set_format_type(hp.FormatType.FLOAT32)
        infer_model.output().set_format_type(hp.FormatType.FLOAT32)

        with infer_model.configure() as configured_model:
            configured_model.activate()
            bindings = configured_model.create_bindings()
            input_name  = infer_model.input().name
            output_name = infer_model.output().name

            out_info = infer_model.output()
            out_shape = out_info.shape             # Hailo NHWC shape
            out_buf = np.empty(out_shape, dtype=np.float32)
            bindings.input(input_name).set_buffer(u_nhwc)
            bindings.output(output_name).set_buffer(out_buf)
            configured_model.run([bindings], 5000)
            configured_model.deactivate()

    return postprocess(out_buf, pp)               # postprocess handles any NHWC shape


def _fmt(arr):
    return np.array2string(arr, precision=6, suppress_small=True, separator=", ")


EDGE_SAMPLES = 20  # ReplicationPad->ZeroPad causal-padding approximation only
                    # affects the first ~16 samples (see 0_onnx_converter.py) —
                    # comparing there mixes in a known, unavoidable transient
                    # instead of measuring steady-state accuracy.


def print_comparison(pt_covs, hef_covs, n_show=8):
    print("\n" + "=" * 72)
    print("MEASUREMENT COVARIANCES  [cov_lat, cov_up]")
    print("=" * 72)
    print(f"(first {EDGE_SAMPLES} samples affected by the ReplicationPad->ZeroPad "
          "causal-padding approximation — shown separately below)")
    for i in range(EDGE_SAMPLES, EDGE_SAMPLES + n_show):
        print(f"\nTimestep {i}:")
        print(f"  PyTorch (f64)  {_fmt(pt_covs[i])}")
        print(f"  HEF            {_fmt(hef_covs[i])}")

    mae_edge = float(np.mean(np.abs(
        hef_covs[:EDGE_SAMPLES] - pt_covs[:EDGE_SAMPLES].astype(np.float32)
    )))
    mae_steady = float(np.mean(np.abs(
        hef_covs[EDGE_SAMPLES:] - pt_covs[EDGE_SAMPLES:].astype(np.float32)
    )))
    print("\n" + "-" * 72)
    print(f"MAE (HEF vs PyTorch), first {EDGE_SAMPLES} samples (padding transient): {mae_edge:.6e}")
    print(f"MAE (HEF vs PyTorch), steady-state (excl. first {EDGE_SAMPLES} samples): {mae_steady:.6e}")
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",   type=Path,
                        default=_ARTIFACTS_ONLINE / "iekfnets.p",
                        help="Causal-trained weights (artifacts/deep_iekf_online/). "
                             "If missing, a single fold in that folder is auto-discovered.")
    parser.add_argument("--hef",       type=Path, default=_HERE / "deep_iekf.hef")
    parser.add_argument("--n-samples", type=int,  default=8)
    args = parser.parse_args()

    # ── Load postprocessing params ────────────────────────────────────────────
    pp = load_postproc(POSTPROC_PATH)

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

    # ── PyTorch ground truth (CAUSAL model) ───────────────────────────────────
    from iekf_ai_imu_online import _find_online_weights
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    from utils_torch_filter import TORCHIEKF

    weights_path = args.weights
    if not weights_path.exists():
        weights_path = _find_online_weights()
        if weights_path is None:
            raise FileNotFoundError(
                f"Causal weights not found: {args.weights}\n"
                f"and none discoverable in {_ARTIFACTS_ONLINE}.\n"
                "Train first: python dl_filters/deep_iekf/train_ai_imu.py --causal "
                "--mode loo --held-out <drive>"
            )

    try:
        from main_kitti import KITTIParameters
        torch_iekf = TORCHIEKF(KITTIParameters)
    except Exception:
        torch_iekf = TORCHIEKF()
    if torch_iekf.cov0_measurement is None:
        torch_iekf.cov0_measurement = torch.tensor([0.2, 300.0]).double()

    attach_causal_mesnet(torch_iekf)                 # swap MesNet → CausalMesNet
    log.info("Loading CAUSAL weights: %s", weights_path)
    torch_iekf.load_state_dict(torch.load(weights_path, map_location="cpu",
                                          weights_only=False))
    torch_iekf.eval()

    norm = _find_norm_factors(weights_path)
    if norm is None:
        raise FileNotFoundError(
            f"Normalisation factors (<stem>_norm.p) not found next to {weights_path}."
        )
    torch_iekf.u_loc = norm["u_loc"].double()
    torch_iekf.u_std = norm["u_std"].double()

    log.info("PyTorch inference ...")
    pt_covs = infer_pytorch(torch_iekf, test_imu, pp)

    # ── HEF on-device inference ───────────────────────────────────────────────
    if not args.hef.exists():
        log.error("HEF not found: %s — run 3_compilation.py first.", args.hef)
        return

    log.info("HEF inference on Hailo-8: %s", args.hef)
    hef_covs = infer_hef(args.hef, test_imu, pp)

    print_comparison(pt_covs, hef_covs, n_show=args.n_samples)


if __name__ == "__main__":
    main()
