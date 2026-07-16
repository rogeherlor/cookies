"""
AI-IMU CausalMesNet HAR optimisation and cross-backend comparison
=================================================================
Runs the same KITTI IMU sequence through multiple backends and compares:

  1. PyTorch      — causal checkpoint, artifacts/deep_iekf_online/ (float64 ref)
  2. SDK_NATIVE   — Hailo emulator, no changes (full precision)
  3. SDK_FP_OPT   — after optimize_full_precision()
  4. SDK_QUANTIZED— after quantization with real KITTI calibration data

What runs on Hailo vs Python
-----------------------------
Hailo executes the WHOLE CausalMesNet (Conv1d converted to 4-D Conv2d):
    Input : u_norm_conv (1, 6, 1, N) NCHW — normalized IMU with H=1 dummy dim
    Output: measurement_covs (1, 2, 1, N) NCHW — final [cov_lat, cov_up]

Python handles only input normalisation (params in deep_iekf_postproc.npz):
    Pre : u_norm = (u - u_loc) / u_std → (1, 6, 1, N) NCHW
    Post: covs = measurement_covs[0, :, 0, :].T    # (N, 2) — reshape only

Note on calibration data
------------------------
Uses real KITTI IMU (gyro_flu + accel_flu) pre-normalized via u_loc/u_std
and transposed to (1, 6, N) Conv1d format.  Synthetic data gives
unrepresentative activation ranges in the Conv1d layers.

Usage
-----
    python 2_optimisation.py [--weights  artifacts/deep_iekf_online/fold_01.p]

Normalisation factors (u_loc, u_std) are auto-discovered from the sibling
`<stem>_norm.p` next to the weights.
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
_IEKF_DIR   = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_iekf"
_AI_IMU_SRC = _REPO_ROOT / "external/ai-imu-dr/src"
_SCRIPTS    = _REPO_ROOT / "scripts/positioning/python"
_ARTIFACTS        = _REPO_ROOT / "artifacts/deep_iekf"          # acausal (diagnostic)
_ARTIFACTS_ONLINE = _REPO_ROOT / "artifacts/deep_iekf_online"   # causal (definitive)

for _p in [str(_IEKF_DIR), str(_AI_IMU_SRC), str(_SCRIPTS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
ONNX_PATH          = _HERE / "deep_iekf.onnx"
HAR_PATH           = _HERE / "deep_iekf_hailo_model.har"
QUANTIZED_HAR_PATH = _HERE / "deep_iekf_quantized_model.har"
POSTPROC_PATH      = _HERE / "deep_iekf_postproc.npz"

SEQ_LEN      = 4544
IMU_CHANNELS = 6
COV_DIM      = 2
N_INFER      = 8   # timesteps used for comparison printout


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("deep_iekf_optimisation")

log = init_logging()


# ── Pre/postprocessing ────────────────────────────────────────────────────────

def load_postproc(npz_path: Path):
    """Load the input-normalisation parameters saved by 0_onnx_converter.py."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Normalisation file not found: {npz_path}\nRun 0_onnx_converter.py first."
        )
    d = np.load(str(npz_path))
    return {k: d[k] for k in d.files}   # u_loc, u_std


def preprocess(imu_np: np.ndarray, pp: dict) -> np.ndarray:
    """(N, 6) float32 → normalized → (1, 6, 1, N) float32 NCHW for Hailo/ONNX."""
    u_norm = (imu_np.astype(np.float32) - pp["u_loc"]) / pp["u_std"]  # (N, 6)
    return u_norm.T[np.newaxis, :, np.newaxis, :]  # (1, 6, 1, N)


def postprocess(covs_dev: np.ndarray, pp: dict) -> np.ndarray:
    """Device output → (N, 2) measurement covariances (reshape only).

    The cov_lin head and output scaling now run on-device, so the model already
    emits the final [cov_lat, cov_up].  Accepts:
      NCHW (1, 2, 1, N)              — onnxruntime output
      NHWC (1, 1, N, 2) or (1, N, 2) — Hailo SDK emulator output
    """
    if covs_dev.ndim == 4 and covs_dev.shape[1] == COV_DIM:
        return covs_dev[0, :, 0, :].T        # NCHW → (N, 2)
    return covs_dev.reshape(-1, COV_DIM)     # NHWC (any shape) → (N, 2)


# ── Inference helpers ─────────────────────────────────────────────────────────

def infer_pytorch(torch_iekf, imu_np):
    """Run float64 MesNet on (N, 6), return (N, 2) covs.

    MesNet.cov_net is Conv1d and expects (1, 6, N) NCL format.
    """
    u = torch.from_numpy(imu_np.astype(np.float64)).T.unsqueeze(0)  # (1, 6, N)
    with torch.no_grad():
        covs = torch_iekf.mes_net(u, torch_iekf)
    return covs.numpy()  # (N, 2)


def infer_onnx(session, imu_np, pp):
    """Run onnxruntime on (N, 6) float32, return (N, 2)."""
    u_nchw = preprocess(imu_np, pp)             # (1, 6, 1, N)
    out = session.run(None, {"u_norm_conv": u_nchw})
    return postprocess(out[0], pp)              # out[0]: (1, 2, 1, N)


def _hailo_input_names(runner):
    return [l.name for l in runner._hn.get_input_layers()]


def infer_hailo(runner, ctx, imu_np, pp):
    """Run Hailo emulator on (N, 6) IMU, return (N, 2) covs.

    Model input  NCHW: (1, 6, 1, N) → Hailo NHWC: (1, 1, N, 6)
    Model output NHWC: (1, 1, N, 2) → (N, 2)  (head runs on-device)
    """
    names = _hailo_input_names(runner)
    if len(names) != 1:
        raise RuntimeError(f"Expected 1 Hailo input, got {len(names)}: {names}")
    u_nchw = preprocess(imu_np, pp)                    # (1, 6, 1, N)
    u_nhwc = u_nchw.transpose(0, 2, 3, 1)             # (1, 1, N, 6) NHWC
    results = runner.infer(ctx, {names[0]: u_nhwc})
    covs    = np.concatenate(results, axis=0)          # Hailo NHWC (..., 2)
    return postprocess(covs, pp)                       # (N, 2)


# ── Comparison printout ───────────────────────────────────────────────────────

def _fmt(arr):
    return np.array2string(arr, precision=6, suppress_small=True, separator=", ")


def print_comparison(backends: dict, reference: str = "PyTorch"):
    ref = backends[reference]
    print("\n" + "=" * 72)
    print("MEASUREMENT COVARIANCES  [cov_lat, cov_up]")
    print("=" * 72)
    for i in range(N_INFER):
        print(f"\nTimestep {i}:")
        for name, arr in backends.items():
            print(f"  {name:<14} {_fmt(arr[i])}")

    print("\n" + "-" * 72)
    print(f"MAE vs {reference}:")
    for name, arr in backends.items():
        if name == reference:
            continue
        mae = float(np.mean(np.abs(arr[:N_INFER] - ref[:N_INFER])))
        print(f"  {name:<14} {mae:.6e}")
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path,
                        default=_ARTIFACTS_ONLINE / "iekfnets.p",
                        help="Causal-trained weights (artifacts/deep_iekf_online/). "
                             "If missing, a single fold in that folder is auto-discovered.")
    args = parser.parse_args()

    # ── Load postprocessing parameters ────────────────────────────────────────
    pp = load_postproc(POSTPROC_PATH)
    log.info("Postprocessing params loaded: %s", POSTPROC_PATH.name)

    # ── Calibration data — real KITTI IMU ─────────────────────────────────────
    # Conv1d kernels need real temporal IMU correlations for proper activation
    # range estimation; synthetic random data gives flat, unrepresentative ranges.
    try:
        import data_loader as _dl
        _nav = _dl.get_kitti_dataset("00")
        _gyro  = _nav.gyro_flu.astype(np.float32)
        _accel = _nav.accel_flu.astype(np.float32)
        calib_imu = np.concatenate([_gyro, _accel], axis=1)  # (N, 6) [gyro|accel]
        if len(calib_imu) < SEQ_LEN:
            reps = int(np.ceil(SEQ_LEN / len(calib_imu)))
            calib_imu = np.tile(calib_imu, (reps, 1))[:SEQ_LEN]
        else:
            calib_imu = calib_imu[:SEQ_LEN]
        log.info("Calibration: %d real KITTI IMU samples", len(calib_imu))
    except Exception as _e_cal:
        log.warning("KITTI unavailable (%s) — using synthetic data.", _e_cal)
        rng = np.random.default_rng(42)
        calib_imu = rng.standard_normal((SEQ_LEN, IMU_CHANNELS)).astype(np.float32)

    backends = {}

    # ── 1. PyTorch (float64 reference — CAUSAL model) ─────────────────────────
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

    # Use the full calibration sequence — MesNet processes the entire sequence
    # at once; ONNX has a static input shape of SEQ_LEN samples.
    # print_comparison displays only the first N_INFER timesteps.
    log.info("PyTorch inference ...")
    backends["PyTorch"] = infer_pytorch(torch_iekf, calib_imu.astype(np.float64))

    # ── 2. ONNX ───────────────────────────────────────────────────────────────
    if ONNX_PATH.exists():
        import onnxruntime as ort
        log.info("ONNX inference: %s", ONNX_PATH)
        session = ort.InferenceSession(str(ONNX_PATH))
        backends["ONNX"] = infer_onnx(session, calib_imu, pp)
    else:
        log.warning("ONNX not found (%s) — skipping.", ONNX_PATH)

    # ── 3-5. Hailo HAR ────────────────────────────────────────────────────────
    if not HAR_PATH.exists():
        log.warning("HAR not found (%s) — skipping Hailo stages.", HAR_PATH)
    else:
        runner = ClientRunner(har=str(HAR_PATH))
        hailo_names = _hailo_input_names(runner)
        log.info("Hailo input layer name(s): %s", hailo_names)

        # Calibration input: (1, 1, N, 6) NHWC
        calib_nchw  = preprocess(calib_imu, pp)                # (1, 6, 1, N)
        calib_nhwc  = calib_nchw.transpose(0, 2, 3, 1)        # (1, 1, N, 6) NHWC
        calib_dataset = {hailo_names[0]: calib_nhwc}

        log.info("Hailo SDK_NATIVE inference ...")
        with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
            backends["SDK_NATIVE"] = infer_hailo(runner, ctx, calib_imu, pp)

        log.info("Running optimize_full_precision() ...")
        runner.optimize_full_precision(calib_dataset)

        log.info("Hailo SDK_FP_OPTIMIZED inference ...")
        with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
            backends["SDK_FP_OPT"] = infer_hailo(runner, ctx, calib_imu, pp)

        runner.load_model_script(
            "pre_quantization_optimization(dead_layers_removal, policy=disabled)\n"
        )
        log.info("Running optimize() (quantization) ...")
        try:
            runner.optimize(calib_dataset)
            runner.save_har(str(QUANTIZED_HAR_PATH))
            log.info("Quantized HAR saved: %s", QUANTIZED_HAR_PATH)

            with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
                backends["SDK_QUANTIZED"] = infer_hailo(runner, ctx, calib_imu, pp)
        except Exception as e:
            log.warning(
                "Quantization failed (%s: %s). "
                "SDK_NATIVE and SDK_FP_OPTIMIZED results are still valid.",
                type(e).__name__, e,
            )

    print_comparison(backends)


if __name__ == "__main__":
    main()
