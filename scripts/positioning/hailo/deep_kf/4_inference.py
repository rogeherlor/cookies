"""
DeepKFNet on-device HEF inference (hailort)
=============================================
Runs the compiled .hef on a physical Hailo-8 device via the hailort Python
bindings and compares the result against PyTorch ground-truth.

Usage
-----
    python 4_inference.py [--artifact artifacts/deep_kf/fold_01.pt]
                          [--hef      scripts/positioning/hailo/deep_kf/deep_kf.hef]
                          [--n-samples 8]

Input interface (matches 0_onxx_converter.py)
---------------------------------------------
    x  [1, 1, 15]   nav_state [p(3)|v(3)|rpy(3)|b_a(3)|b_g(3)]

Output
------
    state  [1, 15]   predicted state x_t^{+-}

Profiler with runtime data
--------------------------
To collect per-layer timing alongside inference, run hailortcli separately:

    hailortcli run2 -m raw measure-fw-actions \\
        --output-path runtime_data_deep_kf.json \\
        set-net deep_kf.hef

Then attach the JSON to the compiled HAR profiler:

    hailo profiler deep_kf_compiled_model.har \\
        --runtime-data runtime_data_deep_kf.json \\
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
_MODEL_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_kf"

if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from model import DeepKFNet  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
NAV_DIM   = 15
STATE_DIM = 15


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("deep_kf_inference")

log = init_logging()


# ── PyTorch wrapper (same as 2_optimisation.py) ───────────────────────────────

class DeepKFNetONNX(torch.nn.Module):
    """Unrolled 2-layer LSTM wrapper — matches the exported ONNX/HEF interface."""

    def __init__(self, model: DeepKFNet):
        super().__init__()
        orig = model.lstm.lstm
        d_in  = orig.input_size
        d_hid = orig.hidden_size

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

    def forward(self, x):           # x: [batch, 1, 15]
        out, _ = self.lstm_l0(x)
        out, _ = self.lstm_l1(out)
        h_out = out[:, 0, :]
        return self.decoder(h_out)  # delta only; caller adds residual


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_x(nav_np: np.ndarray) -> np.ndarray:
    """Add seq dim -> [N, 1, 15]."""
    return nav_np[:, np.newaxis, :]


def infer_pytorch(model, nav_np):
    x = _make_x(nav_np)
    with torch.no_grad():
        delta = model(torch.from_numpy(x).float()).numpy()
    return delta + nav_np   # x_t^{+-} = delta + x_{t-1}^+


def infer_hef(hef_path: Path, nav_np: np.ndarray) -> np.ndarray:
    """Run inference on a physical Hailo-8 device via hailort bindings.

    Parameters
    ----------
    hef_path : Path to the compiled .hef file
    nav_np   : [N, 15] float32

    Returns
    -------
    state : [N, 15] float32
    """
    try:
        import hailo_platform as hp
    except ImportError:
        raise ImportError(
            "hailo_platform not found.  Install the HailoRT Python wheel "
            "(hailort-*.whl) that matches your firmware version."
        )

    x = _make_x(nav_np)   # [N, 1, 15]
    n_samples = x.shape[0]

    params  = hp.VDevice.create_params()
    results = []

    with hp.VDevice(params) as vdevice:
        infer_model = vdevice.create_infer_model(str(hef_path))
        infer_model.set_batch_size(1)

        with infer_model.configure() as configured_model:
            bindings = configured_model.create_bindings()

            input_name  = infer_model.input().name
            output_name = infer_model.output().name

            for i in range(n_samples):
                sample = x[i].astype(np.float32)   # [1, 15]
                bindings.input(input_name).set_buffer(sample)

                out_buf = np.empty((1, STATE_DIM), dtype=np.float32)
                bindings.output(output_name).set_buffer(out_buf)

                configured_model.run([bindings], timeout_ms=1000)
                results.append(out_buf.copy())

    delta = np.concatenate(results, axis=0)   # [N, 15]
    return delta + nav_np                     # x_t^{+-} = delta + x_{t-1}^+


def _fmt(arr):
    return np.array2string(arr, precision=5, suppress_small=True, separator=", ")


STATE_LABELS = "p_e  p_n  p_u  v_e  v_n  v_u  roll  pitch  yaw  ba_x  ba_y  ba_z  bg_x  bg_y  bg_z"


def print_comparison(pt_state: np.ndarray, hef_state: np.ndarray):
    print("\n" + "=" * 80)
    print(f"STATE PREDICTIONS  [{STATE_LABELS}]")
    print("=" * 80)
    for i in range(pt_state.shape[0]):
        print(f"\nSample {i}:")
        print(f"  PyTorch  {_fmt(pt_state[i])}")
        print(f"  HEF      {_fmt(hef_state[i])}")

    mae = float(np.mean(np.abs(hef_state - pt_state)))
    print("\n" + "-" * 80)
    print(f"MAE (HEF vs PyTorch): {mae:.6e}")
    print("=" * 80 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact",  type=Path,
                        default=_REPO_ROOT / "artifacts/deep_kf/fold_01.pt")
    parser.add_argument("--hef",       type=Path,
                        default=_HERE / "deep_kf.hef")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nav-dim",    type=int, default=NAV_DIM)
    args = parser.parse_args()

    # ── Synthetic test data (same seed as 2_optimisation.py) ─────────────
    rng     = np.random.default_rng(42)
    nav_np  = rng.standard_normal((args.n_samples, args.nav_dim)).astype(np.float32)

    # ── PyTorch ground truth ──────────────────────────────────────────────
    log.info("Loading PyTorch checkpoint: %s", args.artifact)
    ckpt       = torch.load(args.artifact, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cfg        = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    hidden_dim = cfg.get("latent_dim", args.hidden_dim)
    num_layers  = cfg.get("num_layers",  args.num_layers)

    pt_model = DeepKFNet(
        nav_state_dim=args.nav_dim,
        hidden_dim=hidden_dim, num_layers=num_layers,
    )
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    wrapped = DeepKFNetONNX(pt_model)
    wrapped.eval()

    log.info("PyTorch inference ...")
    pt_state = infer_pytorch(wrapped, nav_np)

    # ── HEF on-device inference ───────────────────────────────────────────
    if not args.hef.exists():
        log.error("HEF not found: %s — run 3_compilation.py first.", args.hef)
        return

    log.info("HEF inference on Hailo-8: %s", args.hef)
    hef_state = infer_hef(args.hef, nav_np)

    # ── Print comparison ──────────────────────────────────────────────────
    print_comparison(pt_state, hef_state)


if __name__ == "__main__":
    main()
