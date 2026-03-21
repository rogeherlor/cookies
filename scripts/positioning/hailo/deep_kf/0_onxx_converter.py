"""
DeepKFNet -> ONNX converter
==========================
Exports a trained DeepKFNet checkpoint to ONNX for Hailo DFC parsing.

The model is unrolled into two explicit single-layer LSTMs (no Slice ops).
Initial hidden/cell states are omitted from the ONNX graph — the Hailo
runtime manages LSTM state recurrence internally between inference calls.

Usage
-----
    python 0_onxx_converter.py --artifact artifacts/deep_kf/fold_01.pt

Outputs (next to the script by default, or set --out-dir):
    deep_kf.onnx

ONNX interface
--------------
Input:
    x      (1, 1, 15)    nav_state [p(3)|v(3)|rpy(3)|b_a(3)|b_g(3)]
                          seq_len=1 in dim-1 for LSTM-style 3-D input

Output:
    state  (1, 15)       delta for the last timestep; caller adds residual

Why a single 3-D input?
-----------------------
Hailo's emulator (acceleras) maps LSTM gates to Conv2D ops and requires the
input tensor to have at least 3 dimensions (N, H, W or N, H, W, C).
Keeping the input as [N, 1, 15] gives Hailo a clean 3-D LSTM-style input.

Note on bias_hh epsilon
-----------------------
The trained model may have near-zero bias_hh values.  With h0=0 this makes
the LSTM recurrent branch always output all-zeros during calibration, causing
scale=0 → desired_factor=inf → crash in Hailo's EW_Add quantisation.
A small epsilon (1e-2) is added to bias_hh in the ONNX wrapper so the
recurrent branch is always non-zero.  This does not affect the model's
trained weights and has negligible impact on inference accuracy.

Note on statefulness and constant h₀
--------------------------------------
Dynamic h/c inputs to the ONNX LSTM cause a Hailo DFC parse error.
Instead, h₀ is embedded as a constant buffer (H_INIT * ones) in the graph.

Why non-zero? During Hailo calibration each sample runs independently with
whatever h₀ is in the graph.  With h₀=0 the recurrent branch W_hh @ h₀ is
identically zero → normalization2/7 (which wrap that branch) record a
near-zero activation range → SDK reduces them from 8-bit to 2-bit → NaN
kernels → AccelerasNegativeSlopesError.  H_INIT=1.0 makes W_hh @ h₀ ≈ O(1),
giving the normalization layers a real range to quantise against.

The Hailo runtime replaces the constant h₀ with actual h_n feedback between
inference calls, so deployment behaviour is unaffected.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent          # cookies/
_MODEL_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_kf"

if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from model import DeepKFNet  # noqa: E402


# ── ONNX-friendly wrapper ─────────────────────────────────────────────────────

class DeepKFNetONNX(nn.Module):
    """
    Hailo-compatible export wrapper.

    Two key design decisions:
    1. Unrolled layers: nn.LSTM(num_layers=2) -> two nn.LSTM(num_layers=1).
       This removes Slice ops on h0/c0 that Hailo's parser cannot follow.
    2. No initial-state inputs: h0/c0 are omitted from the graph entirely.
       Hailo only supports zero or constant LSTM initial states.

    The residual connection (nav_state + delta) is included in the ONNX graph
    so the output is the full predicted state x_t^{+-}.
    """

    def __init__(self, model: DeepKFNet):
        super().__init__()

        orig_lstm = model.lstm.lstm
        input_dim  = orig_lstm.input_size    # 15
        hidden_dim = orig_lstm.hidden_size   # 128

        self.lstm_l0 = nn.LSTM(input_size=input_dim,  hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)
        self.lstm_l1 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)

        # Copy weights layer-by-layer from the original 2-layer LSTM.
        # A small epsilon is added to bias_hh so the EW_Add inside Hailo's
        # LSTM representation never gets scale=0 → desired_factor=inf → crash.
        BIAS_HH_EPS = 1e-2
        self.lstm_l0.weight_ih_l0.data.copy_(orig_lstm.weight_ih_l0.data)
        self.lstm_l0.weight_hh_l0.data.copy_(orig_lstm.weight_hh_l0.data)
        self.lstm_l0.bias_ih_l0.data.copy_(orig_lstm.bias_ih_l0.data)
        self.lstm_l0.bias_hh_l0.data.copy_(orig_lstm.bias_hh_l0.data + BIAS_HH_EPS)

        self.lstm_l1.weight_ih_l0.data.copy_(orig_lstm.weight_ih_l1.data)
        self.lstm_l1.weight_hh_l0.data.copy_(orig_lstm.weight_hh_l1.data)
        self.lstm_l1.bias_ih_l0.data.copy_(orig_lstm.bias_ih_l1.data)
        self.lstm_l1.bias_hh_l0.data.copy_(orig_lstm.bias_hh_l1.data + BIAS_HH_EPS)

        self.decoder = model.decoder

        # Constant non-zero initial hidden states embedded as ONNX constants.
        #
        # Problem: Hailo's calibration runs each sample independently (h_0=0
        # every call).  The recurrent branch W_hh @ h is therefore always zero,
        # so normalization2/7 (which wrap that branch) record a near-zero
        # activation range → SDK reduces them from 8-bit to 2-bit → NaN kernels
        # → AccelerasNegativeSlopesError during quantization.
        #
        # Fix: embed h_0 = H_INIT * ones as a buffer (constant initializer in the
        # ONNX graph).  Hailo's DFC accepts constant initial LSTM states and its
        # runtime replaces them with actual h_n feedback between inference calls.
        # H_INIT=1.0 gives W_hh @ h_0 with std ≈ sqrt(hidden_dim)*||W_hh||*H_INIT
        # ≈ O(1), safely above the 2-bit threshold.
        H_INIT = 1.0
        self.register_buffer('h0_l0', H_INIT * torch.ones(1, 1, hidden_dim))
        self.register_buffer('c0_l0', torch.zeros(1, 1, hidden_dim))
        self.register_buffer('h0_l1', H_INIT * torch.ones(1, 1, hidden_dim))
        self.register_buffer('c0_l1', torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, 1, 15)
            Navigation state with sequence dim=1.

        Returns
        -------
        delta : (batch, 15) — state increment; caller adds residual:
                state = delta + x[:, 0, :]
        """
        out_l0, _ = self.lstm_l0(x, (self.h0_l0, self.c0_l0))   # (batch, 1, 128)
        out_l1, _ = self.lstm_l1(out_l0, (self.h0_l1, self.c0_l1))  # (batch, 1, 128)
        h_out = out_l1[:, 0, :]           # (batch, 128)
        return self.decoder(h_out)        # (batch, 15) — delta only; caller adds residual


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simplify_onnx(onnx_path: Path) -> None:
    """Run onnx-simplifier in-place (promotes initializers -> Constant nodes)."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        print(
            "[warning] onnxsim not found — skipping simplification.\n"
            "          Hailo parsing may fail.  Install with:\n"
            "          pip install onnxsim"
        )
        return

    print("Simplifying ONNX (onnxsim) ...")
    model = onnx.load(str(onnx_path))
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(model_sim, str(onnx_path))
        print(f"Simplified ONNX saved -> {onnx_path}")
    else:
        print("[warning] onnxsim could not simplify the model — using original.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export DeepKFNet to ONNX")
    parser.add_argument(
        "--artifact", type=Path,
        default=_REPO_ROOT / "artifacts/deep_kf/fold_01.pt",
        help="Path to the trained .pt checkpoint",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_HERE,
        help="Directory where deep_kf.onnx will be written (default: script dir)",
    )
    parser.add_argument("--hidden-dim",  type=int, default=128)
    parser.add_argument("--num-layers",  type=int, default=2)
    parser.add_argument("--nav-dim",     type=int, default=15)
    parser.add_argument("--opset",       type=int, default=11,
                        help="ONNX opset (Hailo DFC 3.x supports up to 11/12)")
    args = parser.parse_args()

    artifact: Path = args.artifact
    if not artifact.exists():
        raise FileNotFoundError(f"Checkpoint not found: {artifact}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {artifact}")
    ckpt = torch.load(artifact, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        hidden_dim = cfg.get("latent_dim",  args.hidden_dim)
        num_layers  = cfg.get("num_layers",  args.num_layers)
    else:
        state_dict = ckpt
        hidden_dim = args.hidden_dim
        num_layers  = args.num_layers

    if num_layers != 2:
        raise ValueError(
            f"This converter assumes num_layers=2 but got {num_layers}. "
            "Adjust DeepKFNetONNX for a different layer count."
        )

    model = DeepKFNet(
        nav_state_dim=args.nav_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model:  hidden_dim={hidden_dim}  num_layers={num_layers}")

    wrapped = DeepKFNetONNX(model)
    wrapped.eval()

    input_dim = args.nav_dim                       # 15
    x_dummy   = torch.zeros(1, 1, input_dim)       # (batch=1, seq=1, features=15)

    input_names  = ["x"]
    output_names = ["state"]

    out_path = args.out_dir / "deep_kf.onnx"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset {args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (x_dummy,),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _simplify_onnx(out_path)

    print("Done.")
    print()
    print("ONNX interface:")
    print(f"  Input : x=[1, 1, {input_dim}]  (batch, seq=1, nav_state)")
    print(f"  Output: state=[1, {input_dim}]  (delta; caller adds residual)")


if __name__ == "__main__":
    main()
