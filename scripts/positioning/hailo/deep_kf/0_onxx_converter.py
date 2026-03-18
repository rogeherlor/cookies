"""
DeepKFNet → ONNX converter
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
    x      (1, 1, 21)    [nav_state(15) | imu_mean(6)] — pre-concatenated,
                          seq_len=1 in dim-1 for LSTM-style 3-D input

Output:
    bias   (1,  6)       [δb_acc(3) | δb_gyr(3)] corrections in FLU

Why a single 3-D input?
-----------------------
Hailo's emulator (acceleras) maps LSTM gates to Conv2D ops and requires the
input tensor to have at least 3 dimensions (N, H, W or N, H, W, C).
When nav_state and imu_mean are passed as separate 2-D tensors [N, 15] and
[N, 6], the internal Concat + Unsqueeze ops that produce [N, 1, 21] are not
propagated correctly through the Hailo emulator, causing:
    ValueError: not enough values to unpack (expected 2, got 1)
in conv_stripped_op._compute_output_shape (input_shape[1:3] has 1 element).

Moving the concatenation OUTSIDE the model (caller concatenates nav_state
and imu_mean, adds seq dim → [N, 1, 21]) gives Hailo a clean 3-D input.

Note on statefulness
--------------------
The LSTM initial states (h0, c0) are intentionally omitted from the ONNX
graph inputs.  Hailo's DFC parser only supports zero or constant initial
states — dynamic h/c inputs cause a parse error.  The Hailo runtime handles
LSTM state recurrence (h_n → h0 feedback) internally across calls.
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
    1. Unrolled layers: nn.LSTM(num_layers=2) → two nn.LSTM(num_layers=1).
       This removes Slice ops on h0/c0 that Hailo's parser cannot follow.
    2. No initial-state inputs: h0/c0 are omitted from the graph entirely.
       Hailo only supports zero or constant LSTM initial states.  Dynamic
       initial states (graph inputs) cause 'ValueInfoProto has no attribute'
       in the parser.

    Weights are copied directly from the original checkpoint using PyTorch's
    weight_ih_l{i} / weight_hh_l{i} / bias_ih_l{i} / bias_hh_l{i} names.
    """

    def __init__(self, model: DeepKFNet):
        super().__init__()

        orig_lstm = model.lstm.lstm
        input_dim  = orig_lstm.input_size    # 21  (15 nav + 6 imu)
        hidden_dim = orig_lstm.hidden_size   # 128

        self.lstm_l0 = nn.LSTM(input_size=input_dim,  hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)
        self.lstm_l1 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
                               num_layers=1, batch_first=True)

        # Copy weights layer-by-layer from the original 2-layer LSTM
        self.lstm_l0.weight_ih_l0.data.copy_(orig_lstm.weight_ih_l0.data)
        self.lstm_l0.weight_hh_l0.data.copy_(orig_lstm.weight_hh_l0.data)
        self.lstm_l0.bias_ih_l0.data.copy_(orig_lstm.bias_ih_l0.data)
        self.lstm_l0.bias_hh_l0.data.copy_(orig_lstm.bias_hh_l0.data)

        self.lstm_l1.weight_ih_l0.data.copy_(orig_lstm.weight_ih_l1.data)
        self.lstm_l1.weight_hh_l0.data.copy_(orig_lstm.weight_hh_l1.data)
        self.lstm_l1.bias_ih_l0.data.copy_(orig_lstm.bias_ih_l1.data)
        self.lstm_l1.bias_hh_l0.data.copy_(orig_lstm.bias_hh_l1.data)

        self.decoder = model.decoder

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, 1, nav_dim + imu_dim)  e.g. (1, 1, 21)
            Pre-concatenated [nav_state | imu_mean] with sequence dim=1.
            Keeping the Concat + Unsqueeze OUTSIDE the ONNX graph gives
            Hailo's emulator a clean 3-D LSTM-style input and avoids the
            'not enough values to unpack' error in conv_stripped_op.

        Returns
        -------
        bias : (batch, 6)
        """
        out_l0, _ = self.lstm_l0(x)       # (batch, 1, 128)
        out_l1, _ = self.lstm_l1(out_l0)  # (batch, 1, 128)
        h_out = out_l1[:, 0, :]           # (batch, 128)
        return self.decoder(h_out)        # (batch, 6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simplify_onnx(onnx_path: Path) -> None:
    """Run onnx-simplifier in-place (promotes initializers → Constant nodes)."""
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
        print(f"Simplified ONNX saved → {onnx_path}")
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
    parser.add_argument("--imu-dim",     type=int, default=6)
    parser.add_argument("--opset",       type=int, default=11,
                        help="ONNX opset (Hailo DFC 3.x supports up to 11/12)")
    args = parser.parse_args()

    artifact: Path = args.artifact
    if not artifact.exists():
        raise FileNotFoundError(f"Checkpoint not found: {artifact}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
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
        imu_dim=args.imu_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model:  hidden_dim={hidden_dim}  num_layers={num_layers}")

    wrapped = DeepKFNetONNX(model)
    wrapped.eval()

    input_dim = args.nav_dim + args.imu_dim   # 21
    x_dummy   = torch.zeros(1, 1, input_dim)  # (batch=1, seq=1, features=21)

    input_names  = ["x"]
    output_names = ["bias"]

    out_path = args.out_dir / "deep_kf.onnx"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset {args.opset}) → {out_path}")
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
    print(f"  Input : x=[1, 1, {input_dim}]  (batch, seq=1, [nav_state | imu_mean])")
    print(f"  Output: bias=[1, 6]")


if __name__ == "__main__":
    main()
