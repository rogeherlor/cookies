"""
DeepKFNet -> ONNX converter (STATEFUL cell export)
==================================================
Exports a trained DeepKFNet checkpoint to ONNX for Hailo DFC parsing, as an
explicit single-step LSTM CELL whose hidden and cell states are ordinary
graph inputs and outputs.

Why not the ONNX LSTM operator
------------------------------
The previous export used two nn.LSTM modules with h0/c0 baked in as constant
buffers, because the DFC rejects dynamic h/c inputs on the ONNX LSTM operator.
That produced a graph with a single input and a single output and therefore NO
WAY to carry state: h0 was re-applied on every inference call, so the deployed
model was memoryless while the trained model is a stateful two-layer LSTM.
Measured directly (feed the same input repeatedly, outputs were bit-identical,
and prior history had no effect), and the cost was large: on the outage
scenario, dropping recurrence alone moved sequence 01 from 82 m to 272 m of
ATE, before any quantisation.

The fix is to stop using the ONNX LSTM operator. The cell is written out as
primitive ops — Gemm, Add, Sigmoid, Tanh, Mul — with x, h and c as ordinary
tensor inputs, which the DFC compiles without complaint. The host then carries
(h, c) between ticks exactly as the CPU runner does, so the accelerated model
computes the same function as the trained one and the CPU-vs-Hailo difference
becomes a measurement of quantisation alone.

Two workarounds the old export needed are consequently GONE:
  * BIAS_HH_EPS (+1e-2 on bias_hh) — was needed because with h==0 the recurrent
    branch was identically zero during calibration, collapsing its quantisation
    range. With h a real input calibrated on real captured states, the branch
    carries a genuine distribution.
  * H_INIT=1.0 constant h0 — same root cause, same resolution.
Both perturbed the trained weights; neither is applied any more.

Usage
-----
    python 0_onxx_converter.py --artifact artifacts/deep_kf/fold_01.pt

Outputs (next to the script by default, or set --out-dir):
    deep_kf.onnx

ONNX interface
--------------
Inputs:
    x       (1, 1, 15)     nav_state [p(3)|v(3)|rpy(3)|b_a(3)|b_g(3)], normalised
    h_l0    (1, 1, 128)    layer-0 hidden state from the previous tick
    c_l0    (1, 1, 128)    layer-0 cell state
    h_l1    (1, 1, 128)    layer-1 hidden state
    c_l1    (1, 1, 128)    layer-1 cell state

Outputs:
    state   (1, 15)        delta for this timestep; caller adds the residual
    h_l0_o, c_l0_o, h_l1_o, c_l1_o   (1, 1, 128)  states to feed back next tick

At the start of a sequence the caller passes zeros, matching the CPU runner.
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
    """Single-step, STATEFUL LSTM cell wrapper for Hailo export.

    forward(x, h_l0, c_l0, h_l1, c_l1) -> (delta, h_l0', c_l0', h_l1', c_l1')

    The two LSTM layers are written out as primitive ops rather than nn.LSTM,
    so h/c are ordinary graph inputs the DFC accepts (the ONNX LSTM operator
    rejects dynamic initial states, which is what forced the old memoryless
    export). The arithmetic is the standard PyTorch cell, with PyTorch's gate
    ordering [i, f, g, o] in the packed weight rows:

        gates = [W_ih | W_hh] [x; h] + (b_ih + b_hh)
        c' = sigmoid(f) * c + sigmoid(i) * tanh(g)
        h' = sigmoid(o) * tanh(c')

    The trained weights are copied verbatim — no bias epsilon, no constant h0.
    """

    def __init__(self, model: DeepKFNet):
        super().__init__()
        orig = model.lstm.lstm
        self.input_dim  = orig.input_size     # 15
        self.hidden_dim = orig.hidden_size    # 128

        # FUSED input+recurrent projection, one Gemm per layer:
        #     W_ih x + b_ih + W_hh h + b_hh  ==  [W_ih | W_hh] [x; h] + (b_ih+b_hh)
        # Algebraically identical, but it removes the elementwise add between two
        # Gemm outputs. The Hailo allocator rejects that add outright — "Can't
        # find mutual format for fc2 -> ew_add1" — because it cannot reconcile
        # the internal layouts of two fully-connected results. Concatenating the
        # operands and projecting once sidesteps the problem instead of working
        # around it, and is cheaper on-device as well.
        def _fused(w_ih, b_ih, w_hh, b_hh):
            m = nn.Linear(w_ih.shape[1] + w_hh.shape[1], w_ih.shape[0], bias=True)
            m.weight.data.copy_(torch.cat([w_ih, w_hh], dim=1))
            m.bias.data.copy_(b_ih + b_hh)
            return m

        self.gate_l0 = _fused(orig.weight_ih_l0.data, orig.bias_ih_l0.data,
                              orig.weight_hh_l0.data, orig.bias_hh_l0.data)
        self.gate_l1 = _fused(orig.weight_ih_l1.data, orig.bias_ih_l1.data,
                              orig.weight_hh_l1.data, orig.bias_hh_l1.data)

        self.decoder = model.decoder

    @staticmethod
    def _cell(gates, c_prev, hidden_dim):
        i = torch.sigmoid(gates[:, 0 * hidden_dim:1 * hidden_dim])
        f = torch.sigmoid(gates[:, 1 * hidden_dim:2 * hidden_dim])
        g = torch.tanh(   gates[:, 2 * hidden_dim:3 * hidden_dim])
        o = torch.sigmoid(gates[:, 3 * hidden_dim:4 * hidden_dim])
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x, h_l0, c_l0, h_l1, c_l1):
        # (1,1,D) -> (1,D); the 3-D shape is kept at the interface because
        # Hailo's parser wants >=3 dims on inputs (see module docstring).
        x2, h0, c0 = x[:, 0, :], h_l0[:, 0, :], c_l0[:, 0, :]
        h1, c1     = h_l1[:, 0, :], c_l1[:, 0, :]

        g0 = self.gate_l0(torch.cat([x2, h0], dim=1))
        h0n, c0n = self._cell(g0, c0, self.hidden_dim)

        g1 = self.gate_l1(torch.cat([h0n, h1], dim=1))
        h1n, c1n = self._cell(g1, c1, self.hidden_dim)

        delta = self.decoder(h1n)               # (1, 15) — caller adds residual
        return (delta,
                h0n.unsqueeze(1), c0n.unsqueeze(1),
                h1n.unsqueeze(1), c1n.unsqueeze(1))


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
    # Zero initial states, exactly what the CPU runner starts a sequence with.
    h_dummy   = torch.zeros(1, 1, hidden_dim)
    dummies   = (x_dummy, h_dummy.clone(), h_dummy.clone(),
                 h_dummy.clone(), h_dummy.clone())

    input_names  = ["x", "h_l0", "c_l0", "h_l1", "c_l1"]
    output_names = ["state", "h_l0_o", "c_l0_o", "h_l1_o", "c_l1_o"]

    out_path = args.out_dir / "deep_kf.onnx"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset {args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummies,
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
    print("ONNX interface (STATEFUL cell — host carries h/c between ticks):")
    print(f"  Inputs : x=[1, 1, {input_dim}], "
          f"h_l0/c_l0/h_l1/c_l1=[1, 1, {hidden_dim}]")
    print(f"  Outputs: state=[1, {input_dim}] (delta; caller adds residual), "
          f"h_l0_o/c_l0_o/h_l1_o/c_l1_o=[1, 1, {hidden_dim}]")


if __name__ == "__main__":
    main()
