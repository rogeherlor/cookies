"""
AI-IMU MesNet CNN -> ONNX converter (Hailo-compatible)
=======================================================
Exports only the MesNet Conv1d backbone (cov_net) to ONNX for Hailo DFC
parsing.  The surrounding normalization, reshape, linear head, and output
scaling are done in Python at runtime.

Why partial export?
-------------------
The full MesNetWrapper forward pass contains three ops that Hailo DFC cannot
parse:

  1. Transpose (2-D .t())  — UnsupportedShuffleLayerError
  2. Squeeze               — UnexpectedNodeError
  3. Pow(10.0)             — only square (power=2) is supported

Hailo's own recommendation is to parse from /mesnet/Unsqueeze to
/mesnet/cov_net/cov_net.6/Relu — i.e. only the Conv1d backbone (cov_net).

Exported ONNX interface
-----------------------
Input:
    u_norm_conv  (1, 6, 1, 4544)  float32 — pre-normalized IMU in Conv2d format
                                             NCHW: N=1, C=6, H=1, W=4544
                                             Hailo NHWC: (1, 1, 4544, 6)
Output:
    cov_features (1, 32, 1, 4544) float32 — raw CNN features before linear head
                                             NCHW: N=1, C=32, H=1, W=4544

Why Conv2d instead of Conv1d?
Hailo's ONNX parser misidentifies the channel dimension in 3-D NCL tensors —
it reads the length axis (4544) as the channel count.  Exporting as 4-D NCHW
(with H=1) gives unambiguous shape information.

Runtime postprocessing (Python — saved to deep_iekf_postproc.npz + deep_iekf_cov_lin.pt)
-----------------------------------------------------------------
  features = cov_features[0, :, 0, :].T      # (N, 32) float32
  z_cov    = cov_lin(features)               # (N, 2)  Sequential(Linear+Tanh)
  z_scaled = beta_measurement * z_cov        # (N, 2)  beta_measurement shape: (2,)
  covs     = cov0_measurement * (10.0 ** z_scaled)  # (N, 2) final output

Usage
-----
    python 0_onnx_converter.py [--weights  artifacts/deep_iekf/iekfnets.p]
                               [--norm     artifacts/deep_iekf/iekfnets_norm.p]
                               [--out-dir  scripts/positioning/hailo/deep_iekf]
                               [--seq-len  4544]
                               [--opset    11]

Outputs:
    deep_iekf.onnx              — Hailo-parseable Conv1d backbone
    deep_iekf_postproc.npz      — pre/post-processing parameters for runtime
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_REPO_ROOT  = _HERE.parent.parent.parent.parent
_IEKF_DIR   = _REPO_ROOT / "scripts/positioning/python/dl_filters/deep_iekf"
_AI_IMU_SRC = _REPO_ROOT / "external/ai-imu-dr/src"
_ARTIFACTS  = _REPO_ROOT / "artifacts/deep_iekf"

for _p in [str(_IEKF_DIR), str(_AI_IMU_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_SEQ_LEN  = 4544   # full KITTI sequence length
IMU_CHANNELS     = 6
COV_NET_FEATURES = 32     # cov_net output channels


# ── Conv1d → Conv2d converter ─────────────────────────────────────────────────

def _make_cov_net_2d(cov_net_1d: nn.Sequential) -> nn.Sequential:
    """
    Convert Sequential(Conv1d / ReplicationPad1d / ...) to Conv2d equivalents.

    Hailo's ONNX parser cannot resolve channel dimensions from 3-D (N, C, L)
    Conv1d graphs; it misidentifies the length axis as the channel count.
    Providing 4-D (N, C, 1, L) Conv2d graphs resolves the ambiguity.

    Weight mapping : Conv1d (C_out, C_in, k) → Conv2d (C_out, C_in, 1, k)
    Padding mapping: ReplicationPad1d(p)      → ZeroPad2d(left=p, right=p)
    """
    layers = []
    for m in cov_net_1d:
        if isinstance(m, nn.Conv1d):
            new = nn.Conv2d(
                m.in_channels, m.out_channels,
                kernel_size=(1, m.kernel_size[0]),
                stride=(1, m.stride[0]),
                dilation=(1, m.dilation[0]),
                groups=m.groups,
                bias=m.bias is not None,
                padding=0,
            )
            new.weight.data = m.weight.data.unsqueeze(2).float()
            if m.bias is not None:
                new.bias.data = m.bias.data.float()
            layers.append(new)
        elif isinstance(m, nn.ReplicationPad1d):
            p = m.padding if isinstance(m.padding, (list, tuple)) else (m.padding, m.padding)
            layers.append(nn.ZeroPad2d((p[0], p[1], 0, 0)))
        else:
            layers.append(m)   # ReLU, Dropout — element-wise, work on any shape
    return nn.Sequential(*layers)


# ── Hailo-compatible wrapper (cov_net only) ───────────────────────────────────

class MesNetCovNetHailo(nn.Module):
    """
    Exports only the cov_net Conv2d backbone (converted from Conv1d).

    Hailo cannot parse Transpose, Squeeze, or Pow(10) — all present in the
    full MesNetWrapper.forward().  This wrapper takes pre-normalized IMU in
    4-D NCHW format (1, 6, 1, N) and returns raw CNN features (1, 32, 1, N).

    Input  : u_norm_conv  (1, 6, 1, N)  float32 — NCHW, H=1 dummy dim
    Output : cov_features (1, 32, 1, N) float32 — NCHW, raw features

    Caller applies postprocessing (stored in deep_iekf_postproc.npz + .pt):
        features = cov_features[0, :, 0, :].T      # (N, 32)
        z_cov    = cov_lin(features)               # (N, 2)
        covs     = cov0 * (10.0 ** (beta * z_cov)) # (N, 2)
    """

    def __init__(self, torch_iekf):
        super().__init__()
        self.cov_net = _make_cov_net_2d(torch_iekf.mes_net.cov_net.float())

    def forward(self, u_norm_conv: torch.Tensor) -> torch.Tensor:
        return self.cov_net(u_norm_conv)   # (1, 32, 1, N)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simplify_onnx(onnx_path: Path) -> None:
    """Run onnx-simplifier in-place."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        print(
            "[warning] onnxsim not found — skipping simplification.\n"
            "          Install with: pip install onnxsim"
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
    parser = argparse.ArgumentParser(
        description="Export AI-IMU MesNet cov_net backbone to ONNX (Hailo-compatible)"
    )
    parser.add_argument("--weights", type=Path,
                        default=_ARTIFACTS / "iekfnets.p")
    parser.add_argument("--norm",    type=Path,
                        default=_ARTIFACTS / "iekfnets_norm.p")
    parser.add_argument("--out-dir", type=Path, default=_HERE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Fixed sequence length (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--opset",   type=int, default=11)
    args = parser.parse_args()

    for f in [args.weights, args.norm]:
        if not f.exists():
            raise FileNotFoundError(
                f"File not found: {f}\n"
                "Train first: python dl_filters/deep_iekf/train_ai_imu.py"
            )

    # ── Load TORCHIEKF ────────────────────────────────────────────────────────
    from utils_torch_filter import TORCHIEKF
    try:
        from main_kitti import KITTIParameters
        torch_iekf = TORCHIEKF(KITTIParameters)
    except Exception:
        torch_iekf = TORCHIEKF()

    print(f"Loading weights: {args.weights}")
    mondict = torch.load(args.weights, map_location="cpu", weights_only=False)
    torch_iekf.load_state_dict(mondict)
    torch_iekf.eval()

    print(f"Loading normalization: {args.norm}")
    norm = torch.load(args.norm, map_location="cpu", weights_only=False)
    torch_iekf.u_loc = norm["u_loc"].double()
    torch_iekf.u_std = norm["u_std"].double()

    if torch_iekf.cov0_measurement is None:
        torch_iekf.set_param_attr()

    # ── Save postprocessing parameters ────────────────────────────────────────
    # These are applied in Python at runtime (cannot go through Hailo).
    # cov_lin is a Sequential (Linear + Tanh) — save the whole module so we
    # don't need to assume its internal structure.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    postproc = {
        "u_loc":            torch_iekf.u_loc.numpy().astype(np.float32),               # (6,)
        "u_std":            torch_iekf.u_std.numpy().astype(np.float32),               # (6,)
        "beta_measurement": torch_iekf.mes_net.beta_measurement.numpy().astype(np.float32),  # (2,)
        "cov0_measurement": torch_iekf.cov0_measurement.numpy().astype(np.float32),   # (2,)
    }
    postproc_path = args.out_dir / "deep_iekf_postproc.npz"
    np.savez(str(postproc_path), **postproc)
    print(f"Postprocessing params saved: {postproc_path}")

    # Save cov_lin module separately (Sequential — structure unknown at save time)
    cov_lin_path = args.out_dir / "deep_iekf_cov_lin.pt"
    torch.save(torch_iekf.mes_net.cov_lin, cov_lin_path)
    print(f"cov_lin module saved: {cov_lin_path}")

    # ── Build Hailo-compatible wrapper ────────────────────────────────────────
    wrapped = MesNetCovNetHailo(torch_iekf)
    wrapped.eval()

    seq_len  = args.seq_len
    dummy_in = torch.zeros(1, IMU_CHANNELS, 1, seq_len, dtype=torch.float32)

    out_path = args.out_dir / "deep_iekf.onnx"

    print(f"Exporting ONNX (opset {args.opset}, seq_len={seq_len}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy_in,),
            str(out_path),
            input_names=["u_norm_conv"],
            output_names=["cov_features"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _simplify_onnx(out_path)

    print("Done.")
    print()
    print("ONNX interface (Hailo-parseable cov_net backbone only):")
    print(f"  Input : u_norm_conv=[1, {IMU_CHANNELS}, 1, {seq_len}]  "
          "NCHW — normalized IMU (H=1 dummy dim for Hailo)")
    print(f"  Output: cov_features=[1, {COV_NET_FEATURES}, 1, {seq_len}]  "
          "NCHW — raw CNN features")
    print()
    print("Runtime pre-processing (Python):")
    print("  u_norm = (u - u_loc) / u_std  # (N,6)")
    print("  u_norm_conv = u_norm.T[None, :, None, :]   # (1,6,1,N) NCHW")
    print("Runtime post-processing (Python, params in deep_iekf_postproc.npz + .pt):")
    print("  features = cov_features[0, :, 0, :].T      # (N, 32)")
    print("  z = cov_lin(features)                      # (N, 2) Sequential(Linear+Tanh)")
    print("  covs = cov0 * (10.0 ** (beta * z))         # (N, 2)")


if __name__ == "__main__":
    main()
