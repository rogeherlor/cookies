"""
AI-IMU CausalMesNet -> ONNX converter (Hailo-compatible, FULL network)
======================================================================
Exports the WHOLE CausalMesNet noise-adapter (cov_net backbone + cov_lin head +
output scaling) to ONNX for Hailo DFC parsing.  Only the per-channel input
normalisation stays in Python; the device now outputs the final measurement
covariances directly.

Full on-device head (DFC ≥ 3.31 capabilities)
---------------------------------------------
Earlier this converter exported only the Conv1d backbone because the head used
ops the DFC was thought not to support.  The DFC user guide (and a parse test
against DFC 3.33.1) show they ARE supported, so the head moves on-device:

  • cov_lin `Linear(32→2)`      → 1x1 Conv2d(32→2)  (Matmul/Dense)
  • `Tanh`                      → supported activation
  • `cov0 · 10**(beta·z)`       → `cov0 · exp(ln10 · beta · z)`  (`Exp` is supported;
                                  variable-exponent `Pow` is not, so we use `exp`)

`exp(ln10·x) == 10**x` exactly, so this is a bit-faithful rewrite of the trained
model — no weights change.  The transpose/squeeze that previously blocked parsing
are avoided by keeping everything in 4-D NCHW (H=1) and using 1x1 conv for the
linear head.

Causal model (the definitive / deployable AI-IMU)
-------------------------------------------------
The definitive AI-IMU is the strictly CAUSAL model (`CausalMesNet`, trained by
`train_ai_imu.py --causal`, weights in `artifacts/deep_iekf_online/`).  Unlike
the acausal batch MesNet (symmetric ReplicationPad1d → each output depends on
~8 FUTURE samples), CausalMesNet left-pads BEFORE each conv, so output[i]
depends only on inputs ≤ i.  It is the only Hailo-deployable form and the one
this pipeline exports.  See `dl_filters/deep_iekf/iekf_ai_imu_online.py` and
`DEVIATIONS.md`.

Note — ReplicationPad → ZeroPad (pre-existing approximation): the Hailo Conv2d
converter maps the left `ReplicationPad1d` to `ZeroPad2d`.  Causality is
preserved exactly (padding is still left-only); only the first ~16 samples of a
sequence differ (zeros vs replicated first sample), identical to how the
acausal pipeline already handled padding.

Exported ONNX interface
-----------------------
Input:
    u_norm_conv  (1, 6, 1, N)  float32 — pre-normalized IMU in Conv2d format
                                          NCHW: N=1, C=6, H=1, W=seq_len
                                          Hailo NHWC: (1, 1, seq_len, 6)
Output:
    measurement_covs (1, 2, 1, N) float32 — final measurement covariances
                                             [cov_lat, cov_up] per timestep
                                             NCHW; Hailo NHWC: (1, 1, N, 2)

Why Conv2d instead of Conv1d?
Hailo's ONNX parser misidentifies the channel dimension in 3-D NCL tensors —
it reads the length axis as the channel count.  Exporting as 4-D NCHW (with
H=1) gives unambiguous shape information.

Runtime processing left in Python (saved to deep_iekf_postproc.npz)
-------------------------------------------------------------------
  Pre : u_norm = (u - u_loc) / u_std          # (N,6) per-channel affine
        u_norm_conv = u_norm.T[None, :, None, :]   # (1,6,1,N) NCHW
  Post: covs = measurement_covs[0, :, 0, :].T # (N, 2) — reshape only; the head
                                              #          now runs on-device

Usage
-----
    python 0_onnx_converter.py [--weights  artifacts/deep_iekf_online/fold_01.p]
                               [--out-dir  scripts/positioning/hailo/deep_iekf]
                               [--seq-len  4544]
                               [--opset    11]

The normalisation factors (u_loc, u_std) are auto-discovered from the sibling
`<stem>_norm.p` next to the weights (as saved by `train_ai_imu.py --causal`).
If `--weights` is omitted or missing, a single causal fold in
`artifacts/deep_iekf_online/` is discovered automatically.

Outputs:
    deep_iekf.onnx              — Hailo-parseable full CausalMesNet
    deep_iekf_postproc.npz      — input normalisation factors (u_loc, u_std)
"""

import argparse
import math
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
_ARTIFACTS        = _REPO_ROOT / "artifacts/deep_iekf"          # acausal (diagnostic)
_ARTIFACTS_ONLINE = _REPO_ROOT / "artifacts/deep_iekf_online"   # causal (definitive)

for _p in [str(_IEKF_DIR), str(_AI_IMU_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_SEQ_LEN  = 4544   # full KITTI sequence length
IMU_CHANNELS     = 6
COV_NET_FEATURES = 32     # cov_net output channels (internal)
COV_DIM          = 2      # measurement covariances [cov_lat, cov_up]


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


# ── Hailo-compatible wrapper (FULL CausalMesNet) ──────────────────────────────

class MesNetFullHailo(nn.Module):
    """
    Exports the WHOLE CausalMesNet noise adapter, in 4-D NCHW so the DFC parses
    it end-to-end (no transpose/squeeze, no variable-exponent Pow):

        cov_net   Conv2d backbone (converted from Conv1d)   → (1, 32, 1, N)
        cov_lin   Linear(32→2) as 1x1 Conv2d, then Tanh      → (1,  2, 1, N)
        scaling   cov0 · exp(ln10 · beta · z)  ( == cov0·10**(beta·z) )

    Input  : u_norm_conv      (1, 6, 1, N)  float32 — NCHW, H=1 dummy dim
    Output : measurement_covs (1, 2, 1, N)  float32 — NCHW, final covariances

    beta_measurement and cov0_measurement are baked in as constant buffers, so
    the device output is the final [cov_lat, cov_up] per timestep — the Python
    caller only reshapes it.  This is a bit-faithful rewrite of the trained
    CausalMesNet.forward (exp(ln10·x) == 10**x); no weights are modified.
    """

    def __init__(self, torch_iekf):
        super().__init__()
        mes = torch_iekf.mes_net
        # NB: _make_cov_net_2d copies weights into fresh float Conv2d layers, so
        # do NOT call mes.cov_net.float() here — that would mutate torch_iekf's
        # cov_net to float in place and corrupt subsequent double-precision use.
        self.cov_net = _make_cov_net_2d(mes.cov_net)

        # cov_lin is Sequential(Linear(32→2), Tanh); express the Linear as a
        # 1x1 Conv2d over the 32 feature channels (Hailo Matmul/Dense).
        lin = mes.cov_lin[0]                       # nn.Linear(32, 2)
        self.cov_lin_conv = nn.Conv2d(COV_NET_FEATURES, COV_DIM, kernel_size=1)
        self.cov_lin_conv.weight.data = lin.weight.data.float().view(COV_DIM, COV_NET_FEATURES, 1, 1)
        self.cov_lin_conv.bias.data   = lin.bias.data.float()
        self.tanh = nn.Tanh()

        # Output scaling constants as (1, 2, 1, 1) buffers for broadcasting.
        beta = mes.beta_measurement.float().reshape(1, COV_DIM, 1, 1)
        cov0 = torch_iekf.cov0_measurement.float().reshape(1, COV_DIM, 1, 1)
        self.register_buffer("beta", beta)
        self.register_buffer("cov0", cov0)
        self.register_buffer("ln10", torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(self, u_norm_conv: torch.Tensor) -> torch.Tensor:
        feat = self.cov_net(u_norm_conv)                 # (1, 32, 1, N)
        z    = self.tanh(self.cov_lin_conv(feat))        # (1,  2, 1, N)
        return self.cov0 * torch.exp(self.ln10 * self.beta * z)   # (1, 2, 1, N)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _finalize_onnx(onnx_path: Path, wrapped: nn.Module) -> None:
    """Simplify (onnx-simplifier) and verify the exported ONNX.

    onnxsim collapses the causal-pad + linear-head graph from ~43 to ~9 nodes,
    which is leaner for the DFC.  We use the PyTorch `wrapped` module as ground
    truth: the simplified graph is kept only if it matches `wrapped` on several
    random inputs; otherwise we fall back to the un-simplified export (which the
    DFC also parses).  Either way the file on disk is verified vs PyTorch.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[warning] onnx/onnxruntime not found — skipping simplify/verify.")
        return

    g0 = onnx.load(str(onnx_path)).graph
    name = g0.input[0].name
    dims = [d.dim_value for d in g0.input[0].type.tensor_type.shape.dim]

    # Ground-truth outputs from the PyTorch wrapper on several random inputs.
    rng = np.random.default_rng(0)
    xs  = [rng.standard_normal(dims).astype(np.float32) for _ in range(8)]
    with torch.no_grad():
        refs = [wrapped(torch.from_numpy(x)).numpy().astype(np.float64) for x in xs]

    def _max_err(model_bytes_or_path) -> float:
        sess = ort.InferenceSession(model_bytes_or_path)
        e = 0.0
        for x, ref in zip(xs, refs):
            out = sess.run(None, {name: x})[0].astype(np.float64)
            e = max(e, float(np.max(np.abs(out - ref))))
        return e

    # Candidate 1: onnxsim-simplified (preferred — fewer nodes).
    try:
        import onnxsim
        model_sim, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
        if ok and _max_err(model_sim.SerializeToString()) < 1e-3:
            onnx.save(model_sim, str(onnx_path))
            print(f"ONNX simplified + verified vs PyTorch "
                  f"(max|Δ|<1e-3, {len(model_sim.graph.node)} nodes).")
            return
    except ImportError:
        print("[info] onnxsim not installed — keeping un-simplified graph.")

    # Candidate 2: the un-simplified export.
    err = _max_err(str(onnx_path))
    tag = "verified" if err < 1e-3 else f"WARNING max|Δ|={err:.3e}"
    print(f"ONNX (un-simplified) vs PyTorch: {tag} ({len(g0.node)} nodes).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export AI-IMU MesNet cov_net backbone to ONNX (Hailo-compatible)"
    )
    parser.add_argument("--weights", type=Path,
                        default=_ARTIFACTS_ONLINE / "iekfnets.p",
                        help="Causal-trained weights (artifacts/deep_iekf_online/). "
                             "If missing, a single fold in that folder is auto-discovered.")
    parser.add_argument("--out-dir", type=Path, default=_HERE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help=f"Fixed sequence length (default: {DEFAULT_SEQ_LEN})")
    parser.add_argument("--opset",   type=int, default=11)
    args = parser.parse_args()

    # ── Resolve causal weights (mirror iekf_ai_imu_online._find_online_weights) ─
    from iekf_ai_imu_online import _find_online_weights
    weights_path = args.weights
    if not weights_path.exists():
        discovered = _find_online_weights()
        if discovered is None:
            raise FileNotFoundError(
                f"Causal weights not found: {weights_path}\n"
                f"and none discoverable in {_ARTIFACTS_ONLINE}.\n"
                "Train first: python dl_filters/deep_iekf/train_ai_imu.py --causal "
                "--mode loo --held-out <drive>"
            )
        weights_path = discovered

    # ── Build TORCHIEKF with a CausalMesNet (mirrors iekf_ai_imu_online) ───────
    from utils_torch_filter import TORCHIEKF
    from causal_mesnet import attach_causal_mesnet
    from iekf_ai_imu import _find_norm_factors
    # cov0_measurement (the base [cov_lat, cov_up]) is BAKED INTO the exported
    # ONNX/HEF here, so it MUST be the trained value. get_kitti_parameters()
    # guarantees the correct [1.0, 10.0] on every machine (see kitti_params.py);
    # the old 'except Exception -> bare TORCHIEKF() -> [0.2, 300.0]' fallback
    # silently baked the WRONG base covariance into the quantized model on any
    # host without navpy.
    from kitti_params import get_kitti_parameters
    torch_iekf = TORCHIEKF(get_kitti_parameters())
    if torch_iekf.cov0_measurement is None:
        torch_iekf.cov0_measurement = torch.tensor([1.0, 10.0]).double()

    attach_causal_mesnet(torch_iekf)                 # swap MesNet → CausalMesNet
    print(f"Loading CAUSAL weights: {weights_path}")
    mondict = torch.load(weights_path, map_location="cpu", weights_only=False)
    torch_iekf.load_state_dict(mondict)
    torch_iekf.eval()

    norm = _find_norm_factors(weights_path)
    if norm is None:
        raise FileNotFoundError(
            f"Normalisation factors (<stem>_norm.p) not found next to {weights_path}.\n"
            "On-device inference requires fixed u_loc/u_std; re-run "
            "train_ai_imu.py --causal to regenerate them."
        )
    torch_iekf.u_loc = norm["u_loc"].double()
    torch_iekf.u_std = norm["u_std"].double()

    # ── Save input-normalisation parameters ───────────────────────────────────
    # Only the per-channel input affine stays in Python; the cov_lin head and
    # output scaling now run on-device, so beta/cov0/cov_lin are no longer saved.
    args.out_dir.mkdir(parents=True, exist_ok=True)

    postproc = {
        "u_loc": torch_iekf.u_loc.numpy().astype(np.float32),   # (6,)
        "u_std": torch_iekf.u_std.numpy().astype(np.float32),   # (6,)
    }
    postproc_path = args.out_dir / "deep_iekf_postproc.npz"
    np.savez(str(postproc_path), **postproc)
    print(f"Input-normalisation params saved: {postproc_path}")

    # ── Build Hailo-compatible wrapper (full network) ─────────────────────────
    wrapped = MesNetFullHailo(torch_iekf)
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
            output_names=["measurement_covs"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _finalize_onnx(out_path, wrapped)

    print("Done.")
    print()
    print("ONNX interface (Hailo-parseable full CausalMesNet):")
    print(f"  Input : u_norm_conv=[1, {IMU_CHANNELS}, 1, {seq_len}]  "
          "NCHW — normalized IMU (H=1 dummy dim for Hailo)")
    print(f"  Output: measurement_covs=[1, {COV_DIM}, 1, {seq_len}]  "
          "NCHW — final covariances [cov_lat, cov_up]")
    print()
    print("Runtime pre-processing (Python, params in deep_iekf_postproc.npz):")
    print("  u_norm = (u - u_loc) / u_std               # (N,6)")
    print("  u_norm_conv = u_norm.T[None, :, None, :]   # (1,6,1,N) NCHW")
    print("Runtime post-processing (Python): reshape only —")
    print("  covs = measurement_covs[0, :, 0, :].T      # (N, 2)  (head runs on-device)")


if __name__ == "__main__":
    main()
