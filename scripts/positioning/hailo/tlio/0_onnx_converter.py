"""
TLIO ResNet1D -> ONNX converter
================================
Exports a trained TLIO ResNet1D checkpoint to ONNX for Hailo DFC parsing.

The ResNet1D is a pure Conv1D architecture — the simplest of the four DL
filter networks to export.  No LSTM state tricks are needed.

The model has two output heads (mean + logstd).  Hailo requires a single
output tensor, so a thin wrapper concatenates them:
  TLIOWrapperHailo(x) -> cat([mean, logstd], dim=-1)  shape: (1, 6)

Usage
-----
    python 0_onnx_converter.py [--artifact artifacts/tlio/tlio_resnet.pt]
                               [--out-dir  scripts/positioning/hailo/tlio]
                               [--window   200]
                               [--opset    11]

Outputs (next to the script by default):
    tlio.onnx

ONNX interface
--------------
Input:
    imu_window  (1, 6, 200)   gravity-aligned IMU [gyro_ga(3) | accel_ga_motion(3)]
                               window_size=200 samples @ 100 Hz = 2 s
Output:
    disp_logstd (1, 6)        cat([mean_disp(3), logstd(3)], dim=-1)
                               in gravity-aligned frame; caller rotates to ENU
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/tlio/network"

if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from model_resnet import ResNet1D, BasicBlock1D  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 200          # IMU samples per window @ 100 Hz = 2 s
IMU_CHANNELS = 6           # [gyro_ga(3) | accel_ga_motion(3)]
OUT_DIM = 3                # displacement xyz
GROUP_SIZES = [2, 2, 2, 2]


# ── Hailo export wrapper ───────────────────────────────────────────────────────

class TLIOWrapperHailo(nn.Module):
    """
    Single-output wrapper for ResNet1D.

    ResNet1D has two head outputs (mean, logstd) which Hailo cannot represent
    as two separate output tensors.  This wrapper concatenates them so the ONNX
    graph has a single output node 'disp_logstd'.

    At inference time, split along the last axis:
        disp_logstd[:, :3]  -> displacement mean  [m] in gravity-aligned frame
        disp_logstd[:, 3:]  -> log-std             [log m]
    """

    def __init__(self, net: ResNet1D):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (1, 6, 200)  gravity-aligned IMU window (float32)

        Returns
        -------
        disp_logstd : (1, 6)  cat([mean(1,3), logstd(1,3)], dim=-1)
        """
        mean, logstd = self.net(x)
        return torch.cat([mean, logstd], dim=-1)  # (1, 6)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simplify_onnx(onnx_path: Path) -> None:
    """Run onnx-simplifier in-place."""
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
    parser = argparse.ArgumentParser(description="Export TLIO ResNet1D to ONNX")
    parser.add_argument(
        "--artifact", type=Path,
        default=_REPO_ROOT / "artifacts/tlio/tlio_resnet.pt",
        help="Path to the trained .pt checkpoint (default: artifacts/tlio/tlio_resnet.pt)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_HERE,
        help="Directory where tlio.onnx will be written (default: script dir)",
    )
    parser.add_argument("--window",  type=int, default=WINDOW_SIZE,
                        help="IMU window size in samples (default: 200)")
    parser.add_argument("--opset",   type=int, default=11,
                        help="ONNX opset (Hailo DFC 3.x supports up to 11/12)")
    args = parser.parse_args()

    artifact: Path = args.artifact
    if not artifact.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {artifact}\n"
            "Train first with: python dl_filters/tlio/train_tlio.py --mode all"
        )

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"Loading checkpoint: {artifact}")
    ckpt = torch.load(artifact, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    window = args.window
    inter_dim = window // 32 + 1   # temporal size after backbone (e.g. 200//32+1 = 7)

    net = ResNet1D(
        block_type=BasicBlock1D,
        in_dim=IMU_CHANNELS,
        out_dim=OUT_DIM,
        group_sizes=GROUP_SIZES,
        inter_dim=inter_dim,
    )
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print(f"Model: ResNet1D  window={window}  inter_dim={inter_dim}  "
          f"params={net.get_num_params():,}")

    wrapped = TLIOWrapperHailo(net)
    wrapped.eval()

    # ── Export ───────────────────────────────────────────────────────────────
    x_dummy = torch.zeros(1, IMU_CHANNELS, window)  # (batch=1, channels=6, window=200)

    out_path = args.out_dir / "tlio.onnx"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting ONNX (opset {args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (x_dummy,),
            str(out_path),
            input_names=["imu_window"],
            output_names=["disp_logstd"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _simplify_onnx(out_path)

    print("Done.")
    print()
    print("ONNX interface:")
    print(f"  Input : imu_window=[1, {IMU_CHANNELS}, {window}]  "
          "(batch, channels=[gyro_ga|accel_ga], window)")
    print(f"  Output: disp_logstd=[1, {OUT_DIM * 2}]  "
          "(cat([mean_disp, logstd], dim=-1))")


if __name__ == "__main__":
    main()
