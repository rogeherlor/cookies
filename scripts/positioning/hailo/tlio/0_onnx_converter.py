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

from model_resnet import ResNet1D, BasicBlock1D, FcBlock  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 200          # IMU samples per window @ 100 Hz = 2 s
IMU_CHANNELS = 6           # [gyro_ga(3) | accel_ga_motion(3)]
OUT_DIM = 3                # displacement xyz
GROUP_SIZES = [2, 2, 2, 2]


# ── Conv1D → Conv2D conversion helpers ────────────────────────────────────────
#
# Hailo's ONNX parser misidentifies the length axis of 3D NCL (Conv1d) tensors
# as the channel count, so every Conv1d/BatchNorm1d/MaxPool1d layer is
# re-expressed as its 2D counterpart with a dummy H=1 dimension (NCHW).  Same
# technique already used for the tartan_imu backbone export.

def _conv1d_to_conv2d(m: nn.Conv1d) -> nn.Conv2d:
    pad = m.padding[0] if isinstance(m.padding, tuple) else m.padding
    new = nn.Conv2d(
        m.in_channels, m.out_channels,
        kernel_size=(1, m.kernel_size[0]),
        stride=(1, m.stride[0]),
        dilation=(1, m.dilation[0]),
        groups=m.groups,
        bias=m.bias is not None,
        padding=(0, pad),
    )
    new.weight.data = m.weight.data.unsqueeze(2).float()
    if m.bias is not None:
        new.bias.data = m.bias.data.float()
    return new


def _bn1d_to_bn2d(m: nn.BatchNorm1d) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine)
    if m.affine:
        new.weight.data = m.weight.data.float()
        new.bias.data   = m.bias.data.float()
    new.running_mean.data = m.running_mean.data.float()
    new.running_var.data  = m.running_var.data.float()
    return new


def _maxpool1d_to_maxpool2d(m: nn.MaxPool1d) -> nn.MaxPool2d:
    ks = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
    st = m.stride[0] if isinstance(m.stride, tuple) else m.stride
    pd = m.padding[0] if isinstance(m.padding, tuple) else m.padding
    return nn.MaxPool2d(kernel_size=(1, ks), stride=(1, st), padding=(0, pd))


class _BasicBlock2D(nn.Module):
    """Conv2d version of BasicBlock1D for Hailo export."""

    def __init__(self, block1d: BasicBlock1D):
        super().__init__()
        self.conv1 = _conv1d_to_conv2d(block1d.conv1)
        self.bn1   = _bn1d_to_bn2d(block1d.bn1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = _conv1d_to_conv2d(block1d.conv2)
        self.bn2   = _bn1d_to_bn2d(block1d.bn2)
        if block1d.downsample is not None:
            ds_conv, ds_bn = block1d.downsample
            self.downsample = nn.Sequential(_conv1d_to_conv2d(ds_conv), _bn1d_to_bn2d(ds_bn))
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class _FcBlock2D(nn.Module):
    """
    Conv2d version of FcBlock for Hailo export.

    Hailo's DFC cannot parse the Flatten->FC head (UnsupportedShuffleLayerError)
    and recommends ending the on-device graph at the prep1 1x1-conv output. So
    only `prep1` runs on Hailo; `forward_post_prep` (bn1 + flatten + fc1/2/3)
    runs on the host from the weights saved to tlio_postproc.pt — same
    CNN/host split pattern as tartan_imu's LSTM head.
    """

    def __init__(self, fc1d: FcBlock):
        super().__init__()
        self.prep1 = _conv1d_to_conv2d(fc1d.prep1)
        self.bn1   = _bn1d_to_bn2d(fc1d.bn1)
        self.relu  = nn.ReLU(inplace=True)
        self.fc1   = fc1d.fc1
        self.fc2   = fc1d.fc2
        self.fc3   = fc1d.fc3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep1(x)                                  # (B, prep_channel, 1, inter_dim) — Hailo ends here
        return self.forward_post_prep(x)

    def forward_post_prep(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = torch.flatten(x, 1)                            # (B, prep_channel * inter_dim)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))                        # dropout is a no-op in eval()
        return self.fc3(x)

    def post_prep_state(self) -> dict:
        """Host-side (non-Hailo) weights: bn1 + fc1 + fc2 + fc3."""
        return {
            "bn1.weight":        self.bn1.weight.detach().clone(),
            "bn1.bias":          self.bn1.bias.detach().clone(),
            "bn1.running_mean":  self.bn1.running_mean.detach().clone(),
            "bn1.running_var":   self.bn1.running_var.detach().clone(),
            "bn1.eps":           self.bn1.eps,
            "fc1.weight": self.fc1.weight.detach().clone(), "fc1.bias": self.fc1.bias.detach().clone(),
            "fc2.weight": self.fc2.weight.detach().clone(), "fc2.bias": self.fc2.bias.detach().clone(),
            "fc3.weight": self.fc3.weight.detach().clone(), "fc3.bias": self.fc3.bias.detach().clone(),
        }


class ResNet1DHailo(nn.Module):
    """
    Conv2D mirror of ResNet1D — identical weights, NCHW (dummy H=1) shapes.

    Input : imu_window (1, 6, 1, W)  NCHW
    Output: (mean, logstd) each (1, 3)
    """

    def __init__(self, net: ResNet1D):
        super().__init__()
        ib_conv, ib_bn, _relu, ib_pool = net.input_block
        self.input_block = nn.Sequential(
            _conv1d_to_conv2d(ib_conv),
            _bn1d_to_bn2d(ib_bn),
            nn.ReLU(inplace=True),
            _maxpool1d_to_maxpool2d(ib_pool),
        )
        self.residual_groups = nn.Sequential(*[
            nn.Sequential(*[_BasicBlock2D(b) for b in group])
            for group in net.residual_groups
        ])
        self.output_block1 = _FcBlock2D(net.output_block1)
        self.output_block2 = _FcBlock2D(net.output_block2)

    def forward(self, x: torch.Tensor):
        x = self.input_block(x)
        x = self.residual_groups(x)
        return self.output_block1(x), self.output_block2(x)


# ── Hailo export wrapper ───────────────────────────────────────────────────────

class TLIOWrapperHailo(nn.Module):
    """
    Single-output wrapper for ResNet1DHailo.

    ResNet1D has two head outputs (mean, logstd) which Hailo cannot represent
    as two separate output tensors.  This wrapper concatenates them so the ONNX
    graph has a single output node 'disp_logstd'.

    At inference time, split along the last axis:
        disp_logstd[:, :3]  -> displacement mean  [m] in gravity-aligned frame
        disp_logstd[:, 3:]  -> log-std             [log m]
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (1, 6, 1, 200)  gravity-aligned IMU window (float32) NCHW

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

    hailo_net = ResNet1DHailo(net)
    hailo_net.eval()

    # ── Sanity check: Conv2D mirror must match the original Conv1D model ─────
    with torch.no_grad():
        x_check = torch.randn(1, IMU_CHANNELS, window)
        mean_1d, logstd_1d = net(x_check)
        mean_2d, logstd_2d = hailo_net(x_check.unsqueeze(2))  # (1, 6, 1, window)
        max_err = max(
            (mean_1d - mean_2d).abs().max().item(),
            (logstd_1d - logstd_2d).abs().max().item(),
        )
    print(f"Conv2D mirror max abs error vs original Conv1D model: {max_err:.3e}")
    if max_err > 1e-4:
        raise RuntimeError(
            f"Conv1D->Conv2D conversion diverges from the original model (max_err={max_err:.3e}). "
            "Aborting export — check the conversion helpers."
        )

    wrapped = TLIOWrapperHailo(hailo_net)
    wrapped.eval()

    # ── Postprocessing weights (host-side head, runs after Hailo's prep1 conv) ─
    postproc_path = args.out_dir / "tlio_postproc.pt"
    torch.save({
        "head1": hailo_net.output_block1.post_prep_state(),
        "head2": hailo_net.output_block2.post_prep_state(),
    }, postproc_path)
    print(f"Postprocessing weights saved -> {postproc_path}")

    # ── Export ───────────────────────────────────────────────────────────────
    x_dummy = torch.zeros(1, IMU_CHANNELS, 1, window)  # (batch=1, channels=6, H=1, window=200) NCHW

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
            # do_constant_folding=True fuses prep1's Conv with the immediately
            # following BatchNorm2d into a single Conv node. Hailo's graph is
            # truncated to end exactly at that prep1 Conv (see 1_parsing.py),
            # so a fused node would silently bake bn1 into Hailo's output —
            # and the host-side head (forward_post_prep) applies bn1 again,
            # double-counting it. Keep folding off so prep1 stays a pure,
            # un-fused conv and the Hailo/host split matches the code on
            # both sides of the boundary.
            do_constant_folding=False,
            dynamo=False,
        )

    # NOTE: onnxsim is intentionally NOT run here. For this graph (ResNet
    # backbone + FcBlock heads + Concat), onnxsim's folding corrupts Conv node
    # metadata in a way that's invisible to plain onnxruntime (which still
    # loads and runs it fine) but breaks Hailo's ONNX parser — it misreads the
    # first conv's input features as the window size (200) instead of the
    # channel count (6). The raw, unsimplified export parses correctly.

    print("Done.")
    print()
    print("ONNX interface:")
    print(f"  Input : imu_window=[1, {IMU_CHANNELS}, 1, {window}]  NCHW "
          "(batch, channels=[gyro_ga|accel_ga], H=1, window)")
    print(f"  Output: disp_logstd=[1, {OUT_DIM * 2}]  "
          "(cat([mean_disp, logstd], dim=-1))")


if __name__ == "__main__":
    main()
