"""
Tartan IMU CNN backbone -> ONNX converter (Hailo DFC)
=====================================================
Exports only the Conv1D ResNet backbone of TartanIMU to ONNX for Hailo DFC.

Architecture split
------------------
Hailo executes the CNN backbone only:
    Input : imu_step  (1, 6, 1, 200)  NCHW float32 — one LSTM step of IMU
    Output: cnn_feat  (1, 128, 1, 13) NCHW float32 — ResNet features

Python handles the rest:
    For each of 10 LSTM steps: imu_step → Hailo → cnn_feat (1, 128, 1, 13)
    Flatten:     (1, 1664) per step → stack to (1, 10, 1664)
    LSTM:        (1, 10, 1664) → (1, 10, 64)
    IMU_Trunk:   6 × TransformerBlock → (1, 10, 64)
    Head[car]:   last step (1, 64) → v_body (1, 3) + log_std (1, 3)

Why CNN only?
--------------
Hailo DFC does not support LSTM or MultiheadAttention.  Only the Conv1D backbone
is Hailo-compatible.  The Conv1D layers are exported as Conv2D with a dummy H=1
dimension (shape NCHW) because Hailo's ONNX parser misidentifies the length axis
of 3D NCL tensors as the channel count.

Prerequisite: base model weights
---------------------------------
The Tartan IMU BASE MODEL must be downloaded from HuggingFace before running.

    python -c "
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    for f in ['checkpoint_28.pt', 'checkpoint_14.pt', 'checkpoint_07.pt']:
        hf_hub_download('raphael-blanchard/TartanIMU', f'checkpoints/foundation_model/{f}',
                        repo_type='dataset', local_dir='external/tartan_imu')
    "

Or set TARTAN_IMU_WEIGHTS to the .pt file path.

Usage
-----
    python 0_onnx_converter.py [--out-dir scripts/positioning/hailo/tartan_imu]
                               [--opset   11]

Outputs (next to the script by default):
    tartan_imu.onnx           — CNN backbone for Hailo
    tartan_imu_postproc.pt    — LSTM + Transformer + head weights for Python
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_REPO_ROOT  = _HERE.parent.parent.parent.parent
_TARTAN_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/tartan_imu"
_ARTIFACTS  = _REPO_ROOT / "artifacts/tartan_imu"

if str(_TARTAN_DIR) not in sys.path:
    sys.path.insert(0, str(_TARTAN_DIR))

# ── Constants ─────────────────────────────────────────────────────────────────
STEP_SAMPLES = 200
IMU_CHANNELS = 6    # [accel_gf(3) | gyro(3)]
CNN_OUT_CH   = 128  # resnet_post_pro output channels
CNN_OUT_T    = 13   # time steps: 200→50→25→13 (stride-4, stride-2, stride-2)


# ── Conv1D → Conv2D conversion helpers ────────────────────────────────────────

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


class _ResBlock2D(nn.Module):
    """Conv2d version of _ResBlock1D for Hailo export."""

    def __init__(self, block1d):
        super().__init__()
        c0, c1, _relu, c3, c4 = block1d.convs   # Conv1d BN ReLU Conv1d BN
        self.convs = nn.Sequential(
            _conv1d_to_conv2d(c0),
            _bn1d_to_bn2d(c1),
            nn.ReLU(inplace=True),
            _conv1d_to_conv2d(c3),
            _bn1d_to_bn2d(c4),
        )
        if block1d.downsample is not None:
            d0, d1 = block1d.downsample
            self.downsample = nn.Sequential(
                _conv1d_to_conv2d(d0),
                _bn1d_to_bn2d(d1),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convs(x)
        res = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + res)


# ── CNN backbone wrapper for Hailo ─────────────────────────────────────────────

class TartanCNNHailo(nn.Module):
    """
    Conv2D ResNet backbone — processes one LSTM step for Hailo.

    Input : imu_step (1, 6, 1, 200)  NCHW float32
    Output: cnn_feat (1, 128, 1, 13) NCHW float32

    Time steps: 200 → 50 (stride-4) → 25 (stride-2) → 13 (stride-2)
    """

    def __init__(self, backbone):
        super().__init__()
        ib_conv, ib_bn = backbone.input_block
        self.input_block = nn.Sequential(
            _conv1d_to_conv2d(ib_conv),
            _bn1d_to_bn2d(ib_bn),
        )
        self.residual_groups = nn.ModuleList([
            nn.ModuleList([_ResBlock2D(b) for b in group])
            for group in backbone.residual_groups
        ])
        p0, p1, _relu, p3, p4 = backbone.resnet_post_pro
        self.resnet_post_pro = nn.Sequential(
            _conv1d_to_conv2d(p0),
            _bn1d_to_bn2d(p1),
            nn.ReLU(inplace=True),
            _conv1d_to_conv2d(p3),
            _bn1d_to_bn2d(p4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_block(x))
        for group in self.residual_groups:
            for block in group:
                x = block(x)
        return self.resnet_post_pro(x)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simplify_onnx(onnx_path: Path) -> None:
    try:
        import onnx, onnxsim
    except ImportError:
        print("[warning] onnxsim not found — skipping. Install: pip install onnxsim")
        return
    print("Simplifying ONNX (onnxsim) ...")
    m = onnx.load(str(onnx_path))
    m_sim, ok = onnxsim.simplify(m)
    if ok:
        onnx.save(m_sim, str(onnx_path))
        print(f"Simplified ONNX saved -> {onnx_path}")
    else:
        print("[warning] onnxsim could not simplify — using original.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export TartanIMU CNN backbone to ONNX for Hailo DFC"
    )
    parser.add_argument("--out-dir", type=Path, default=_HERE,
                        help="Directory where outputs will be written")
    parser.add_argument("--opset",   type=int, default=11,
                        help="ONNX opset version (Hailo DFC 3.x supports up to 11/12)")
    args = parser.parse_args()

    from tartan_runner import (
        _find_tartan_weights, _find_lora_adapter, _load_tartan_model, _TartanIMUModel,
    )

    weights_path = _find_tartan_weights()
    lora_path    = _find_lora_adapter()
    model, use_lora = _load_tartan_model(weights_path, lora_path, lora_rank=8, device="cpu")

    if not isinstance(model, _TartanIMUModel):
        raise RuntimeError(
            f"Loaded model is {type(model).__name__}, expected _TartanIMUModel.\n"
            "Download the base model from HuggingFace (see script docstring)."
        )

    print(f"Base model: {weights_path.name}")
    if use_lora and lora_path:
        print(f"LoRA adapter: {lora_path.name}")

    # ── Build Hailo-compatible CNN-only wrapper ────────────────────────────────
    cnn_hailo = TartanCNNHailo(model.model)
    cnn_hailo.eval()

    # ── Save LSTM + Transformer + heads for Python-side inference ─────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    postproc_path = args.out_dir / "tartan_imu_postproc.pt"
    torch.save({
        "lstm_state":  model.model.lstm.state_dict(),
        "trunk_state": model.model.IMU_Trunk.state_dict(),
        "heads_state": model.heads.state_dict(),
    }, postproc_path)
    print(f"Postprocessing weights saved -> {postproc_path}")

    # ── Export ONNX ───────────────────────────────────────────────────────────
    x_dummy  = torch.zeros(1, IMU_CHANNELS, 1, STEP_SAMPLES)
    out_path = args.out_dir / "tartan_imu.onnx"

    print(f"Exporting ONNX (opset {args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            cnn_hailo,
            (x_dummy,),
            str(out_path),
            input_names=["imu_step"],
            output_names=["cnn_feat"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _simplify_onnx(out_path)

    print("Done.")
    print()
    print("ONNX interface (CNN backbone only — LSTM/Transformer run in Python):")
    print(f"  Input : imu_step=[1, {IMU_CHANNELS}, 1, {STEP_SAMPLES}]  NCHW (one LSTM step)")
    print(f"  Output: cnn_feat=[1, {CNN_OUT_CH}, 1, {CNN_OUT_T}]    NCHW (ResNet features)")
    print()
    print(f"Postprocessing: {postproc_path.name}  (lstm_state, trunk_state, heads_state)")


if __name__ == "__main__":
    main()
