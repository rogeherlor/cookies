"""
Tartan IMU velocity predictor -> ONNX converter
=================================================
Exports the Tartan IMU velocity prediction network to ONNX for Hailo DFC parsing.

The network takes 10 seconds of IMU history (at 200 Hz) and predicts body-frame
velocity + log-std.  The Hailo ESKF loop remains in Python/NumPy.

Two-output problem
------------------
The Tartan network returns (velocity, log_std) as two separate tensors.  Hailo
requires a single output tensor.  A thin wrapper concatenates them:
    TartanWrapperHailo(x) -> cat([v(1,3), log_std(1,3)], dim=-1)  shape: (1, 6)

At inference time split: [:, :3] = velocity [m/s] body, [:, 3:] = log_std.

Prerequisite: base model weights
---------------------------------
The Tartan IMU BASE MODEL must be downloaded from HuggingFace before running this
script.  It is NOT included in the repository (the model is too large).

    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('raphael-blanchard/TartanIMU', repo_type='dataset',
                      local_dir='external/tartan_imu')
    "

Or set the TARTAN_IMU_WEIGHTS environment variable to point to the .pt file.

If the base model is unavailable, this script exports the _TartanImuStub
(physics-based fallback) with a warning.

Usage
-----
    python 0_onnx_converter.py [--out-dir scripts/positioning/hailo/tartan_imu]
                               [--opset   11]

Outputs (next to the script by default):
    tartan_imu.onnx

ONNX interface
--------------
Input:
    imu_lstm  (1, 10, 200, 6)  float32 — 10 LSTM steps × 200 samples × [accel_gf(3)|gyro(3)]
Output:
    vel_logstd (1, 6)          float32 — cat([v_body(3), log_std(3)], dim=-1)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent.parent
_TARTAN_DIR = _REPO_ROOT / "scripts/positioning/python/dl_filters/tartan_imu"
_ARTIFACTS  = _REPO_ROOT / "artifacts/tartan_imu"
_EXTERNAL   = _REPO_ROOT / "external/tartan_imu"

if str(_TARTAN_DIR) not in sys.path:
    sys.path.insert(0, str(_TARTAN_DIR))

# ── Constants ─────────────────────────────────────────────────────────────────
LSTM_STEPS   = 10
STEP_SAMPLES = 200
IMU_CHANNELS = 6    # [accel_gf(3) | gyro(3)]


# ── Hailo export wrapper ───────────────────────────────────────────────────────

class TartanWrapperHailo(nn.Module):
    """
    Single-output wrapper for the Tartan velocity predictor.

    The Tartan model returns two tensors (velocity, log_std); Hailo cannot
    represent a model with two separate output nodes.  This wrapper concatenates
    them along the last axis so the ONNX graph has a single output 'vel_logstd'.

    Input : imu_lstm  (1, 10, 200, 6) — LSTM steps × step_samples × [accel_gf|gyro]
    Output: vel_logstd (1, 6)          — cat([v_body(3), log_std(3)])
    """

    def __init__(self, model, robot_type: str = 'car'):
        super().__init__()
        self.model = model
        self.robot_type = robot_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (1, 10, 200, 6)  LSTM input

        Returns
        -------
        vel_logstd : (1, 6)  cat([velocity(1,3), log_std(1,3)])
        """
        v, log_std = self.model(x, robot_type=self.robot_type)
        return torch.cat([v, log_std], dim=-1)


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
    parser = argparse.ArgumentParser(
        description="Export Tartan IMU velocity predictor to ONNX"
    )
    parser.add_argument("--out-dir", type=Path, default=_HERE,
                        help="Directory where tartan_imu.onnx will be written")
    parser.add_argument("--opset",   type=int, default=11,
                        help="ONNX opset (Hailo DFC 3.x supports up to 11/12)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    from tartan_runner import _find_tartan_weights, _find_lora_adapter, _load_tartan_model

    model = None
    using_stub = False
    try:
        weights_path = _find_tartan_weights()
        lora_path = _find_lora_adapter()
        model, use_lora = _load_tartan_model(
            weights_path, lora_path, lora_rank=8, device="cpu"
        )
        print(f"Base model: {weights_path.name}")
        if use_lora and lora_path:
            print(f"LoRA adapter: {lora_path.name}")
    except RuntimeError as e:
        print(f"\n[WARNING] {e}")
        print("\nFalling back to _TartanImuStub (physics-based placeholder).")
        print("The exported ONNX will NOT use learned weights.")
        print("Download the base model from HuggingFace to get a real export.\n")
        from tartan_runner import _TartanImuStub
        model = _TartanImuStub()
        model.eval()
        using_stub = True

    wrapped = TartanWrapperHailo(model, robot_type='car')
    wrapped.eval()

    # ── Dummy input ───────────────────────────────────────────────────────────
    # Shape: (batch=1, lstm_steps=10, step_samples=200, imu_channels=6)
    x_dummy = torch.zeros(1, LSTM_STEPS, STEP_SAMPLES, IMU_CHANNELS)

    out_path = args.out_dir / "tartan_imu.onnx"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if using_stub:
        print("[WARNING] Exporting stub model — replace with real weights before deployment.")

    print(f"Exporting ONNX (opset {args.opset}) -> {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (x_dummy,),
            str(out_path),
            input_names=["imu_lstm"],
            output_names=["vel_logstd"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    _simplify_onnx(out_path)

    print("Done.")
    print()
    print("ONNX interface:")
    print(f"  Input : imu_lstm=[1, {LSTM_STEPS}, {STEP_SAMPLES}, {IMU_CHANNELS}]  "
          "(batch, lstm_steps, step_samples, [accel_gf|gyro])")
    print("  Output: vel_logstd=[1, 6]  (cat([v_body(3), log_std(3)]))")
    if using_stub:
        print()
        print("  [WARNING] This export uses the physics stub, not trained weights.")
        print("  Download base model from HuggingFace and re-run to get real weights.")


if __name__ == "__main__":
    main()
