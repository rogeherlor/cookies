"""
Tartan IMU ONNX -> Hailo HAR (parsing step)
=============================================
Parses the ONNX produced by 0_onnx_converter.py into a Hailo HAR archive
using the Hailo Dataflow Compiler SDK.

Usage
-----
    python 1_parsing.py

Outputs (next to the script):
    tartan_imu_hailo_model.har   — parsed model ready for optimization

ONNX interface expected (must match 0_onnx_converter.py output):
    Input : imu_lstm    [1, 10, 200, 6]  (batch, lstm_steps, step_samples, channels)
    Output: vel_logstd  [1, 6]           (cat([v_body(3), log_std(3)]))

HAR representation
------------------
The Hailo DFC maps all linear/Conv ops to Conv2D in the HAR.  The 4-D input
[1, 10, 200, 6] is already in NHWC format (batch, H=lstm_steps, W=step_samples,
C=imu_channels), which Hailo handles natively.
"""

import subprocess
from pathlib import Path

from hailo_sdk_client import ClientRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_DIR  = Path(__file__).resolve().parent
ONNX_PATH = FILE_DIR / "tartan_imu.onnx"

# ── Config ────────────────────────────────────────────────────────────────────
CHOSEN_HW_ARCH  = "hailo8"
ONNX_MODEL_NAME = "tartan_imu"

LSTM_STEPS   = 10
STEP_SAMPLES = 200
IMU_CHANNELS = 6

START_NODES = ["imu_lstm"]
END_NODES   = ["vel_logstd"]

NET_INPUT_SHAPES = {
    "imu_lstm": [1, LSTM_STEPS, STEP_SAMPLES, IMU_CHANNELS],  # [1, 10, 200, 6]
}

# ── Parse ─────────────────────────────────────────────────────────────────────
if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {ONNX_PATH}\n"
        "Run 0_onnx_converter.py first."
    )

print(f"Parsing {ONNX_PATH} for {CHOSEN_HW_ARCH} ...")

runner = ClientRunner(hw_arch=CHOSEN_HW_ARCH)

hn, npz = runner.translate_onnx_model(
    str(ONNX_PATH),
    ONNX_MODEL_NAME,
    start_node_names=START_NODES,
    end_node_names=END_NODES,
    net_input_shapes=NET_INPUT_SHAPES,
)

har_path = FILE_DIR / f"{ONNX_MODEL_NAME}_hailo_model.har"
runner.save_har(str(har_path))
print(f"HAR saved: {har_path}")

# ── Visualise (optional, requires graphviz) ───────────────────────────────────
svg_path = FILE_DIR / f"{ONNX_MODEL_NAME}.svg"
try:
    subprocess.run(
        ["hailo", "visualizer", str(har_path), "--no-browser",
         "--out-path", str(svg_path)],
        cwd=str(FILE_DIR),
        check=True,
    )
    print(f"Graph visualisation: {svg_path}")
except FileNotFoundError:
    print("hailo CLI not found — skipping visualisation.")
except subprocess.CalledProcessError as e:
    print(f"Visualisation failed (non-fatal): {e}")
