"""
TLIO ONNX -> Hailo HAR (parsing step)
======================================
Parses the ONNX produced by 0_onnx_converter.py into a Hailo HAR archive
using the Hailo Dataflow Compiler SDK.

Usage
-----
    python 1_parsing.py

Outputs (next to the script):
    tlio_hailo_model.har   — parsed model ready for optimization

ONNX interface expected (must match 0_onnx_converter.py output):
    Input : imu_window  [1, 6, 200]  (batch=1, channels=6, window=200)
    Output: disp_logstd [1, 6]       (cat([mean_disp(3), logstd(3)]))

HAR representation
------------------
The Hailo DFC maps Conv1D ops to Conv2D in the HAR (Hailo hardware uses 2-D
convolutional engines).  Seeing Conv2D in Netron is expected and correct.

ResNet1D topology (approx):
    input_block : Conv1d(6→64, k=7, s=2) + MaxPool(s=2)
    group 0     : 2× BasicBlock1D(64→64)
    group 1     : 2× BasicBlock1D(64→128, s=2)
    group 2     : 2× BasicBlock1D(128→256, s=2)
    group 3     : 2× BasicBlock1D(256→512, s=2)
    output_block1+2 : FcBlock(512→3) × 2  → cat → (1,6)
"""

import subprocess
from pathlib import Path

from hailo_sdk_client import ClientRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_DIR  = Path(__file__).resolve().parent
ONNX_PATH = FILE_DIR / "tlio.onnx"

# ── Config ────────────────────────────────────────────────────────────────────
CHOSEN_HW_ARCH  = "hailo8"
ONNX_MODEL_NAME = "tlio"

WINDOW_SIZE  = 200
IMU_CHANNELS = 6

START_NODES = ["imu_window"]
END_NODES   = ["disp_logstd"]

NET_INPUT_SHAPES = {
    "imu_window": [1, IMU_CHANNELS, WINDOW_SIZE],  # [1, 6, 200]
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
