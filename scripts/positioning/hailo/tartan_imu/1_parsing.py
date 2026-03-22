"""
Tartan IMU CNN backbone ONNX -> Hailo HAR (parsing step)
=========================================================
Parses the ONNX produced by 0_onnx_converter.py into a Hailo HAR archive.

Hailo executes only the Conv1D ResNet backbone (exported as Conv2D with 4D NCHW):
    Input : imu_step  [1, 6, 1, 200]   NCHW — one LSTM step of IMU
    Output: cnn_feat  [1, 128, 1, 13]  NCHW — ResNet features before LSTM

Python handles LSTM + Transformer + head using tartan_imu_postproc.pt.

Usage
-----
    python 1_parsing.py

Outputs (next to the script):
    tartan_imu_hailo_model.har   — parsed model ready for optimization
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

STEP_SAMPLES = 200
IMU_CHANNELS = 6
CNN_OUT_CH   = 128
CNN_OUT_T    = 13   # 200 → 50 (stride-4) → 25 (stride-2) → 13 (stride-2)

START_NODES = ["imu_step"]
END_NODES   = ["cnn_feat"]

NET_INPUT_SHAPES = {
    "imu_step": [1, IMU_CHANNELS, 1, STEP_SAMPLES],  # [1, 6, 1, 200] NCHW
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
