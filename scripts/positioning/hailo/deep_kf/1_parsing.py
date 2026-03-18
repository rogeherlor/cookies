"""
DeepKFNet ONNX -> Hailo HAR (parsing step)
==========================================
Parses the ONNX produced by 0_onxx_converter.py into a Hailo HAR archive
using the Hailo Dataflow Compiler SDK.

Usage
-----
    python 1_parsing.py

Outputs (next to the script):
    deep_kf_hailo_model.har   — parsed model ready for optimization

ONNX interface expected (must match 0_onxx_converter.py output):
    Input : x     [1, 1, 15]  (batch=1, seq=1, nav_state)
    Output: state [1, 15]     (predicted state x_t^{+-})

HAR representation (Conv layers in Netron)
------------------------------------------
The Hailo DFC maps all matrix multiplications to 1x1 Conv2D ops in the HAR,
because Hailo's hardware is built around convolutional engines.  Seeing Conv
layers in Netron is expected and correct — it is not a bug.

    LSTM gate  (W · x  or  U · h)  ->  Conv 1x1
    StateDecoder Linear(128->128)  ->  Conv 1x1
    StateDecoder Linear(128->15)   ->  Conv 1x1

Each LSTM has 4 gates (i, f, g, o) x 2 operands (input weight, recurrent
weight), so 2 layers x 8 Conv nodes = ~16 Conv nodes from the LSTMs alone,
plus 2 more from the decoder MLP.  The model is functionally identical to
the original — only the internal representation differs.
"""

import subprocess
from pathlib import Path

from hailo_sdk_client import ClientRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_DIR   = Path(__file__).resolve().parent
ONNX_PATH  = FILE_DIR / "deep_kf.onnx"

# ── Config ────────────────────────────────────────────────────────────────────
CHOSEN_HW_ARCH  = "hailo8"
ONNX_MODEL_NAME = "deep_kf"

NAV_DIM    = 15
INPUT_DIM  = NAV_DIM   # 15
BATCH_SIZE = 1

# Single 3-D input: (batch, seq=1, features) — required for Hailo conv ops.
START_NODES = ["x"]
END_NODES   = ["state"]

NET_INPUT_SHAPES = {
    "x": [BATCH_SIZE, 1, INPUT_DIM],   # [1, 1, 15]
}

# ── Parse ─────────────────────────────────────────────────────────────────────
if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {ONNX_PATH}\n"
        "Run 0_onxx_converter.py first."
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
