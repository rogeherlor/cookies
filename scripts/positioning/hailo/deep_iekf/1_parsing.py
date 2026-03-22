"""
AI-IMU MesNet cov_net ONNX -> Hailo HAR (parsing step)
========================================================
Parses the ONNX produced by 0_onnx_converter.py into a Hailo HAR archive.

The exported ONNX contains only the Conv1d backbone (cov_net) — the
normalization, linear head, and output scaling are handled in Python.
See 0_onnx_converter.py for the full pre/postprocessing pipeline.

Usage
-----
    python 1_parsing.py

Outputs (next to the script):
    deep_iekf_hailo_model.har   — parsed model ready for optimization

ONNX interface expected (must match 0_onnx_converter.py output):
    Input : u_norm_conv  [1, 6, 1, 4544]   NCHW — normalized IMU (H=1 dummy dim)
    Output: cov_features [1, 32, 1, 4544]  NCHW — raw CNN features

Why Conv2d / 4-D?
-----------------
Hailo's ONNX parser cannot resolve channel dimensions from 3-D NCL tensors —
it treats the length axis as the channel count.  0_onnx_converter.py exports
the Conv1d layers as Conv2d with kernel (1, k) and a dummy H=1 dimension.

cov_net topology (Hailo-visible part only):
    Conv2d(6→32, k=(1,5), pad=ZeroPad2d)
    ReLU
    Conv2d(32→32, k=(1,5), dilation=(1,3), pad=ZeroPad2d)
    ReLU
"""

import subprocess
from pathlib import Path

from hailo_sdk_client import ClientRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_DIR  = Path(__file__).resolve().parent
ONNX_PATH = FILE_DIR / "deep_iekf.onnx"

# ── Config ────────────────────────────────────────────────────────────────────
CHOSEN_HW_ARCH  = "hailo8"
ONNX_MODEL_NAME = "deep_iekf"

SEQ_LEN          = 4544   # fixed — must match 0_onnx_converter.py
IMU_CHANNELS     = 6
COV_NET_FEATURES = 32

START_NODES = ["u_norm_conv"]
END_NODES   = ["cov_features"]

NET_INPUT_SHAPES = {
    "u_norm_conv": [1, IMU_CHANNELS, 1, SEQ_LEN],   # [1, 6, 1, 4544] NCHW
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
