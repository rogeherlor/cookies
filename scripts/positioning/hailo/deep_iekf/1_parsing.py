"""
AI-IMU CausalMesNet ONNX -> Hailo HAR (parsing step)
===================================================
Parses the ONNX produced by 0_onnx_converter.py into a Hailo HAR archive.

The exported ONNX is the WHOLE CausalMesNet (cov_net backbone + cov_lin head +
`cov0·exp(ln10·beta·z)` scaling); only the per-channel input normalisation is
handled in Python.  The device outputs the final measurement covariances.  See
0_onnx_converter.py for why the causal model is the definitive Hailo target and
why the head now runs on-device.

Usage
-----
    python 1_parsing.py

Outputs (next to the script):
    deep_iekf_hailo_model.har   — parsed model ready for optimization

ONNX interface expected (must match 0_onnx_converter.py output):
    Input : u_norm_conv      [1, 6, 1, 4544]  NCHW — normalized IMU (H=1 dummy dim)
    Output: measurement_covs [1, 2, 1, 4544]  NCHW — final [cov_lat, cov_up]

Why Conv2d / 4-D?
-----------------
Hailo's ONNX parser cannot resolve channel dimensions from 3-D NCL tensors —
it treats the length axis as the channel count.  0_onnx_converter.py exports
the Conv1d layers as Conv2d with kernel (1, k) and a dummy H=1 dimension.

Network topology (all on-device — CAUSAL, pad BEFORE conv):
    ZeroPad2d(left=4)                                   # causal left-pad
    Conv2d(6→32, k=(1,5)) → ReLU
    ZeroPad2d(left=12)                                  # causal left-pad, dilated
    Conv2d(32→32, k=(1,5), dilation=(1,3)) → ReLU
    Conv2d(32→2, k=1)  (== cov_lin Linear) → Tanh       # cov_lin head
    cov0 · exp(ln10 · beta · z)                         # output scaling
"""

import subprocess
from pathlib import Path

from hailo_sdk_client import ClientRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_DIR  = Path(__file__).resolve().parent
ONNX_PATH = FILE_DIR / "deep_iekf.onnx"

# ── Config ────────────────────────────────────────────────────────────────────
CHOSEN_HW_ARCH  = "hailo8l"
ONNX_MODEL_NAME = "deep_iekf"

SEQ_LEN          = 4544   # fixed — must match 0_onnx_converter.py
IMU_CHANNELS     = 6
COV_DIM          = 2      # output channels: [cov_lat, cov_up]

START_NODES = ["u_norm_conv"]
END_NODES   = ["measurement_covs"]

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
