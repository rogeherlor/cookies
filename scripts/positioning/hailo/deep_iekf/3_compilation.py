"""
AI-IMU MesNet HAR → HEF compilation
======================================
Compiles the quantized HAR (produced by 2_optimisation.py) into a Hailo
Executable Format (.hef) binary ready for on-device inference.

Also saves a compiled HAR and runs the Hailo profiler so you can inspect
layer-level latency and resource utilisation.

Usage
-----
    python 3_compilation.py

Inputs  (next to the script):
    deep_iekf_quantized_model.har   — output of 2_optimisation.py

Outputs (next to the script):
    deep_iekf.hef                   — on-device executable
    deep_iekf_compiled_model.har    — HAR with compilation metadata
    deep_iekf_compiled_model.html   — profiler report  (hailo profiler)
    deep_iekf_compiled_model.svg    — graph visualisation (hailo visualizer)
"""

import logging
import subprocess
from pathlib import Path

from hailo_sdk_client import ClientRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
FILE_DIR   = Path(__file__).resolve().parent
MODEL_NAME = "deep_iekf"

QUANTIZED_HAR_PATH = FILE_DIR / f"{MODEL_NAME}_quantized_model.har"
HEF_PATH           = FILE_DIR / f"{MODEL_NAME}.hef"
COMPILED_HAR_PATH  = FILE_DIR / f"{MODEL_NAME}_compiled_model.har"
SVG_PATH           = FILE_DIR / f"{MODEL_NAME}_compiled_model.svg"


# ── Logging ───────────────────────────────────────────────────────────────────

def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("deep_iekf_compilation")

log = init_logging()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not QUANTIZED_HAR_PATH.exists():
        raise FileNotFoundError(
            f"Quantized HAR not found: {QUANTIZED_HAR_PATH}\n"
            "Run 2_optimisation.py first."
        )

    log.info("Loading quantized HAR: %s", QUANTIZED_HAR_PATH)
    runner = ClientRunner(har=str(QUANTIZED_HAR_PATH))

    log.info("Compiling to HEF ...")
    hef = runner.compile()

    with open(HEF_PATH, "wb") as f:
        f.write(hef)
    log.info("HEF saved: %s", HEF_PATH)

    runner.save_har(str(COMPILED_HAR_PATH))
    log.info("Compiled HAR saved: %s", COMPILED_HAR_PATH)

    # ── Profiler ──────────────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["hailo", "profiler", str(COMPILED_HAR_PATH)],
            cwd=str(FILE_DIR),
            check=True,
        )
        log.info("Profiler report written next to the HAR.")
    except FileNotFoundError:
        log.warning("hailo CLI not found — skipping profiler.")
    except subprocess.CalledProcessError as e:
        log.warning("Profiler failed (non-fatal): %s", e)

    # ── Visualiser ────────────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["hailo", "visualizer", str(COMPILED_HAR_PATH),
             "--no-browser", "--out-path", str(SVG_PATH)],
            cwd=str(FILE_DIR),
            check=True,
        )
        log.info("Graph visualisation: %s", SVG_PATH)
    except FileNotFoundError:
        log.warning("hailo CLI not found — skipping visualiser.")
    except subprocess.CalledProcessError as e:
        log.warning("Visualisation failed (non-fatal): %s", e)


if __name__ == "__main__":
    main()
