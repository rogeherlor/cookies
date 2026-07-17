# syntax=docker/dockerfile:1
#
# HailoRT runtime image — multi-arch: builds for this PC (linux/amd64) and for
# Raspberry Pi 5 (linux/arm64). Runs compiled .hef files via 4_inference.py-style
# scripts. It does NOT include the Hailo Dataflow Compiler (DFC) — the DFC is
# x86_64-only and does not run on the Pi. Compile .hef files on the PC using the
# hailo_ai_sw_suite_2025-01 image (steps 0-3 in ../<approach>/), then copy the
# .hef + this image to the Pi for step-4-style inference. See docker/README.md.
#
# Build for this PC:
#   docker buildx build --platform linux/amd64 -t cookies-hailo-runtime:amd64 \
#       -f scripts/positioning/hailo/docker/Dockerfile scripts/positioning/hailo/docker
#
# Build for Raspberry Pi 5 (run natively ON the Pi — safest, no cross-compile
# wheel mismatches):
#   docker build -t cookies-hailo-runtime:arm64 \
#       -f scripts/positioning/hailo/docker/Dockerfile scripts/positioning/hailo/docker
#
# Both builds expect a HailoRT Python wheel for the target architecture in
# hailort-packages/ (see docker/README.md for where to get it — it is not on
# PyPI and cannot be fetched automatically).

FROM python:3.11-slim-bookworm

ARG TARGETARCH

# ── System packages ───────────────────────────────────────────────────────────
# libgl1/libusb are common HailoRT runtime deps (device access, image I/O if the
# inference script does any pre/post-processing with OpenCV-like libs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libusb-1.0-0 \
        udev \
        pciutils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── HailoRT (from Hailo Developer Zone — see docker/README.md) ───────────────
# Not on PyPI, so it must be supplied locally per architecture before build.
COPY hailort-packages/ /tmp/hailort-packages/
RUN set -eux; \
    whl=$(find /tmp/hailort-packages -maxdepth 1 -iname "hailort*${TARGETARCH}*.whl" -o -iname "hailort*.whl" | head -n1); \
    if [ -z "$whl" ]; then \
        echo "ERROR: no HailoRT wheel found in hailort-packages/ for arch=${TARGETARCH}."; \
        echo "See scripts/positioning/hailo/docker/README.md to download one."; \
        exit 1; \
    fi; \
    pip install --no-cache-dir "$whl"

# ── Python inference dependencies ─────────────────────────────────────────────
# Keep this list minimal and in sync with what 4_inference.py-style scripts
# actually import at runtime (numpy is the only hard requirement; torch is only
# needed if the deployed script keeps a PyTorch-side reference/postproc step).
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt

WORKDIR /workspace/cookies

ENTRYPOINT ["python3"]
