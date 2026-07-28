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
# The amd64 build expects a HailoRT Python wheel in hailort-packages/ (see
# docker/README.md for where to get it — not on PyPI, gated behind a Hailo
# Developer Zone account). The arm64 build needs nothing pre-placed — it
# pulls a version-matched HailoRT straight from Raspberry Pi's public apt repo.

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
        curl \
        bzip2 \
    && rm -rf /var/lib/apt/lists/*

# ── HailoRT ────────────────────────────────────────────────────────────────
# Pinned to 4.20.0 — the firmware version flashed on the target Hailo-8/8L
# devices this repo has been validated against (see hailort.log in the
# ../<approach>/ dirs). A HailoRT version mismatched against the on-device
# firmware will fail at runtime even if it installs cleanly, so don't bump
# this without reflashing firmware to match (see docker/README.md).
#
# arm64 (Raspberry Pi 5): libhailort.so + hailortcli come from Raspberry
# Pi's public apt repo (mirrors Hailo's release, no Developer Zone account
# needed). The python3-hailort *package* there targets Debian's system
# python3 though (a different interpreter than this image's
# /usr/local/bin/python3, built from source by the upstream
# python:3.11-slim-bookworm image) — rather than install a second full
# Python just to get one package, we extract python3-hailort's .deb with
# dpkg -x (no install, no dependency resolution) and drop its
# hailo_platform/ package straight into /usr/local's site-packages. Both
# pythons are CPython 3.11 with the same ABI tag
# (cpython-311-aarch64-linux-gnu), so the compiled _pyhailort extension
# loads fine either way. Its only two runtime deps not already pulled in —
# netifaces, netaddr (see hailo_platform/pyhailort/ethernet_utils.py) — are
# added below: netifaces the same dpkg -x way (it's a C extension with no
# prebuilt PyPI wheel for arm64, and this image has no compiler toolchain),
# netaddr via plain pip (pure Python, ships a universal wheel).
#
# amd64 (this PC): no public apt repo exists for generic x86_64, so this
# still requires a manually downloaded HailoRT Python wheel (matching
# 4.20.0, not a newer version) placed in hailort-packages/ — see
# docker/README.md. Note the wheel alone only provides the Python bindings;
# it dynamically links against libhailort.so.4.20.0, which must also be
# installed (via Hailo's HailoRT .deb/.run installer for x86_64).
ARG HAILORT_VERSION=4.20.0-1
COPY hailort-packages/ /tmp/hailort-packages/
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
        rpi_pool="https://archive.raspberrypi.com/debian/pool/main/r/raspberrypi-archive-keyring"; \
        rpi_keyring_deb=$(curl -fsSL "${rpi_pool}/" | grep -o 'raspberrypi-archive-keyring_[^"]*_all\.deb' | sort -V | tail -1); \
        curl -fsSL -o /tmp/rpi-keyring.deb "${rpi_pool}/${rpi_keyring_deb}"; \
        apt-get update; \
        apt-get install -y --no-install-recommends /tmp/rpi-keyring.deb; \
        echo "deb http://archive.raspberrypi.com/debian/ bookworm main" \
            > /etc/apt/sources.list.d/raspi.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends "hailort=${HAILORT_VERSION}"; \
        ( cd /tmp && apt-get download "python3-hailort=${HAILORT_VERSION}" ); \
        dpkg -x /tmp/python3-hailort_*.deb /tmp/pyhailort-extract; \
        cp -r /tmp/pyhailort-extract/usr/lib/python3/dist-packages/hailo_platform \
            /usr/local/lib/python3.11/site-packages/; \
        cp -r /tmp/pyhailort-extract/usr/lib/python3/dist-packages/hailo_tutorials \
            /usr/local/lib/python3.11/site-packages/ 2>/dev/null || true; \
        ( cd /tmp && apt-get download python3-netifaces ); \
        dpkg -x /tmp/python3-netifaces_*.deb /tmp/netifaces-extract; \
        cp /tmp/netifaces-extract/usr/lib/python3/dist-packages/netifaces*.so \
            /usr/local/lib/python3.11/site-packages/; \
        pip install --no-cache-dir netaddr; \
        rm -rf /var/lib/apt/lists/* /tmp/rpi-keyring.deb /tmp/pyhailort-extract \
            /tmp/python3-hailort_*.deb /tmp/netifaces-extract /tmp/python3-netifaces_*.deb; \
    else \
        whl=$(find /tmp/hailort-packages -maxdepth 1 -iname "hailort*${TARGETARCH}*.whl" -o -iname "hailort*.whl" | head -n1); \
        if [ -z "$whl" ]; then \
            echo "ERROR: no HailoRT wheel found in hailort-packages/ for arch=${TARGETARCH}."; \
            echo "See scripts/positioning/hailo/docker/README.md to download one."; \
            exit 1; \
        fi; \
        pip install --no-cache-dir "$whl"; \
    fi; \
    rm -rf /tmp/hailort-packages

# ── Python inference dependencies ─────────────────────────────────────────────
# Keep this list minimal and in sync with what 4_inference.py-style scripts
# actually import at runtime (numpy is the only hard requirement; torch is only
# needed if the deployed script keeps a PyTorch-side reference/postproc step).
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runtime.txt

# ── GTSAM (for smoothers/isam2_runner.py, isam2_fixedlag_runner.py) ──────────
# gtsam has no aarch64 wheel on PyPI; conda-forge is the only aarch64 build
# available. Installed via micromamba — a small (~20MB) standalone
# conda-compatible package manager, NOT a full Anaconda/Miniconda — into an
# isolated prefix at /opt/conda-gtsam, kept separate from this image's main
# python3 (a completely different interpreter/ABI lineage than either
# /usr/local's official build or the apt-bridged HailoRT one above; unlike
# HailoRT's single small extension module, gtsam pulls in a large C++
# dependency graph — boost, suitesparse, tbb, metis — not worth bridging).
# isam2_runner.py/isam2_fixedlag_runner.py/fgo_batch_runner.py only import
# numpy + pymap3d + gtsam (checked — no other repo-module dependencies);
# isam2_map_runner.py ("iSAM2 Track") additionally needs scipy.spatial.KDTree
# for its point-to-curve track projection. Otherwise this stays a small,
# self-contained environment; the eval harness invokes it as a subprocess via
# /opt/conda-gtsam/bin/python3 (see _isam2_conda_worker.py), not by importing
# gtsam into the main interpreter. amd64 gets it too (conda-forge publishes
# linux-64 as well) for parity, though this repo's primary amd64 use is local
# dev/testing, not deployment.
RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then mmb_plat="linux-aarch64"; \
    else mmb_plat="linux-64"; fi; \
    curl -sL "https://micro.mamba.pm/api/micromamba/${mmb_plat}/latest" \
        | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba; \
    micromamba create -y -p /opt/conda-gtsam -c conda-forge \
        python=3.11 gtsam numpy pymap3d scipy; \
    micromamba clean -y --all

WORKDIR /workspace/cookies

ENTRYPOINT ["python3"]
