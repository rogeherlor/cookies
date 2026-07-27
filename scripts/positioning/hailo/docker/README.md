# HailoRT runtime image (this PC + Raspberry Pi 5)

One `Dockerfile`, built twice — once per target — for running compiled `.hef`
files via HailoRT. This image does **not** include the Hailo Dataflow Compiler
(DFC): the DFC only runs on x86_64 and is what the `hailo_ai_sw_suite_2025-01`
container (used by `0_onnx_converter.py`–`3_compilation.py` in `../<approach>/`)
is for. This image is the deployment target for step 4 (`4_inference.py`-style
scripts) — on this PC for local testing, and on the Raspberry Pi 5 for the real
target hardware.

## 1. Get a HailoRT Python wheel (per architecture)

HailoRT is **not on PyPI** — it's a manual download, gated behind an account,
from the **Hailo Developer Zone** (hailo.ai's developer portal, "Software
Downloads" → HailoRT). Download the Python wheel that matches:

- **This PC**: `linux_x86_64`, Python 3.11 (or rebuild the image with a
  `python:3.10-slim-bookworm` base to match the wheel you have — this repo's
  existing `hailo_ai_sw_suite_2025-01` container uses Python 3.10).
- **Raspberry Pi 5**: `linux_aarch64`/`arm64`, matching the Python version on
  Raspberry Pi OS (Bookworm ships Python 3.11 by default — matches this
  Dockerfile's base image as-is).

Place the wheel(s) in this directory:

```
scripts/positioning/hailo/docker/hailort-packages/
    hailort-4.20.0-cp311-cp311-linux_x86_64.whl      # for the PC build
    hailort-4.20.0-cp311-cp311-linux_aarch64.whl     # for the Pi build
```

(`hailort-packages/` is gitignored — see repo `.gitignore`. Only the wheel you
need for the build you're currently running has to be present; the Dockerfile
picks whichever one matches `$TARGETARCH`.)

## 2. Build

**This PC** (amd64):
```bash
cd scripts/positioning/hailo/docker
docker buildx build --platform linux/amd64 -t cookies-hailo-runtime:amd64 .
```

**Raspberry Pi 5** (arm64) — build natively **on the Pi** to avoid
cross-compilation wheel mismatches (copy this `docker/` directory, including
the arm64 wheel, over to the Pi first):
```bash
cd scripts/positioning/hailo/docker
docker build -t cookies-hailo-runtime:arm64 .
```
Cross-building from this PC via `buildx --platform linux/arm64` also works if
QEMU user-mode emulation is set up (`docker buildx create --use`), but native
build on the Pi is simpler and doesn't depend on emulation.

## 3. Run

Both platforms need the Hailo PCIe/USB device passed through. On the
Raspberry Pi 5 the Hailo-8L usually attaches via the M.2 HAT+ or the Hailo AI
HAT, exposing the same kind of `/dev/hailo0` char device seen on this PC:

```bash
docker run --rm -it \
  --device=/dev/hailo0 \
  -v /lib/firmware:/lib/firmware \
  -v "$(pwd)/../<approach>:/workspace/cookies/scripts/positioning/hailo/<approach>" \
  cookies-hailo-runtime:<amd64|arm64> \
  scripts/positioning/hailo/<approach>/4_inference.py --hef .../<approach>.hef
```

If `/dev/hailo0` isn't present, check `hailortcli fw-control identify` on the
host first — the PCIe/USB driver (`hailort-pcie-driver` or the USB
equivalent) has to be installed on the host OS (Pi or PC) before the
container can see the device; this image only provides the userspace
HailoRT library, not the kernel driver.

## What ships in the image vs. what doesn't

| | This image (`docker/Dockerfile`) | `hailo_ai_sw_suite_2025-01` (used for steps 0-3) |
|---|---|---|
| Runs on | x86_64 **and** ARM64 (Raspberry Pi 5) | x86_64 only |
| HailoRT (runtime, `.hef` inference) | ✅ | ✅ |
| Dataflow Compiler (ONNX→HAR, quantization, HAR→HEF) | ❌ | ✅ |
| Use for | Deploying/running compiled models | Compiling/quantizing models |
