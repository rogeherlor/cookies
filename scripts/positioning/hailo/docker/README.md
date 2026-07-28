# HailoRT runtime image (this PC + Raspberry Pi 5)

One `Dockerfile`, built twice — once per target — for running compiled `.hef`
files via HailoRT. This image does **not** include the Hailo Dataflow Compiler
(DFC): the DFC only runs on x86_64 and is what the `hailo_ai_sw_suite_2025-01`
container (used by `0_onnx_converter.py`–`3_compilation.py` in `../<approach>/`)
is for. This image is the deployment target for step 4 (`4_inference.py`-style
scripts) — on this PC for local testing, and on the Raspberry Pi 5 for the real
target hardware.

**Version pin: 4.20.0.** That's the firmware flashed on the Hailo-8/8L
devices this repo has actually been run against (see `hailort.log` in each
`../<approach>/` dir — it logs `firmware_version is: 4.20.0` from a prior
successful run). HailoRT talking to mismatched firmware fails at runtime even
when everything installs cleanly, so keep the installed HailoRT version and
the on-device firmware in lockstep. Check the device's firmware version with
`hailortcli fw-control identify` before changing `HAILORT_VERSION` in the
Dockerfile.

## 1. Get HailoRT (per architecture)

**Raspberry Pi 5 (arm64): nothing to do.** The Dockerfile installs
`hailort` + `python3-hailort` straight from Raspberry Pi's public apt repo
(`archive.raspberrypi.com`) — no Developer Zone account, no manual wheel.
This mirrors Hailo's official release and is what this repo's Pi is
currently running (firmware 4.20.0, installed the same way — see below).

**This PC (amd64): still needs a manual wheel.** HailoRT is **not on PyPI**
for x86_64, and there's no public apt mirror for it — it's a gated download
from the **Hailo Developer Zone** (hailo.ai's developer portal, "Software
Downloads" → HailoRT). Get the **4.20.0** wheel (not a newer version — see
the pin note above) for `linux_x86_64`, Python 3.11, and place it here:

```
scripts/positioning/hailo/docker/hailort-packages/
    hailort-4.20.0-cp311-cp311-linux_x86_64.whl
```

(`hailort-packages/` is gitignored — see repo `.gitignore`.) Note the wheel
only provides the Python bindings — it dynamically links against
`libhailort.so.4.20.0`, which must be installed separately too (Hailo's
HailoRT `.deb`/`.run` installer for x86_64, matching version).

## 2. Build

Via `docker-compose.yml` (recommended — picks the right `platform` and wires
up the device/volume mounts for you):

```bash
cd scripts/positioning/hailo/docker
docker compose --profile arm64 build   # on the Pi
docker compose --profile amd64 build   # on the PC
```

Or directly with `docker build`/`buildx`:

**This PC** (amd64):
```bash
cd scripts/positioning/hailo/docker
docker buildx build --platform linux/amd64 -t cookies-hailo-runtime:amd64 .
```

**Raspberry Pi 5** (arm64) — build natively **on the Pi** (safest — no
cross-compile mismatches, and nothing needs to be copied over since the
arm64 path no longer depends on a local wheel):
```bash
cd scripts/positioning/hailo/docker
docker build -t cookies-hailo-runtime:arm64 .
```
Cross-building from the PC via `buildx --platform linux/arm64` also works if
QEMU user-mode emulation is set up (`docker buildx create --use`), but native
build on the Pi is simpler and doesn't depend on emulation.

## 3. Run

Both platforms need the Hailo PCIe/USB device passed through. Via compose
(mounts the whole repo to `/workspace/cookies` and passes `/dev/hailo0`
automatically):

```bash
docker compose --profile arm64 run --rm hailo-arm64 \
  scripts/positioning/hailo/<approach>/4_inference.py --hef .../<approach>.hef
```

Or directly:
```bash
docker run --rm -it \
  --device=/dev/hailo0 \
  -v /lib/firmware:/lib/firmware:ro \
  -v "$(pwd)/../../../..:/workspace/cookies" \
  cookies-hailo-runtime:<amd64|arm64> \
  scripts/positioning/hailo/<approach>/4_inference.py --hef .../<approach>.hef
```

If `/dev/hailo0` isn't present, check `hailortcli fw-control identify` on the
host first — the PCIe/USB driver (`hailo_pci`, in-tree on Raspberry Pi OS's
kernel) has to be loaded **and** matching firmware installed
(`sudo apt install hailofw` on the Pi) before the container can see the
device; this image only provides the userspace HailoRT library, not the
kernel driver or firmware.

## What ships in the image vs. what doesn't

| | This image (`docker/Dockerfile`) | `hailo_ai_sw_suite_2025-01` (used for steps 0-3) |
|---|---|---|
| Runs on | x86_64 **and** ARM64 (Raspberry Pi 5) | x86_64 only |
| HailoRT (runtime, `.hef` inference) | ✅ | ✅ |
| Dataflow Compiler (ONNX→HAR, quantization, HAR→HEF) | ❌ | ✅ |
| Use for | Deploying/running compiled models | Compiling/quantizing models |
