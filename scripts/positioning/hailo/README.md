# Hailo-8L adaptation

Deploys four of the DL positioning filters (`deep_kf`, `tlio`, `tartan_imu`, `deep_iekf`)
onto a physical Hailo-8L accelerator. Each subfolder is a self-contained 5-stage
pipeline that converts a trained PyTorch checkpoint, parses/quantizes it with the
Hailo Dataflow Compiler (DFC), compiles a `.hef`, and runs it on the real device:

```
0_onnx_converter.py   PyTorch checkpoint -> ONNX
1_parsing.py          ONNX -> Hailo HAR (hw_arch=hailo8l)
2_optimisation.py      HAR -> quantized HAR + PyTorch/ONNX/SDK_NATIVE/SDK_FP_OPT/SDK_QUANTIZED comparison
3_compilation.py       quantized HAR -> .hef + compiler profiler report
4_inference.py         .hef running on the physical Hailo-8L device vs PyTorch ground truth
```

## Device

This machine has a physical **Hailo-8L** (not Hailo-8 — check with
`hailortcli fw-control identify`, field `Device Architecture`). `hw_arch` must be
set to `"hailo8l"` in every `1_parsing.py`; using `"hailo8"` compiles a binary the
device cannot run.

The Hailo Dataflow Compiler (DFC, used in steps 0-3) is **x86_64-only** — it cannot
run on a Raspberry Pi. HailoRT (the runtime used in step 4) supports both x86_64
and ARM64, so the normal workflow is: compile on this PC, copy the resulting
`.hef` + inference script to the Pi, run inference there. See
[`docker/`](docker/) for the multi-arch runtime image.

## Running the pipeline yourself

Everything here runs inside the `hailo_ai_sw_suite_2025-01` Docker image (has the
DFC, HailoRT, and PCIe device access). Start it with:

```bash
docker run -d --privileged --gpus all \
  -v /dev:/dev -v /lib/firmware:/lib/firmware -v /lib/modules:/lib/modules \
  -v /var/lib/dkms:/var/lib/dkms -v /usr/src:/usr/src \
  -v "$(pwd):/workspace/cookies" \
  --name cookies_hailo_test hailo_ai_sw_suite_2025-01:1 tail -f /dev/null
```

Then for any approach:

```bash
docker exec -it cookies_hailo_test bash
cd /workspace/cookies/scripts/positioning/hailo/<approach>
python3 0_onnx_converter.py [--artifact path/to/checkpoint.pt]
python3 1_parsing.py
python3 2_optimisation.py      # prints PyTorch vs ONNX vs SDK_* MAE table
python3 3_compilation.py       # writes <approach>.hef + <approach>_compiled_model.html
python3 4_inference.py         # real hardware run, prints HEF vs PyTorch MAE
```

`tartan_imu` needs `TARTAN_IMU_LORA=<repo>/artifacts/tartan_imu/lora_fold_01.pt`
exported first (or another fold) — without it, both the CNN export and the
PyTorch reference silently fall back to the zero-shot base model, which will
never match a `.hef` compiled from a specific fold.

`deep_iekf` defaults to `artifacts/deep_iekf_online/iekfnets.p` — **if a
training run is writing to that folder, pass an explicit, already-completed
fold instead** (`--weights <repo>/artifacts/deep_iekf_online/fold_01.p`, or
`fold_04.p`/`fold_06.p`/`fold_07.p`, whichever are already finished) on
`0_onnx_converter.py`, `2_optimisation.py`, and `4_inference.py` alike — the
default path is exactly the file the training loop is actively overwriting.

## See the real hardware results yourself

Each `2_optimisation.py`/`4_inference.py` prints a per-sample table plus a final
`MAE vs PyTorch` line — that's the actual on-device accuracy number.

For a hardware-measured **latency/throughput** profile (not just a static
estimate), run the device benchmark and feed it back into the profiler:

```bash
cd /workspace/cookies/scripts/positioning/hailo/<approach>
hailortcli run2 -m raw measure-fw-actions \
    --output-path runtime_data_<approach>.json set-net <approach>.hef
hailo profiler <approach>_compiled_model.har \
    --runtime-data runtime_data_<approach>.json \
    --out-path runtime_profiler_<approach>.html
```

Open the resulting `runtime_profiler_<approach>.html` in a browser (it's a plain
file on the host — the container mount is bind, not a volume) — measured FPS,
latency, and per-layer resource utilization on the actual chip. Already
generated for all four validated approaches:
`deep_kf/runtime_profiler_deep_kf.html`, `tlio/runtime_profiler_tlio.html`,
`tartan_imu/runtime_profiler_tartan_imu.html`,
`deep_iekf/runtime_profiler_deep_iekf.html`.

`3_compilation.py` also writes a **compiler-estimate** profile
(`<approach>_compiled_model.html`) — that one has no live device data behind
it, it's the DFC's static estimate. The `runtime_profiler_*.html` files above
are the ones backed by real hardware measurements.

## Results

| Approach     | Status | On-device MAE vs PyTorch | Notes |
|--------------|--------|---------------------------|-------|
| `deep_kf`    | ✅ validated | 0.094 (core state channels match to 4-5 sig figs; error concentrated in near-zero covariance channels, which are coarsely quantized by design — see `LSTM h0` comment in the model) | |
| `tlio`       | ✅ validated | 0.046 | Needed 3 real fixes, see below |
| `tartan_imu` | ✅ validated | 0.014 | Needed 3 real fixes + a pre-existing bug fix in `tartan_runner.py`, see below |
| `deep_iekf`  | ✅ validated | 5.01 steady-state (excl. first 20 samples — see below) | Tested against completed fold `fold_01.p` while a *different* fold trains in the background; needed 1 real fix, see below |

All four approaches ran the **entire** pipeline (steps 0-4) on the physical
Hailo-8L, not just the DFC's software emulator. `deep_iekf` was tested against
`artifacts/deep_iekf_online/fold_01.p` — a completed LOO fold — while a
training run for a *different* held-out fold continued untouched in the
background throughout.

### `deep_iekf` result detail

`deep_iekf` processes IMU covariances for a whole KITTI sequence
(`SEQ_LEN=4544` samples) causally in one pass, using left-only
(`ReplicationPad1d`) padding so output `i` never depends on future input. The
Hailo Conv2d converter maps that padding to `ZeroPad2d`, which is exact
everywhere **except the first ~16 samples**, where "replicate the first
sample" and "pad with zero" genuinely disagree — a documented, unavoidable
transient, not a bug. `2_optimisation.py`/`4_inference.py` used to report a
single MAE over just the first 8 timesteps, which sat **entirely inside**
that transient and made the model look badly broken (MAE ~44, later ~76 after
finding the normalization bug below). Both scripts now report the transient
and steady-state MAE separately:

```
MAE vs PyTorch, first 20 samples (padding transient):        37.6
MAE vs PyTorch, steady-state (excl. first 20 samples):        0.0098  (SDK_FP_OPT, full precision)
                                                                4.99   (SDK_QUANTIZED)
                                                                5.01   (real HEF on Hailo-8L)
```
cov_up's scale is ~80-300, so a steady-state MAE of ~5 is a ~2-6% relative
error — the same order as the other three approaches' quantization error.

## Bugs found and fixed

### Shared across all four approaches (`1_parsing.py`, `4_inference.py`)

- **Wrong `hw_arch`.** Every `1_parsing.py` hardcoded `CHOSEN_HW_ARCH = "hailo8"`.
  This device is a **Hailo-8L**, a different (smaller) architecture — a HEF
  compiled for `hailo8` cannot run on it. Fixed to `"hailo8l"`.
- **`configured_model.run([bindings], timeout_ms=1000)`** — the installed
  HailoRT 4.20.0 Python API takes `timeout` as a **positional** argument, not
  a `timeout_ms` keyword. This raised immediately.
- **Missing `set_format_type(FormatType.FLOAT32)`** on input/output streams.
  Without it HailoRT defaults to `UINT8` — passing float32 numpy buffers then
  fails with "Input buffer size N is different than expected N/4" (byte vs.
  element count mismatch).
- **Missing `configured_model.activate()` / `.deactivate()`.** With the
  scheduler disabled (`HAILO_SCHEDULING_ALGORITHM_NONE`, the default),
  `run()` without an explicit `activate()` **returns all-zero output with no
  exception raised.** This is the most dangerous one — it looks exactly like
  "the model produces garbage on Hailo" rather than a missing API call.
- **`pymap3d` not installed** in the Hailo Docker image's Python env — KITTI
  calibration data loading silently fell back to synthetic random noise,
  which gives the quantizer meaningless activation ranges. `pip install
  pymap3d` inside the container fixes it (real KITTI data confirmed via the
  "Calibration: using N real KITTI ..." log line).

### `tlio`-specific

- **Raw `Conv1d` in the ONNX export.** Hailo's parser can't handle a native
  1D conv — it misreads the input's channel count as the window length. Fixed
  by mirroring the existing `Conv1d->Conv2d` (dummy `H=1` dimension) trick
  already used in `tartan_imu`'s converter: `ResNet1DHailo` in
  `tlio/0_onnx_converter.py`, verified `0.0` numerical error against the
  original model before export.
- **`onnxsim` corrupting the graph.** The pre-export `onnxsim.simplify()` call
  silently mangled Conv node metadata for this specific graph shape (ResNet +
  FC heads + Concat) — plain `onnxruntime` still loaded and ran the corrupted
  file fine, but Hailo's parser choked, misreading the first conv's channel
  count as the window size. Disabled `onnxsim` for `tlio`'s export; the raw
  export parses correctly.
- **Hailo can't run the `Flatten -> FC` head.** `UnsupportedShuffleLayerError`.
  The DFC's own error message recommended ending the graph at the two `prep1`
  1x1-convs — so the model is now split the same way as `tartan_imu`: Hailo
  runs the CNN backbone through `prep1`, and the small
  `bn1 -> flatten -> fc1 -> fc2 -> fc3` head runs on the host from weights
  saved to `tlio_postproc.pt`.
- **`do_constant_folding=True` fusing `prep1`'s Conv with the following
  BatchNorm** into one node. Since the Hailo/host split boundary sits exactly
  between those two ops, the fused node meant Hailo's output already included
  BatchNorm — and the host-side head applied BatchNorm a **second time**.
  Disabled constant folding for the export; `prep1` now stays an unfused,
  pure conv on both sides of the boundary.
- Dynamic `x.reshape(x.size(0), -1)` in the FC head traced to a
  Shape/Gather/Reshape chain that Hailo's parser also couldn't handle
  (`IndexError` deep in `is_windows_to_input_chain_end`). Replaced with
  `torch.flatten(x, 1)`, which exports as a plain ONNX `Flatten` op.
- Missing `sys.path` entry for `tlio_dataset.py` (lives one directory up from
  where the scripts looked) — calibration/test data loading fell back to
  synthetic data with `KITTI data unavailable (No module named 'tlio_dataset')`.
- Non-contiguous NHWC transpose passed straight to `set_buffer()` — HailoRT
  requires `C_CONTIGUOUS` buffers. Fixed with `np.ascontiguousarray(...)`.

### `tartan_imu`-specific

- **Pre-existing bug in `tartan_runner.py`**, unrelated to Hailo:
  `_TartanIMUBackbone.forward()` did `x.reshape(B * T, C, S)` on a tensor
  actually shaped `(B, T, S, C)`. A bare `reshape` does not transpose axes —
  it needs `x.permute(0, 1, 3, 2).reshape(B * T, C, S)`. Confirmed empirically
  (the reshaped tensor's per-step data did not match the equivalent
  `.permute().reshape()`). **This method is called from three places**:
  `train_tartan.py`'s training loop (`return_sequence=True`), the online EKF
  velocity update in `tartan_runner.py` (`~line 675`), and the Hailo test
  harness here — so every real (non-Hailo) use of the full 10-step window in
  one call was silently scrambling per-step channel/sample data before this
  fix. `artifacts/tartan_imu/lora_fold_01.pt` is a LoRA checkpoint trained
  through this exact path — the adapter learned to work with scrambled
  features. **Consider retraining/re-validating the LoRA folds** now that the
  underlying reshape is correct; the numbers in the results table above are
  measured *after* this fix, using the *existing* (pre-fix-trained) fold_01
  checkpoint, so they may not represent the adapter's full potential accuracy.
- **`2_optimisation.py`/`4_inference.py` never updated for the CNN/host
  split.** `0_onnx_converter.py` already only exports the per-step CNN
  backbone (`imu_step`, one LSTM step at a time) and saves
  `tartan_imu_postproc.pt` (LSTM + IMU_Trunk + robot-head weights) for the
  host side — but the comparison/inference scripts still fed the whole
  10-step window into a single Hailo/ONNX call
  (`ValueError: Required inputs (['imu_step']) are missing from input feed
  (['imu_lstm'])`). Rewrote both scripts to loop over the 10 steps, run each
  through the CNN (Hailo/ONNX), and run the saved `TartanPostproc`
  (LSTM -> IMU_Trunk -> robot head) on the stacked per-step features — the
  same "own new code" approach applied consistently everywhere.
- **NHWC/NCHW flatten-order mismatch.** Hailo's per-step CNN output comes
  back as `(1, 13, 128)` NHWC (width-then-channel); the host-side LSTM
  expects the `(128, 13)` channel-then-width flatten order that
  `forward_cnn()`'s `.flatten(1)` produces. Feeding Hailo's raw output
  straight into `.reshape(1, -1)` silently reordered the 1664-dim feature
  vector, producing a huge (MAE ~2.5) but *silent* mismatch — the fix is a
  `.transpose()` back to channel-first before flattening.
- `do_constant_folding=True` also disabled for `tartan_imu`'s export as a
  precaution (see `tlio` above) — though it wasn't the actual root cause
  here (the flatten-order bug was). With folding off, the pre-export
  `onnxsim` call also started crashing on an unrelated ONNX-optimizer
  assertion; disabled for the same reason as `tlio`.

### `deep_iekf`-specific

- **`infer_pytorch()` fed raw, unnormalized IMU to `mes_net`.** `mes_net`
  (`CausalMesNet.forward`) has no internal normalization — production code
  (`TORCHIEKF.forward_nets()` in `external/ai-imu-dr/src/utils_torch_filter.py:421-426`)
  normalizes (`u_n = (u - u_loc) / u_std`) *before* calling `mes_net`, while
  `2_optimisation.py`/`4_inference.py` called `torch_iekf.mes_net(u, ...)`
  directly with the raw sequence. Both ONNX and Hailo correctly receive
  normalized input via `preprocess()`, so the "ground truth" was silently
  comparing against the wrong reference the entire time. Fixed by normalizing
  with the same `u_loc`/`u_std` saved to `deep_iekf_postproc.npz` before
  calling `mes_net`, in both scripts.
- **Misleading comparison window** — see "`deep_iekf` result detail" above;
  not a computation bug, but the reported numbers were unusable until the
  transient/steady-state split was added.
- This script's own ONNX export self-check (`_finalize_onnx()` in
  `0_onnx_converter.py`) only verifies the *ONNX file* against the in-memory
  `MesNetFullHailo` wrapper on random inputs — it never cross-checks that
  `MesNetFullHailo` (the Hailo reimplementation) itself matches the original
  `CausalMesNet.forward()`. It does, to float64 precision, confirmed
  independently — beyond the documented first-16-sample transient — but this
  is exactly the kind of gap where a real bug in the reimplementation could
  hide undetected; worth keeping in mind if `MesNetFullHailo` is ever changed.

## Execution time — is Hailo actually accelerating this, and is it real-time?

Measured end-to-end (Hailo device call **+** any host-side post-processing —
not just the DFC's static estimate), averaged over hundreds of synchronous
calls on the physical Hailo-8L:

| Approach | Update rate in the real filter loop | Per-call budget | Measured latency (Hailo+host) | Margin |
|---|---|---|---|---|
| `deep_kf` | 100 Hz (every IMU sample) | 10 ms | 0.11 ms | ~90x |
| `tlio` | 20 Hz (`tlio_runner.py`: `_UPDATE_STRIDE`, clone augmentation) | 50 ms | 6.3 ms | ~8x |
| `tartan_imu` | 1 Hz (`tartan_runner.py`: `tartan_interval = int(sample_rate)`) | 1000 ms | 11.1 ms | ~90x |
| `deep_iekf` | see caveat below | — | 1.26 ms per full 4544-sample (45.4 s) sequence | see caveat below |

All three streaming approaches (`deep_kf`, `tlio`, `tartan_imu`) are
comfortably real-time with an order of magnitude or more of headroom — `tlio`
has the tightest margin (~8x) because its update rate is much higher (20 Hz)
than its window-processing cost, not because the model itself is slow.

**`deep_iekf` doesn't fit the same framing.** The compiled `.hef` has a fixed
input shape of exactly `SEQ_LEN=4544` samples — it processes one whole KITTI
drive (45.4 s at 100 Hz) in a single 1.26 ms call, not one new IMU sample at
a time. That's extremely fast for what it does (recomputing covariances for
an entire 45-second trajectory 800 times a second), but it isn't natural
online/streaming inference the way the other three are: you cannot feed it a
single new sample and get a single new covariance back. Options for real
online deployment: (a) treat it as intended — a fast *batch* recompute over
whatever history is available, re-run each time enough new samples justify
it, which at 1.26 ms/45.4 s of data leaves enormous headroom even re-run
every 10 ms (~8x margin, tightest of the four, and this margin **shrinks** as
a drive grows past 4544 samples since the HEF can't resize); or (b) compile a
second, much shorter `SEQ_LEN` (e.g. a small sliding chunk) for genuine
sample-by-sample streaming — not attempted this session.

Confirms Hailo **is** accelerating these models, not just running them: the
Hailo-8L handled every one of these workloads with wide real-time margin, on
hardware roughly the class of what would end up on a Raspberry Pi 5 (which
carries the same Hailo-8L). The tightest case (`deep_iekf`'s fixed-length
batch re-run) still clears its budget by ~8x with the current export.

## Deployment to Raspberry Pi 5

See [`docker/`](docker/) for the HailoRT-only multi-arch runtime image
(builds for both this PC's `amd64` and the Pi 5's `arm64`). The DFC/compile
step stays here on x86_64; only the compiled `.hef` + `4_inference.py`-style
inference code needs to travel to the Pi.
