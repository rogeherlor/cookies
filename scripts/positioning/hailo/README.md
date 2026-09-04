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

`tartan_imu` takes `--artifact <repo>/artifacts/tartan_imu/lora_fold_01.pt` on
both `0_onnx_converter.py` and `2_optimisation.py`, mirroring `tlio`'s flag.
Without it, both the CNN export and the PyTorch reference silently fall back to
the zero-shot base model, which will never match a `.hef` compiled from a
specific fold. (The `TARTAN_IMU_LORA` env var still works as an escape hatch,
but `--artifact` is what `build_per_fold_hefs.py` uses and what these scripts
document — an env var whose name has to match a *different* module's spelling
is too easy to get silently wrong.)

`deep_iekf` defaults to `artifacts/deep_iekf_online/iekfnets.p` — **if a
training run is writing to that folder, pass an explicit, already-completed
fold instead** (`--weights <repo>/artifacts/deep_iekf_online/fold_01.p`, or
`fold_04.p`/`fold_06.p`/`fold_07.p`, whichever are already finished) on
`0_onnx_converter.py`, `2_optimisation.py`, and `4_inference.py` alike — the
default path is exactly the file the training loop is actively overwriting.

## Per-LOO-fold compilation (required)

**There is no single `<model>.hef`.** Each of the four models is compiled once
per leave-one-out fold, and the evaluator loads the fold matching the sequence
under test.

This is not a refinement — it is the difference between measuring quantisation
and measuring train/test leakage. The Hailo side originally shipped one
`tlio.hef` / `deep_kf.hef` / `tartan_imu.hef` used for all seven sequences,
traced to `fold_01`. On sequence 06 that gave ATE 5.3955 where the LOO-correct
`fold_06` weights give 7.2033; the "Hailo beats CPU" result on five of six
sequences was the model recognising data it had trained on. The CPU backends
never had this problem — `tlio_runner._find_weights` raises rather than
substitute a fold — so the two columns were not comparable.

```bash
python3 build_per_fold_hefs.py --dry-run     # inspect the plan
python3 build_per_fold_hefs.py               # 21 builds (3 models x 7 folds)
python3 build_per_fold_hefs.py --models tlio --folds 06     # rebuild just one
```

Produces, for seq in {01,04,06,07,08,09,10}:

```
tlio/tlio_fold_<seq>.hef              + tlio/tlio_postproc_fold_<seq>.pt
deep_kf/deep_kf_fold_<seq>.hef        (no postproc — whole net is on-device)
tartan_imu/tartan_imu_fold_<seq>.hef  + tartan_imu/tartan_imu_postproc_fold_<seq>.pt
deep_iekf_stream/deep_iekf_stream_fold_<seq>.hef + ..._postproc.npz  (built by build_stream.py)
```

### The postproc file is half the model

For `tlio` and `tartan_imu` the accelerator holds only the backbone; the head
runs on the host (`tlio`: bn1 + fc1/2/3, `tartan`: LSTM + Trunk + robot head)
from `<model>_postproc_fold_<seq>.pt`, which stage 0 writes **straight from the
fold checkpoint**. Those weights are exactly as fold-specific as the backbone.
Pairing a per-fold `.hef` with a single shared postproc would run fold F's
backbone into whichever fold's head was exported last — a second leak, and a
numerically incoherent model on top. `_fold_hef()` therefore requires **both**
files and treats a lone `.hef` as "not built".

### Why the driver wipes intermediates

Stages 1-3 hardcode canonical filenames (`tlio.onnx` -> `tlio_hailo_model.har`
-> `tlio.hef`), so the driver runs the normal pipeline once per fold and moves
the result aside. That makes stale files dangerous: if stage 2 failed and
exited 0, stage 3 would compile the **previous fold's** quantized HAR and the
driver would report success. Two guards close this:

* every `2_optimisation.py` now re-raises on a quantisation failure instead of
  warning and exiting 0;
* the driver deletes all intermediates, **including the fold being rebuilt**,
  before each fold. A missing file fails loudly; a stale one does not.

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
| `deep_iekf`  | ⚠️ superseded | 5.01 steady-state (excl. first 20 samples — see below) | Fixed `SEQ_LEN=4544` input — covers only `min(N,4544)` of a real sequence (8–40% of most KITTI drives, `N≠4544` raises). Not a deployable online path; kept for history. **Use `deep_iekf_stream/` instead.** |
| `deep_iekf` (**streaming, `deep_iekf_stream/`**) | ✅ validated, **on real Hailo-8L, all 7 LOO folds** | 0.003–0.855 m end-to-end trajectory ATE vs float reference (per-fold, real device) | Fixed `SEQ_LEN=32`, driven per-tick or block-16 over any sequence length — genuinely online. See `deep_iekf_stream/README.md`. |

All four original approaches ran the **entire** pipeline (steps 0-4) on the
physical Hailo-8L, not just the DFC's software emulator. `deep_iekf` was
tested against `artifacts/deep_iekf_online/fold_01.p` — a completed LOO fold —
while a training run for a *different* held-out fold continued untouched in
the background throughout. **`deep_iekf`'s fixed 4544-sample whole-sequence
HEF was later found unusable as an online deployment** (see below) and
replaced by `deep_iekf_stream/` (small fixed window, driven over the whole
sequence one window at a time) — validated end-to-end, per LOO fold, on this
PC's physical Hailo-8L card (firmware 4.20.0, same as the Pi's).

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

**This whole-sequence 4544-sample HEF was later found unusable for real
deployment**: its fixed input length only covers `min(N,4544)` of an actual
KITTI sequence (8–40% of most drives; the runtime raises outright if
`N≠4544`), and it cannot be driven per-tick as new IMU samples arrive. See
`deep_iekf_stream/README.md` for the small-fixed-window (`SEQ_LEN=32`)
replacement, validated end-to-end on real Hailo-8L hardware across all 7 LOO
folds, that supersedes it.

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

## Timing measurement protocol

The per-event timings in the thesis tables are only valid if the run was not
competing with anything else on the board. An earlier sweep that ignored this
reported figures inflated by 2-4x, and the error was not uniform: it changed a
verdict (TLIO read 64.51 ms against its 50 ms budget and appeared to FAIL,
where a clean measurement gives 15.96 ms and PASSes) and it invented a
difference between the outage and no-outage scenarios that does not exist
(ES-EKF Groves' outage event median read 0.48 +/- 0.56 ms against a true
0.23 +/- 0.02 ms). A GNSS correction cannot cost more during a GNSS outage —
the outage removes events, it does not make each one more expensive.

Rules, in order of importance:

1. **One measurement process at a time.** `run_full_benchmark.py` already runs
   its jobs sequentially, but nothing stops a second sweep, an editor's
   language server, or a forgotten background job from running alongside it.
   Check with `ps aux` before starting and after finishing.
2. **Single-threaded BLAS.** `_full_eval_worker.py` now sets
   `OMP_NUM_THREADS=1` and friends at import time, before numpy/torch load, so
   this is automatic. Do not override it. Multi-threaded BLAS is slower here,
   not faster: the tensors are small enough that thread fan-out/join costs more
   than the arithmetic.
3. **Pin the cores, identically for every row.** Set `BENCH_TASKSET` once and
   both `run_full_benchmark.py` and `_hailo_rerun.sh` apply it to every job:

   ```bash
   BENCH_TASKSET=1,2,3 python3 run_full_benchmark.py    # leaves core 0 for the OS
   BENCH_TASKSET=1,2,3 ./_hailo_rerun.sh
   ```

   This used to be split — `_hailo_rerun.sh` hardcoded `taskset -c 1,2,3` while
   `run_full_benchmark.py` pinned nothing — so the Hailo and CPU halves of the
   same table were measured on different effective machines with nothing in the
   output recording it. Every result JSON now carries `n_cpus_affinity` and the
   1-minute load average before and after the run, so an inconsistency is
   visible after the fact instead of invisible. Pinning alone is not sufficient
   — it stops migration, not preemption.
4. **Check the board is cool and idle first.** `vcgencmd measure_temp` and
   `vcgencmd get_throttled`. A long prior sweep can leave it thermally
   throttled, which silently inflates everything that follows.

### The self-check

Every run records thread CPU time beside wall-clock and emits `wall_cpu_ratio`
in its JSON. CPU time counts only cycles actually retired on the thread, so a
run that was descheduled shows wall >> cpu. **A ratio near 1.0 is positive
evidence the measurement is clean; the worker prints a warning above 1.25.** A
clean full sweep of the seven classical filters holds
`max |wall/cpu - 1| = 0.033`.

This now covers **all eleven CPU-backed filters**, not just the seven classical
ones. The four DL runners emit `net_cpu_s` alongside `net_latency_s`; before
that, a preempted TLIO/DKF/Tartan/Deep-IEKF CPU run was reported as a genuine
latency with nothing to contradict it — the exact failure the check exists to
catch, left uncovered on the rows most likely to be quoted.

On the **Hailo** rows the same field is recorded but the warning is gated off
(`wall_cpu_ratio_gated: false` in the JSON). There the host thread really is
blocked on the accelerator, so wall >> cpu is the expected offload signature
rather than contamination — but having the number still lets you separate
"waiting on the device" from "waiting on some other process", which a bare
wall-clock figure cannot do.

A **pre-run load check** runs before any timed work: if the 1-minute load
average already exceeds half the assigned core count, the worker warns that the
machine is not idle. This catches interference that starts before the run and
so never shows up in a per-event ratio.

### Statistics

Event counts differ by three orders of magnitude between filters (about 120k
for a per-sample update, 30-540 for a 1 Hz GNSS correction). A 95th percentile
over ~56 events is roughly the third-largest sample and is dominated by
scheduler noise; pool events across sequences before quoting a tail statistic.
