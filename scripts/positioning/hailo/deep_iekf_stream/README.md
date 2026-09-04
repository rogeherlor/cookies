# Deep-IEKF streaming HEFs (W=32) — genuine online covariance predictor

Companion to `../deep_iekf/` (the fixed **4544-sample whole-sequence** HEF).
That HEF has one fixed input length, so it can only cover `min(N, 4544)` of a
real KITTI sequence — **8–40 %** of most drives (seq08 = 8 %, seq00 = 10 %) — and
the runtime *raises* if `N ≠ 4544`. It is not a viable online deployment.

These HEFs compile the **same causal MesNet** at a small fixed window
`W = SEQ_LEN = 32` samples and are driven over the whole sequence one window at a
time, so they process **any length** and are genuinely online:

* **per-tick** — slide by 1 each new IMU sample, keep the last output
  (one HEF call per sample, ~1-tick latency). The real online mode.
* **block**    — slide by `FRESH = 16`, feed 32 (= 16 carried context + 16 new),
  keep the last 16 outputs (16 covariances per HEF call). Amortised; default.

Both are bit-faithful to the whole-sequence pass in steady state (receptive
field = 17, so an output keeps only its own 16 samples of real left context;
proven to ≤ 2e-6 in `verify_chunk_math.py`). Only the first ~16 samples of the
whole stream differ (ZeroPad warmup — identical to the 4544 HEF).

## Per-fold (LOO)

One HEF per leave-one-out fold — the covariance network weights (and input
normalisation) are baked in, so each held-out benchmark sequence uses its own
fold, exactly like the 7 CPU `artifacts/deep_iekf_online/fold_XX.p` weights:

```
deep_iekf_stream_fold_{01,04,06,07,08,09,10}.hef
deep_iekf_stream_fold_{01,04,06,07,08,09,10}_postproc.npz
```

`_full_eval_worker.py` auto-selects `deep_iekf_stream_fold_<seq>.hef` per
sequence when present (else falls back to the 4544 HEF).
`DEEP_IEKF_STREAM_MODE=per_tick|block` (default `block`) picks the drive mode.

## On-device output is a bounded z, not the scaled covariance

The HEF emits `z = tanh(cov_lin(cov_net(u))) ∈ [-1,1]`; the host reconstructs
`cov = cov0 · 10**(beta · z)`. `deep_iekf_stream_fold_XX_postproc.npz` carries
`u_loc, u_std` (per-fold input normalisation) **and** `beta, cov0` (the host
scaling constants). `HailoDeepIEKFStream` does this reconstruction.

Baking the `cov0·10**(beta·z)` exponential on-device, as the 4544 HEF does, makes the
HEF emit a heavy-tailed 0…~10000 range that INT8 quantises coarsely. Folds whose
covariances spike suffer badly: fold_06 (cov_up up to ~8800) drifts 153 m. A bounded
`z` quantises uniformly for every fold, and fold_06 then reproduces the float
trajectory to 2.9 cm.

## Validation (SDK_QUANTIZED emulator vs float64 CausalMesNet, end-to-end IEKF ATE)

Each sequence driven through the real IEKF with its own fold's streaming HEF,
no outage, horizontal-RMSE vs the float64 Deep-IEKF trajectory:

| seq | fold | ATE stream-vs-float-ref | stream-vs-KITTI (float ref) |
|-----|------|-------------------------|-----------------------------|
| 04  | 04   | 0.003 m                 | 0.399 m (0.401) |
| 07  | 07   | 0.010 m                 | 0.313 m (0.315) |
| 09  | 09   | 0.021 m                 | 1.733 m (1.734) |
| 01  | 01   | 0.025 m                 | 6.203 m (6.197) |
| 10  | 10   | 0.029 m                 | 0.638 m (0.630) |
| 06  | 06   | 0.029 m                 | 2.133 m (2.126) |
| 08  | 08   | 0.855 m                 | 3.496 m (3.822) |

So the streaming HEF reproduces the CPU Deep-IEKF result to cm level on 6/7
sequences (sub-metre on the 537 s seq08), with absolute accuracy matching the
float reference on every sequence.

## REAL Hailo-8L hardware confirmation (not just the SDK emulator)

This dev PC has a physical Hailo-8L M.2 module on its PCIe bus (`lspci`: "Hailo-8
AI Processor"; `hailortcli fw-control identify` inside the DFC container reports
firmware 4.20.0, Device Architecture HAILO8L — the same chip/firmware documented
for the Pi). `verify_real_device.py` runs `hailo_backend.HailoDeepIEKFStream`
(genuine HailoRT calls, not `SDK_QUANTIZED`) over full real KITTI sequences, per
fold, via `docker run --privileged -v /dev:/dev ...` device passthrough:

| seq | fold | covariance MAE (real HW) | real-device calls | mean ms/call | wall time | real-time× | **ATE real-HW-vs-float-ref** |
|-----|------|--------------------------|--------------------|--------------|-----------|------------|-------------------------------|
| 04  | 04   | 0.062 | 186  | 0.133 ms | 35 ms  | 847× | **0.003 m** |
| 07  | 07   | 0.201 | 723  | 0.136 ms | 143 ms | 808× | **0.010 m** |
| 01  | 01   | 0.068 | 763  | 0.146 ms | 164 ms | 744× | **0.025 m** |
| 10  | 10   | 0.557 | 798  | 0.139 ms | 162 ms | 787× | **0.029 m** |
| 09  | 09   | 0.187 | 1038 | 0.144 ms | 218 ms | 762× | **0.021 m** |
| 06  | 06   | 24.36 ⚠️ | 711 | 0.138 ms | 141 ms | 808× | **0.030 m** |
| 08  | 08   | 0.220 | 3354 | 0.143 ms | 699 ms | 768× | **0.894 m** |

`mode=block` (16 fresh covariances/call). **fold_06's raw covariance MAE is an
outlier on real silicon** (24.36 vs the SDK emulator's 16.78) — its trained
network outputs unusually heavy-tailed covariances (`cov_up` up to ~8800), so
even the bounded-`z` INT8 quantisation is coarser there than for other folds.
**This does NOT propagate into the trajectory**: end-to-end IEKF ATE using the
*real-device* covariances is 0.030 m, matching the SDK-emulator prediction
(0.029 m) almost exactly — the IEKF's own covariance weighting absorbs the
larger per-sample noise-estimate error without moving the state estimate.
Verified with `run_end2end_realdevice.py` (real-device covariance dumps in
`covs_dump/covs_realdevice_*.npz`).

**Real per-call latency (0.13–0.15 ms) is ~15–20× the DFC's static estimate**
(~0.01 ms) — the estimate ignores fixed per-call DMA/activation overhead. Even
so, per-tick throughput is 700–850× real-time; the CPU IEKF loop itself
(~0.2–0.3 ms/sample, separately measured) is the actual bottleneck, and total
per-tick cost is still a small fraction of the 10 ms/sample budget at 100 Hz.

## Rebuild

```bash
# inside the hailo_ai_sw_suite DFC container (x86), for each fold:
python3 build_stream.py --weights ../../../../artifacts/deep_iekf_online/fold_01.p \
        --window 32 --calib-seq 00 --tag fold_01
```

Real-hardware validation (requires a physical Hailo-8L, e.g. via
`docker run --privileged -v /dev:/dev ...`):
```bash
python3 verify_real_device.py           # per-fold accuracy + genuine device latency
```
