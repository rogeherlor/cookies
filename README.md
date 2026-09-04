# Cookies — INS/GNSS positioning with deep filters on an edge accelerator

Vehicle positioning on the KITTI raw dataset, comparing classical inertial/GNSS
filters, factor-graph smoothers, and four deep-learning filters — then deploying the
deep filters onto a Hailo-8L accelerator attached to a Raspberry Pi 5 and measuring
what quantisation and the accelerator cost in accuracy and latency.

Thirteen estimators are evaluated on seven KITTI drives, with and without a 60 s GNSS
outage:

| Group | Estimators |
|---|---|
| Classical | ES-EKF (Groves), ES-EKF (Solà), left-invariant EKF — each vanilla and with NHC + ZUPT — plus IMU-only dead reckoning |
| Smoothers | iSAM2, iSAM2 fixed-lag (GTSAM) |
| Deep | Deep IEKF (AI-IMU, Brossard 2020), TLIO (Liu 2020), Deep KF (Hosseinyalamdary 2018), Tartan IMU (Zhao 2025) |

The four deep filters each run twice: once on CPU, once with the network on the
Hailo-8L.

## Layout

```
scripts/positioning/python/     filters, DL filters, smoothers, training and evaluation
  filters/                      the six classical filters + imu_only
  dl_filters/                   deep_iekf, tlio, deep_kf, tartan_imu — models and trainers
  smoothers/                    iSAM2 and FGO batch runners (GTSAM)
  ins_genetic_cv.py             genetic tuning of the classical filters
  ins_train.py                  DL training orchestrator across LOO folds
  ins_compare.py                runs every filter on one sequence and one outage window
  emit_journal_tables.py        per-run JSONs -> outputs/tables/*.tex
scripts/positioning/hailo/      ONNX export, quantisation, compilation, on-device benchmark
  <model>/0..4_*.py             per-model 5-stage pipeline (export -> parse -> optimise -> compile -> infer)
  build_per_fold_hefs.py        builds all 28 per-fold .hef binaries
  run_full_benchmark.py         the full sweep: 13 filters x 7 seqs x 2 scenarios
  build_latex_tables.py         sweep results -> outputs/tables_hailo/*.tex
  figures/                      trajectory and ground-truth figure generators
  docker/                       HailoRT runtime image (amd64 + arm64)
scripts/positioning/legacy/     the original MATLAB and TensorFlow code this grew out of
scripts/positioning/c/ekf/      the EKF in C, shared with the firmware
scripts/utils/                  serial capture and coordinate-conversion helpers
simplicity/                     EFR32 firmware for the sensor board (see its README)
datasets/                       KITTI raw extracts (.p) and the legacy .mat drives
artifacts/                      trained checkpoints, one per LOO fold
outputs/                        generated LaTeX tables and comparison plots
tests/                          pytest suite over the filters and data loader
docs/                           tuning, training, and the open-questions list
```

## Setup

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

GTSAM must come from conda, not pip — the pip wheel segfaults against numpy ≥ 2.0:

```bash
conda install -c conda-forge gtsam
```

`datasets/raw_kitti/` holds the KITTI raw drives already converted to pickles. The
`external/` submodules provide the AI-IMU, TLIO and Tartan IMU upstream code — source
only, no weights.

The AI-IMU acausal checkpoint is tracked in this repo at
`artifacts/deep_iekf/iekfnets.p`, so a clone gets it. It drives the diagnostic batch
filter only; the Deep IEKF that is actually evaluated is the causal variant and must be
trained (stage 2).

**The Tartan IMU foundation model does not.** `external/tartan_imu` is a HuggingFace
dataset repository that keeps its checkpoints in Git LFS, so a plain
`git submodule update` gives you ~130-byte pointer files rather than weights. Fetch the
real ones either way round:

```bash
# either: let git-lfs resolve the pointers already checked out
git lfs install && git -C external/tartan_imu lfs pull

# or: download the snapshot directly (no git-lfs needed)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('raphael-blanchard/TartanIMU', repo_type='dataset', \
                    local_dir='external/tartan_imu')"
```

Either route lands the checkpoints in
`external/tartan_imu/checkpoints/foundation_model/`, which is where the runner looks.
Running the filter without them raises an error naming the download command, so a
missing model fails loudly rather than silently falling back.

The Hailo work needs two containers, because the toolchain is split by architecture:
the Dataflow Compiler (stages 1–4 below) is x86_64-only, while HailoRT (stage 5) runs
on both. Build instructions for the runtime image are in
[scripts/positioning/hailo/docker/README.md](scripts/positioning/hailo/docker/README.md).
HailoRT and the on-device firmware must be the same version — 4.20.0 here; check with
`hailortcli fw-control identify`.

## What a clone gives you

| | In the repo | How to get it |
|---|---|---|
| KITTI drives (7 clean folds) | yes | `datasets/raw_kitti/*.p`, tracked |
| Tuned classical parameters | yes | `filter_params.json`, tracked — stage 1 is optional |
| AI-IMU acausal weights | yes | `artifacts/deep_iekf/iekfnets.p`, tracked |
| Sample DL checkpoint | fold 01 only | the other six folds come from stage 2 |
| Tartan IMU foundation model | no | Git LFS or `snapshot_download` — see Setup |
| Trained DL checkpoints | no | stage 2 (or copy from a machine that has them) |
| Compiled `.hef` binaries | no | stage 4, x86_64 host with the DFC |
| Ground-truth cache | no | `_precompute_batch_gt.py`, stage 5 |
| Result tables and figures | no | stages 3 and 5 |
| Hailo DFC image, HailoRT x86 wheel | no | gated downloads from the Hailo Developer Zone |

So evaluation of the classical filters and smoothers runs from a clone with no training
step: the tuned parameters and the KITTI data are both tracked. The deep filters need
stage 2 first, and anything touching the accelerator needs Hailo's toolchain.

## Reproducing the results

Seven clean KITTI sequences are used: `01 04 06 07 08 09 10`. Sequences `00`, `02` and
`05` have ~2 s data gaps and `03` has no raw data. Evaluation runs twice per sequence:
no outage, and a 60 s GNSS outage starting at 40 s. Sequence 04 is only 29.7 s long, so
it has no outage row.

Everything is leave-one-out: seven folds, and the held-out drive is never loaded during
training or tuning.

### 1. Tune the classical filters

```bash
cd scripts/positioning/python
./run_genetic_loo.sh
```

Differential evolution against the normalised cost in `ins_cost.py`, one search per
held-out sequence. Writes `filter_params.json`. Budget 6–10 hours per fold; seven folds
is an overnight run. Details in [docs/tuning.md](docs/tuning.md).

### 2. Train the deep filters

```bash
./run_dl_training.sh                     # sequential
./run_dl_training_parallel.sh            # folds in parallel — same results, less wall time
```

Produces `artifacts/tlio/fold_<SEQ>.pt`, `artifacts/deep_kf/fold_<SEQ>.pt`,
`artifacts/tartan_imu/lora_fold_<SEQ>.pt` and `artifacts/deep_iekf_online/fold_<SEQ>.p`
— 28 checkpoints. Training is outage-free by design; outages are simulated only at
evaluation. Details in [docs/training.md](docs/training.md).

### 3. Evaluate on CPU

```bash
./run_journal_evaluation.sh
python emit_journal_tables.py
```

Runs `ins_compare.py` once per (sequence × regime), then aggregates the per-run JSONs
into `outputs/tables/no_outage_kitti.tex` and `outputs/tables/outage_60s_kitti.tex`.

### 4. Compile the Hailo binaries — x86_64 host

```bash
cd scripts/positioning/hailo
python3 build_per_fold_hefs.py
```

Inside the `hailo_ai_sw_suite_2025-01` image. Produces 28 `.hef` (one per model per
fold) and 21 postproc files. For TLIO and Tartan IMU the accelerator holds only the
backbone and the head runs on the host from those weights, so a `.hef` and its
fold-matched postproc file must travel together. To step through one model by hand
instead, run its `0_onnx_converter.py` … `4_inference.py` in order — see
[scripts/positioning/hailo/README.md](scripts/positioning/hailo/README.md).

### 5. Benchmark on the Raspberry Pi 5

```bash
python3 _precompute_batch_gt.py          # once — the sweep raises without it
BENCH_TASKSET=1,2,3 python3 run_full_benchmark.py
python3 build_latex_tables.py
```

`_precompute_batch_gt.py` builds the FGO-Batch ground-truth trajectory each sequence is
scored against, and needs the `/opt/conda-gtsam` interpreter that the runtime image
provisions. It is independent of filter, backend and outage window, so it runs once and
the cache is reused.

Accuracy is host-independent: the same `.hef` on the same Hailo-8L at the same firmware
gives bit-identical outputs. Timing is not — every latency, real-time factor and
speedup is a property of the host CPU and its PCIe link, so those numbers only count
when measured on the Pi. `build_latex_tables.py` refuses to build a table from rows
measured on more than one architecture. Writes `outputs/tables_hailo/*.tex`.

Copy lists, prerequisites and the throttling checks are in
[scripts/positioning/hailo/RUNBOOK_PI.md](scripts/positioning/hailo/RUNBOOK_PI.md).

## Where the numbers live

`outputs/` and `scripts/positioning/hailo/full_benchmark_results/` are not tracked —
they are regenerated by stages 3 and 5. Trained checkpoints under `artifacts/` are not
tracked either, apart from one fold kept as a sample. Re-running the sweep is both
necessary and sufficient to refresh every table: after retraining, nothing in
`outputs/` needs touching by hand, but the sweep does have to run again or the tables
keep the old trajectory metrics.

Figures for the sequence-08 comparison come from `scripts/positioning/hailo/figures/`.

## Tests

```bash
pytest                  # everything
pytest -m "not slow"    # skip the slow ones
```

The suite covers the data loader, rotation conventions, the invariant-EKF properties,
and the NHC/ZUPT constraints. Markers are declared in `pytest.ini`.

## Further reading

- [docs/tuning.md](docs/tuning.md) — genetic tuning of the classical filters
- [docs/training.md](docs/training.md) — the four DL filters, fold by fold
- [docs/open-questions.md](docs/open-questions.md) — methodological caveats and what is still unverified
- [docs/hailo-dfc-notes.md](docs/hailo-dfc-notes.md) — working notes on the Hailo Dataflow Compiler
- [scripts/positioning/hailo/README.md](scripts/positioning/hailo/README.md) — the deployment pipeline
- [scripts/positioning/hailo/deep_iekf_stream/README.md](scripts/positioning/hailo/deep_iekf_stream/README.md) — the streaming Deep IEKF export
- [simplicity/README.md](simplicity/README.md) — the sensor-board firmware

## Licence

MIT — see [LICENSE](LICENSE).
