# Runbook — final positioning evaluation on the Raspberry Pi 5

## Why anything runs on the Pi at all

Accuracy is host-independent: the same `.hef` on the same Hailo-8L at the same
firmware gives bit-identical outputs, and that was confirmed on the build box
(also `HAILO8L`, fw `4.20.0`). **Timing is not.** Every latency, real-time
factor and speedup is a property of the host CPU and its PCIe link, so those
numbers must be measured here and nowhere else.

`build_latex_tables.py` refuses to build a table from rows measured on more than
one architecture, so a half-and-half sweep cannot silently reach the paper.

---

## 1. What to copy

From the build machine, preserving paths:

```
artifacts/tlio/                 fold_<seq>.pt                      (7 folds)
artifacts/deep_kf/              fold_<seq>.pt                      (7)
artifacts/tartan_imu/           lora_fold_<seq>.pt                 (7)
artifacts/deep_iekf_online/     fold_<seq>.p, fold_<seq>_norm.p    (7 + 7)

scripts/positioning/hailo/tlio/          tlio_fold_<seq>.hef
                                         tlio_postproc_fold_<seq>.pt
scripts/positioning/hailo/deep_kf/       deep_kf_fold_<seq>.hef
scripts/positioning/hailo/tartan_imu/    tartan_imu_fold_<seq>.hef
                                         tartan_imu_postproc_fold_<seq>.pt
scripts/positioning/hailo/deep_iekf_stream/
                                         deep_iekf_stream_fold_<seq>.hef
                                         deep_iekf_stream_fold_<seq>_postproc.npz

scripts/positioning/hailo/full_benchmark_results/gt_cache/<seq>.npz   (7)

external/tlio/          external/ai-imu-dr/          (imported by the runners)
external/tartan_imu/checkpoints/foundation_model/    (frozen base model)
datasets/raw_kitti/     scripts/   config/
```

seq ∈ {01, 04, 06, 07, 08, 09, 10}. Expect **28 `.hef`** and **21 postproc**
files in total.

The `_postproc` files are **not optional and not interchangeable between
folds**: for `tlio` and `tartan_imu` the accelerator holds only the backbone and
the head runs on the host from those weights, exported from the same fold
checkpoint. `_fold_hef()` refuses a `.hef` whose matching postproc is missing.

Do **not** copy `full_benchmark_results/*.json` — the sweep must be measured
fresh here. Keep `gt_cache/`.

## 2. Prerequisites on the Pi

```bash
# arm64 runtime container with HailoRT and /dev/hailo0 (see docker/)
docker start cookies_bench
docker exec cookies_bench hailortcli fw-control identify | grep -E "Architecture|Firmware"
#   expect: HAILO8L / 4.20.0   — must match what the .hef was compiled for
docker exec cookies_bench python3 -c "import pymap3d, torch, numpy"
ls /opt/conda-gtsam/bin/python3        # needed ONLY for the isam2 rows
```

The Dataflow Compiler is x86-only and is **not** needed here — nothing on the Pi
compiles.

## 3. Ground-truth cache

Required; the sweep raises without it. If you copied `gt_cache/` you can skip.

```bash
cd scripts/positioning/hailo
python3 _precompute_batch_gt.py        # needs /opt/conda-gtsam
ls full_benchmark_results/gt_cache/    # 01 04 06 07 08 09 10 .npz
```

## 4. Quiet the machine

Timings here are sub-millisecond in places, so scheduling noise is not ignorable.

1. **Nothing else running.** `ps aux`; stop editors, browsers, background jobs.
   The worker samples the load average before it starts and warns if the machine
   is not idle.
2. **Single-threaded BLAS** — automatic (`_full_eval_worker.py` sets it before
   numpy/torch import). Do not override.
3. **Thermals** — `vcgencmd measure_temp`, `vcgencmd get_throttled`. A throttled
   board silently inflates everything measured after it.

## 5. Run the sweep — once, whole, in one go

```bash
cd scripts/positioning/hailo
rm -f full_benchmark_results/*.json          # keep gt_cache/
BENCH_TASKSET=1,2,3 python3 run_full_benchmark.py
```

`BENCH_TASKSET` pins every job — CPU and Hailo alike — to the same cores,
leaving core 0 for the OS. Both this script and `_hailo_rerun.sh` read it, so
the two halves of a table are measured under identical affinity.

238 jobs. To re-run only the Hailo rows later, use the same variable:
`BENCH_TASKSET=1,2,3 ./_hailo_rerun.sh`.

## 6. Check the run before trusting it

```bash
python3 - <<'PY'
import json, pathlib
rows = json.loads(pathlib.Path("full_benchmark_results/all_results.json").read_text())
bad = [r for r in rows if (r['timing'].get('wall_cpu_ratio') or 0) > 1.25
       and r['timing'].get('wall_cpu_ratio_gated')]
print(f"{len(rows)} rows, {len(bad)} contaminated")
for r in bad:
    print(" ", r['approach'], r['backend'], r['scenario'], r['seq'],
          round(r['timing']['wall_cpu_ratio'], 2))
print("architectures:", {(r.get('host') or {}).get('machine') for r in rows})
PY
```

* `wall_cpu_ratio` near 1.0 on a CPU-backed row is positive evidence the
  measurement was clean. Above 1.25 the worker already warned — re-measure that
  row rather than reporting it.
* On **Hailo** rows the ratio is legitimately >1 (the thread blocks on the
  device) and is recorded but not gated (`wall_cpu_ratio_gated: false`). The
  `cpu_median_ms` beside it is the host share, which is what separates "waiting
  on the accelerator" from "waiting on another process".
* `architectures` must be a single value.

## 7. Build the tables

```bash
python3 build_latex_tables.py
ls ../../../outputs/tables_hailo/
```

Every table, accuracy and timing alike, comes from
`full_benchmark_results/all_results.json` alone. The script prints a provenance
line and aborts if the rows mix architectures.

Then sync the numbers into `3.tex` / `4.tex` — the tables are inline there, not
`\input`, so they are edited by hand.

## 8. Known gaps

* **Deep IEKF protocol.** If it was copied before its retrain finished, it uses
  the upstream protocol (trains on all remaining drives, deploys the final
  epoch) while the other three use a nested 5/1/1 split with best-J selection.
  This is documented in 4.tex §"Differences with respect to the original
  formulations". Its CPU weights and its `.hef` must come from the *same*
  generation — mixing them measures the protocol change, not quantisation.
* **isam2 / isam2_fixedlag** need `/opt/conda-gtsam`; without it those rows are
  absent and the tables show them blank.
* **Sequence 04** is 29.7 s, shorter than the 40 s outage start, so its outage
  row is blank by construction for every filter.

## Troubleshooting

**"no per-fold HEF for \<model\> seq \<n\>"** — that fold's `.hef` or its
`_postproc` was not copied. The run continues on a non-LOO binary and its
accuracy is meaningless; fix the files, not the warning.

**Tables abort with "more than one architecture"** — the results mix the build
host and the Pi. Delete `full_benchmark_results/*.json` (keep `gt_cache/`) and
re-run the whole sweep here.

**deep_kf HEF has 1 input** — that is the old memoryless export. The current
graph is a stateful cell with five inputs (`x, h_l0, c_l0, h_l1, c_l1`); the
backend raises rather than run it. Re-copy the binaries.
