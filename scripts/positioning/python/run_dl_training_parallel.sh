#!/usr/bin/env bash
# run_dl_training_parallel.sh — LOO training for the DL filters, N folds at once.
#
# Trains exactly what run_dl_training.sh / ins_train.py train: same scripts, same
# epochs, same nested LOO split (dl_filters/_validation.py::inner_split). Only the
# scheduling differs — folds are independent processes writing to distinct
# fold_<seq>.pt files, so concurrency changes the wall time and nothing else.
#
# It is worth it: Deep KF is ~0.85 min/epoch x 150 epochs ~= 2 h per fold, and a single
# job leaves the GPU at ~32 % because the LSTM loop is host-bound. Sequentially that is
# ~15 h for Deep KF alone; at concurrency 7, ~2 h.
#
# Usage:
#   ./run_dl_training_parallel.sh                      # all 3 models, 7 folds
#   ./run_dl_training_parallel.sh tlio                 # one model
#   JOBS=4 ./run_dl_training_parallel.sh               # cap concurrency
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACTS="${REPO_ROOT}/artifacts"
PY="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-7}"
# Per-model concurrency cap, in host RAM — NOT GPU memory.
#
# This exists because sizing the fan-out against the GPU (16 GB, ~0.6 GB/job)
# and ignoring system RAM took the machine down. deep_iekf's MultiSequenceDataset
# holds every clean KITTI drive as float64 torch tensors in EACH worker: the
# kernel's own OOM record put one such process at 4.24 GB resident, so seven of
# them need ~29 GB on a 31 GB box. The OOM killer took gnome-shell, VS Code,
# Chrome and finally one of the trainers. tlio/deep_kf/tartan are an order of
# magnitude lighter and are fine at the default.
declare -A MAX_JOBS=( [tlio]=7 [deep_kf]=7 [tartan_imu]=7 [deep_iekf]=2 )
# Free RAM required before another fold is allowed to start, per model (MB).
declare -A MIN_FREE_MB=( [tlio]=4000 [deep_kf]=4000 [tartan_imu]=4000 [deep_iekf]=9000 )
SEQS="${SEQS:-01 04 06 07 08 09 10}"
LOGDIR="${REPO_ROOT}/logs/dl_training"

# Epoch counts must match ins_train.py's defaults or the runs are not
# comparable to each other or to what the paper reports.
# TLIO's budget was 50. Upstream (external/tlio/src/main_net.py) does not cap at
# 50: it runs to convergence (max 10000, ReduceLROnPlateau patience 10, best-val
# checkpoint). The 50-epoch cap was defended on the grounds that longer schedules
# collapse log-sigma^2 and disable the SCEKF Mahalanobis gate — but that failure
# mode is now caught by selection, which scores the closed-loop journal metric on
# the inner-validation drive and simply would not pick such an epoch. Under
# J-selection a larger budget is strictly more search, not more overfitting risk,
# so the cap is raised to close the fidelity gap with upstream.
declare -A EPOCHS=( [tlio]=200 [deep_kf]=150 [tartan_imu]=50 [deep_iekf]=400 )
# How often to score the journal metric J on the inner-validation sequence.
# J is the SELECTION criterion (see the trainers), so this is the granularity at
# which the deployed epoch can be chosen — not a logging interval. Each
# evaluation closes the loop over a whole sequence with the 40s/60s outage, so
# it costs real time; these values keep that under ~20% of training while still
# sampling the run densely enough to catch the minimum.
declare -A VAL_EVERY=( [tlio]=4 [deep_kf]=5 [tartan_imu]=2 [deep_iekf]=20 )
declare -A SCRIPTS=(
  [tlio]="dl_filters/tlio/train_tlio.py"
  [deep_kf]="dl_filters/deep_kf/train_deep_kf.py"
  [tartan_imu]="dl_filters/tartan_imu/train_tartan.py"
  [deep_iekf]="dl_filters/deep_iekf/train_ai_imu.py"
)
# deep_iekf takes --held-out <drive> rather than --val-seq <id>, and writes to
# artifacts/deep_iekf_online/ (the causal model's folder).
declare -A SEQ_FLAG=( [tlio]="--val-seq" [deep_kf]="--val-seq" [tartan_imu]="--val-seq" [deep_iekf]="--held-out" )
declare -A OUTDIR=( [tlio]="tlio" [deep_kf]="deep_kf" [tartan_imu]="tartan_imu" [deep_iekf]="deep_iekf_online" )
declare -A DRIVE=( [01]=2011_10_03_drive_0042_extract [04]=2011_09_30_drive_0016_extract [06]=2011_09_30_drive_0020_extract [07]=2011_09_30_drive_0027_extract [08]=2011_09_30_drive_0028_extract [09]=2011_09_30_drive_0033_extract [10]=2011_09_30_drive_0034_extract )

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then MODELS=(tlio deep_kf tartan_imu); fi

mkdir -p "$LOGDIR"
echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO TRAINING (parallel, ${JOBS} concurrent folds)"
echo "  Models    : ${MODELS[*]}"
echo "  Sequences : ${SEQS}"
echo "  Split     : 5 train / 1 inner-val / 1 held-out(test, never loaded)"
echo "  Selection : journal metric J (ATE_outage + t_rel + r_rel) on inner-val"
echo "  Logs      : ${LOGDIR}"
echo "════════════════════════════════════════════════════════════════"

fail=0
for model in "${MODELS[@]}"; do
  script="${SCRIPTS[$model]:-}"
  if [ -z "$script" ]; then echo "unknown model: $model" >&2; fail=1; continue; fi
  echo ""
  echo "── ${model} (${EPOCHS[$model]} epochs x $(echo $SEQS | wc -w) folds) ──"
  pids=()
  markers=()
  model_jobs="${MAX_JOBS[$model]:-$JOBS}"
  if [ "$model_jobs" -gt "$JOBS" ]; then model_jobs="$JOBS"; fi
  need_mb="${MIN_FREE_MB[$model]:-4000}"
  echo "  concurrency: ${model_jobs}, requires ${need_mb} MB free per fold (host-RAM caps)"
  for seq in $SEQS; do
    # Throttle to $model_jobs concurrent children.
    while [ "$(jobs -rp | wc -l)" -ge "$model_jobs" ]; do wait -n; done
    # Refuse to start another fold if free memory is already low; a fold that
    # cannot fit is better delayed than OOM-killed halfway through.
    while :; do
      avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      [ "${avail_mb:-99999}" -ge "$need_mb" ] && break
      echo "  waiting: only ${avail_mb} MB available, need ${need_mb} MB to start a fold"
      wait -n 2>/dev/null || sleep 60
    done
    log="${LOGDIR}/${model}_fold_${seq}.log"
    done_marker="${ARTIFACTS}/${OUTDIR[$model]}/.${model}_fold_${seq}.done"
    # Resumability. A 400-epoch x 7-fold run is long enough that an interruption
    # is likely; without this, a restart would redo folds that already finished.
    # The sentinel is written ONLY after the trainer exits 0, so a fold killed
    # part-way is correctly retried rather than silently accepted -- its
    # checkpoint file exists by then (it is rewritten on every J improvement)
    # and would otherwise look complete.
    if [ "${SKIP_DONE:-1}" = "1" ] && [ -f "$done_marker" ]; then
      echo "  fold ${seq}: already complete (${done_marker##*/}) — skipping"
      continue
    fi
    if [ "$model" = "deep_iekf" ]; then
      PYTHONPATH="${SCRIPT_DIR}" "$PY" -u "${SCRIPT_DIR}/${script}" \
          --mode loo --causal --epochs "${EPOCHS[$model]}" \
          --output "${ARTIFACTS}/${OUTDIR[$model]}" \
          --held-out "${DRIVE[$seq]}" \
          --val-metric-every "${VAL_EVERY[$model]}" > "$log" 2>&1 &
    else
      PYTHONPATH="${SCRIPT_DIR}" "$PY" -u "${SCRIPT_DIR}/${script}" \
          --mode loo --epochs "${EPOCHS[$model]}" --dataset kitti \
          --output "${ARTIFACTS}/${OUTDIR[$model]}" \
          --val-seq "$seq" --val-metric-every "${VAL_EVERY[$model]}" > "$log" 2>&1 &
    fi
    child=$!
    pids+=($child)
    markers+=("$done_marker")
    # Make the kernel prefer these over the desktop if memory runs short. The
    # previous run's OOM killed gnome-shell, VS Code and Chrome before it
    # touched a trainer, because the trainers were not the worst offenders by
    # oom_score. Raising their own score is unprivileged and reverses that.
    echo 700 > "/proc/$child/oom_score_adj" 2>/dev/null || true
    renice -n 5 -p "$child" >/dev/null 2>&1 || true
    echo "  started fold ${seq} (pid $child) -> ${log##*/}"
  done
  # Wait for this model's folds before starting the next model, so a failure is
  # attributable and the GPU is not oversubscribed across model types.
  # Guard the empty case: when every fold was skipped as already complete there
  # is nothing to wait on, and expanding an empty array into `wait` reports a
  # spurious failure that would abort the downstream chain.
  if [ "${#pids[@]}" -gt 0 ]; then
    idx=0
    for pid in "${pids[@]}"; do
      if wait "$pid"; then
        touch "${markers[$idx]}"    # completion sentinel — see SKIP_DONE above
      else
        echo "  FAILED pid ${pid}"; fail=1
      fi
      idx=$((idx + 1))
    done
  fi
  echo "  ${model}: done"
done

echo ""
if [ "$fail" -eq 0 ]; then
  echo "DL_TRAINING_PARALLEL_DONE ok"
else
  echo "DL_TRAINING_PARALLEL_DONE with failures — check ${LOGDIR}"
fi
exit "$fail"
