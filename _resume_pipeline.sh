#!/bin/bash
# Serial resume: deep_iekf (400 epochs, concurrency 2, resumable) -> its HEFs ->
# final sweep -> tables. One stage at a time. If this is interrupted, rerunning
# it is safe: completed folds are skipped via their .done sentinels.
set -uo pipefail
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies

echo "STAGE 1: deep_iekf retrain, 400 epochs, concurrency 2  $(date '+%m-%d %H:%M')"
PYTHON_BIN=/home/inartrans2/miniconda3/envs/cookies/bin/python \
  ./scripts/positioning/python/run_dl_training_parallel.sh deep_iekf
grep -q "DL_TRAINING_PARALLEL_DONE ok" <(tail -5 "$0.marker" 2>/dev/null) 2>/dev/null || true
n_done=$(ls artifacts/deep_iekf_online/.deep_iekf_fold_*.done 2>/dev/null | wc -l)
echo "STAGE 1 END: ${n_done}/7 folds complete  $(date '+%m-%d %H:%M')"
[ "$n_done" -eq 7 ] || { echo "ABORT: only ${n_done}/7 folds finished — rerun this script to resume"; exit 1; }

echo "STAGE 2: deep_iekf streaming HEFs  $(date '+%m-%d %H:%M')"
fail=0
for s in 01 04 06 07 08 09 10; do
  docker exec -w /workspace/cookies/scripts/positioning/hailo/deep_iekf_stream cookies_build \
    python3 build_stream.py --weights /workspace/cookies/artifacts/deep_iekf_online/fold_${s}.p \
    --tag fold_${s} >/dev/null 2>&1 && echo "  hef ok $s" || { echo "  hef FAIL $s"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "ABORT: deep_iekf HEF build failed"; exit 1; }

echo "STAGE 3: final full sweep  $(date '+%m-%d %H:%M')"
rm -f scripts/positioning/hailo/full_benchmark_results/*.json
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 -u run_full_benchmark.py 2>&1 | tail -3
echo "STAGE 4: tables  $(date '+%m-%d %H:%M')"
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 build_latex_tables.py 2>&1 | grep -E "Provenance|REFUS|WARN"
echo "RESUME_PIPELINE_DONE  $(date '+%m-%d %H:%M')"
