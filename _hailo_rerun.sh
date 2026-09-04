#!/bin/bash
# Re-run only the Hailo half of the sweep (56 jobs).
#
# CPU pinning comes from BENCH_TASKSET, the SAME variable run_full_benchmark.py
# reads, so the Hailo rows and the CPU rows are measured under identical
# affinity. This script used to hardcode `taskset -c 1,2,3` while
# run_full_benchmark.py pinned nothing, which made the two halves of every
# timing table incomparable without anything in the output recording it.
# Each result JSON now records n_cpus_affinity and the load average.
#
#   BENCH_TASKSET=1,2,3 ./_hailo_rerun.sh      # Pi: leave core 0 for the OS
set -u

CONTAINER="${BENCH_CONTAINER:-cookies_bench}"
WORKDIR=/workspace/cookies/scripts/positioning/hailo

if [ -n "${BENCH_TASKSET:-}" ]; then
  PIN=(taskset -c "$BENCH_TASKSET")
  echo "Pinning every job to cores ${BENCH_TASKSET}"
else
  PIN=()
  echo "BENCH_TASKSET unset — running unpinned (must match how the CPU rows were run)"
fi

fail=0
for ap in deep_iekf tlio deep_kf tartan_imu; do
  for sc in no_outage outage; do
    for sq in 01 04 06 07 08 09 10; do
      out="${WORKDIR}/full_benchmark_results/${ap}_hailo_${sc}_${sq}.json"
      if docker exec -w "$WORKDIR" "$CONTAINER" \
           "${PIN[@]}" python3 _full_eval_worker.py \
           --approach "$ap" --backend hailo --scenario "$sc" --seq "$sq" \
           --out "$out" >/dev/null 2>&1; then
        echo "ok $ap $sc $sq"
      else
        echo "FAIL $ap $sc $sq"
        fail=$((fail + 1))
      fi
    done
  done
done

echo "HAILO_RERUN_DONE failures=${fail}"
exit $(( fail > 0 ))
