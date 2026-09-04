#!/bin/bash
# After the deep_kf/tartan quantisation measurement, run the complete 13-filter
# sweep on THIS host and build the tables. Purpose is validation of the pipeline
# and of the accuracy numbers (which are host-independent: same HEF, same INT8
# weights, same HAILO8L at fw 4.20.0). The TIMING columns produced here are x86
# and are NOT the reported numbers — those must be measured on the Pi.
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -qE "MEASURE_DONE|ABORT" _measure.log; do sleep 30; done
if grep -q ABORT _measure.log; then echo "ABORT upstream"; exit 1; fi
echo "SWEEP_START $(date +%H:%M:%S)"
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 -u run_full_benchmark.py 2>&1 | tail -40
echo "SWEEP_END $(date +%H:%M:%S)"
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 build_latex_tables.py 2>&1 | tail -25
echo "SWEEP_CHAIN_DONE"
