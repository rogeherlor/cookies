#!/bin/bash
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_bench \
  taskset -c 1,2,3 python3 run_full_benchmark.py
echo "SWEEP_EXIT=$?"
echo "FULL_SWEEP_FINISHED"
