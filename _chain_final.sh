#!/bin/bash
# Final pass: runs only after deep_iekf has retrained AND its streaming HEFs are
# rebuilt. Everything upstream (tlio/dkf retrain, their HEFs, an intermediate
# sweep) has already run; this discards that intermediate sweep and measures all
# four models together, so the tables come from one consistent set of artefacts.
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -q "DEEPIEKF_CHAIN_DONE" _chain_di.log; do sleep 120; done
if grep -q "hef FAIL" _chain_di.log; then echo "ABORT: deep_iekf HEF build failed"; exit 1; fi
echo "FINAL_SWEEP_START $(date +%H:%M)"
rm -f scripts/positioning/hailo/full_benchmark_results/*.json
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 -u run_full_benchmark.py 2>&1 | tail -4
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 build_latex_tables.py 2>&1 | grep -E "Provenance|REFUS|WARN|Wrote" | tail -12
echo "FINAL_CHAIN_DONE $(date +%H:%M)"
