#!/bin/bash
# After the fidelity retrain (TLIO 200 epochs, DKF scheduled sampling off):
# rebuild only the two models whose checkpoints changed, then re-run the whole
# sweep and rebuild the tables. Tartan IMU and Deep IEKF are untouched, so their
# HEFs remain valid and are deliberately not rebuilt.
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -q "DL_TRAINING_PARALLEL_DONE" _retrain_fid.log; do sleep 60; done
echo "TRAINING_FINISHED: $(grep DL_TRAINING_PARALLEL_DONE _retrain_fid.log)"
if grep -q "with failures" _retrain_fid.log; then echo "ABORT: training failures"; exit 1; fi
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 -u build_per_fold_hefs.py --models tlio deep_kf 2>&1 \
  | tee _fid_hef.log | grep -E "fold [0-9]+: ->|FAILED|folds built"
if ! grep -q "tlio: 7/7" _fid_hef.log || ! grep -q "deep_kf: 7/7" _fid_hef.log; then
  echo "ABORT: HEF rebuild incomplete"; exit 1; fi
rm -f scripts/positioning/hailo/full_benchmark_results/*.json
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 -u run_full_benchmark.py 2>&1 | tail -5
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 build_latex_tables.py 2>&1 | grep -E "Provenance|REFUS|WARN"
echo "FIDELITY_CHAIN_DONE"
