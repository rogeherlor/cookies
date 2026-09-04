#!/bin/bash
# Waits for the journal-metric retraining to finish, then rebuilds ALL 21
# per-fold HEFs from the newly selected checkpoints. All three models are
# rebuilt because J-based selection changed every fold's deployed epoch.
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -q "DL_TRAINING_PARALLEL_DONE" _retrain_j.log; do sleep 60; done
echo "TRAINING_FINISHED: $(grep DL_TRAINING_PARALLEL_DONE _retrain_j.log)"
if grep -q "DL_TRAINING_PARALLEL_DONE with failures" _retrain_j.log; then
  echo "ABORT: training reported failures — not building HEFs from suspect checkpoints"
  exit 1
fi
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 -u build_per_fold_hefs.py 2>&1 \
  | tee _all_hef_rebuild.log | grep -E "fold [0-9]+: ->|FAILED|folds built"
echo "CHAIN_DONE"
