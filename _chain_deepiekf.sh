#!/bin/bash
# Retrain deep_iekf on the same protocol as the other three (nested split,
# best-J selection) once the TLIO/DKF run has finished, then rebuild its
# streaming HEFs from the newly selected weights.
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -q "DL_TRAINING_PARALLEL_DONE" _retrain_fid.log; do sleep 60; done
echo "PRIOR_RUN_DONE"
PYTHON_BIN=/home/inartrans2/miniconda3/envs/cookies/bin/python \
  ./scripts/positioning/python/run_dl_training_parallel.sh deep_iekf
echo "DEEPIEKF_TRAIN_DONE"
for s in 01 04 06 07 08 09 10; do
  docker exec -w /workspace/cookies/scripts/positioning/hailo/deep_iekf_stream cookies_build \
    python3 build_stream.py --weights /workspace/cookies/artifacts/deep_iekf_online/fold_${s}.p \
    --tag fold_${s} >/dev/null 2>&1 && echo "hef ok $s" || echo "hef FAIL $s"
done
echo "DEEPIEKF_CHAIN_DONE"
