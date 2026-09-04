#!/bin/bash
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -q "CHAIN_DONE\|ABORT" _chain.log; do sleep 30; done
if grep -q "ABORT" _chain.log; then echo "ABORT: HEF rebuild aborted"; exit 1; fi
grep -E "folds built|FAILED" _chain.log
docker exec cookies_build rm -rf /tmp/q; docker exec cookies_build mkdir -p /tmp/q
for ap in deep_kf tartan_imu; do
  for s in 01 04 06 07 08 09 10; do
    for b in cpu hailo; do
      for sc in no_outage outage; do
        docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
          python3 _full_eval_worker.py --approach $ap --backend $b --scenario $sc \
          --seq $s --out /tmp/q/${ap}_${b}_${sc}_${s}.json >/dev/null 2>&1 \
          || echo "FAIL $ap $b $sc $s"
      done
    done
  done
  echo "done $ap"
done
echo "MEASURE_DONE"
