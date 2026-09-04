#!/bin/bash
cd /media/inartrans2/6c8f6887-9acf-4794-b990-8de964c59e871/rhl/cookies
until grep -q "Aggregate rebuilt" _sweep_now.log 2>/dev/null; do sleep 60; done
echo "SWEEP_DONE $(grep -c '' /dev/null)"
tail -2 _sweep_now.log
docker exec -w /workspace/cookies/scripts/positioning/hailo cookies_build \
  python3 build_latex_tables.py 2>&1 | grep -E "Provenance|REFUS|WARN|Wrote"
echo "TABLES_DONE"
