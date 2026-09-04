#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-measure the rows that were measured while the SoC was throttling.

`run_full_benchmark.py` stamps every result with the firmware's live throttle
bits sampled for the duration of that job. On a passively cooled Pi 5 a long
job can cross the soft temperature limit mid-run, which caps the ARM clock and
inflates that row's timings. Those rows are flagged, not discarded, and this
script re-measures them one at a time from a deliberately cold start.

Comparability is preserved: each retry runs the SAME worker under the SAME
`taskset` affinity as the original sweep. Below the throttle point the clock is
a flat 2.4 GHz, so a row measured at 65 C and one measured at 78 C are directly
comparable -- what is NOT comparable is a row measured on a capped clock, which
is exactly what this removes.

Usage (inside the container, from scripts/positioning/hailo):
    BENCH_TASKSET=1,2,3 python3 _remeasure_throttled.py [--max-passes N]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_full_benchmark as R

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WORKER = _HERE / "_full_eval_worker.py"
OUT_DIR = _HERE / "full_benchmark_results"

# Deliberately colder than the sweep's gate: a retry is worth waiting for.
COLD_START_C = float(os.environ.get("REMEASURE_START_C", "66"))
COLD_MAXWAIT = float(os.environ.get("REMEASURE_MAXWAIT", "900"))


def _throttled_rows():
    rows = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if (r.get('thermal') or {}).get('throttled_during'):
            rows.append((p, r))
    return rows


def _wait_cold(cores):
    t0 = time.time()
    t = R._temp_c()
    if t is not None and t <= COLD_START_C:
        return t
    print(f"    [cold] {t:.1f}C -- waiting for <= {COLD_START_C:.0f}C", flush=True)
    while time.time() - t0 < COLD_MAXWAIT:
        time.sleep(5)
        t = R._temp_c()
        if t is not None and t <= COLD_START_C:
            break
    print(f"    [cold] starting at {t:.1f}C after {time.time()-t0:.0f}s", flush=True)
    return t


def main():
    max_passes = 3
    if "--max-passes" in sys.argv:
        max_passes = int(sys.argv[sys.argv.index("--max-passes") + 1])
    cores = R._gate_cores()

    for attempt in range(1, max_passes + 1):
        rows = _throttled_rows()
        if not rows:
            print(f"No throttled rows remain (pass {attempt}).")
            break
        print(f"\n=== pass {attempt}: {len(rows)} throttled row(s) to re-measure ===")
        for p, r in rows:
            approach, backend = r['approach'], r['backend']
            scenario, seq = r['scenario'], r['seq']
            print(f"  re-measuring {approach}/{backend}/{scenario}/seq{seq} "
                  f"(was {r['thermal']['temp_c_max_during']}C, "
                  f"bits=0x{r['thermal']['throttle_bits_during']:x})", flush=True)
            _wait_cold(cores)
            new = R.run_one(approach, backend, scenario, seq, cores, None)
            if new is None:
                print("    FAILED — leaving previous result in place")
                continue
            if (new.get('thermal') or {}).get('throttled_during'):
                print("    still throttled; will retry on the next pass")
    else:
        rows = _throttled_rows()
        if rows:
            print(f"\n[WARN] {len(rows)} row(s) still throttled after "
                  f"{max_passes} passes:")
            for p, r in rows:
                print(f"    {r['approach']}/{r['backend']}/{r['scenario']}/"
                      f"seq{r['seq']}")

    # Rebuild the aggregate so the tables see the re-measured rows.
    all_results = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            all_results.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"  [WARN] could not parse {p}: {exc}")
    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nAggregate rebuilt: {len(all_results)} results.")
    still = [r for r in all_results if (r.get('thermal') or {}).get('throttled_during')]
    print(f"Rows still flagged as throttled: {len(still)}")


if __name__ == "__main__":
    main()
