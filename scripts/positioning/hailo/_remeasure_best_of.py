#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-of re-measurement for rows that cannot be measured unthrottled.

Some rows (on this rig, TLIO) run long enough under full load that a passively
cooled Pi 5 crosses its soft temperature limit no matter how cold the start --
TLIO CPU seq08 is 216 s of sustained compute and the board idles at ~63 C.
Those rows cannot be made clean on this hardware.

What CAN be done is to report the least-distorted measurement available.
Thermal throttling only ever caps the clock, so it only ever INFLATES a
timing: of several repeats of the same row, the smallest `median_ms` is the
one closest to the true unthrottled value. This script therefore keeps the
MINIMUM across repeats rather than the most recent one.

That matters because a naive re-measure loop does the opposite: each pass
starts from a hotter board than the last, so repeated passes drift upward and
overwriting with the newest result systematically keeps the WORST run.

Usage (inside the container, from scripts/positioning/hailo):
    BENCH_TASKSET=1,2,3 python3 _remeasure_best_of.py [--rounds N]
"""
import json
import os
import sys
import time
from pathlib import Path

import run_full_benchmark as R

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "full_benchmark_results"

COLD_C = float(os.environ.get("BESTOF_START_C", "65"))
COLD_MAXWAIT = float(os.environ.get("BESTOF_MAXWAIT", "900"))


def _throttled_rows():
    out = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if (r.get('thermal') or {}).get('throttled_during'):
            out.append((p, r))
    return out


def _wait_cold():
    t0 = time.time()
    t = R._temp_c()
    if t is not None and t <= COLD_C:
        return t
    print(f"    [cold] {t:.1f}C -- waiting for <= {COLD_C:.0f}C", flush=True)
    while time.time() - t0 < COLD_MAXWAIT:
        time.sleep(5)
        t = R._temp_c()
        if t is not None and t <= COLD_C:
            break
    print(f"    [cold] starting at {t:.1f}C after {time.time()-t0:.0f}s", flush=True)
    return t


def main():
    rounds = 2
    if "--rounds" in sys.argv:
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1])
    cores = R._gate_cores()

    targets = [(p, r) for p, r in _throttled_rows()]
    print(f"{len(targets)} row(s) cannot be measured unthrottled; "
          f"taking best-of-{rounds + 1} on each.\n")

    for p, orig in targets:
        approach, backend = orig['approach'], orig['backend']
        scenario, seq = orig['scenario'], orig['seq']
        best = orig
        best_med = orig['timing']['median_ms']
        print(f"  {approach}/{backend}/{scenario}/seq{seq}: "
              f"stored median={best_med:.3f}ms "
              f"(peak {orig['thermal']['temp_c_max_during']}C)", flush=True)

        for k in range(rounds):
            _wait_cold()
            new = R.run_one(approach, backend, scenario, seq, cores, None)
            if new is None:
                print("      run failed; keeping previous")
                continue
            med = new['timing']['median_ms']
            if med < best_med:
                print(f"      round {k+1}: {med:.3f}ms  <-- better, keeping")
                best, best_med = new, med
            else:
                print(f"      round {k+1}: {med:.3f}ms  (worse, discarding)")

        # run_one already wrote whatever ran last; put the BEST back on disk.
        p.write_text(json.dumps(best, indent=2))
        print(f"    -> kept median={best_med:.3f}ms "
              f"(peak {best['thermal']['temp_c_max_during']}C, "
              f"throttled={best['thermal']['throttled_during']})\n", flush=True)

    all_results = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            all_results.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"  [WARN] could not parse {p}: {exc}")
    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    still = [r for r in all_results if (r.get('thermal') or {}).get('throttled_during')]
    print(f"Aggregate rebuilt: {len(all_results)} results; "
          f"{len(still)} still flagged as throttled (best-of retained).")


if __name__ == "__main__":
    main()
