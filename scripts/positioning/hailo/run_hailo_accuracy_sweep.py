#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-runs just the Hailo/DL subset (4 filters x 7 seqs x 2 scenarios = 56
jobs) of _full_eval_worker.py — CPU and classical/smoother timing were
already collected by run_full_benchmark.py and don't need re-running; this
just adds accuracy (now computed for Hailo runs too, see
_full_eval_worker.py's `if args.backend == "hailo"` block) on top of the
existing full_benchmark_results/*.json for those 56 combinations (the JSON
files get overwritten with a superset that includes both timing and
accuracy).
"""
import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WORKER = _HERE / "_full_eval_worker.py"
OUT_DIR = _HERE / "full_benchmark_results"

DL = ['deep_kf', 'tlio', 'tartan_imu', 'deep_iekf']
SEQS = ['01', '04', '06', '07', '08', '09', '10']
SCENARIOS = ['no_outage', 'outage']


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(a, s, sq) for a in DL for s in SCENARIOS for sq in SEQS]
    print(f"Total jobs: {len(jobs)}\n")

    ok = 0
    for i, (approach, scenario, seq) in enumerate(jobs, 1):
        out_path = OUT_DIR / f"{approach}_hailo_{scenario}_{seq}.json"
        cmd = [sys.executable, str(_WORKER),
              "--approach", approach, "--backend", "hailo",
              "--scenario", scenario, "--seq", seq, "--out", str(out_path)]
        print(f"[{i}/{len(jobs)}] {approach}/hailo/{scenario}/{seq}", flush=True)
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '?'
            print(f"    FAILED: {last}")
            continue
        ok += 1
        with open(out_path) as fh:
            r = json.load(fh)
        acc = r.get('accuracy', {})
        print(f"    ok  wall={r['timing']['wall_s']:.2f}s  "
              f"t_rel={acc.get('kitti_metrics', {}).get('t_rel', float('nan')):.2f}%")

    print(f"\nDone. {ok}/{len(jobs)} succeeded.")

    # Rebuild the aggregate all_results.json from every *.json in the dir
    # (mirrors run_full_benchmark.py's final step).
    results = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            results.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"  [WARN] could not parse {p}: {exc}")
    (OUT_DIR / "all_results.json").write_text(json.dumps(results, indent=2))
    print(f"Aggregate rebuilt: {len(results)} total results -> {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
