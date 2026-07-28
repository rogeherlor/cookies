#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-off rerun of ONLY the deep_iekf combinations, after fixing the
online-weights selection bug in _full_eval_worker.py's _run_dl() (fold-specific
AI_IMU_ONLINE_WEIGHTS was not being set, so every seq silently got whatever
generic artifacts/deep_iekf_online/iekfnets.p happened to exist).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WORKER = _HERE / "_full_eval_worker.py"
OUT_DIR = _HERE / "full_benchmark_results"

SEQS = ['01', '04', '06', '07', '08', '09', '10']
SCENARIOS = ['no_outage', 'outage']


def run_one(approach, backend, scenario, seq):
    out_path = OUT_DIR / f"{approach}_{backend}_{scenario}_{seq}.json"
    cmd = [sys.executable, str(_WORKER),
          "--approach", approach, "--backend", backend,
          "--scenario", scenario, "--seq", seq, "--out", str(out_path)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAILED {approach}/{backend}/{scenario}/{seq} ({dt:.1f}s): "
              f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else '?'}")
        return None
    with open(out_path) as fh:
        r = json.load(fh)
    ate = r.get('accuracy', {}).get('ate', {}).get('rmse')
    print(f"    ok {backend:5s} {scenario:10s} seq{seq}  "
          f"wall={r['timing']['wall_s']:.2f}s  ATE={ate}")
    return r


def main():
    jobs = []
    for backend in ['cpu', 'hailo']:
        for scenario in SCENARIOS:
            for seq in SEQS:
                jobs.append(('deep_iekf', backend, scenario, seq))

    print(f"Total jobs: {len(jobs)}\n")
    results = []
    for i, (approach, backend, scenario, seq) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {approach}/{backend}/{scenario}/seq{seq}")
        r = run_one(approach, backend, scenario, seq)
        if r is not None:
            results.append(r)
    print(f"\nDone. {len(results)}/{len(jobs)} succeeded.")

    # Rebuild the full aggregate from every *.json on disk (same convention
    # as run_full_benchmark.py) so downstream table-building sees the fix.
    all_results = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name == "all_results.json":
            continue
        try:
            all_results.append(json.loads(p.read_text()))
        except Exception as exc:
            print(f"  [WARN] could not parse {p}: {exc}")
    (OUT_DIR / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"Aggregate rebuilt: {len(all_results)} total results -> {OUT_DIR}/all_results.json")


if __name__ == "__main__":
    main()
