#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrates _full_eval_worker.py across (approach x backend x scenario x
KITTI sequence), each in its own subprocess (see _full_eval_worker.py's
docstring for why). Timing only — accuracy is read from this repo's
already-cached, LOO-tuned outputs/<filter_key>/... results (see
build_latex_tables.py).

Usage (inside the arm64 Hailo docker image, /dev/hailo0 required for the
Hailo-backed rows):
    python3 scripts/positioning/hailo/run_full_benchmark.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WORKER = _HERE / "_full_eval_worker.py"

CLASSICAL = ['esekfg_vanilla', 'esekfg_enhanced', 'esekfs_vanilla', 'esekfs_enhanced',
             'iekf_vanilla', 'iekf_enhanced', 'imu_only']
DL         = ['deep_kf', 'tlio', 'tartan_imu', 'deep_iekf']
SMOOTHERS  = ['isam2', 'isam2_fixedlag']
ALL_APPROACHES = CLASSICAL + DL + SMOOTHERS

SEQS = ['01', '04', '06', '07', '08', '09', '10']
SCENARIOS = ['no_outage', 'outage']

OUT_DIR = _HERE / "full_benchmark_results"


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
    print(f"    ok {approach:16s} {backend:5s} {scenario:10s} seq{seq}  "
          f"wall={r['timing']['wall_s']:.2f}s  ({dt:.1f}s incl. subprocess overhead)")
    return r


def main():
    cpu_only = '--cpu-only' in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {OUT_DIR}\n")

    jobs = []
    for approach in ALL_APPROACHES:
        backends = (['cpu', 'hailo'] if approach in DL else ['cpu'])
        if cpu_only:
            backends = ['cpu']
        for backend in backends:
            for scenario in SCENARIOS:
                for seq in SEQS:
                    jobs.append((approach, backend, scenario, seq))

    print(f"Total jobs: {len(jobs)}\n")

    results = []
    for i, (approach, backend, scenario, seq) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {approach}/{backend}/{scenario}/seq{seq}")
        r = run_one(approach, backend, scenario, seq)
        if r is not None:
            results.append(r)

    print(f"\nDone. {len(results)}/{len(jobs)} succeeded.")

    # Rebuild the aggregate from EVERY *.json on disk, not just this run's
    # results — a partial (e.g. --cpu-only) rerun must not drop previously
    # collected results (e.g. the Hailo rows) from the aggregate.
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
