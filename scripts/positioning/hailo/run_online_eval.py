#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online, causal, real-hardware CPU-vs-Hailo-8L comparison.

For each of the 4 approaches (deep_kf, tlio, tartan_imu, deep_iekf), runs
the SAME production dl_filters/*/​*_runner.py causal simulation loop twice
per outage scenario — once with the network forward pass on CPU (torch),
once on the physical Hailo-8L (hailo_backend.py) — timing each network call
as it happens (as if IMU/GNSS samples were arriving live, not read from a
file ahead of time) and scoring the resulting trajectory against ground
truth with this repo's standard metrics.py.

Each (approach, backend, scenario) combination runs in its own subprocess
via _eval_worker.py — see that module's docstring for why (sys.path/
sys.modules contamination across approaches when run in one process).

Usage (inside the arm64 Hailo docker image, /dev/hailo0 required for the
'hailo' backend rows):
    python3 scripts/positioning/hailo/run_online_eval.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_WORKER = _HERE / "_eval_worker.py"

APPROACHES = ["deep_kf", "tlio", "tartan_imu", "deep_iekf"]
BACKENDS   = ["cpu", "hailo"]
SCENARIOS  = ["no_outage", "outage"]


def run_one(approach, backend, scenario, out_dir):
    out_path = out_dir / f"{approach}_{backend}_{scenario}.json"
    cmd = [sys.executable, str(_WORKER),
          "--approach", approach, "--backend", backend,
          "--scenario", scenario, "--out", str(out_path)]
    print(f">>> {approach} / {backend} / {scenario}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}, {dt:.1f}s)")
        return None
    print(f"    done ({dt:.1f}s)")
    with open(out_path) as fh:
        return json.load(fh)


def fmt(x, nd=3):
    return "N/A" if x is None else f"{x:.{nd}f}"


def main():
    # Persistent, repo-mounted location — NOT /tmp inside the container,
    # which is discarded when `docker compose run --rm` exits.
    out_dir = _HERE / "online_eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {out_dir}\n")

    results = []
    for approach in APPROACHES:
        for backend in BACKENDS:
            for scenario in SCENARIOS:
                r = run_one(approach, backend, scenario, out_dir)
                if r is not None:
                    results.append(r)

    lines = []
    lines.append("## Accuracy — outage vs. no-outage, CPU vs. Hailo-8L\n")
    lines.append("| Approach | Backend | Scenario | ATE 2D (m) | ATE 3D (m) | Pos RMSE 2D (m) | Outage-segment mean (m) | Outage-segment max (m) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r["metrics"]
        oa = m.get("outage_analysis")
        lines.append(
            f"| {r['approach']} | {r['backend']} | {r['scenario']} "
            f"| {fmt(m['ate']['rmse_2D'])} | {fmt(m['ate']['rmse_3D'])} "
            f"| {fmt(m['position_rmse']['2D'])} "
            f"| {fmt(oa['mean']) if oa else 'N/A'} "
            f"| {fmt(oa['max']) if oa else 'N/A'} |"
        )

    lines.append("\n## Real per-network-call execution time (no-outage run)\n")
    lines.append("| Approach | Backend | # calls | mean (ms) | median (ms) | p95 (ms) | total net time (s) | wall time (s) | dataset duration (s) | real-time factor |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in results:
        key = (r["approach"], r["backend"])
        if key in seen or r["scenario"] != "no_outage":
            continue
        seen.add(key)
        t = r["timing"]
        lines.append(
            f"| {r['approach']} | {r['backend']} | {t['n_calls']} "
            f"| {fmt(t['mean_ms'])} | {fmt(t['median_ms'])} | {fmt(t['p95_ms'])} "
            f"| {fmt(t['total_net_s'])} | {fmt(t['wall_s'])} "
            f"| {fmt(t['dataset_duration_s'])} "
            f"| {fmt(t['real_time_factor'])}x |"
        )

    report_md = "\n".join(lines)
    print("\n" + report_md)

    (out_dir / "report.json").write_text(json.dumps(results, indent=2))
    (out_dir / "report.md").write_text(report_md)
    print(f"\nFull results: {out_dir}/report.json")
    print(f"Markdown report: {out_dir}/report.md")


if __name__ == "__main__":
    main()
