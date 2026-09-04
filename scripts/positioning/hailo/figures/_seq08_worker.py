#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run exactly ONE filter on seq08 (standard 40s/60s outage) and save its
estimated trajectory to an .npz file. Deliberately a fresh process per
invocation — matching _full_eval_worker.py's own subprocess-per-combination
architecture — after discovering that running deep_iekf as the 8th filter
in a shared long-lived process (following 6 classical filters) silently
produced a wrong trajectory (5.04 m RMSE instead of the verified-correct
208.86 m). Whatever the root cause of that state leak, per-process
isolation is the same fix the rest of this codebase already relies on.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_HAILO_DIR = _HERE.parent
_REPO_ROOT = _HERE.parent.parent.parent.parent
_PY_DIR = _REPO_ROOT / "scripts/positioning/python"
sys.path.insert(0, str(_PY_DIR))
sys.path.insert(0, str(_HAILO_DIR))

import data_loader
import filter_params as fp
import _full_eval_worker as w

ap = argparse.ArgumentParser()
ap.add_argument("--approach", required=True)
ap.add_argument("--kind", required=True, choices=["classical", "dl", "smoother"])
ap.add_argument("--seq", required=True)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()

SEQ = args.seq
OUTAGE_CFG = {"start": w.OUTAGE_START, "duration": w.OUTAGE_DURATION}

nav = data_loader.get_kitti_dataset(SEQ)
tuned = w._load_tuned_params(nav, True, fp)
tuned_key = w.DL_TUNED_KEY[args.approach] if args.approach in w.DL else args.approach
params = tuned.get(tuned_key)
print(f"DEBUG dataset_name={nav.dataset_name!r} tuned_key={tuned_key!r} params={params!r}", flush=True)

if args.kind == "classical":
    _, _, result, _ = w._run_classical(args.approach, nav, params, OUTAGE_CFG, use_3d=True)
elif args.kind == "dl":
    _, _, result, _ = w._run_dl(args.approach, nav, params, OUTAGE_CFG, backend="cpu", seq=SEQ, use_3d=True)
else:
    _, _, result = w._run_smoother(args.approach, nav, params, OUTAGE_CFG, use_3d=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, p=result["p"])
print(f"OK {args.approach} -> {args.out}  (p shape {result['p'].shape})")
