#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs isam2_runner.py / isam2_fixedlag_runner.py under the isolated
/opt/conda-gtsam Python (gtsam has no aarch64 PyPI wheel — see
docker/Dockerfile's GTSAM section). Invoked as a subprocess by
_general_eval_worker.py's main-interpreter process, which owns data
loading and metrics evaluation; this script's only job is the timed
run() call itself, handed a minimal navigation-data namespace via a
.npz file so it doesn't need data_loader.py or any other repo module
(isam2_runner.py itself only imports numpy + pymap3d + gtsam).
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SMOOTHERS_DIR = _HERE.parent.parent.parent / "scripts/positioning/python/smoothers"
sys.path.insert(0, str(_SMOOTHERS_DIR))

RUNNERS = {
    "isam2": "isam2_runner",
    "isam2_fixedlag": "isam2_fixedlag_runner",
    "isam2_map": "isam2_map_runner",
    "fgo_batch": "fgo_batch_runner",
}


def _load_nav(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    nav = types.SimpleNamespace()
    for k in d.files:
        v = d[k]
        if v.shape == () and v.dtype == object:
            v = v.item()
        setattr(nav, k, v)
    return nav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, choices=list(RUNNERS))
    ap.add_argument("--nav-npz", required=True, type=Path)
    ap.add_argument("--params-json", required=True, type=Path)
    ap.add_argument("--outage-start", type=float, required=True)
    ap.add_argument("--outage-duration", type=float, required=True)
    ap.add_argument("--use-3d", type=int, default=1)
    ap.add_argument("--out-npz", required=True, type=Path)
    args = ap.parse_args()

    import importlib
    runner = importlib.import_module(RUNNERS[args.runner])

    nav = _load_nav(args.nav_npz)
    params = json.loads(args.params_json.read_text()) if args.params_json.stat().st_size else None
    outage_cfg = {'start': args.outage_start, 'duration': args.outage_duration}

    t0 = time.perf_counter()
    result = runner.run(nav, params=params, outage_config=outage_cfg,
                        use_3d_rotation=bool(args.use_3d))
    wall_s = time.perf_counter() - t0

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_npz,
        p=result['p'], v=result['v'], r=result['r'],
        bias_acc=result['bias_acc'], bias_gyr=result['bias_gyr'],
        wall_s=np.array(wall_s),
    )
    print(f"OK {args.runner} -> {args.out_npz} ({wall_s:.3f}s)")


if __name__ == "__main__":
    main()
