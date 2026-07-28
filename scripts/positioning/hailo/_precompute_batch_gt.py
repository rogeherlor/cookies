#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precomputes the FGO-Batch ground truth trajectory (GTSAM factor graph, all
GPS, non-causal — see smoothers/fgo_batch_runner.py) for each KITTI
sequence, ONCE (it's independent of filter/backend/outage scenario — see
that module's docstring: "outage_config is intentionally ignored — this is
the ground truth, not a filter"). This is what ins_compare.py uses as
`p_gt` for ATE/RMSE when ins_config.GT_SOURCE == 'batch' (the current
setting, and what the cached accuracy tables were generated with) — as
opposed to `p_kitti` (raw GPS), which is always used for t_rel/r_rel
regardless of GT_SOURCE. _full_eval_worker.py reads this cache to replicate
that exact methodology so its accuracy numbers are comparable to the
historical cached ones.

Usage:
    python3 scripts/positioning/hailo/_precompute_batch_gt.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PY_DIR = _REPO_ROOT / "scripts/positioning/python"
sys.path.insert(0, str(_PY_DIR))

CONDA_PYTHON = "/opt/conda-gtsam/bin/python3"
CACHE_DIR = _HERE / "full_benchmark_results" / "gt_cache"
KITTI_SEQS = ['01', '04', '06', '07', '08', '09', '10']


def main():
    import data_loader
    import filter_params as fp

    sys.path.insert(0, str(_HERE))
    from _full_eval_worker import _load_tuned_params

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for seq in KITTI_SEQS:
        out_npz = CACHE_DIR / f"{seq}.npz"
        nav = data_loader.get_kitti_dataset(seq)
        tuned = _load_tuned_params(nav, True, fp)
        params = tuned.get('isam2')

        nav_npz = CACHE_DIR / f"_nav_{seq}.npz"
        params_json = CACHE_DIR / f"_params_{seq}.json"
        save_kwargs = {}
        for f in ("accel_flu", "gyro_flu", "vel_enu", "lla", "orient",
                  "gps_available", "gps_speed_mps", "gps_cog_rad", "time"):
            v = getattr(nav, f, None)
            if v is not None:
                save_kwargs[f] = v
        save_kwargs["sample_rate"] = np.array(nav.sample_rate)
        save_kwargs["gps_rate"] = np.array(nav.gps_rate)
        save_kwargs["lla0"] = nav.lla0
        save_kwargs["dataset_name"] = np.array(nav.dataset_name)
        np.savez(nav_npz, **save_kwargs)
        params_json.write_text(json.dumps(params) if params else "")

        cmd = [CONDA_PYTHON, str(_HERE / "_isam2_conda_worker.py"),
              "--runner", "fgo_batch", "--nav-npz", str(nav_npz),
              "--params-json", str(params_json),
              "--outage-start", "0", "--outage-duration", "0",
              "--use-3d", "1", "--out-npz", str(out_npz)]
        print(f"seq {seq}: running FGO-Batch...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  FAILED: {proc.stdout}\n{proc.stderr}")
            continue
        print(f"  ok -> {out_npz}")
        nav_npz.unlink()
        params_json.unlink()


if __name__ == "__main__":
    main()
