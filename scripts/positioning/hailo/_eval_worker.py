#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-(approach, backend, scenario) worker for run_online_eval.py.

Runs in its OWN process, invoked as a subprocess by the orchestrator. This
is deliberate, not incidental: several dl_filters/*/*_runner.py modules
(and the external repos they wrap — external/tlio, external/ai-imu-dr)
manipulate sys.path and import generically-named modules (`utils`, `model`,
...). Importing more than one approach into the same interpreter causes
stale entries in sys.modules to shadow a different approach's same-named
module (observed directly while wiring this up: deep_iekf's `from utils
import prepare_data` picked up tlio's `external/tlio/src/utils/__init__.py`
after tlio was imported first in the same process). One process per
approach avoids this entirely and matches how these filters are actually
invoked in production (ins_runner.py runs one filter per process).
"""
import argparse
import copy
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PY_DIR = _REPO_ROOT / "scripts/positioning/python"
sys.path.insert(0, str(_PY_DIR))

APPROACH_CFG = {
    'deep_kf': dict(
        runner_path=_PY_DIR / "dl_filters/deep_kf",
        runner_mod="deep_kf_runner",
        hef=_HERE / "deep_kf/deep_kf.hef",
    ),
    'tlio': dict(
        runner_path=_PY_DIR / "dl_filters/tlio",
        runner_mod="tlio_runner",
        hef=_HERE / "tlio/tlio.hef",
        postproc=_HERE / "tlio/tlio_postproc.pt",
    ),
    'tartan_imu': dict(
        runner_path=_PY_DIR / "dl_filters/tartan_imu",
        runner_mod="tartan_runner",
        hef=_HERE / "tartan_imu/tartan_imu.hef",
        postproc=_HERE / "tartan_imu/tartan_imu_postproc.pt",
    ),
    'deep_iekf': dict(
        runner_path=_PY_DIR / "dl_filters/deep_iekf",
        runner_mod="iekf_ai_imu_online",
        hef=_HERE / "deep_iekf/deep_iekf.hef",
        postproc=_HERE / "deep_iekf/deep_iekf_postproc.npz",
    ),
}

# KITTI '01' — a "clean" sequence (ins_config.py's own default; no data gaps,
# unlike '00'/'02'/'05'), ~121.9s long, comfortably covering the default
# 80s-start/40s-duration outage window with margin.
DATASET_ID   = "01"
FULL_OUTAGE  = {'start': 80.0, 'duration': 40.0}

# deep_iekf's HEF has a FIXED input length (see hailo_backend.HailoDeepIEKF)
# — the CPU causal runner ALSO evaluates its causal MesNet as one
# whole-sequence pass (see iekf_ai_imu_online.py's own docstring), so both
# backends run on the SAME truncated slice for a fair comparison. The
# default 80s outage window doesn't fit inside this ~45.44s slice, so a
# smaller, in-range window is used instead.
IEKF_SEQ_LEN = 4544
IEKF_OUTAGE  = {'start': 15.0, 'duration': 10.0}


def _truncate(nav, n):
    nd = copy.copy(nav)
    for f in ("accel_flu", "gyro_flu", "vel_enu", "lla", "orient",
              "gps_available", "gps_speed_mps", "gps_cog_rad", "time"):
        v = getattr(nd, f, None)
        if v is not None:
            setattr(nd, f, v[:n])
    return nd


def _strip_arrays(d):
    """Drop numpy-array-valued keys so json.dump works (mirrors ins_runner.py)."""
    if isinstance(d, dict):
        return {k: _strip_arrays(v) for k, v in d.items()
                if not isinstance(v, np.ndarray)}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=list(APPROACH_CFG))
    ap.add_argument("--backend", required=True, choices=["cpu", "hailo"])
    ap.add_argument("--scenario", required=True, choices=["no_outage", "outage"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = APPROACH_CFG[args.approach]
    sys.path.insert(0, str(cfg["runner_path"]))
    runner = importlib.import_module(cfg["runner_mod"])

    import data_loader
    nav = data_loader.get_kitti_dataset(DATASET_ID)

    if args.approach == "deep_iekf":
        nav = _truncate(nav, IEKF_SEQ_LEN)
        outage_cfg = IEKF_OUTAGE if args.scenario == "outage" else {'start': 0., 'duration': 0.}
    else:
        outage_cfg = FULL_OUTAGE if args.scenario == "outage" else {'start': 0., 'duration': 0.}

    dataset_duration_s = len(nav.accel_flu) / nav.sample_rate

    hailo_net = None
    hailo_cm = None
    if args.backend == "hailo":
        sys.path.insert(0, str(_HERE))
        import hailo_backend
        if args.approach == "deep_kf":
            hailo_cm = hailo_backend.HailoDeepKF(cfg["hef"])
        elif args.approach == "tlio":
            hailo_cm = hailo_backend.HailoTLIO(cfg["hef"], cfg["postproc"])
        elif args.approach == "tartan_imu":
            hailo_cm = hailo_backend.HailoTartanIMU(cfg["hef"], cfg["postproc"])
        elif args.approach == "deep_iekf":
            hailo_cm = hailo_backend.HailoDeepIEKF(cfg["hef"], cfg["postproc"])
        hailo_net = hailo_cm.__enter__()

    try:
        t_wall0 = time.perf_counter()
        result = runner.run(nav, outage_config=outage_cfg,
                            backend=args.backend, hailo_net=hailo_net)
        wall_s = time.perf_counter() - t_wall0
    finally:
        if hailo_cm is not None:
            hailo_cm.__exit__(None, None, None)

    # ── Ground truth + metrics — mirrors ins_runner.py exactly ─────────────
    import pymap3d as pm
    import metrics as _metrics

    lla, lla0 = nav.lla, nav.lla0
    e, n, u = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], lla0[0], lla0[1], lla0[2])
    p_gt = np.column_stack([e, n, u])

    t1 = outage_cfg.get('start', 0.)
    d  = outage_cfg.get('duration', 0.)
    gnss_outage_info = {
        'start': t1, 'end': t1 + d, 'duration': d,
        'start_idx': int(t1 * nav.sample_rate),
        'end_idx':   int((t1 + d) * nav.sample_rate),
    }

    metrics_result = _metrics.evaluate_navigation_performance(
        p_est=result['p'], v_est=result['v'], r_est=result['r'],
        p_gt=p_gt, v_gt=nav.vel_enu, r_gt=nav.orient,
        dataset_name=nav.dataset_name,
        gnss_outage_info=gnss_outage_info,
        sample_rate=nav.sample_rate,
    )

    net_latency_s = np.asarray(result['net_latency_s'], dtype=float)
    timing = {
        'n_calls':    int(len(net_latency_s)),
        'mean_ms':    float(np.mean(net_latency_s) * 1000)   if len(net_latency_s) else None,
        'median_ms':  float(np.median(net_latency_s) * 1000) if len(net_latency_s) else None,
        'p95_ms':     float(np.percentile(net_latency_s, 95) * 1000) if len(net_latency_s) else None,
        'total_net_s': float(np.sum(net_latency_s)),
        'wall_s':      float(wall_s),
        'dataset_duration_s': float(dataset_duration_s),
        'real_time_factor': float(dataset_duration_s / wall_s) if wall_s > 0 else None,
    }

    out = {
        'approach': args.approach,
        'backend':  args.backend,
        'scenario': args.scenario,
        'metrics':  _strip_arrays(metrics_result),
        'timing':   timing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"OK {args.approach}/{args.backend}/{args.scenario} -> {args.out}")


if __name__ == "__main__":
    main()
