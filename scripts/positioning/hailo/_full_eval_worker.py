#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timing-only worker (approach x backend x scenario x KITTI sequence),
covering ALL 13 filters this repo supports — the 4 already covered by
_eval_worker.py (deep_kf, tlio, tartan_imu, deep_iekf) plus the 7 classical
EKF/ESKF/IMU-only filters and the 2 GTSAM smoothers (isam2, isam2_fixedlag,
via the isolated /opt/conda-gtsam interpreter — see
_isam2_conda_worker.py; gtsam has no aarch64 PyPI wheel).

Accuracy is NOT recomputed here — this repo already has cached, LOO-tuned
accuracy results for all 13 filters x 7 KITTI sequences x {no_outage,
outage} in outputs/<filter_key>/... (see emit_journal_tables.py). This
worker's only job is real per-run wall-clock timing on THIS hardware,
using the SAME tuned params (`_load_tuned_params`, inlined from
ins_compare.py rather than imported — importing ins_compare directly would
eagerly import every DL filter module at once and reintroduce the
cross-approach sys.modules contamination _eval_worker.py's docstring
describes) and the SAME outage window (40s start / 60s duration, matching
emit_journal_tables.py's default) that produced those cached numbers —
except deep_iekf's Hailo row, whose HEF has a fixed 4544-sample input
length (~45.44s) and needs its own compatible outage window.

Runs in its own process per (approach, backend, scenario, seq) combination
for the same reason _eval_worker.py does — see that module's docstring.
"""
import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
_PY_DIR = _REPO_ROOT / "scripts/positioning/python"
sys.path.insert(0, str(_PY_DIR))

CLASSICAL = ['esekfg_vanilla', 'esekfg_enhanced', 'esekfs_vanilla', 'esekfs_enhanced',
             'iekf_vanilla', 'iekf_enhanced', 'imu_only']
DL = ['deep_kf', 'tlio', 'tartan_imu', 'deep_iekf']
SMOOTHERS = ['isam2', 'isam2_fixedlag']

DL_RUNNER_MOD = {
    'deep_kf': 'deep_kf_runner', 'tlio': 'tlio_runner',
    'tartan_imu': 'tartan_runner', 'deep_iekf': 'iekf_ai_imu_online',
}
DL_RUNNER_DIR = {
    'deep_kf': 'dl_filters/deep_kf', 'tlio': 'dl_filters/tlio',
    'tartan_imu': 'dl_filters/tartan_imu', 'deep_iekf': 'dl_filters/deep_iekf',
}
DL_TUNED_KEY = {  # key used in filter_params.json / _load_tuned_params()'s dict
    'deep_kf': 'deep_kf', 'tlio': 'tlio', 'tartan_imu': 'tartan_imu',
    'deep_iekf': 'iekf_ai_imu_online',
}
HEF_PATH = {
    'deep_kf':    _HERE / "deep_kf/deep_kf.hef",
    'tlio':       _HERE / "tlio/tlio.hef",
    'tartan_imu': _HERE / "tartan_imu/tartan_imu.hef",
    'deep_iekf':  _HERE / "deep_iekf/deep_iekf.hef",
}
POSTPROC_PATH = {
    'tlio':       _HERE / "tlio/tlio_postproc.pt",
    'tartan_imu': _HERE / "tartan_imu/tartan_imu_postproc.pt",
    'deep_iekf':  _HERE / "deep_iekf/deep_iekf_postproc.npz",
}

OUTAGE_START      = 40.0
OUTAGE_DURATION   = 60.0
DEEP_IEKF_SEQ_LEN = 4544
DEEP_IEKF_HAILO_OUTAGE = {'start': 15.0, 'duration': 10.0}

CONDA_PYTHON = "/opt/conda-gtsam/bin/python3"


def _load_tuned_params(nav_data, mode_3d, fp):
    """Inlined from ins_compare.py::_load_tuned_params — see that function's
    docstring for the full priority/derivation rules. Reimplemented here
    (rather than `import ins_compare`) to avoid eagerly importing every DL
    filter module into this process at once."""
    result = {}
    loo_key = f'__loo_held_{nav_data.dataset_name}__'
    cv_key = '__cv_kitti__'

    for key in ['esekfg_vanilla', 'esekfg_enhanced', 'esekfs_vanilla', 'esekfs_enhanced',
                'iekf_vanilla', 'iekf_enhanced']:
        p = (fp.get(key, mode_3d, loo_key)
             or fp.get(key, mode_3d, nav_data.dataset_name)
             or fp.get(key, mode_3d, cv_key))
        if p is not None:
            result[key] = p

    for key in ['isam2', 'isam2_fixedlag', 'isam2_map']:
        p = (fp.get(key, mode_3d, loo_key)
             or fp.get(key, mode_3d, nav_data.dataset_name)
             or fp.get(key, mode_3d, cv_key))
        if p is not None:
            result[key] = p
    _base_isam2 = result.get('isam2')
    if _base_isam2 is not None:
        for k in ['isam2_fixedlag', 'isam2_map']:
            if k not in result:
                result[k] = dict(_base_isam2)

    _best_classical = (result.get('esekfs_enhanced') or result.get('esekfs_vanilla')
                       or result.get('esekfg_enhanced') or result.get('esekfg_vanilla'))
    if _best_classical is not None:
        _shared = ['Qpos', 'Qvel', 'Qacc', 'Rpos', 'beta_acc', 'beta_gyr',
                   'P_pos_std', 'P_vel_std', 'P_orient_std', 'P_acc_std', 'P_gyr_std']
        _dl_base = {k: _best_classical[k] for k in _shared if k in _best_classical}
        if 'QorientXY' in _best_classical:
            _dl_base['Qorient'] = (_best_classical.get('QorientXY', 1e-5)
                                   + _best_classical.get('QorientZ', 1e-5)) / 2.0
        if 'QgyrXY' in _best_classical:
            _dl_base['Qgyr'] = (_best_classical.get('QgyrXY', 1e-7)
                                + _best_classical.get('QgyrZ', 1e-7)) / 2.0
        for dl_key in ['tlio', 'deep_kf', 'tartan_imu', 'iekf_ai_imu', 'iekf_ai_imu_online']:
            if dl_key not in result:
                result[dl_key] = _dl_base.copy()

    return result


def _truncate(nav, n):
    nd = copy.copy(nav)
    for f in ("accel_flu", "gyro_flu", "vel_enu", "lla", "orient",
              "gps_available", "gps_speed_mps", "gps_cog_rad", "time"):
        v = getattr(nd, f, None)
        if v is not None:
            setattr(nd, f, v[:n])
    return nd


def _run_classical(approach, nav, params, outage_cfg, use_3d=True):
    mod = importlib.import_module(f"filters.{approach}")
    t0 = time.perf_counter()
    result = mod.run(nav, params=params, outage_config=outage_cfg, use_3d_rotation=use_3d)
    return time.perf_counter() - t0, None, result


def _run_dl(approach, nav, params, outage_cfg, backend, seq, use_3d=True):
    sys.path.insert(0, str(_PY_DIR / DL_RUNNER_DIR[approach]))
    mod = importlib.import_module(DL_RUNNER_MOD[approach])

    # deep_iekf's run() resolves its causal-MesNet weights via
    # _find_online_weights(), whose OWN priority order checks a generic
    # artifacts/deep_iekf_online/iekfnets.p file (a leftover single-file
    # copy from whatever training run happened last) BEFORE it ever looks
    # at the real per-LOO-fold fold_<seq>.p files — so left alone, every
    # sequence silently gets the same wrong (non-held-out) weights on both
    # backends, since even the Hailo path still calls _load_causal_torch_iekf
    # for the process-noise/init-covariance scaling. ins_compare.py avoids
    # this by exporting AI_IMU_ONLINE_WEIGHTS itself before calling run();
    # replicate that exact resolution here.
    prev_env = os.environ.get('AI_IMU_ONLINE_WEIGHTS')
    if approach == "deep_iekf":
        fold_weights = _REPO_ROOT / f"artifacts/deep_iekf_online/fold_{seq}.p"
        held_weights = _REPO_ROOT / f"artifacts/deep_iekf_online/iekfnets_held_{nav.dataset_name}.p"
        weights = fold_weights if fold_weights.exists() else held_weights
        if not weights.exists():
            raise RuntimeError(f"No per-fold deep_iekf weights found for seq {seq} "
                               f"(tried {fold_weights} and {held_weights})")
        os.environ['AI_IMU_ONLINE_WEIGHTS'] = str(weights)

    hailo_cm = hailo_net = None
    if backend == "hailo":
        sys.path.insert(0, str(_HERE))
        import hailo_backend
        if approach == "deep_kf":
            hailo_cm = hailo_backend.HailoDeepKF(HEF_PATH[approach])
        elif approach == "tlio":
            hailo_cm = hailo_backend.HailoTLIO(HEF_PATH[approach], POSTPROC_PATH[approach])
        elif approach == "tartan_imu":
            hailo_cm = hailo_backend.HailoTartanIMU(HEF_PATH[approach], POSTPROC_PATH[approach])
        elif approach == "deep_iekf":
            hailo_cm = hailo_backend.HailoDeepIEKF(HEF_PATH[approach], POSTPROC_PATH[approach])
        hailo_net = hailo_cm.__enter__()

    try:
        t0 = time.perf_counter()
        result = mod.run(nav, params=params, outage_config=outage_cfg,
                         use_3d_rotation=use_3d, backend=backend, hailo_net=hailo_net)
        wall_s = time.perf_counter() - t0
    finally:
        if hailo_cm is not None:
            hailo_cm.__exit__(None, None, None)
        if approach == "deep_iekf":
            if prev_env is None:
                os.environ.pop('AI_IMU_ONLINE_WEIGHTS', None)
            else:
                os.environ['AI_IMU_ONLINE_WEIGHTS'] = prev_env

    net_latency_s = np.asarray(result.get('net_latency_s', []), dtype=float)
    return wall_s, net_latency_s, result


def _run_smoother(approach, nav, params, outage_cfg, use_3d=True):
    with tempfile.TemporaryDirectory() as td:
        nav_npz = Path(td) / "nav.npz"
        params_json = Path(td) / "params.json"
        out_npz = Path(td) / "out.npz"

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
              "--runner", approach, "--nav-npz", str(nav_npz),
              "--params-json", str(params_json),
              "--outage-start", str(outage_cfg['start']),
              "--outage-duration", str(outage_cfg['duration']),
              "--use-3d", "1" if use_3d else "0",
              "--out-npz", str(out_npz)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        if proc.returncode != 0:
            raise RuntimeError(f"{approach} conda worker failed:\n{proc.stdout}\n{proc.stderr}")
        out = np.load(out_npz)
        if "wall_s" in out.files:
            wall_s = float(out["wall_s"])
        result = {k: out[k] for k in ("p", "v", "r", "bias_acc", "bias_gyr") if k in out.files}
    return wall_s, None, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True, choices=CLASSICAL + DL + SMOOTHERS)
    ap.add_argument("--backend", required=True, choices=["cpu", "hailo"])
    ap.add_argument("--scenario", required=True, choices=["no_outage", "outage"])
    ap.add_argument("--seq", required=True, help="KITTI seq id, e.g. '01'")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.approach in CLASSICAL + SMOOTHERS and args.backend == "hailo":
        raise ValueError(f"{args.approach} has no Hailo variant")

    import data_loader
    import filter_params as fp

    nav = data_loader.get_kitti_dataset(args.seq)
    full_duration_s = len(nav.accel_flu) / nav.sample_rate

    truncated = False
    if args.approach == "deep_iekf" and args.backend == "hailo":
        nav = _truncate(nav, DEEP_IEKF_SEQ_LEN)
        truncated = True
        outage_cfg = DEEP_IEKF_HAILO_OUTAGE if args.scenario == "outage" else {'start': 0., 'duration': 0.}
    else:
        outage_cfg = ({'start': OUTAGE_START, 'duration': OUTAGE_DURATION}
                      if args.scenario == "outage" else {'start': 0., 'duration': 0.})

    tuned = _load_tuned_params(nav, True, fp)
    tuned_key = DL_TUNED_KEY[args.approach] if args.approach in DL else args.approach
    params = tuned.get(tuned_key)

    if args.approach in CLASSICAL:
        wall_s, net_latency_s, result = _run_classical(args.approach, nav, params, outage_cfg)
    elif args.approach in SMOOTHERS:
        wall_s, net_latency_s, result = _run_smoother(args.approach, nav, params, outage_cfg)
    else:
        wall_s, net_latency_s, result = _run_dl(args.approach, nav, params, outage_cfg, args.backend, args.seq)

    duration_s = (DEEP_IEKF_SEQ_LEN / nav.sample_rate) if truncated else full_duration_s
    timing = {
        'wall_s': float(wall_s),
        'dataset_duration_s': float(duration_s),
        'real_time_factor': float(duration_s / wall_s) if wall_s > 0 else None,
        'truncated': truncated,
    }
    if net_latency_s is not None and len(net_latency_s):
        timing.update({
            'n_calls': int(len(net_latency_s)),
            'mean_ms': float(np.mean(net_latency_s) * 1000),
            'median_ms': float(np.median(net_latency_s) * 1000),
            'p95_ms': float(np.percentile(net_latency_s, 95) * 1000),
        })

    out = {'approach': args.approach, 'backend': args.backend,
          'scenario': args.scenario, 'seq': args.seq, 'timing': timing}

    # Computed uniformly for every backend (CPU and Hailo alike) so every
    # number in every table comes from a run on THIS device — using the
    # EXACT same methodology ins_compare.py uses: ATE/RMSE against the
    # FGO-Batch ground truth (ins_config.GT_SOURCE == 'batch', the setting
    # the historical cached tables were generated with — a GTSAM factor-
    # graph-smoothed trajectory, NOT raw GPS; see
    # _precompute_batch_gt.py, which must be run first), while t_rel/r_rel
    # (kitti_metrics) are — per ins_compare.py's own comment — "always vs
    # raw KITTI GPS", regardless of GT_SOURCE.
    if result is not None:
        import pymap3d as pm
        import metrics as _metrics

        lla, lla0 = nav.lla, nav.lla0
        e, n, u = pm.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], lla0[0], lla0[1], lla0[2])
        p_kitti = np.column_stack([e, n, u])

        gt_cache = _HERE / "full_benchmark_results" / "gt_cache" / f"{args.seq}.npz"
        if not gt_cache.exists():
            raise RuntimeError(
                f"Batch ground truth cache missing: {gt_cache}. "
                f"Run scripts/positioning/hailo/_precompute_batch_gt.py first.")
        p_gt_batch = np.load(gt_cache)['p']
        if truncated:
            p_gt_batch = p_gt_batch[:DEEP_IEKF_SEQ_LEN]

        t1, d = outage_cfg['start'], outage_cfg['duration']
        gnss_outage_info = {
            'start': t1, 'end': t1 + d, 'duration': d,
            'start_idx': int(t1 * nav.sample_rate), 'end_idx': int((t1 + d) * nav.sample_rate),
        }
        mets = _metrics.evaluate_navigation_performance(
            p_est=result['p'], v_est=result['v'], r_est=result['r'],
            p_gt=p_gt_batch, v_gt=nav.vel_enu, r_gt=nav.orient,
            dataset_name=nav.dataset_name, gnss_outage_info=gnss_outage_info,
            sample_rate=nav.sample_rate,
        )
        kitti_mets = _metrics.compute_kitti_metrics(
            p_est=result['p'], r_est=result['r'], p_gt=p_kitti, r_gt=nav.orient,
        )
        out['accuracy'] = {
            'ate': {k: v for k, v in mets['ate'].items() if k != 'errors'},
            'position_rmse': mets['position_rmse'],
            'kitti_metrics': kitti_mets,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"OK {args.approach}/{args.backend}/{args.scenario}/{args.seq} -> {args.out} ({wall_s:.3f}s)")


if __name__ == "__main__":
    main()
