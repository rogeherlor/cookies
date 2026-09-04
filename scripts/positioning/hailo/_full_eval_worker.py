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
import os

# ── Measurement hygiene — MUST run before numpy/torch are imported ───────────
# BLAS/OMP default to one thread per core. On this board that is actively
# counterproductive for the tensor sizes these filters use (15x15 covariances,
# small 1-D conv windows): thread fan-out/join costs more than the arithmetic,
# and the resulting scheduling jitter lands squarely in the per-event timing.
# Measured effect of NOT setting these: TLIO's median network call read
# 64.51 ms instead of 15.96 ms (4x), Tartan IMU 76.06 instead of 36.02 ms,
# and the once-per-second classical filters showed 15-20x inflated 95th
# percentiles that moved randomly from run to run (ES-EKF Groves' outage
# event median read 0.48 +/- 0.56 ms against a true 0.23 +/- 0.02 ms).
# Setting them here rather than in the caller means a plain
# `python3 _full_eval_worker.py ...` is already correct and the protocol
# cannot be forgotten. See README.md, "Timing measurement protocol".
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import importlib
import json
import platform
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

# Fraction of the assigned cores that may already be busy before a timed run is
# considered contaminated. 0.5 means: on a 3-core taskset, a 1-minute load above
# 1.5 says something else is running and these latencies will include waiting
# for it. This is the "measuring other processes" failure the wall/cpu ratio
# catches after the fact — checking up front means a bad run is flagged even
# when the interference lands somewhere the per-event check cannot see it.
LOAD_WARN_FRACTION = 0.5


def _loadavg():
    """1-minute load average, or None where /proc is unavailable."""
    try:
        with open('/proc/loadavg') as fh:
            return float(fh.read().split()[0])
    except Exception:
        return None

_DEEP_IEKF_STREAM_DIR = _HERE / "deep_iekf_stream"

_LOADAVG_BEFORE = None   # set in main() before any timed work


def _deep_iekf_stream_paths(seq):
    """(hef, postproc) for the per-LOO-fold STREAMING deep_iekf HEF (W=32,
    any sequence length, one call per IMU sample), or None if this fold was
    never compiled. When it exists it is strictly preferred over the old
    fixed 4544-sample whole-sequence HEF: that one forces the caller to
    truncate the drive to min(N,4544) samples AND to substitute a shorter
    outage window that fits inside it, which makes the resulting row
    incomparable to every other row in the tables."""
    hef = _DEEP_IEKF_STREAM_DIR / f"deep_iekf_stream_fold_{seq}.hef"
    postproc = _DEEP_IEKF_STREAM_DIR / f"deep_iekf_stream_fold_{seq}_postproc.npz"
    return (hef, postproc) if hef.exists() and postproc.exists() else None


def _fold_hef(approach, seq):
    """(hef, postproc) for the per-LOO-fold build of tlio/deep_kf/tartan_imu, or
    None if this fold was never compiled. postproc is None for deep_kf, which
    runs entirely on-device.

    The CPU backends load per-fold weights and refuse any other checkpoint
    (tlio_runner.py::_find_weights raises rather than leak). Until
    build_per_fold_hefs.py has been run on an x86 host with the Dataflow
    Compiler, the Hailo side has only ONE .hef per model, traced to fold_01,
    so 6 of the 7 sequences are evaluated on a model that trained on them.
    Once the per-fold files exist this picks them up automatically and the
    two backends become comparable; until then the caller falls back to the
    single HEF and the accuracy comparison for these three models is not
    leave-one-out. See todo.md.

    BOTH files are required together. For tlio and tartan_imu the accelerator
    holds only the backbone — the head (tlio's bn1+fc1/2/3, tartan's
    LSTM+Trunk+robot head) runs on the host from <model>_postproc.pt, exported
    from the same fold checkpoint and therefore just as fold-specific. Accepting
    a per-fold .hef while silently keeping the single-fold postproc would pair
    fold F's backbone with another fold's head: still leaking, and now also
    numerically incoherent. If only one of the pair is present, treat the fold as
    not built."""
    base = HEF_PATH.get(approach)
    if base is None:
        return None
    hef = base.with_name(base.name.replace('.hef', f'_fold_{seq}.hef'))
    if not hef.exists():
        return None
    pp_base = POSTPROC_PATH.get(approach)
    if pp_base is None:
        return (hef, None)
    pp = pp_base.with_name(pp_base.name.replace('.pt', f'_fold_{seq}.pt'))
    if not pp.exists():
        print(f"  WARNING: {hef.name} exists but {pp.name} does not; the host-side "
              f"head is fold-specific, so this pair is unusable. Treating "
              f"{approach} seq {seq} as not built.", file=sys.stderr)
        return None
    return (hef, pp)


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
    wall_s = time.perf_counter() - t0
    # One entry per sample on which a measurement update fired (GPS correction,
    # and for the enhanced variants the NHC/ZUPT tier too) — see each filter's
    # `event_latency_s` instrumentation.
    event_latency_s = np.asarray(result.get('event_latency_s', []), dtype=float)
    # Thread CPU time for the same events (see the filters' `event_cpu_s`).
    # Counts only cycles actually retired on this thread, so preemption is
    # invisible to it — main() turns the wall/CPU ratio into a self-check.
    _cpu = np.asarray(result.get('event_cpu_s', []), dtype=float)
    return wall_s, event_latency_s, result, _cpu


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
        # Per-fold (HEF, postproc) when built, else the single all-folds pair
        # (which is NOT leave-one-out for 6 of 7 sequences — see _fold_hef and
        # todo.md).
        _pair = _fold_hef(approach, seq)
        if _pair is not None:
            _hef, _pp = _pair
        else:
            _hef, _pp = HEF_PATH.get(approach), POSTPROC_PATH.get(approach)
            if approach in ("deep_kf", "tlio", "tartan_imu"):
                print(f"  WARNING: no per-fold HEF for {approach} seq {seq}; using "
                      f"{HEF_PATH[approach].name}, which was compiled from a single "
                      f"fold — this row is not LOO and its CPU-vs-Hailo accuracy "
                      f"comparison measures train/test leakage, not quantisation.",
                      file=sys.stderr)
        if approach == "deep_kf":
            hailo_cm = hailo_backend.HailoDeepKF(_hef)
        elif approach == "tlio":
            hailo_cm = hailo_backend.HailoTLIO(_hef, _pp)
        elif approach == "tartan_imu":
            hailo_cm = hailo_backend.HailoTartanIMU(_hef, _pp)
        elif approach == "deep_iekf":
            # Prefer the per-fold STREAMING HEF (W=32, any-length, genuine
            # online) when present; fall back to the fixed 4544 whole-sequence
            # HEF otherwise. The streaming HEF emits bounded z and reconstructs
            # cov=cov0*10**(beta*z) on the host (see HailoDeepIEKFStream / the
            # deep_iekf_stream build) — the only path that covers full KITTI
            # sequences (the 4544 HEF only reaches min(N,4544) = 8..40% of most).
            _stream = _deep_iekf_stream_paths(seq)
            if _stream is not None:
                _shef, _spp = _stream
                # 'per_tick', not 'block': one HEF call per newly arrived IMU
                # sample, which is what a live sensor stream actually looks
                # like and what makes the measured Event latency comparable to
                # the 10ms sample deadline. 'block' is faster per sample but
                # waits for FRESH=16 samples before it can emit any of them —
                # 160ms of input buffering that a real-time deployment does not
                # have, and it would make the reported per-call latency cover
                # 16 samples' worth of work rather than one.
                _mode = os.environ.get("DEEP_IEKF_STREAM_MODE", "per_tick")
                hailo_cm = hailo_backend.HailoDeepIEKFStream(_shef, _spp, mode=_mode)
            else:
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
    # Thread CPU time for the same events. On the CPU backend this is the same
    # contamination self-check the classical filters have always had; on the
    # Hailo backend it is the host-side share of each call (the device wait is
    # not this thread's CPU time), which is reported but never warned on.
    net_cpu_s = np.asarray(result.get('net_cpu_s', []), dtype=float)
    return wall_s, net_latency_s, result, net_cpu_s


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
        # Factor-graph solve time per GPS epoch (ms -> s). update_ms is a
        # full-length array that is zero on the ~99 of 100 IMU samples where no
        # solve runs; only the epochs that actually solved are Events.
        event_latency_s = np.zeros(0)
        if "update_ms" in out.files:
            update_ms = np.asarray(out["update_ms"], dtype=float)
            event_latency_s = update_ms[update_ms > 0.0] / 1e3
    return wall_s, event_latency_s, result


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

    # Sample the machine BEFORE any work, so the reading reflects pre-existing
    # load rather than this process's own.
    global _LOADAVG_BEFORE
    _LOADAVG_BEFORE = _loadavg()
    _n_aff = len(os.sched_getaffinity(0))
    if _LOADAVG_BEFORE is not None and _LOADAVG_BEFORE > LOAD_WARN_FRACTION * _n_aff:
        print(f"  WARNING: 1-min load average {_LOADAVG_BEFORE:.2f} on "
              f"{_n_aff} assigned core(s) BEFORE this run — the machine is not "
              f"idle. Latencies measured now include contention with whatever "
              f"else is running; re-measure on a quiet machine.", file=sys.stderr)

    import data_loader
    import filter_params as fp

    nav = data_loader.get_kitti_dataset(args.seq)
    full_duration_s = len(nav.accel_flu) / nav.sample_rate

    # Truncation + the substitute short outage window are ONLY needed by the
    # legacy fixed-length (4544-sample) deep_iekf HEF. With the streaming HEF
    # compiled for this fold, the Hailo run covers the whole drive under the
    # exact same 40s/60s window as every other filter, so its row is directly
    # comparable — which is the whole point of having compiled it.
    truncated = False
    if (args.approach == "deep_iekf" and args.backend == "hailo"
            and _deep_iekf_stream_paths(args.seq) is None):
        nav = _truncate(nav, DEEP_IEKF_SEQ_LEN)
        truncated = True
        outage_cfg = DEEP_IEKF_HAILO_OUTAGE if args.scenario == "outage" else {'start': 0., 'duration': 0.}
    else:
        outage_cfg = ({'start': OUTAGE_START, 'duration': OUTAGE_DURATION}
                      if args.scenario == "outage" else {'start': 0., 'duration': 0.})

    tuned = _load_tuned_params(nav, True, fp)
    tuned_key = DL_TUNED_KEY[args.approach] if args.approach in DL else args.approach
    params = tuned.get(tuned_key)

    event_cpu_s = None
    if args.approach in CLASSICAL:
        wall_s, event_latency_s, result, event_cpu_s = _run_classical(args.approach, nav, params, outage_cfg)
    elif args.approach in SMOOTHERS:
        wall_s, event_latency_s, result = _run_smoother(args.approach, nav, params, outage_cfg)
    else:
        wall_s, event_latency_s, result, event_cpu_s = _run_dl(
            args.approach, nav, params, outage_cfg, args.backend, args.seq)

    duration_s = (DEEP_IEKF_SEQ_LEN / nav.sample_rate) if truncated else full_duration_s
    timing = {
        'wall_s': float(wall_s),
        'dataset_duration_s': float(duration_s),
        'real_time_factor': float(duration_s / wall_s) if wall_s > 0 else None,
        'truncated': truncated,
        # Measurement conditions, recorded per run so a number can be audited
        # after the fact instead of trusted. A row measured on N cores is not
        # comparable to one measured on M, and _hailo_rerun.sh used to pin to
        # `taskset -c 1,2,3` while run_full_benchmark.py pinned nothing — which
        # made the Hailo and CPU halves of the same table incomparable without
        # anything in the output saying so.
        'n_cpus_affinity': len(os.sched_getaffinity(0)),
        'loadavg_1min_before': _LOADAVG_BEFORE,
        'loadavg_1min_after': _loadavg(),
    }
    # Every one of the 13 filters is now Event-instrumented (network call for
    # the DL ones, GPS/NHC/ZUPT correction for the classical ones, factor-graph
    # solve for the smoothers), so these fields are always emitted — a filter
    # that legitimately fired no event records n_calls=0 rather than dropping
    # the fields, which build_latex_tables.py would otherwise read as "not
    # instrumented" and print as TBD.
    if event_latency_s is not None:
        n = int(len(event_latency_s))
        timing.update({
            'n_calls': n,
            'mean_ms': float(np.mean(event_latency_s) * 1000) if n else 0.0,
            'median_ms': float(np.median(event_latency_s) * 1000) if n else 0.0,
            'p95_ms': float(np.percentile(event_latency_s, 95) * 1000) if n else 0.0,
        })
        # Contamination self-check. CPU time excludes any interval this thread
        # was descheduled, so wall/cpu ~= 1 is positive evidence the run was
        # not preempted; a ratio well above 1 means the numbers are polluted
        # by other load and should be re-measured on a quiet machine rather
        # than reported. All 11 CPU-backed filters (7 classical + 4 DL) now
        # expose per-event CPU time, so the check covers every row that is
        # pure host compute — previously only the classical ones did, and a
        # preempted TLIO/Tartan/DKF/DeepIEKF CPU run was reported as a genuine
        # latency. On the Hailo backend the same field is the HOST share of
        # each call: the thread really is blocked on the device for the rest,
        # so a high ratio there is the expected offload signature, not
        # contamination, and is recorded without a warning.
        if event_cpu_s is not None and len(event_cpu_s) == n and n:
            _cm = float(np.median(event_cpu_s) * 1000)
            timing['cpu_median_ms'] = _cm
            timing['cpu_p95_ms'] = float(np.percentile(event_cpu_s, 95) * 1000)
            timing['wall_cpu_ratio'] = (timing['median_ms'] / _cm) if _cm > 0 else None
            _is_device_backed = (args.backend == 'hailo')
            timing['wall_cpu_ratio_gated'] = not _is_device_backed
            if (not _is_device_backed and timing['wall_cpu_ratio']
                    and timing['wall_cpu_ratio'] > 1.25):
                print(f"  WARNING: wall/cpu = {timing['wall_cpu_ratio']:.2f} "
                      f"(>1.25) — this run was preempted; treat the timing as "
                      f"contaminated and re-measure on an idle machine.",
                      file=sys.stderr)

    # Provenance. Accuracy is machine-independent (same HEF, same INT8 weights,
    # bit-identical results on any Hailo-8L), but TIMING is not: a sweep run on
    # the x86 build host and a sweep run on the Pi write to the same
    # all_results.json, and without this nothing in the file says which is which.
    # build_latex_tables.py refuses to mix them.
    # 'machine' is the comparability key, NOT 'node': inside Docker node() is the
    # container ID, which changes every time the container is recreated, so
    # keying on it would flag a perfectly consistent Pi sweep as mixed. The
    # architecture (x86_64 build host vs aarch64 Pi) is what actually makes two
    # timing numbers incomparable. BENCH_HOST_ID overrides it when you need a
    # finer distinction (e.g. two different aarch64 boards).
    out = {'approach': args.approach, 'backend': args.backend,
          'scenario': args.scenario, 'seq': args.seq, 'timing': timing,
          'host': {'node': platform.node(),
                   'machine': os.environ.get('BENCH_HOST_ID') or platform.machine()}}

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
