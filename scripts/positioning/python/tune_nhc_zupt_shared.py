# -*- coding: utf-8 -*-
"""
tune_nhc_zupt_shared.py — pooled (non-LOO-per-fold) search for a single shared
NHC/ZUPT parameter set per enhanced filter.

Why this exists
----------------
`iekf_enhanced` / `esekfg_enhanced` / `esekfs_enhanced` each augment their
vanilla counterpart with two pseudo-measurements (Non-Holonomic Constraints
and Zero-Velocity Update). ins_genetic_cv.py's search space (ins_cost.BOUNDS,
15-D) never included the four NHC/ZUPT knobs (Rnhc, Rzupt,
zupt_accel_threshold, zupt_gyro_threshold) — they've been running at
hardcoded, never-validated textbook defaults (0.1, 0.01, 0.3, 0.05) for every
sequence and every fold.

Rather than adding them to the per-fold LOO search (which would let a test
sequence's own held-out-ness be diluted across 4 more free dimensions per
fold, and risks overfitting these "physical assumption" constants to KITTI's
specific 7 sequences), this script finds ONE shared value per filter,
evaluated by pooling cost across all 7 clean KITTI sequences — using each
sequence's ALREADY per-fold-tuned base 15 parameters (from filter_params.json)
as the backdrop, varying only the 4 NHC/ZUPT knobs on top. This keeps the
existing, working LOO tuning of the base parameters untouched.

Usage
-----
    python tune_nhc_zupt_shared.py iekf_enhanced
    python tune_nhc_zupt_shared.py iekf_enhanced esekfg_enhanced esekfs_enhanced
    python tune_nhc_zupt_shared.py --maxiter 20 --popsize 12 iekf_enhanced
"""
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ins_cost
from data_loader import get_kitti_dataset
from filters import esekfg_enhanced, esekfs_enhanced, iekf_enhanced

_FILTER_MODULES = {
    'iekf_enhanced':   iekf_enhanced,
    'esekfg_enhanced': esekfg_enhanced,
    'esekfs_enhanced': esekfs_enhanced,
}

# seq id -> KITTI drive name (must match ins_genetic_cv.KITTI_CLEAN_DRIVES)
SEQ_TO_DRIVE = {
    '01': '2011_10_03_drive_0042_extract',
    '04': '2011_09_30_drive_0016_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}

# 4-D log10 search space for the shared NHC/ZUPT parameters.
# Current textbook defaults: Rnhc=0.1, Rzupt=0.01, accel_thr=0.3, gyro_thr=0.05.
NHC_ZUPT_BOUNDS = [
    (-3, 1),     # log10(Rnhc)                  [(m/s)^2]
    (-4, 1),     # log10(Rzupt)                 [(m/s)^2]
    (-2, 1),     # log10(zupt_accel_threshold)  [m/s^2]
    (-3, 0),     # log10(zupt_gyro_threshold)   [rad/s]
]
NHC_ZUPT_KEYS = ['Rnhc', 'Rzupt', 'zupt_accel_threshold', 'zupt_gyro_threshold']


def decode_nhc_zupt(x):
    return dict(zip(NHC_ZUPT_KEYS, (10.0 ** xi for xi in x)))


# Sequences long enough for a 40s-start/60s-duration outage window (matches
# the actual benchmark in ins_compare.py / emit_journal_tables.py). seq04
# (29.7s) is excluded — it's excluded from the outage table for the same
# reason (too short), so including it here would just inject a guaranteed
# COST_REJECT into every candidate's average regardless of NHC/ZUPT values.
OUTAGE_START, OUTAGE_DURATION = 40.0, 60.0


class _PooledCost:
    """Picklable objective for scipy DE with multiprocessing workers.

    Stores the filter as a string key (not the module object) — modules
    aren't picklable, but each worker process can re-resolve the key via
    _FILTER_MODULES after importing this file.
    """

    def __init__(self, filter_key, base_params_per_seq, nav_by_seq):
        self.filter_key = filter_key
        self.base_params_per_seq = base_params_per_seq
        self.nav_by_seq = nav_by_seq

    def __call__(self, x):
        module = _FILTER_MODULES[self.filter_key]
        cand = decode_nhc_zupt(x)
        costs = []
        for seq, base in self.base_params_per_seq.items():
            nav = self.nav_by_seq[seq]
            # single_window_cost requires d>0 (t1=0,d=0 short-circuits to an
            # instant, uninformative COST_REJECT) — it has no "no outage"
            # mode, so we can only evaluate sequences long enough for the
            # actual 40s/60s outage window used by the real benchmark.
            if (OUTAGE_START + OUTAGE_DURATION) * nav.sample_rate >= len(nav.lla):
                continue
            merged = {**base, **cand}
            costs.append(ins_cost.single_window_cost(
                module, nav, merged, t1=OUTAGE_START, d=OUTAGE_DURATION, use_3d=True))
        return float(np.mean(costs)) if costs else ins_cost.COST_REJECT


_HERE_PARAMS = _HERE / 'filter_params.json'


def load_base_params_per_seq(filter_key: str) -> dict:
    """Per-sequence base (15-D) params from that sequence's own LOO fold —
    i.e. the params already used to *test* on this sequence in ins_compare.py."""
    all_params = json.load(open(_HERE_PARAMS))
    entry = all_params[filter_key]['3d']
    out = {}
    for seq, drive in SEQ_TO_DRIVE.items():
        fold_key = f'__loo_held_{drive}__'
        if fold_key not in entry:
            raise KeyError(f"{filter_key}: missing tuned params for {fold_key}")
        out[seq] = entry[fold_key]['params']
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filters', nargs='+', choices=list(_FILTER_MODULES.keys()))
    parser.add_argument('--maxiter', type=int, default=20)
    parser.add_argument('--popsize', type=int, default=12)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--workers', type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)

    log.info("Loading all 7 clean KITTI sequences ...")
    nav_by_seq = {seq: get_kitti_dataset(drive) for seq, drive in SEQ_TO_DRIVE.items()}

    for filt in args.filters:
        module = _FILTER_MODULES[filt]
        base_params = load_base_params_per_seq(filt)
        default_nhc_zupt = {k: module.DEFAULT_PARAMS[k] for k in NHC_ZUPT_KEYS}
        log.info(f"=== {filt} ===  current defaults: {default_nhc_zupt}")

        pooled_cost = _PooledCost(filt, base_params, nav_by_seq)

        # Baseline: current hardcoded textbook defaults, for comparison.
        x0 = [np.log10(default_nhc_zupt[k]) for k in NHC_ZUPT_KEYS]
        baseline_cost = pooled_cost(x0)
        log.info(f"{filt}: baseline (textbook defaults) pooled cost = {baseline_cost:.4f}")

        result = differential_evolution(
            pooled_cost, NHC_ZUPT_BOUNDS,
            maxiter=args.maxiter, popsize=args.popsize,
            seed=args.seed, workers=args.workers, updating='deferred',
            polish=True, tol=1e-6,
        )

        best = decode_nhc_zupt(result.x)
        log.info(f"{filt}: DE best pooled cost = {result.fun:.4f} "
                 f"(baseline {baseline_cost:.4f}, "
                 f"{'IMPROVED' if result.fun < baseline_cost else 'NOT improved'})")
        log.info(f"{filt}: shared NHC/ZUPT params -> {json.dumps(best, indent=2)}")

        out_path = _HERE / f'{filt}_shared_nhc_zupt.json'
        json.dump({
            'filter': filt,
            'baseline_defaults': default_nhc_zupt,
            'baseline_cost': baseline_cost,
            'tuned_params': best,
            'tuned_cost': float(result.fun),
            'per_seq_base_params_source': 'filter_params.json (own LOO fold)',
            'maxiter': args.maxiter, 'popsize': args.popsize, 'seed': args.seed,
        }, open(out_path, 'w'), indent=2)
        log.info(f"{filt}: saved -> {out_path}")


if __name__ == '__main__':
    main()
