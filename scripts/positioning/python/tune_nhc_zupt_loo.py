# -*- coding: utf-8 -*-
"""
tune_nhc_zupt_loo.py — properly nested leave-one-out tuning of the shared
NHC/ZUPT parameters for the enhanced filters.

Supersedes tune_nhc_zupt_shared.py, which pooled cost across ALL 6
outage-eligible sequences in a single search — meaning each sequence's own
test performance was part of the objective that produced the value later
applied to it (real, if diluted, leakage). This version instead runs one
search PER held-out fold, using only the OTHER sequences to tune that fold's
shared NHC/ZUPT value — matching the same LOO discipline already used for
the base 15 parameters in filter_params.json.

For each of the 3 enhanced filters and each held-out sequence in
{01, 06, 07, 08, 09, 10} (seq 04 excluded — too short for the 40s/60s outage
window under any parameters, same reason it's excluded from the outage
benchmark table):
    pool = the other 5 of those 6 sequences (seq 04 is never in the pool
           either — it can never contribute a valid outage-window cost)
    For each seq in pool: use ITS OWN per-fold-tuned base 15 params (from
        filter_params.json, from the fold where THAT seq was held out).
    DE-search the 4-D NHC/ZUPT space minimizing mean cost over the pool.
    Write the result into filter_params.json's existing
    "__loo_held_<held-out drive>__" entry for this filter (merging the 4
    new keys into its "params" dict) — so ins_compare.py picks it up via
    the exact same fp.get()/_load_tuned_params() path already used for
    the base parameters, no new plumbing needed.

seq04's own fold entry is left untouched (falls back to DEFAULT_PARAMS
textbook values) since it's never evaluated on the outage scenario anyway.

Performance note
-----------------
Workers are a persistent multiprocessing.Pool with an initializer: each
worker process loads its own copy of the 7 KITTI NavigationData objects
directly from disk ONCE, rather than having the ~2.5 MB+ arrays re-pickled
over IPC on every one of the ~400+ evaluations per fold (this was the
dominant cost in an earlier version — workers sat at ~30% CPU utilization,
not ~100%, because most wall-clock time went to (de)serializing the
objective's closure state instead of computing).

Usage
-----
    python tune_nhc_zupt_loo.py iekf_enhanced
    python tune_nhc_zupt_loo.py iekf_enhanced esekfg_enhanced esekfs_enhanced
    python tune_nhc_zupt_loo.py --maxiter 10 --popsize 10 iekf_enhanced
"""
import os

# Must be set BEFORE numpy/BLAS is imported anywhere in this process (forked
# workers inherit an already-initialized BLAS runtime, so setting this later —
# e.g. in the worker initializer — is too late). Without this, each of the 16
# worker processes' own OpenBLAS backend tries to multi-thread even tiny
# 15x15 covariance matmuls, wildly oversubscribing the 32 physical cores —
# observed as ~30% CPU utilization per worker instead of ~100%.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import sys
import json
import argparse
import logging
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ins_cost  # noqa: E402
from data_loader import get_kitti_dataset  # noqa: E402
from filters import esekfg_enhanced, esekfs_enhanced, iekf_enhanced  # noqa: E402

_FILTER_MODULES = {
    'iekf_enhanced':   iekf_enhanced,
    'esekfg_enhanced': esekfg_enhanced,
    'esekfs_enhanced': esekfs_enhanced,
}

SEQ_TO_DRIVE = {
    '01': '2011_10_03_drive_0042_extract',
    '04': '2011_09_30_drive_0016_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}
# seq04 is too short (29.7s) for the 40s-start/60s-duration outage window
# under any parameters — same reason it's "--" in the outage benchmark table.
OUTAGE_ELIGIBLE_SEQS = ['01', '06', '07', '08', '09', '10']

OUTAGE_START, OUTAGE_DURATION = 40.0, 60.0

NHC_ZUPT_BOUNDS = [
    (-3, 1),     # log10(Rnhc)
    (-4, 1),     # log10(Rzupt)
    (-2, 1),     # log10(zupt_accel_threshold)
    (-3, 0),     # log10(zupt_gyro_threshold)
]
NHC_ZUPT_KEYS = ['Rnhc', 'Rzupt', 'zupt_accel_threshold', 'zupt_gyro_threshold']


def decode_nhc_zupt(x):
    return dict(zip(NHC_ZUPT_KEYS, (10.0 ** xi for xi in x)))


_PARAMS_PATH = _HERE / 'filter_params.json'


def load_base_params_per_seq(filter_key: str) -> dict:
    """Per-sequence base (15-D) params from that sequence's own LOO fold."""
    all_params = json.load(open(_PARAMS_PATH))
    entry = all_params[filter_key]['3d']
    out = {}
    for seq, drive in SEQ_TO_DRIVE.items():
        fold_key = f'__loo_held_{drive}__'
        if fold_key not in entry:
            raise KeyError(f"{filter_key}: missing tuned params for {fold_key}")
        out[seq] = entry[fold_key]['params']
    return out


# ── Worker-local globals — populated ONCE per worker process via the pool
# initializer, never re-pickled per evaluation ─────────────────────────────
_W_NAV_BY_SEQ = None
_W_FILTER_KEY = None
_W_POOL_SEQS = None
_W_BASE_PARAMS = None


def _worker_init(filter_key, pool_seqs, base_params_per_seq):
    global _W_NAV_BY_SEQ, _W_FILTER_KEY, _W_POOL_SEQS, _W_BASE_PARAMS
    _W_FILTER_KEY = filter_key
    _W_POOL_SEQS = pool_seqs
    _W_BASE_PARAMS = base_params_per_seq
    # Each worker loads its own copy directly from disk — avoids shipping
    # large NavigationData arrays over IPC on every evaluation.
    _W_NAV_BY_SEQ = {seq: get_kitti_dataset(SEQ_TO_DRIVE[seq]) for seq in pool_seqs}


def _worker_eval(x):
    module = _FILTER_MODULES[_W_FILTER_KEY]
    cand = decode_nhc_zupt(x)
    costs = []
    for seq in _W_POOL_SEQS:
        merged = {**_W_BASE_PARAMS[seq], **cand}
        costs.append(ins_cost.single_window_cost(
            module, _W_NAV_BY_SEQ[seq], merged,
            t1=OUTAGE_START, d=OUTAGE_DURATION, use_3d=True))
    return float(np.mean(costs)) if costs else ins_cost.COST_REJECT


def _pool_map(func, iterable):
    """map-like callable for scipy's `workers=` — func is always _worker_eval,
    already bound to worker-local globals, so only the tiny `x` vectors
    actually cross the IPC boundary per call."""
    return _ACTIVE_POOL.map(_worker_eval, iterable)


_ACTIVE_POOL = None


def main():
    global _ACTIVE_POOL

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filters', nargs='+', choices=list(_FILTER_MODULES.keys()))
    parser.add_argument('--maxiter', type=int, default=10)
    parser.add_argument('--popsize', type=int, default=10)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the fold/pool plan without running DE.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    log = logging.getLogger(__name__)

    n_folds = len(OUTAGE_ELIGIBLE_SEQS)
    n_evals_est = n_folds * (args.maxiter + 1) * (args.popsize * len(NHC_ZUPT_BOUNDS))
    log.info(f"Plan: {len(args.filters)} filter(s) x {n_folds} held-out folds "
             f"(seq04 skipped) = {len(args.filters) * n_folds} DE searches, "
             f"~{n_evals_est} evaluations each filter.")

    for filt in args.filters:
        base_params = load_base_params_per_seq(filt)
        all_params = json.load(open(_PARAMS_PATH))
        default_nhc_zupt = {k: _FILTER_MODULES[filt].DEFAULT_PARAMS[k] for k in NHC_ZUPT_KEYS}

        for held_out in OUTAGE_ELIGIBLE_SEQS:
            pool = [s for s in OUTAGE_ELIGIBLE_SEQS if s != held_out]
            log.info(f"=== {filt}  held-out={held_out}  pool={pool} ===")

            if args.dry_run:
                continue

            _ACTIVE_POOL = multiprocessing.Pool(
                args.workers, initializer=_worker_init,
                initargs=(filt, pool, base_params))
            try:
                x0 = [np.log10(default_nhc_zupt[k]) for k in NHC_ZUPT_KEYS]
                baseline_cost = _pool_map(_worker_eval, [x0])[0]

                result = differential_evolution(
                    _worker_eval, NHC_ZUPT_BOUNDS,
                    maxiter=args.maxiter, popsize=args.popsize,
                    seed=args.seed, workers=_pool_map, updating='deferred',
                    polish=True, tol=1e-6,
                )
            finally:
                _ACTIVE_POOL.close()
                _ACTIVE_POOL.join()
                _ACTIVE_POOL = None

            best = decode_nhc_zupt(result.x)
            log.info(f"{filt} held-out={held_out}: baseline={baseline_cost:.4f} "
                     f"-> tuned={result.fun:.4f} "
                     f"({'IMPROVED' if result.fun < baseline_cost else 'not improved'})  "
                     f"{json.dumps(best)}")

            drive = SEQ_TO_DRIVE[held_out]
            fold_key = f'__loo_held_{drive}__'
            all_params[filt]['3d'][fold_key]['params'].update(best)
            all_params[filt]['3d'][fold_key]['nhc_zupt_tuning'] = {
                'method': 'nested_loo_pooled', 'pool_seqs': pool,
                'baseline_cost': baseline_cost, 'tuned_cost': float(result.fun),
                'maxiter': args.maxiter, 'popsize': args.popsize, 'seed': args.seed,
            }
            json.dump(all_params, open(_PARAMS_PATH, 'w'), indent=2)
            log.info(f"{filt} held-out={held_out}: written to {_PARAMS_PATH}")

    log.info("All done.")


if __name__ == '__main__':
    main()
