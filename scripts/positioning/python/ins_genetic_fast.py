# -*- coding: utf-8 -*-
"""
ins_genetic_fast.py — Quick multi-filter parameter sweep.

Runs a reduced differential-evolution optimisation for every requested
filter × mode (2D / 3D) combination.  Results are saved automatically to
filter_params.json via filter_params.py so that ins_runner.py and
ins_compare.py can pick them up without any copy-pasting.

Without --seq: optimises on the single dataset configured in ins_config.py.

With --seq SEQ_ID: applies the LOO training-split protocol —
  • clean test sequence (01, 04, 06, 07, 08, 09, 10):
      train on the other 6 clean sequences (leave-one-out)
  • not-clean test sequence (00, 02, 05):
      train on all 7 clean sequences
  On every fitness evaluation one training sequence is sampled at random,
  keeping the per-evaluation cost identical to the single-dataset case.
  Parameters are saved under the held-out sequence key in filter_params.json.

Typical use:
    python ins_genetic_fast.py                        # single dataset from ins_config
    python ins_genetic_fast.py --seq 02               # LOO split, test on seq 02
    python ins_genetic_fast.py --seq 01 --3d          # only 3D mode
    python ins_genetic_fast.py --seq 08 eskf_enhanced # specific filter

Tune quality vs. speed with MAXITER and POPSIZE at the top of this file.
For production-quality parameters, use ins_genetic_cv.py (40 generations,
full average over all training pairs).
"""
import sys
import json
import logging
import argparse
import numpy as np
import pymap3d as pm
from pathlib import Path
from datetime import datetime
from scipy.optimize import differential_evolution

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import ins_config
import filter_params as fp
from data_loader import get_kitti_dataset, KITTI_SEQ_TO_DRIVE
from filters import (
    ekf_vanilla, ekf_enhanced,
    eskf_vanilla, eskf_enhanced,
    iekf_vanilla, iekf_enhanced,
)

# ── Speed / quality trade-off ─────────────────────────────────────────────────
# Fast:       MAXITER=10, POPSIZE=8   → ~1 200 evals/filter  (~1–3 min)
# Balanced:   MAXITER=20, POPSIZE=10  → ~3 000 evals/filter  (~5–10 min)
# Full:       MAXITER=40, POPSIZE=15  → ~9 000 evals/filter  (use ins_genetic_cv.py)
MAXITER = 10
POPSIZE = 8

# ── Filters to sweep (edit or pass as CLI args) ───────────────────────────────
# Deep learning filters (tlio, deep_kf, tartan_imu) have tunable params too:
#   Rpos, window_seconds/stride_seconds (TLIO), latent_dim (Deep KF), etc.
# Add them here to include in the genetic sweep.
ALL_FILTERS = [
    'ekf_vanilla',
    'ekf_enhanced',
    'eskf_vanilla',
    'eskf_enhanced',
    'iekf_vanilla',
    'iekf_enhanced',
    # 'tlio',       # uncomment after training weights are available
    # 'deep_kf',    # uncomment after training weights are available
    # 'tartan_imu', # uncomment after pretrained weights are downloaded
]

_FILTER_MODULES = {
    'ekf_vanilla':   ekf_vanilla,
    'ekf_enhanced':  ekf_enhanced,
    'eskf_vanilla':  eskf_vanilla,
    'eskf_enhanced': eskf_enhanced,
    'iekf_vanilla':  iekf_vanilla,
    'iekf_enhanced': iekf_enhanced,
}

# ── LOO protocol — sequence classification ────────────────────────────────────
# Clean sequences: no data gaps, reliable for both training and test.
# Not-clean sequences: ~2-second data gaps (logging artefact).
_KITTI_CLEAN_SEQS    = ['01', '04', '06', '07', '08', '09', '10']
_KITTI_NOT_CLEAN_SEQS = {'00', '02', '05'}


def _get_training_seqs(held_out_seq: str) -> list[str]:
    """
    Return the list of KITTI sequence IDs to use as training for held_out_seq.

    • Held-out is clean     → train on the other 6 clean sequences (LOO)
    • Held-out is not-clean → train on all 7 clean sequences
    """
    if held_out_seq in _KITTI_NOT_CLEAN_SEQS:
        return list(_KITTI_CLEAN_SEQS)
    return [s for s in _KITTI_CLEAN_SEQS if s != held_out_seq]


# ── Parameter search bounds (log₁₀) ──────────────────────────────────────────
BOUNDS = [
    (-2, 2),     # log10(Qpos)
    (-2, 2),     # log10(Qvel)
    (-5, -1),    # log10(QorientXY)
    (-2, 1),     # log10(QorientZ)
    (-3, 0),     # log10(Qacc)
    (-6, -2),    # log10(QgyrXY)
    (-3, 0),     # log10(QgyrZ)
    (-1, 2),     # log10(Rpos)
    (-8, -5),    # log10(|beta_acc|)  — stored negative
    (-2, 1),     # log10(|beta_gyr|)  — stored negative
    (-1, 1.5),   # log10(P_pos_std)
    (-1, 0.5),   # log10(P_vel_std)
    (-2, -0.5),  # log10(P_orient_std)
    (-3, -1),    # log10(P_acc_std)
    (-4, -2),    # log10(P_gyr_std)
]


def decode_params(x: np.ndarray) -> dict:
    return {
        'Qpos':         10**x[0],
        'Qvel':         10**x[1],
        'QorientXY':    10**x[2],
        'QorientZ':     10**x[3],
        'Qacc':         10**x[4],
        'QgyrXY':       10**x[5],
        'QgyrZ':        10**x[6],
        'Rpos':         10**x[7],
        'beta_acc':    -10**x[8],
        'beta_gyr':    -10**x[9],
        'P_pos_std':    10**x[10],
        'P_vel_std':    10**x[11],
        'P_orient_std': 10**x[12],
        'P_acc_std':    10**x[13],
        'P_gyr_std':    10**x[14],
    }


def _make_fitness(filter_module, nav_data_pool, t1, d, use_3d, logger, rng=None):
    """
    Return a fitness closure for one (filter, mode) combination.

    nav_data_pool: list of NavigationData objects (training datasets).
    rng: np.random.Generator used to sample one dataset per fitness call.
         If None (single-element pool), always uses pool[0].
    """
    # Pre-compute ENU ground-truth and index bounds for every dataset in the pool.
    pool_items = []
    for nd in nav_data_pool:
        lla0    = nd.lla0
        lla     = nd.lla
        frecIMU = nd.sample_rate
        A = int(t1 * frecIMU)
        B = int((t1 + d) * frecIMU)
        f = pm.geodetic2enu(lla[:,0], lla[:,1], lla[:,2], lla0[0], lla0[1], lla0[2])
        p_gt = np.column_stack([f[0], f[1], f[2]])
        N = len(p_gt)
        B = min(B, N)   # guard against sequences shorter than the outage window
        pool_items.append((nd, A, min(B, N), p_gt, N))

    single = (len(pool_items) == 1)

    counter = [0]
    best    = [float('inf')]

    def fitness(x):
        counter[0] += 1
        # Sample one training dataset per evaluation (stochastic, keeps cost ≈ single-dataset)
        if single or rng is None:
            nav_data, A, B, p_gt, N = pool_items[0]
        else:
            nav_data, A, B, p_gt, N = pool_items[int(rng.integers(len(pool_items)))]

        params = decode_params(x)
        try:
            res     = filter_module.run(nav_data, params,
                                        {'start': t1, 'duration': d}, use_3d)
            p       = res['p'];  v = res['v'];  r = res['r']
            std_pos = res['std_pos']
            pos_err = p - p_gt

            err_out = np.sqrt(pos_err[A:B,0]**2 + pos_err[A:B,1]**2)
            ate_2d  = np.sqrt(np.mean(err_out**2)) if B > A else 0.0
            ate_up  = np.sqrt(np.mean(pos_err[A:B,2]**2)) if B > A else 0.0

            mask = np.ones(N, dtype=bool); mask[A:B] = False
            err_gps = np.sqrt(pos_err[mask,0]**2 + pos_err[mask,1]**2)
            ate_gps = np.sqrt(np.mean(err_gps**2))

            oe      = (r - nav_data.orient + np.pi) % (2*np.pi) - np.pi
            rmse_r  = np.sqrt(np.mean(oe[:,0]**2))
            rmse_p_ = np.sqrt(np.mean(oe[:,1]**2))
            rmse_y  = np.sqrt(np.mean(oe[:,2]**2))
            rmse_v  = np.sqrt(np.mean((v - nav_data.vel_enu)[:,0]**2 +
                                      (v - nav_data.vel_enu)[:,1]**2))

            eps = 1e-12; ns, nc = 0.0, 0
            for k in range(0, N, 10):
                if A <= k < B: continue
                ns += np.sum(pos_err[k]**2 / (std_pos[k]**2 + eps))
                nc += 1
            anees   = ns / max(nc, 1) / 3.0
            penalty = abs(np.log10(max(anees, 1e-6)))

            cost = (5*ate_2d + 5*ate_up + 5*ate_gps +
                    5*rmse_r + 5*rmse_p_ + 2*rmse_y +
                    1*rmse_v + 3*penalty)

            if np.isnan(cost) or np.isinf(cost) or cost > 10_000:
                return 1e6

            if cost < best[0]:
                best[0] = cost
                logger.debug(f"  eval {counter[0]:5d}: NEW BEST cost={cost:.3f}  "
                             f"seq={nav_data.dataset_name}")

            return cost
        except Exception as e:
            logger.debug(f"  eval {counter[0]:5d}: ERROR {e}")
            return 1e6

    return fitness


def run_one(filter_name: str, mode_3d: bool,
            nav_data_pool: list, t1: float, d: float,
            logger: logging.Logger, seed: int) -> tuple[dict, float]:
    """
    Optimise one (filter, mode) combination.

    nav_data_pool: list of NavigationData (training datasets).
                   One is sampled at random per fitness evaluation.
    Returns (best_params, best_cost).
    """
    module   = _FILTER_MODULES[filter_name]
    mode_str = '3D' if mode_3d else '2D'
    pool_names = [nd.dataset_name for nd in nav_data_pool]
    logger.info(f"\n{'─'*55}")
    logger.info(f"  {filter_name}  [{mode_str}]")
    logger.info(f"  training pool ({len(nav_data_pool)}): {pool_names}")
    logger.info(f"  maxiter={MAXITER}  popsize={POPSIZE}  "
                f"evals≈{MAXITER * POPSIZE * len(BOUNDS)}")
    logger.info(f"{'─'*55}")

    rng     = np.random.default_rng(seed)
    fitness = _make_fitness(module, nav_data_pool, t1, d, mode_3d, logger, rng)

    result = differential_evolution(
        fitness, BOUNDS,
        strategy='best1bin',
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed % (2**31),
        disp=False,
        polish=False,
        workers=1,
    )

    best_params = decode_params(result.x)
    best_cost   = float(result.fun)

    logger.info(f"  → cost={best_cost:.3f}  evals={result.nfev}  "
                f"success={result.success}")

    return best_params, best_cost


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global MAXITER, POPSIZE
    parser = argparse.ArgumentParser(
        description='Quick parameter sweep for all filter × mode combinations.')
    parser.add_argument('filters', nargs='*',
                        help=f'Filters to run (default: all). '
                             f'Choices: {ALL_FILTERS}')
    parser.add_argument('--seq', metavar='SEQ_ID', default=None,
                        help='KITTI held-out sequence ID (e.g. "02"). '
                             'Enables LOO training split: optimises on the '
                             'appropriate training sequences and saves params '
                             'under the held-out sequence key. '
                             'If omitted, uses ins_config.NAV_DATA directly.')
    parser.add_argument('--3d',  dest='do_3d', action='store_true',  default=None)
    parser.add_argument('--2d',  dest='do_2d', action='store_true',  default=None)
    parser.add_argument('--maxiter', type=int, default=MAXITER)
    parser.add_argument('--popsize', type=int, default=POPSIZE)
    args = parser.parse_args()

    MAXITER = args.maxiter
    POPSIZE = args.popsize

    filters_to_run = args.filters if args.filters else ALL_FILTERS
    invalid = [f for f in filters_to_run if f not in _FILTER_MODULES]
    if invalid:
        print(f"Unknown filters: {invalid}\nAvailable: {ALL_FILTERS}")
        sys.exit(1)

    # Determine which modes to run
    if args.do_3d and not args.do_2d:
        modes = [True]
    elif args.do_2d and not args.do_3d:
        modes = [False]
    else:
        modes = [True, False]   # both by default

    # ── Outage config from ins_config ─────────────────────────────────────────
    t1 = ins_config.OUTAGE_START
    d  = ins_config.OUTAGE_DURATION

    # ── Build training pool ───────────────────────────────────────────────────
    held_out_seq  = args.seq
    held_out_name = None   # dataset_name key under which to save results

    if held_out_seq is not None:
        if held_out_seq not in KITTI_SEQ_TO_DRIVE:
            print(f"Unknown sequence: '{held_out_seq}'. "
                  f"Available: {sorted(KITTI_SEQ_TO_DRIVE.keys())}")
            sys.exit(1)
        training_seqs = _get_training_seqs(held_out_seq)
        held_out_name = KITTI_SEQ_TO_DRIVE[held_out_seq]
        print(f"LOO mode: held-out seq={held_out_seq}  "
              f"training seqs={training_seqs}")
        print("Loading training datasets …", flush=True)
        nav_data_pool = [get_kitti_dataset(s) for s in training_seqs]
    else:
        nav_data_pool = [ins_config.NAV_DATA]
        held_out_name = ins_config.NAV_DATA.dataset_name

    # ── Logging ───────────────────────────────────────────────────────────────
    logs_dir = _HERE / '../../../logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = logs_dir / f'ins_genetic_fast_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)

    total = len(filters_to_run) * len(modes)
    logger.info("=" * 60)
    logger.info("INS GENETIC FAST — MULTI-FILTER PARAMETER SWEEP")
    logger.info("=" * 60)
    if held_out_seq:
        logger.info(f"Mode     : LOO  (held-out seq={held_out_seq})")
        logger.info(f"Train on : {[nd.dataset_name for nd in nav_data_pool]}")
        logger.info(f"Save key : {held_out_name}")
    else:
        logger.info(f"Dataset  : {nav_data_pool[0].dataset_name}")
    logger.info(f"Outage   : {t1}s + {d}s")
    logger.info(f"Filters  : {filters_to_run}")
    logger.info(f"Modes    : {'3D' if True in modes else ''}{'/' if len(modes)==2 else ''}{'2D' if False in modes else ''}")
    logger.info(f"Runs     : {total}  (maxiter={MAXITER}, popsize={POPSIZE})")
    logger.info(f"Log      : {log_file}")
    logger.info("=" * 60)

    # ── Main sweep ────────────────────────────────────────────────────────────
    results_summary = []
    completed = 0
    base_seed = int(datetime.now().timestamp()) % (2**31)

    for filter_name in filters_to_run:
        for mode_3d in modes:
            completed += 1
            mode_str = '3D' if mode_3d else '2D'
            logger.info(f"\n[{completed}/{total}] {filter_name}  [{mode_str}]")

            seed = (base_seed + completed * 1000) % (2**31)

            try:
                best_params, best_cost = run_one(
                    filter_name, mode_3d, nav_data_pool, t1, d, logger, seed)

                # Check if this improves on any existing stored result
                prev_cost = fp.get_cost(filter_name, mode_3d, held_out_name)
                improved  = (prev_cost is None) or (best_cost < prev_cost)

                if improved:
                    fp.set(
                        filter_name, mode_3d, held_out_name,
                        params=best_params, cost=best_cost,
                        metadata={
                            'optimiser':      'ins_genetic_fast',
                            'maxiter':        MAXITER,
                            'popsize':        POPSIZE,
                            'timestamp':      timestamp,
                            'outage':         {'start': t1, 'duration': d},
                            'held_out_seq':   held_out_seq,
                            'training_seqs':  [nd.dataset_name for nd in nav_data_pool],
                        },
                    )
                    status = 'SAVED (new best)'
                else:
                    status = f'skipped (prev={prev_cost:.3f} ≤ {best_cost:.3f})'

                logger.info(f"  → {status}")
                results_summary.append((filter_name, mode_str, best_cost, status))

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results_summary.append((filter_name, mode_str, float('nan'), f'ERROR: {e}'))

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("SWEEP COMPLETE — SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Filter':<20} {'Mode':<5} {'Cost':>8}  Status")
    logger.info("-" * 60)
    for fname, mode_str, cost, status in results_summary:
        cost_s = f"{cost:8.3f}" if not np.isnan(cost) else "    FAIL"
        logger.info(f"{fname:<20} {mode_str:<5} {cost_s}  {status}")
    logger.info("=" * 60)

    print("\n")
    fp.print_summary()
    print(f"All results saved to: {fp._STORE}")
    print(f"Log: {log_file}")
    print("\nNext steps:")
    print("  1. Review filter_params.json")
    print("  2. Run ins_runner.py or ins_compare.py — params are loaded automatically")
    print("  3. For better results: run ins_genetic_cv.py (full LOO, 40 generations)")


if __name__ == '__main__':
    main()
