# -*- coding: utf-8 -*-
"""
ins_genetic_cv.py — Cross-Validation parameter optimiser for INS/GNSS filters.

Splits all available datasets of one type (KITTI or COOKIES) into an 80/20
train/validation partition, then optimises filter noise covariances using
scipy differential_evolution evaluated over ALL training (dataset × outage)
pairs per fitness call (deterministic, full-quality, parallel via workers=-1).

After optimisation, the best parameters are validated on the held-out
datasets and saved to filter_params.json under a sentinel dataset key
"__cv_<type>__" (or "__loo_held_<drive>__" when --held-out is used).

Typical usage:
    python ins_genetic_cv.py                            # kitti, all filters, both modes
    python ins_genetic_cv.py --type kitti               # kitti (default)
    python ins_genetic_cv.py --type cookies             # cookies datasets
    python ins_genetic_cv.py --split 80                 # train % (default 80)
    python ins_genetic_cv.py --outages 2                # outage configs per dataset
    python ins_genetic_cv.py --3d                       # only 3D mode
    python ins_genetic_cv.py --2d                       # only 2D mode
    python ins_genetic_cv.py eskf_enhanced iekf_vanilla # specific filters
    python ins_genetic_cv.py --seed 42                  # random seed
    python ins_genetic_cv.py --maxiter 40 --popsize 15  # DE quality (defaults)
    python ins_genetic_cv.py --workers 8                # parallel workers (-1 = all CPUs)

Leave-one-out (LOO) protocol for paper-comparable results:
    python ins_genetic_cv.py --held-out 2011_10_03_drive_0042_extract  # seq 01
    # Trains on the 6 other clean KITTI sequences; test seq 01 is never seen.
    # Results stored under key "__loo_held_2011_10_03_drive_0042_extract__".
    # Use run_loo_evaluation.py to run all 7 folds automatically.
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

import filter_params as fp
from data_loader import get_kitti_dataset, get_cookies_dataset, NavigationData
from filters import (
    ekf_vanilla, ekf_enhanced,
    eskf_vanilla, eskf_enhanced,
    iekf_vanilla, iekf_enhanced,
)

# ── Speed / quality ───────────────────────────────────────────────────────────
# Full quality (same as ins_genetic.py): ~9 000 evals × N_train_pairs filter runs
MAXITER  = 40
POPSIZE  = 15

# ── KITTI LOO protocol ────────────────────────────────────────────────────────
# Clean sequences for leave-one-out: no data gaps, raw OXTS available.
# Sequences 00, 02, 05 have ~2-second data gaps; 03 has no raw data.
KITTI_CLEAN_DRIVES = [
    '2011_10_03_drive_0042_extract',  # seq 01 — highway
    '2011_09_30_drive_0016_extract',  # seq 04 — country
    '2011_09_30_drive_0020_extract',  # seq 06 — urban
    '2011_09_30_drive_0027_extract',  # seq 07 — urban
    '2011_09_30_drive_0028_extract',  # seq 08 — urban/country
    '2011_09_30_drive_0033_extract',  # seq 09 — urban/country
    '2011_09_30_drive_0034_extract',  # seq 10 — urban/country
]

# ── Filters ───────────────────────────────────────────────────────────────────
ALL_FILTERS = [
    'ekf_vanilla', 'ekf_enhanced',
    'eskf_vanilla', 'eskf_enhanced',
    'iekf_vanilla', 'iekf_enhanced',
]

_FILTER_MODULES = {
    'ekf_vanilla':   ekf_vanilla,
    'ekf_enhanced':  ekf_enhanced,
    'eskf_vanilla':  eskf_vanilla,
    'eskf_enhanced': eskf_enhanced,
    'iekf_vanilla':  iekf_vanilla,
    'iekf_enhanced': iekf_enhanced,
}

# ── Parameter search bounds (log₁₀ scale) ─────────────────────────────────────
BOUNDS = [
    (-2, 2),     # log10(Qpos):      0.01 – 100
    (-2, 2),     # log10(Qvel):      0.01 – 100
    (-5, -1),    # log10(QorientXY): 1e-5 – 0.1
    (-2, 1),     # log10(QorientZ):  0.01 – 10
    (-3, 0),     # log10(Qacc):      0.001 – 1
    (-6, -2),    # log10(QgyrXY):    1e-6 – 0.01
    (-3, 0),     # log10(QgyrZ):     0.001 – 1
    (-1, 2),     # log10(Rpos):      0.1 – 100 m²
    (-8, -5),    # log10(|beta_acc|)
    (-2, 1),     # log10(|beta_gyr|)
    (-1, 1.5),   # log10(P_pos_std):  0.1 – 30 m
    (-1, 0.5),   # log10(P_vel_std):  0.1 – 3 m/s
    (-2, -0.5),  # log10(P_orient_std)
    (-3, -1),    # log10(P_acc_std)
    (-4, -2),    # log10(P_gyr_std)
]


def decode_params(x: np.ndarray) -> dict:
    """Convert log₁₀ parameter vector → filter parameter dict."""
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


# ── Dataset discovery ─────────────────────────────────────────────────────────

def list_kitti_datasets(base_dir: Path = None, held_out: str = None) -> list:
    """
    Return sorted list of KITTI dataset drive names (pickle file stems).

    When held_out is given (LOO mode): restricts to KITTI_CLEAN_DRIVES and
    removes the held-out sequence so it is never seen during optimisation.
    Without held_out: returns all available .p file stems (backward compat).
    """
    if base_dir is None:
        base_dir = _HERE / '../../../datasets/raw_kitti'
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"KITTI dataset directory not found: {base_dir}")
    if held_out is not None:
        # LOO mode: use only clean drives, exclude the held-out one
        available = {p.stem for p in base_dir.glob('*.p')}
        drives = [d for d in KITTI_CLEAN_DRIVES
                  if d in available and d != held_out]
        return drives
    return sorted(p.stem for p in base_dir.glob('*.p'))


def list_cookies_datasets(base_dir: Path = None) -> list:
    """Return sorted list of COOKIES dataset folder names."""
    if base_dir is None:
        base_dir = _HERE / '../../../datasets/raw_cookies'
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"COOKIES dataset directory not found: {base_dir}")
    return sorted(p.name for p in base_dir.iterdir() if p.is_dir())


def load_datasets(ids: list, dataset_type: str,
                  sample_rate: float = 10.0) -> list:
    """
    Load all datasets of the given type, skipping failures with a warning.

    Returns list of NavigationData objects (same order as ids).
    """
    loaded = []
    for ds_id in ids:
        try:
            if dataset_type == 'kitti':
                nd = get_kitti_dataset(ds_id, sample_rate=sample_rate)
            else:
                nd = get_cookies_dataset(ds_id, sample_rate=sample_rate)
            loaded.append(nd)
        except Exception as e:
            print(f"[WARNING] Could not load dataset '{ds_id}': {e}")
    return loaded


# ── Outage configuration generation ──────────────────────────────────────────

def generate_outage_configs(nav_data: NavigationData, n_outages: int,
                             rng: np.random.Generator) -> list:
    """
    Generate n_outages valid (start_sec, duration_sec) pairs for nav_data.

    Constraints (scaled to dataset length):
      - start in [0.20 * T, 0.70 * T]
      - duration in [min_dur, 90] s  where min_dur = min(30, T * 0.10)
      - start + duration <= 0.90 * T

    Uses rejection sampling (max 50 attempts per outage).
    Falls back to a single default config if constraints can't be satisfied.
    Returns an empty list (with a warning) if the dataset is too short for
    any outage — the calling code should skip datasets with no configs.
    """
    T         = len(nav_data) / nav_data.sample_rate
    min_start = 0.20 * T
    max_start = 0.70 * T
    # Scale min duration to dataset length so short datasets can still be used
    min_dur   = min(30.0, T * 0.10)
    max_dur   = min(90.0, T * 0.50)
    max_end   = 0.90 * T
    configs   = []

    # Sanity check: dataset must have enough room for at least one outage
    if max_end - min_start < min_dur or min_start >= max_start:
        print(f"[WARNING] Dataset '{nav_data.dataset_name}' (T={T:.1f}s) is too "
              f"short for outage generation — skipping.")
        return []

    for _ in range(n_outages):
        for attempt in range(50):
            t1     = float(rng.uniform(min_start, max_start))
            d_high = min(max_dur, max_end - t1)
            if d_high < min_dur:
                continue    # this t1 leaves no room for min duration; retry
            d = float(rng.uniform(min_dur, d_high))
            if t1 + d <= max_end:
                configs.append((t1, d))
                break
        else:
            # Fallback: fixed conservative config
            t1_fb = min_start
            d_fb  = min(min_dur, max_end - t1_fb)
            if d_fb >= 5.0:
                configs.append((t1_fb, d_fb))
                print(f"[WARNING] Fallback outage config for '{nav_data.dataset_name}': "
                      f"start={t1_fb:.1f}s dur={d_fb:.1f}s")
            else:
                print(f"[WARNING] Dataset '{nav_data.dataset_name}' is too short "
                      f"(T={T:.1f}s) — skipping one outage config.")

    if not configs:
        print(f"[WARNING] No valid outage configs for '{nav_data.dataset_name}' "
              f"(T={T:.1f}s) — this dataset will be skipped.")

    return configs


# ── Per-pair cost (module-level for picklability) ─────────────────────────────

def _single_cost(filter_name: str, nd: NavigationData, params: dict,
                 t1: float, d: float, use_3d: bool) -> float:
    """
    Run one filter on one (dataset, outage) pair and return the cost.

    Cost formula (same as ins_genetic.py):
        5*ate_2d_out + 5*ate_up_out + 5*ate_2d_gps
      + 5*rmse_roll  + 5*rmse_pitch + 2*rmse_yaw
      + 1*rmse_vel   + 3*anees_penalty
    """
    try:
        module  = _FILTER_MODULES[filter_name]
        frecIMU = nd.sample_rate
        A       = int(t1 * frecIMU)
        B       = int((t1 + d) * frecIMU)

        f    = pm.geodetic2enu(nd.lla[:,0], nd.lla[:,1], nd.lla[:,2],
                               nd.lla0[0], nd.lla0[1], nd.lla0[2])
        p_gt = np.column_stack([f[0], f[1], f[2]])
        N    = len(p_gt)

        res     = module.run(nd, params, {'start': t1, 'duration': d}, use_3d)
        p       = res['p'];  v = res['v'];  r = res['r']
        std_pos = res['std_pos']
        pos_err = p - p_gt

        # Outage errors
        err_out = np.sqrt(pos_err[A:B, 0]**2 + pos_err[A:B, 1]**2)
        ate_2d  = float(np.sqrt(np.mean(err_out**2))) if B > A else 0.0
        ate_up  = float(np.sqrt(np.mean(pos_err[A:B, 2]**2))) if B > A else 0.0

        # GPS-aided errors
        mask    = np.ones(N, dtype=bool); mask[A:B] = False
        err_gps = np.sqrt(pos_err[mask, 0]**2 + pos_err[mask, 1]**2)
        ate_gps = float(np.sqrt(np.mean(err_gps**2))) if mask.any() else 0.0

        # Orientation
        oe      = (r - nd.orient + np.pi) % (2 * np.pi) - np.pi
        rmse_r  = float(np.sqrt(np.mean(oe[:, 0]**2)))
        rmse_p  = float(np.sqrt(np.mean(oe[:, 1]**2)))
        rmse_y  = float(np.sqrt(np.mean(oe[:, 2]**2)))

        # Velocity
        rmse_v  = float(np.sqrt(np.mean(
            (v - nd.vel_enu)[:, 0]**2 + (v - nd.vel_enu)[:, 1]**2)))

        # ANEES consistency (GPS-aided phase, sampled every 10th epoch)
        eps = 1e-12; ns, nc = 0.0, 0
        for k in range(0, N, 10):
            if A <= k < B:
                continue
            ns += float(np.sum(pos_err[k]**2 / (std_pos[k]**2 + eps)))
            nc += 1
        anees   = ns / max(nc, 1) / 3.0
        penalty = abs(np.log10(max(anees, 1e-6)))

        cost = (5.0 * ate_2d  + 5.0 * ate_up + 5.0 * ate_gps
              + 5.0 * rmse_r  + 5.0 * rmse_p + 2.0 * rmse_y
              + 1.0 * rmse_v  + 3.0 * penalty)

        if np.isnan(cost) or np.isinf(cost) or cost > 10_000:
            return 1e6
        return float(cost)

    except Exception:
        return 1e6


# ── Picklable CV fitness class (required for workers > 1) ────────────────────

class CVFitness:
    """
    Deterministic fitness averaged over ALL training (dataset × outage) pairs.

    Using a class (not a closure) makes the callable picklable so that
    scipy's differential_evolution can distribute evaluations across
    multiple processes with workers=-1.
    """

    def __init__(self, filter_name: str, train_data: list,
                 train_outages: list, use_3d: bool):
        self.filter_name   = filter_name
        self.train_data    = train_data    # list[NavigationData]
        self.train_outages = train_outages # list[list[tuple(float,float)]]
        self.use_3d        = use_3d

        # Build flat list of (NavigationData, t1, d) pairs for fast iteration
        self._pairs = []
        for nd, outages in zip(train_data, train_outages):
            for (t1, d) in outages:
                self._pairs.append((nd, t1, d))

    def __call__(self, x: np.ndarray) -> float:
        params = decode_params(x)
        costs  = [
            _single_cost(self.filter_name, nd, params, t1, d, self.use_3d)
            for (nd, t1, d) in self._pairs
        ]
        if not costs:
            return 1e6
        return float(np.mean(costs))


# ── Validation ────────────────────────────────────────────────────────────────

def validate_params(filter_name: str, best_params: dict,
                    val_data: list, val_outages: list,
                    use_3d: bool, logger: logging.Logger) -> dict:
    """
    Evaluate best_params on all (validation_dataset × outage) pairs.

    Returns a summary dict:
    {
        'mean_cost': float,
        'per_dataset': [{'dataset': str, 'outages': [...], 'mean_cost': float}, ...]
    }
    """
    per_dataset = []
    all_costs   = []

    for nd, outages in zip(val_data, val_outages):
        ds_costs = []
        outage_details = []
        for (t1, d) in outages:
            c = _single_cost(filter_name, nd, best_params, t1, d, use_3d)
            ds_costs.append(c)
            all_costs.append(c)
            outage_details.append({'start': t1, 'duration': d, 'cost': c})
            logger.info(f"    val  {nd.dataset_name}  outage={t1:.0f}s+{d:.0f}s  cost={c:.3f}")

        per_dataset.append({
            'dataset':   nd.dataset_name,
            'outages':   outage_details,
            'mean_cost': float(np.mean(ds_costs)) if ds_costs else float('nan'),
        })

    return {
        'mean_cost':   float(np.mean(all_costs)) if all_costs else float('nan'),
        'per_dataset': per_dataset,
    }


# ── One (filter, mode) optimisation run ──────────────────────────────────────

def run_cv_one(filter_name: str, mode_3d: bool,
               train_data: list, val_data: list,
               train_outages: list, val_outages: list,
               seed: int, maxiter: int, popsize: int,
               workers: int, logger: logging.Logger) -> tuple:
    """
    Run differential_evolution for one (filter, mode) combination.

    Returns (best_params, train_cost, val_summary).
    """
    mode_str    = '3D' if mode_3d else '2D'
    n_pairs     = sum(len(o) for o in train_outages)
    n_val_pairs = sum(len(o) for o in val_outages)

    logger.info(f"\n{'─'*60}")
    logger.info(f"  {filter_name}  [{mode_str}]")
    logger.info(f"  train datasets : {[nd.dataset_name for nd in train_data]}")
    logger.info(f"  train pairs    : {n_pairs}  (datasets × outages per eval)")
    logger.info(f"  val datasets   : {[nd.dataset_name for nd in val_data]}")
    logger.info(f"  val pairs      : {n_val_pairs}")
    logger.info(f"  maxiter={maxiter}  popsize={popsize}  "
                f"evals≈{maxiter * popsize * len(BOUNDS)}")
    logger.info(f"  workers={workers}")
    logger.info(f"{'─'*60}")

    fitness_obj = CVFitness(filter_name, train_data, train_outages, mode_3d)

    try:
        result = differential_evolution(
            fitness_obj,
            BOUNDS,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=seed,
            disp=True,
            polish=False,
            workers=workers,
        )
    except Exception as e:
        if workers != 1:
            logger.warning(f"  workers={workers} failed ({e}), retrying with workers=1")
            result = differential_evolution(
                fitness_obj,
                BOUNDS,
                strategy='best1bin',
                maxiter=maxiter,
                popsize=popsize,
                tol=0.01,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=seed,
                disp=True,
                polish=False,
                workers=1,
            )
        else:
            raise

    best_params = decode_params(result.x)
    train_cost  = float(result.fun)

    logger.info(f"  → train cost={train_cost:.3f}  evals={result.nfev}  "
                f"success={result.success}")

    logger.info("  Validating on held-out datasets…")
    val_summary = validate_params(
        filter_name, best_params, val_data, val_outages, mode_3d, logger)
    logger.info(f"  → val  mean cost={val_summary['mean_cost']:.3f}")

    return best_params, train_cost, val_summary


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global MAXITER, POPSIZE

    parser = argparse.ArgumentParser(
        description='Cross-validation genetic parameter optimiser for INS/GNSS filters.')
    parser.add_argument('filters', nargs='*',
                        help=f'Filters to optimise (default: all). Choices: {ALL_FILTERS}')
    parser.add_argument('--type',    choices=['kitti', 'cookies'], default='kitti',
                        help='Dataset type (default: kitti)')
    parser.add_argument('--split',   type=int, default=80,
                        help='Training percentage 50-90 (default: 80)')
    parser.add_argument('--outages', type=int, default=2,
                        help='Random outage configs per dataset (default: 2)')
    parser.add_argument('--3d',  dest='do_3d', action='store_true', default=None)
    parser.add_argument('--2d',  dest='do_2d', action='store_true', default=None)
    parser.add_argument('--seed',    type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--maxiter', type=int, default=MAXITER)
    parser.add_argument('--popsize', type=int, default=POPSIZE)
    parser.add_argument('--workers', type=int, default=-1,
                        help='Parallel workers for DE (-1 = all CPUs, default: -1)')
    parser.add_argument('--held-out', dest='held_out', default=None,
                        help='LOO: drive name to hold out as test set (e.g. '
                             '2011_10_03_drive_0042_extract). Restricts training '
                             'to KITTI_CLEAN_DRIVES minus this sequence.')
    args = parser.parse_args()

    MAXITER = args.maxiter
    POPSIZE = args.popsize

    # ── Validate filters ──────────────────────────────────────────────────────
    filters_to_run = args.filters if args.filters else ALL_FILTERS
    if 'imu_only' in filters_to_run:
        print("imu_only has no tunable parameters — skipping.")
        filters_to_run = [f for f in filters_to_run if f != 'imu_only']
    invalid = [f for f in filters_to_run if f not in _FILTER_MODULES]
    if invalid:
        print(f"Unknown filters: {invalid}\nAvailable: {ALL_FILTERS}")
        sys.exit(1)
    if not filters_to_run:
        print("No filters to run.")
        sys.exit(0)

    # ── Modes ─────────────────────────────────────────────────────────────────
    if args.do_3d and not args.do_2d:
        modes = [True]
    elif args.do_2d and not args.do_3d:
        modes = [False]
    else:
        modes = [True, False]

    # ── Split fraction ────────────────────────────────────────────────────────
    split_pct  = max(50, min(90, args.split))
    split_frac = split_pct / 100.0

    # ── Dataset discovery & loading ───────────────────────────────────────────
    if args.type == 'kitti':
        all_ids = list_kitti_datasets(held_out=args.held_out)
    else:
        all_ids = list_cookies_datasets()

    if len(all_ids) < 2:
        print(f"Need at least 2 {args.type} datasets for cross-validation. "
              f"Found: {all_ids}")
        sys.exit(1)

    # ── Logging ───────────────────────────────────────────────────────────────
    logs_dir = _HERE / '../../../logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = logs_dir / f'ins_genetic_cv_{args.type}_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)

    # ── Train / validation split ──────────────────────────────────────────────
    rng          = np.random.default_rng(args.seed)
    ids_shuffled = list(rng.permutation(all_ids))
    n_train      = max(1, int(np.ceil(len(ids_shuffled) * split_frac)))
    n_val        = max(1, len(ids_shuffled) - n_train)
    # Guard: ensure there is at least 1 in each partition
    if n_train + n_val > len(ids_shuffled):
        n_train = len(ids_shuffled) - 1
        n_val   = 1
    train_ids = ids_shuffled[:n_train]
    val_ids   = ids_shuffled[n_train:n_train + n_val]

    logger.info("=" * 65)
    logger.info("INS GENETIC CV — CROSS-VALIDATION PARAMETER OPTIMISER")
    logger.info("=" * 65)
    logger.info(f"Dataset type : {args.type}  ({len(all_ids)} available)")
    if args.held_out:
        logger.info(f"LOO held-out : {args.held_out}  (excluded from training)")
    logger.info(f"Train ({split_pct}%) : {train_ids}")
    logger.info(f"Val   ({100-split_pct}%) : {val_ids}")
    logger.info(f"Outages/ds   : {args.outages}")
    logger.info(f"Filters      : {filters_to_run}")
    logger.info(f"Modes        : {['3D' if m else '2D' for m in modes]}")
    logger.info(f"Seed         : {args.seed}")
    logger.info(f"DE           : maxiter={MAXITER}, popsize={POPSIZE}, "
                f"workers={args.workers}")
    logger.info(f"Log          : {log_file}")
    logger.info("=" * 65)

    # ── Load datasets ─────────────────────────────────────────────────────────
    logger.info("\nLoading training datasets…")
    train_data = load_datasets(train_ids, args.type)
    logger.info(f"  Loaded {len(train_data)}/{len(train_ids)} train datasets.")

    logger.info("Loading validation datasets…")
    val_data = load_datasets(val_ids, args.type)
    logger.info(f"  Loaded {len(val_data)}/{len(val_ids)} val datasets.")

    if not train_data:
        logger.error("No training datasets loaded — aborting.")
        sys.exit(1)

    # ── Generate outage configurations ────────────────────────────────────────
    logger.info("\nGenerating outage configurations…")
    train_data_valid, train_outages = [], []
    for nd in train_data:
        cfgs = generate_outage_configs(nd, args.outages, rng)
        if not cfgs:
            logger.warning(f"  train {nd.dataset_name}: no valid outage configs — skipped.")
            continue
        train_data_valid.append(nd)
        train_outages.append(cfgs)
        logger.info(f"  train {nd.dataset_name}: {len(cfgs)} outage(s) → "
                    + ", ".join(f"{t1:.0f}s+{d:.0f}s" for t1, d in cfgs))
    train_data = train_data_valid

    val_data_valid, val_outages = [], []
    for nd in val_data:
        cfgs = generate_outage_configs(nd, args.outages, rng)
        if not cfgs:
            logger.warning(f"  val   {nd.dataset_name}: no valid outage configs — skipped.")
            continue
        val_data_valid.append(nd)
        val_outages.append(cfgs)
        logger.info(f"  val   {nd.dataset_name}: {len(cfgs)} outage(s) → "
                    + ", ".join(f"{t1:.0f}s+{d:.0f}s" for t1, d in cfgs))
    val_data = val_data_valid

    n_train_pairs = sum(len(o) for o in train_outages)
    logger.info(f"\nTotal training pairs per fitness eval: {n_train_pairs}")
    logger.info(f"Estimated DE evals: {MAXITER * POPSIZE * len(BOUNDS)}")

    # ── CV sentinel dataset key ───────────────────────────────────────────────
    # LOO mode: tag key with the held-out drive so each fold is stored separately.
    if args.held_out:
        cv_dataset_key = f"__loo_held_{args.held_out}__"
    else:
        cv_dataset_key = f"__cv_{args.type}__"

    # ── Output directory for audit JSONs ──────────────────────────────────────
    out_dir = _HERE / '../../../outputs/genetic_cv'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Main optimisation loop ────────────────────────────────────────────────
    results_summary = []
    total    = len(filters_to_run) * len(modes)
    done     = 0

    for filter_name in filters_to_run:
        for mode_3d in modes:
            done += 1
            mode_str = '3D' if mode_3d else '2D'
            logger.info(f"\n[{done}/{total}] {filter_name}  [{mode_str}]")

            try:
                best_params, train_cost, val_summary = run_cv_one(
                    filter_name=filter_name,
                    mode_3d=mode_3d,
                    train_data=train_data,
                    val_data=val_data,
                    train_outages=train_outages,
                    val_outages=val_outages,
                    seed=args.seed + done,   # distinct seed per run
                    maxiter=MAXITER,
                    popsize=POPSIZE,
                    workers=args.workers,
                    logger=logger,
                )

                # Save to central store (only if better than existing CV result)
                prev_cost = fp.get_cost(filter_name, mode_3d, cv_dataset_key)
                improved  = (prev_cost is None) or (train_cost < prev_cost)

                meta = {
                    'optimiser':       'ins_genetic_cv',
                    'dataset_type':    args.type,
                    'train_ids':       [nd.dataset_name for nd in train_data],
                    'val_ids':         [nd.dataset_name for nd in val_data],
                    'split_pct':       split_pct,
                    'n_outages':       args.outages,
                    'seed':            args.seed,
                    'maxiter':         MAXITER,
                    'popsize':         POPSIZE,
                    'workers':         args.workers,
                    'train_cost':      train_cost,
                    'val_summary':     val_summary,
                    'timestamp':       timestamp,
                    'train_outages':   [[list(cfg) for cfg in o] for o in train_outages],
                    'val_outages':     [[list(cfg) for cfg in o] for o in val_outages],
                }

                if improved:
                    fp.set(filter_name, mode_3d, cv_dataset_key,
                           params=best_params, cost=train_cost, metadata=meta)
                    status = 'SAVED (new best CV)'
                else:
                    status = f'skipped (prev={prev_cost:.3f} ≤ {train_cost:.3f})'

                logger.info(f"  → {status}")

                # Always write per-run audit JSON
                audit_path = (out_dir /
                    f'{filter_name}_{mode_str}_{args.type}_{timestamp}.json')
                with open(audit_path, 'w') as fh:
                    json.dump({
                        'filter':       filter_name,
                        'mode':         mode_str,
                        'dataset_type': args.type,
                        'best_params':  {k: float(v) for k, v in best_params.items()},
                        'train_cost':   train_cost,
                        'val_summary':  val_summary,
                        'metadata':     meta,
                    }, fh, indent=2)
                logger.info(f"  Audit JSON: {audit_path}")

                results_summary.append(
                    (filter_name, mode_str, train_cost,
                     val_summary['mean_cost'], status))

            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results_summary.append(
                    (filter_name, mode_str, float('nan'), float('nan'),
                     f'ERROR: {e}'))

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("CV OPTIMISATION COMPLETE — SUMMARY")
    logger.info("=" * 65)
    logger.info(f"{'Filter':<20} {'Mode':<5} {'TrainCost':>10} {'ValCost':>10}  Status")
    logger.info("-" * 65)
    for fname, mstr, tc, vc, status in results_summary:
        tc_s = f"{tc:10.3f}" if np.isfinite(tc) else "      FAIL"
        vc_s = f"{vc:10.3f}" if np.isfinite(vc) else "      FAIL"
        logger.info(f"{fname:<20} {mstr:<5} {tc_s} {vc_s}  {status}")
    logger.info("=" * 65)

    print("\n")
    fp.print_summary()
    if args.held_out:
        print(f"\nLOO fold results stored under dataset key: '{cv_dataset_key}'")
        print(f"Held-out test sequence: {args.held_out}")
        print("  → Evaluate on this sequence with ins_compare.py")
    else:
        print(f"\nCV results stored under dataset key: '{cv_dataset_key}'")
    print(f"Audit JSONs: {out_dir}")
    print(f"Log:         {log_file}")
    print("\nNext steps:")
    print("  To use CV params in ins_compare.py, load them with:")
    print(f"    fp.get('<filter>', mode_3d=True, dataset='{cv_dataset_key}')")
    print("  Or run ins_compare.py and override TUNED_PARAMS manually.")


if __name__ == '__main__':
    main()
