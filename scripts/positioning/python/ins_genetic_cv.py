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
    python ins_genetic_cv.py esekfs_enhanced iekf_vanilla # specific filters
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
import ins_cost
from data_loader import (get_kitti_dataset, get_cookies_dataset,
                         get_cookies_dataset_by_id, COOKIES_CLEAN_SEQS, NavigationData)
from filters import (
    esekfg_vanilla, esekfg_enhanced,
    esekfs_vanilla, esekfs_enhanced,
    iekf_vanilla, iekf_enhanced,
)

# ── Speed / quality ───────────────────────────────────────────────────────────
# Defaults sized for full DE convergence on the journal-grade three-component
# normalised cost defined in ins_cost.py. Per-pair fitness call cost is one
# filter run; CVFitness then averages across all (nav_data × outage) training
# pairs. With 6 KITTI training drives × 2 random outage windows = 12 pairs per
# fitness call, a 15 × 40 = 600-evaluation DE run amounts to ~7 200 filter
# runs per (filter, mode, fold) — long enough for stable convergence.
MAXITER  = 15
POPSIZE  = 10

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
    'esekfg_vanilla', 'esekfg_enhanced',
    'esekfs_vanilla', 'esekfs_enhanced',
    'iekf_vanilla', 'iekf_enhanced',
]

_FILTER_MODULES = {
    'esekfg_vanilla':   esekfg_vanilla,
    'esekfg_enhanced':  esekfg_enhanced,
    'esekfs_vanilla':  esekfs_vanilla,
    'esekfs_enhanced': esekfs_enhanced,
    'iekf_vanilla':  iekf_vanilla,
    'iekf_enhanced': iekf_enhanced,
}

# ── Parameter search bounds + decoder shared via ins_cost ────────────────────
# (reuse the canonical 15-dimensional log10 search space)
BOUNDS        = ins_cost.BOUNDS
decode_params = ins_cost.decode_params


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
                  sample_rate: float = 100.0) -> list:
    """
    Load all datasets of the given type, skipping failures with a warning.

    The default 100 Hz matches the native KITTI pickle rate and the COOKIES
    downsampling target used everywhere else in the pipeline
    (see data_loader.load_kitti_pickle / load_cookies_data). Passing 10 Hz
    here mis-stamps `nav_data.sample_rate` without resampling the arrays,
    which silently breaks every time-vs-index conversion downstream
    (outage window generation, single_window_cost A/B indexing, etc.).

    Returns list of NavigationData objects (same order as ids).
    """
    loaded = []
    for ds_id in ids:
        try:
            if dataset_type == 'kitti':
                nd = get_kitti_dataset(ds_id, sample_rate=sample_rate)
            else:
                nd = get_cookies_dataset_by_id(ds_id, sample_rate=sample_rate)
            loaded.append(nd)
        except Exception as e:
            print(f"[WARNING] Could not load dataset '{ds_id}': {e}")
    return loaded


# ── Outage configuration generation ──────────────────────────────────────────

CONVERGENCE_S   = 20.0   # GPS-aided seconds the filter gets before any outage
TARGET_OUTAGE_S = 60.0   # preferred outage length (the journal protocol)
MIN_OUTAGE_S    = 10.0   # never simulate an outage shorter than this
TAIL_SETTLE_S   = 10.0   # GPS-aided seconds after the outage (for ANEES diag)
ABS_MIN_TRAJ_S  = 20.0   # drives shorter than this can't carry a useful outage


def generate_outage_configs(nav_data: NavigationData, n_outages: int,
                             rng: np.random.Generator) -> list:
    """
    Generate n_outages valid (start_sec, duration_sec) pairs for nav_data.

    Constraints are absolute seconds, not fractions of trajectory length:

      - start >= CONVERGENCE_S         filter must see GPS for ≥ this many
                                       seconds before the outage so it
                                       converges its state + covariance.
      - start + duration <= T - TAIL_SETTLE_S
                                       brief GPS-aided phase after the outage
                                       so post-outage metrics are meaningful.
      - MIN_OUTAGE_S <= duration <= TARGET_OUTAGE_S

    Short-trajectory degradation
    ----------------------------
    For drives that can't fit the full 20 + 60 + 10 = 90 s layout, the
    convergence / target_outage / tail_settle values are scaled
    proportionally to T (with hard floors). This keeps short drives in the
    evaluation rather than dropping them — they get a correspondingly
    shorter outage, with the convergence prefix preserved as much as
    possible.

    Drives shorter than ABS_MIN_TRAJ_S are skipped — there's no useful
    outage you can place on them.
    """
    T = len(nav_data) / nav_data.sample_rate

    if T < ABS_MIN_TRAJ_S:
        print(f"[WARNING] Dataset '{nav_data.dataset_name}' (T={T:.1f}s) is below "
              f"the absolute minimum of {ABS_MIN_TRAJ_S:.0f}s — skipping.")
        return []

    conv_s   = CONVERGENCE_S
    target_s = TARGET_OUTAGE_S
    tail_s   = TAIL_SETTLE_S

    full_layout_s = conv_s + target_s + tail_s
    if T < full_layout_s:
        # Scale the layout proportionally to T, with floors so the
        # convergence prefix never vanishes.
        scale  = T / full_layout_s
        conv_s   = max(5.0, conv_s   * scale)
        target_s = max(MIN_OUTAGE_S, target_s * scale)
        tail_s   = max(2.0, tail_s   * scale)
        print(f"[INFO] '{nav_data.dataset_name}' (T={T:.1f}s) shorter than the "
              f"{full_layout_s:.0f}s reference layout — scaled to "
              f"convergence={conv_s:.1f}s, outage≤{target_s:.1f}s, "
              f"settle={tail_s:.1f}s.")

    min_start = conv_s
    max_end   = T - tail_s
    budget    = max_end - min_start
    if budget < MIN_OUTAGE_S:
        print(f"[WARNING] '{nav_data.dataset_name}' (T={T:.1f}s) can't fit "
              f"{conv_s:.0f}s convergence + {MIN_OUTAGE_S:.0f}s outage + "
              f"{tail_s:.0f}s settle — skipping.")
        return []

    # Duration is biased toward target_s (journal protocol = 60s) with a
    # small ±jitter, NOT a uniform [10, 60] draw — earlier code averaged
    # 35s outages, which dilutes the benchmark.
    max_dur = min(target_s, budget)
    jitter  = min(10.0, max_dur * 0.20)
    d_lo    = max(MIN_OUTAGE_S, max_dur - jitter)
    d_hi    = min(budget, max_dur + jitter)
    if d_hi <= d_lo:
        d_lo, d_hi = max_dur, max_dur     # degenerate: force the target

    configs = []
    for _ in range(n_outages):
        for attempt in range(50):
            d  = float(rng.uniform(d_lo, d_hi)) if d_hi > d_lo else max_dur
            latest_start = max_end - d
            if latest_start < min_start:
                continue
            t1 = float(rng.uniform(min_start, latest_start))
            if t1 + d <= max_end:
                configs.append((t1, d))
                break
        else:
            t1_fb = min_start
            d_fb  = min(max_dur, max_end - t1_fb)
            if d_fb >= MIN_OUTAGE_S * 0.5:
                configs.append((t1_fb, d_fb))
                print(f"[WARNING] Fallback outage for '{nav_data.dataset_name}': "
                      f"start={t1_fb:.1f}s dur={d_fb:.1f}s")
            else:
                print(f"[WARNING] '{nav_data.dataset_name}' couldn't synthesise "
                      f"one outage config — dropping it.")

    if not configs:
        print(f"[WARNING] No valid outage configs for '{nav_data.dataset_name}' "
              f"(T={T:.1f}s) — this dataset will be skipped.")

    return configs


# ── Per-pair cost (module-level for picklability) ─────────────────────────────

def _single_cost(filter_name: str, nd: NavigationData, params: dict,
                 t1: float, d: float, use_3d: bool,
                 gate_anees: bool = True) -> float:
    """
    Run one filter on one (dataset, outage) pair and return the cost.

    Delegates to `ins_cost.single_window_cost`, which implements the
    journal-grade three-component normalised cost
        J = ATE_outage / 1 m  +  t_rel / 1 %  +  r_rel / 1 deg/km
    plus the ANEES consistency band [0.1, 10] as a hard rejection
    constraint. See `ins_cost.py` for the rationale.

    `gate_anees=True` (default) gates the cost on ANEES — used during GA
    training so the optimiser can't game the cost by under-reporting
    covariance. `gate_anees=False` is used during validation where we
    want a finite number to report regardless of consistency.
    """
    module = _FILTER_MODULES[filter_name]
    return ins_cost.single_window_cost(module, nd, params, t1, d, use_3d,
                                       gate_anees=gate_anees)


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
            # Validation reports the raw cost — no ANEES gate. The training
            # GA used the gate, so the chosen params are still consistent on
            # the training drives; on a held-out short drive ANEES may drift
            # outside [0.1, 10] without it being a parameter problem.
            c = _single_cost(filter_name, nd, best_params, t1, d, use_3d,
                             gate_anees=False)
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
    parser.add_argument('--outages', type=int, default=1,
                        help='Random outage configs per TRAINING dataset (default: 1). '
                             'Each fitness eval runs (train datasets × this many) filter '
                             'simulations, so 1 outage halves the GA cost vs 2.')
    parser.add_argument('--val-outages', dest='val_outages', type=int, default=2,
                        help='Random outage configs per VALIDATION dataset (default: 2). '
                             'Higher than train so the val cost averages over more windows.')
    parser.add_argument('--3d',  dest='do_3d', action='store_true', default=None)
    parser.add_argument('--2d',  dest='do_2d', action='store_true', default=None)
    parser.add_argument('--seed',    type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--maxiter', type=int, default=MAXITER)
    parser.add_argument('--popsize', type=int, default=POPSIZE)
    parser.add_argument('--workers', type=int, default=-1,
                        help='Parallel workers for DE (-1 = all CPUs, default: -1)')
    parser.add_argument('--held-out', dest='held_out', default=None,
                        help='LOO: sequence to hold out as test set. '
                             'kitti: full drive name (e.g. 2011_10_03_drive_0042_extract). '
                             'cookies: short ID (e.g. c01). '
                             'Restricts training to the clean sequence list minus this entry.')
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
        # Use the curated clean-sequence list; honour --held-out for cookies LOO
        all_ids = list(COOKIES_CLEAN_SEQS.keys())   # ['c01'..'c06']
        if args.held_out and args.held_out in all_ids:
            all_ids = [s for s in all_ids if s != args.held_out]

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
    logger.info(f"Outages/ds   : train={args.outages}, val={args.val_outages}")
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
        cfgs = generate_outage_configs(nd, args.val_outages, rng)
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
