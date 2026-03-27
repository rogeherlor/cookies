# -*- coding: utf-8 -*-
"""
run_loo_evaluation.py — Leave-One-Out (LOO) evaluation across all 7 clean KITTI sequences.

Orchestrates all LOO folds for a fair multi-filter comparison:

  For each held-out sequence k in [01, 04, 06, 07, 08, 09, 10]:
    1. Run ins_genetic_cv.py  --held-out <drive_k>  (tune classical filters on 6 seqs)
    2. Run train_ai_imu.py    --held-out <drive_k>  (train CNN on 6 seqs)
    3. Run ins_compare.py     --test-seq <k>         (GPS-aided, outage simulation)
    4. Run ins_compare.py     --test-seq <k> --dr-mode  (dead-reckoning, paper-comparable)

  Collects per-fold metrics from the JSON outputs, computes mean ± std across all 7
  folds, and prints a paper-ready summary table.

Usage
-----
  # Full evaluation (genetic CV + CNN training + comparison):
  python run_loo_evaluation.py

  # Skip genetic CV (use existing filter_params.json):
  python run_loo_evaluation.py --skip-genetic

  # Skip CNN training (use existing iekfnets_held_*.p weights):
  python run_loo_evaluation.py --skip-train

  # Skip both — only run ins_compare and collect results:
  python run_loo_evaluation.py --skip-genetic --skip-train

  # Limit to specific sequences:
  python run_loo_evaluation.py --seqs 01 04 07

  # Genetic CV options forwarded to ins_genetic_cv.py:
  python run_loo_evaluation.py --genetic-epochs 100 --genetic-pop 30

  # CNN training options:
  python run_loo_evaluation.py --train-epochs 400 --kitti-raw-dir /data/kitti/raw

Notes
-----
- Genetic CV is run once per filter per fold. With 6 classical filters × 7 folds,
  this is 42 genetic CV runs — expect several hours on a CPU cluster.
- CNN training is run once per fold (400 epochs each) — expect GPU.
- Use --skip-genetic / --skip-train to resume a partial evaluation.
- Results JSON files are written by ins_compare.py to:
    outputs/<filter_key>/<seq>_outage_*/  (GPS-aided)
    outputs/<filter_key>/<seq>_dr_mode/  (DR mode)
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

_HERE      = Path(__file__).parent
_REPO_ROOT = _HERE / '../../..'

# ── Clean KITTI sequences for LOO ─────────────────────────────────────────────
# Maps KITTI seq ID → full drive name (must match data_loader.KITTI_SEQ_TO_DRIVE)
KITTI_CLEAN_SEQS = {
    '01': '2011_10_03_drive_0042_extract',
    '04': '2011_09_30_drive_0016_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}

# Classical filters tuned by genetic CV
CLASSICAL_FILTERS = [
    'ekf_vanilla', 'ekf_enhanced',
    'eskf_vanilla', 'eskf_enhanced',
    'iekf_vanilla', 'iekf_enhanced',
]

# All filter keys reported in the comparison table
ALL_FILTER_KEYS = CLASSICAL_FILTERS + ['imu_only', 'iekf_ai_imu']


def _run(cmd, desc, dry_run=False):
    """Run a shell command, streaming output. Raises on non-zero exit."""
    print(f"\n{'[DRY RUN] ' if dry_run else ''}>>> {desc}")
    print(f"    CMD: {' '.join(str(c) for c in cmd)}\n")
    if dry_run:
        return
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(_HERE),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}")


def _results_json_path(fkey, seq_id, outage_start, outage_dur, dr_mode):
    """Return the path ins_compare.py writes results to."""
    from ins_config import OUTAGE_START, OUTAGE_DURATION
    t1 = outage_start
    d  = outage_dur
    if dr_mode:
        subdir = f'{seq_id}_dr_mode'
    elif t1 > 0 or d > 0:
        subdir = f'{seq_id}_outage_{t1}s_{d}s'
    else:
        subdir = f'{seq_id}_no_outage'
    run_id = f'outage_{t1}s_{d}s' if (t1 > 0 or d > 0) else 'no_outage'
    if dr_mode:
        run_id = 'no_outage'   # ins_compare uses run_id from outage_config, which is None in dr_mode
    return _REPO_ROOT / f'outputs/{fkey}/{subdir}/{run_id}_results.json'


def _collect_metrics(seq_id, outage_start, outage_dur, dr_mode):
    """
    Read per-filter results JSON for one test sequence.

    Returns dict: {filter_key: {'t_rel': float, 'r_rel': float, 'ate_outage': float}}
    """
    t1 = outage_start
    d  = outage_dur
    if dr_mode:
        subdir = f'{seq_id}_dr_mode'
        run_id = 'no_outage'
    elif t1 > 0 or d > 0:
        subdir = f'{seq_id}_outage_{t1}s_{d}s'
        run_id = f'outage_{t1}s_{d}s'
    else:
        subdir = f'{seq_id}_no_outage'
        run_id = 'no_outage'

    results = {}
    for fkey in ALL_FILTER_KEYS:
        json_path = _REPO_ROOT / f'outputs/{fkey}/{subdir}/{run_id}_results.json'
        if not json_path.exists():
            print(f"  WARNING: missing {json_path}")
            continue
        with open(json_path) as fh:
            data = json.load(fh)
        km = data.get('kitti_metrics', {})
        oa = data.get('outage_analysis', {})
        results[fkey] = {
            't_rel':      km.get('t_rel', float('nan')),
            'r_rel':      km.get('r_rel', float('nan')),
            'ate_outage': oa.get('max', float('nan')),
        }
    return results


def _print_table(fold_results, mode_label):
    """
    fold_results: list of dicts {filter_key: {t_rel, r_rel, ate_outage}}
                  one dict per fold (test sequence)
    """
    print(f"\n{'='*95}")
    print(f"LOO RESULTS — {mode_label}  ({len(fold_results)} folds)")
    print(f"{'='*95}")
    print(f"{'Filter':<20} {'t_rel mean':>12} {'t_rel std':>10} "
          f"{'r_rel mean':>12} {'r_rel std':>10} {'ATE-outage mean':>16} {'ATE-outage std':>15}")
    print(f"{'-'*95}")

    for fkey in ALL_FILTER_KEYS:
        trels  = [f[fkey]['t_rel']      for f in fold_results if fkey in f]
        rrels  = [f[fkey]['r_rel']      for f in fold_results if fkey in f]
        ates   = [f[fkey]['ate_outage'] for f in fold_results if fkey in f]
        n = len(trels)
        if n == 0:
            continue
        t_m  = np.nanmean(trels);  t_s  = np.nanstd(trels,  ddof=min(1, n-1))
        r_m  = np.nanmean(rrels);  r_s  = np.nanstd(rrels,  ddof=min(1, n-1))
        a_m  = np.nanmean(ates);   a_s  = np.nanstd(ates,   ddof=min(1, n-1))
        print(f"{fkey:<20} {t_m:>10.2f}% {t_s:>9.2f}%"
              f" {r_m:>10.2f}°/km {r_s:>8.2f}°/km"
              f" {a_m:>13.2f} m {a_s:>12.2f} m")
    print(f"{'='*95}\n")


def main():
    parser = argparse.ArgumentParser(
        description='LOO evaluation: genetic CV + CNN training + comparison for all LOO folds')
    parser.add_argument('--dataset', choices=['kitti', 'cookies'], default='kitti',
                        help='Dataset family to evaluate (default: kitti). '
                             'kitti and cookies results are never mixed.')
    parser.add_argument('--seqs', nargs='+', default=None,
                        metavar='SEQ', help='Seq IDs to evaluate (default: all clean seqs for the dataset). '
                                            'For kitti: "01".."10". For cookies: "c01".."c06".')
    parser.add_argument('--skip-genetic', action='store_true',
                        help='Skip genetic tuning (use existing filter_params.json)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip CNN training — cookies: always skipped (not yet supported)')
    parser.add_argument('--skip-compare', action='store_true',
                        help='Skip ins_compare; only collect existing JSON results')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing them')

    # Genetic CV forwarding (kitti only)
    parser.add_argument('--genetic-epochs', type=int, default=None,
                        help='Generations for genetic CV (forwarded to ins_genetic_cv.py --generations)')
    parser.add_argument('--genetic-pop', type=int, default=None,
                        help='Population size for genetic CV (forwarded)')
    parser.add_argument('--genetic-filter', type=str, default=None,
                        help='Run genetic tuning only for this filter key (forwarded)')

    # CNN training forwarding (kitti only)
    parser.add_argument('--train-epochs', type=int, default=400,
                        help='Epochs for CNN training (default: 400, matches paper)')
    parser.add_argument('--kitti-raw-dir', type=str, default=None,
                        help='Path to KITTI raw data root (required for CNN training)')

    args = parser.parse_args()

    # ── Resolve sequence list based on dataset ─────────────────────────────────
    sys.path.insert(0, str(_HERE))
    import ins_config
    from data_loader import COOKIES_CLEAN_SEQS

    if args.dataset == 'cookies':
        valid_seqs = list(COOKIES_CLEAN_SEQS.keys())
    else:
        valid_seqs = list(KITTI_CLEAN_SEQS.keys())

    seqs = args.seqs if args.seqs is not None else valid_seqs
    for s in seqs:
        if s not in valid_seqs:
            parser.error(f"Unknown sequence '{s}' for dataset '{args.dataset}'. Valid: {valid_seqs}")

    outage_start = ins_config.OUTAGE_START
    outage_dur   = ins_config.OUTAGE_DURATION

    print(f"\n{'='*65}")
    print(f"LOO EVALUATION — {len(seqs)} folds  [dataset={args.dataset}]")
    print(f"Sequences : {seqs}")
    print(f"Skip genetic    : {args.skip_genetic}")
    print(f"Skip CNN train  : {args.skip_train}")
    print(f"Skip compare    : {args.skip_compare}")
    print(f"Dry run         : {args.dry_run}")
    print(f"{'='*65}\n")

    genetic_fast_script = _HERE / 'ins_genetic_fast.py'
    genetic_cv_script   = _HERE / 'ins_genetic_cv.py'
    train_script        = _HERE / 'dl_filters/deep_iekf/train_ai_imu.py'
    compare_script      = _HERE / 'ins_compare.py'

    gps_fold_results = []   # list of per-fold dicts (GPS-aided mode)
    dr_fold_results  = []   # list of per-fold dicts (DR mode)

    for seq_id in seqs:
        print(f"\n{'#'*65}")
        if args.dataset == 'cookies':
            print(f"# FOLD [cookies]: held-out = {seq_id}")
        else:
            drive = KITTI_CLEAN_SEQS[seq_id]
            print(f"# FOLD [kitti]: held-out = {seq_id}  ({drive})")
        print(f"{'#'*65}")

        # ── 1. Genetic tuning ──────────────────────────────────────────────────
        if not args.skip_genetic:
            if args.dataset == 'cookies':
                # Use ins_genetic_fast.py for cookies LOO
                filters_to_tune = ([args.genetic_filter] if args.genetic_filter
                                   else CLASSICAL_FILTERS)
                for filt in filters_to_tune:
                    cmd = [sys.executable, str(genetic_fast_script),
                           '--dataset', 'cookies', '--seq', seq_id, filt]
                    if args.genetic_epochs:
                        cmd += ['--maxiter', str(args.genetic_epochs)]
                    if args.genetic_pop:
                        cmd += ['--popsize', str(args.genetic_pop)]
                    _run(cmd, f"Genetic fast — {filt} (held-out: {seq_id})", args.dry_run)
            else:
                # Use ins_genetic_cv.py for kitti LOO (original flow)
                filters_to_tune = ([args.genetic_filter] if args.genetic_filter
                                   else CLASSICAL_FILTERS)
                for filt in filters_to_tune:
                    cmd = [sys.executable, str(genetic_cv_script),
                           '--filter', filt,
                           '--held-out', KITTI_CLEAN_SEQS[seq_id]]
                    if args.genetic_epochs:
                        cmd += ['--generations', str(args.genetic_epochs)]
                    if args.genetic_pop:
                        cmd += ['--population', str(args.genetic_pop)]
                    _run(cmd, f"Genetic CV — {filt} (held-out: {seq_id})", args.dry_run)

        # ── 2. CNN training (kitti only) ───────────────────────────────────────
        if not args.skip_train:
            if args.dataset == 'cookies':
                print(f"  INFO: CNN training not yet supported for cookies — skipping fold {seq_id}.")
            else:
                drive = KITTI_CLEAN_SEQS[seq_id]
                if args.kitti_raw_dir is None:
                    print(f"  WARNING: --kitti-raw-dir not provided. Skipping CNN training for fold {seq_id}.")
                    print(f"           Run manually: python {train_script} --mode kitti "
                          f"--kitti-raw-dir <path> --held-out {drive} --epochs {args.train_epochs}")
                else:
                    cmd = [sys.executable, str(train_script),
                           '--mode', 'kitti',
                           '--kitti-raw-dir', args.kitti_raw_dir,
                           '--epochs', str(args.train_epochs),
                           '--held-out', drive]
                    _run(cmd, f"CNN training (held-out: {seq_id})", args.dry_run)

        # ── 3. ins_compare — GPS-aided mode ───────────────────────────────────
        if not args.skip_compare:
            cmd = [sys.executable, str(compare_script),
                   '--dataset', args.dataset, '--test-seq', seq_id]
            _run(cmd, f"ins_compare GPS-aided (test seq: {seq_id})", args.dry_run)

        # ── 4. ins_compare — Dead-reckoning mode ──────────────────────────────
        if not args.skip_compare:
            cmd = [sys.executable, str(compare_script),
                   '--dataset', args.dataset, '--test-seq', seq_id, '--dr-mode']
            _run(cmd, f"ins_compare DR mode (test seq: {seq_id})", args.dry_run)

        # ── Collect metrics for this fold ─────────────────────────────────────
        if not args.dry_run:
            gps_mets = _collect_metrics(seq_id, outage_start, outage_dur, dr_mode=False)
            dr_mets  = _collect_metrics(seq_id, outage_start, outage_dur, dr_mode=True)
            if gps_mets:
                gps_fold_results.append(gps_mets)
            if dr_mets:
                dr_fold_results.append(dr_mets)

    # ── Final summary tables ──────────────────────────────────────────────────
    if not args.dry_run:
        if gps_fold_results:
            _print_table(gps_fold_results, 'GPS-aided INS (outage simulation)')
        if dr_fold_results:
            _print_table(dr_fold_results, 'Dead-reckoning (Brossard et al. 2020 Table I comparable)')

        # Save aggregated results to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = _REPO_ROOT / f'outputs/loo_summary_{timestamp}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            'dataset':  args.dataset,
            'seqs':     seqs,
            'gps_mode': gps_fold_results,
            'dr_mode':  dr_fold_results,
        }
        with open(out_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f"Aggregated LOO results saved to: {out_path}\n")
    else:
        print("\n[DRY RUN] No commands were executed. Remove --dry-run to run for real.\n")


if __name__ == '__main__':
    main()
