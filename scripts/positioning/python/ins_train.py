# -*- coding: utf-8 -*-
"""
ins_train.py — DL model training orchestrator for INS/GNSS deep learning filters.

Trains all deep learning filter models (TLIO, Deep KF, Tartan IMU, AI-IMU) across
KITTI clean sequences in leave-one-out (LOO) mode, analogous to how ins_genetic_cv.py
tunes classical filter parameters.

Clean KITTI sequences used for LOO: 01, 04, 06, 07, 08, 09, 10
(Seqs 00, 02, 05 have 2s data gaps; 03 has no raw data)

Usage
-----
# Train all DL models, all LOO folds:
python ins_train.py

# Train specific models:
python ins_train.py tlio deep_kf

# Train specific folds:
python ins_train.py --seqs 01 04

# Also train all-sequences models (for non-LOO testing, e.g. seq 02):
python ins_train.py --train-all

# Skip folds that already have weight files:
python ins_train.py --skip-existing

# Preview commands without executing:
python ins_train.py --dry-run

# Custom epoch counts:
python ins_train.py tlio --epochs-tlio 300

# AI-IMU requires raw KITTI data path:
python ins_train.py ai_imu --kitti-raw-dir /path/to/kitti/raw
"""
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

_HERE      = Path(__file__).parent
_REPO_ROOT = (_HERE / '../../..').resolve()
_ARTIFACTS = _REPO_ROOT / 'artifacts'
_LOGS_DIR  = _REPO_ROOT / 'logs'

# Clean KITTI sequences for LOO training (no data gaps, raw data available).
CLEAN_SEQS = ['01', '04', '06', '07', '08', '09', '10']

# KITTI sequence ID → full drive name (for AI-IMU --held-out argument).
KITTI_SEQ_TO_DRIVE = {
    '00': '2011_10_03_drive_0027_extract',
    '01': '2011_10_03_drive_0042_extract',
    '02': '2011_10_03_drive_0034_extract',
    '04': '2011_09_30_drive_0016_extract',
    '05': '2011_09_30_drive_0018_extract',
    '06': '2011_09_30_drive_0020_extract',
    '07': '2011_09_30_drive_0027_extract',
    '08': '2011_09_30_drive_0028_extract',
    '09': '2011_09_30_drive_0033_extract',
    '10': '2011_09_30_drive_0034_extract',
}

ALL_FILTERS = ['tlio', 'deep_kf', 'tartan_imu', 'ai_imu']


# ── Weight file existence checks ─────────────────────────────────────────────

def _weight_exists(filter_name, seq, mode='loo'):
    """Check if a trained weight file already exists for this filter/fold."""
    if mode == 'all':
        paths = {
            'tlio':       _ARTIFACTS / 'tlio' / 'tlio_resnet.pt',
            'deep_kf':    _ARTIFACTS / 'deep_kf' / 'deep_kf.pt',
            'tartan_imu': _ARTIFACTS / 'tartan_imu' / 'lora_adapters.pt',
            'ai_imu':     _ARTIFACTS / 'deep_iekf' / 'iekfnets.p',
        }
    else:
        drive = KITTI_SEQ_TO_DRIVE.get(seq, seq)
        paths = {
            'tlio':       _ARTIFACTS / 'tlio' / f'fold_{seq}.pt',
            'deep_kf':    _ARTIFACTS / 'deep_kf' / f'fold_{seq}.pt',
            'tartan_imu': _ARTIFACTS / 'tartan_imu' / f'lora_fold_{seq}.pt',
            'ai_imu':     _ARTIFACTS / 'deep_iekf' / f'iekfnets_held_{drive}.p',
        }
    return paths[filter_name].exists()


# ── Training command builders ─────────────────────────────────────────────────

def _build_cmd_tlio(seq, epochs, mode='loo', dataset='kitti'):
    cmd = [sys.executable, str(_HERE / 'dl_filters/tlio/train_tlio.py'),
           '--mode', mode, '--epochs', str(epochs),
           '--dataset', dataset,
           '--output', str(_ARTIFACTS / 'tlio')]
    if mode == 'loo':
        cmd += ['--val-seq', seq]
    return cmd


def _build_cmd_deep_kf(seq, epochs, mode='loo', dataset='kitti'):
    cmd = [sys.executable, str(_HERE / 'dl_filters/deep_kf/train_deep_kf.py'),
           '--mode', mode, '--epochs', str(epochs),
           '--dataset', dataset,
           '--output', str(_ARTIFACTS / 'deep_kf')]
    if mode == 'loo':
        cmd += ['--val-seq', seq]
    return cmd


def _build_cmd_tartan(seq, epochs, mode='loo', dataset='kitti'):
    cmd = [sys.executable, str(_HERE / 'dl_filters/tartan_imu/train_tartan.py'),
           '--mode', mode, '--epochs', str(epochs),
           '--dataset', dataset,
           '--output', str(_ARTIFACTS / 'tartan_imu')]
    if mode == 'loo':
        cmd += ['--val-seq', seq]
    return cmd


def _build_cmd_ai_imu(seq, epochs, kitti_raw_dir, mode='loo'):
    cmd = [sys.executable, str(_HERE / 'dl_filters/deep_iekf/train_ai_imu.py'),
           '--mode', 'kitti', '--epochs', str(epochs),
           '--output', str(_ARTIFACTS / 'deep_iekf')]
    if kitti_raw_dir:
        cmd += ['--kitti-raw-dir', str(kitti_raw_dir)]
    if mode == 'loo':
        drive = KITTI_SEQ_TO_DRIVE[seq]
        cmd += ['--held-out', drive]
    return cmd


CMD_BUILDERS = {
    'tlio':       _build_cmd_tlio,
    'deep_kf':    _build_cmd_deep_kf,
    'tartan_imu': _build_cmd_tartan,
    'ai_imu':     _build_cmd_ai_imu,
}

FILTER_LABELS = {
    'tlio':       'TLIO (ResNet1D displacement)',
    'deep_kf':    'Deep KF (LSTM state prediction)',
    'tartan_imu': 'Tartan IMU (LoRA fine-tuning)',
    'ai_imu':     'AI-IMU (Deep IEKF CNN)',
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train all DL filter models across KITTI/cookies sequences (LOO).')
    parser.add_argument('filters', nargs='*', default=None,
                        help=f'DL filters to train: {", ".join(ALL_FILTERS)} (default: all)')
    parser.add_argument('--dataset', choices=['kitti', 'cookies'], default='kitti',
                        help='Dataset family (default: kitti). '
                             'cookies: tlio/deep_kf/tartan_imu supported; ai_imu skipped.')
    parser.add_argument('--seqs', nargs='+', default=None,
                        help='LOO validation sequence IDs (default: all clean seqs for the dataset). '
                             f'kitti default: {" ".join(CLEAN_SEQS)}. cookies default: c01..c06.')
    parser.add_argument('--epochs-tlio',    type=int, default=200)
    parser.add_argument('--epochs-deep-kf', type=int, default=150)
    parser.add_argument('--epochs-tartan',  type=int, default=50)
    parser.add_argument('--epochs-ai-imu',  type=int, default=400)
    parser.add_argument('--kitti-raw-dir',  type=str, default=None,
                        help='Path to raw KITTI data root (required for ai_imu)')
    parser.add_argument('--train-all', action='store_true',
                        help='Also train all-sequences models after LOO folds')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip folds whose weight files already exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without executing')
    args = parser.parse_args()
    if args.filters is None or len(args.filters) == 0:
        args.filters = list(ALL_FILTERS)
    for f in args.filters:
        if f not in ALL_FILTERS:
            parser.error(f"Unknown filter '{f}'. Choose from: {', '.join(ALL_FILTERS)}")

    # Resolve default sequence list based on dataset
    if args.seqs is None:
        if args.dataset == 'cookies':
            sys.path.insert(0, str(_HERE))
            from data_loader import COOKIES_CLEAN_SEQS
            args.seqs = list(COOKIES_CLEAN_SEQS.keys())
        else:
            args.seqs = list(CLEAN_SEQS)

    epoch_map = {
        'tlio':       args.epochs_tlio,
        'deep_kf':    args.epochs_deep_kf,
        'tartan_imu': args.epochs_tartan,
        'ai_imu':     args.epochs_ai_imu,
    }

    # Validate: AI-IMU needs raw KITTI data path
    if 'ai_imu' in args.filters and not args.dry_run:
        if args.kitti_raw_dir is None:
            print("WARNING: ai_imu training requires --kitti-raw-dir pointing to raw KITTI data.\n"
                  "         AI-IMU folds will be skipped unless --kitti-raw-dir is provided.\n"
                  "         (Use --dry-run to see planned commands without this check.)")

    # ── Logging ───────────────────────────────────────────────────────────────
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = _LOGS_DIR / f'ins_train_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 65)
    logger.info("DL FILTER TRAINING ORCHESTRATOR")
    logger.info("=" * 65)
    logger.info(f"Filters    : {', '.join(args.filters)}")
    logger.info(f"LOO seqs   : {', '.join(args.seqs)}")
    logger.info(f"Train-all  : {args.train_all}")
    logger.info(f"Skip exist : {args.skip_existing}")
    logger.info(f"Dry run    : {args.dry_run}")
    for f in args.filters:
        logger.info(f"  {f:12s} epochs={epoch_map[f]}")
    logger.info(f"Log file   : {log_file}")
    logger.info("=" * 65)

    # ── Build task list ───────────────────────────────────────────────────────
    tasks = []

    # LOO folds
    for seq in args.seqs:
        if args.dataset == 'kitti' and seq not in KITTI_SEQ_TO_DRIVE:
            logger.warning(f"Sequence '{seq}' not in KITTI_SEQ_TO_DRIVE, skipping.")
            continue
        for filt in args.filters:
            tasks.append({'filter': filt, 'seq': seq, 'mode': 'loo'})

    # All-sequences models
    if args.train_all:
        for filt in args.filters:
            tasks.append({'filter': filt, 'seq': None, 'mode': 'all'})

    # ── Execute ───────────────────────────────────────────────────────────────
    n_total   = len(tasks)
    n_skipped = 0
    n_ok      = 0
    n_fail    = 0

    for i, task in enumerate(tasks, 1):
        filt = task['filter']
        seq  = task['seq']
        mode = task['mode']
        epochs = epoch_map[filt]

        if mode == 'loo':
            label = f"{FILTER_LABELS[filt]}  fold={seq}"
        else:
            label = f"{FILTER_LABELS[filt]}  mode=all"

        # Skip AI-IMU for cookies (no CookiesDataset adapter yet)
        if filt == 'ai_imu' and args.dataset == 'cookies':
            logger.info(f"[{i}/{n_total}] SKIP (ai_imu cookies not supported) {label}")
            n_skipped += 1
            continue

        # Skip if weights exist
        if args.skip_existing and seq is not None and _weight_exists(filt, seq, mode):
            logger.info(f"[{i}/{n_total}] SKIP (exists) {label}")
            n_skipped += 1
            continue
        if args.skip_existing and mode == 'all' and _weight_exists(filt, '', mode):
            logger.info(f"[{i}/{n_total}] SKIP (exists) {label}")
            n_skipped += 1
            continue

        # Skip AI-IMU if no raw dir
        if filt == 'ai_imu' and args.kitti_raw_dir is None and not args.dry_run:
            logger.info(f"[{i}/{n_total}] SKIP (no --kitti-raw-dir) {label}")
            n_skipped += 1
            continue

        # Build command
        builder = CMD_BUILDERS[filt]
        if filt == 'ai_imu':
            cmd = builder(seq, epochs, args.kitti_raw_dir, mode)
        else:
            cmd = builder(seq, epochs, mode, args.dataset)

        cmd_str = ' '.join(cmd)

        if args.dry_run:
            logger.info(f"[{i}/{n_total}] DRY-RUN {label}")
            logger.info(f"  $ {cmd_str}")
            n_ok += 1
            continue

        logger.info(f"[{i}/{n_total}] START {label}")
        logger.info(f"  $ {cmd_str}")

        try:
            result = subprocess.run(cmd, cwd=str(_HERE), check=True,
                                    stdout=sys.stdout, stderr=sys.stderr)
            logger.info(f"[{i}/{n_total}] DONE  {label}")
            n_ok += 1
        except subprocess.CalledProcessError as e:
            logger.error(f"[{i}/{n_total}] FAIL  {label}  (exit code {e.returncode})")
            n_fail += 1
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 65)
    logger.info("TRAINING SUMMARY")
    logger.info(f"  Total tasks : {n_total}")
    logger.info(f"  Completed   : {n_ok}")
    logger.info(f"  Skipped     : {n_skipped}")
    logger.info(f"  Failed      : {n_fail}")
    logger.info("=" * 65)

    # ── Verification: check which weight files exist ──────────────────────────
    logger.info("")
    logger.info("Weight file status:")
    for filt in args.filters:
        for seq in args.seqs:
            exists = _weight_exists(filt, seq, 'loo')
            status = "OK" if exists else "MISSING"
            logger.info(f"  {filt:12s} fold_{seq}: {status}")
        if args.train_all:
            exists = _weight_exists(filt, '', 'all')
            status = "OK" if exists else "MISSING"
            logger.info(f"  {filt:12s} all-seqs: {status}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
