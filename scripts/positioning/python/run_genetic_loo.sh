#!/usr/bin/env bash
# run_genetic_loo.sh — LOO genetic parameter tuning for all classical filters.
#
# Drives `ins_genetic_cv.py` once per held-out sequence, training each filter
# on the other clean drives (LOO). The cost function is the journal-grade
# three-component normalised cost defined in `ins_cost.py`:
#
#     J = ATE_outage / 1 m  +  t_rel / 1 %  +  r_rel / 1 deg/km
#         subject to  ANEES ∈ [0.1, 10]  (consistency constraint)
#
# DE budget defaults: popsize = 10, maxiter = 15. The cost surface is
# smooth and 30+ generation runs have been observed to plateau by step 14
# in earlier experiments. ANEES is gated only during training; validation
# always reports a finite cost (gate disabled in validate_params).
# DL filters are NOT tuned here — they have their own training scripts.
#
# Usage:
#   ./run_genetic_loo.sh                                            # kitti, all folds, all filters
#   ./run_genetic_loo.sh --dataset cookies                          # cookies, all folds
#   ./run_genetic_loo.sh --filters esekfs_enhanced iekf_enhanced     # only listed filters
#   ./run_genetic_loo.sh --maxiter 60 --popsize 20                  # bigger budget
#
# Per-fold runtime (KITTI, 6 filters × 12 train pairs × 600 evals × ~1 s):
# expect ~6–10 hours per fold on a 6-core workstation. Eight folds = overnight.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use python3 by default (codebase requires Python 3.5+ for type hints).
# Override with PYTHON=python ./run_genetic_loo.sh when running inside a venv
# whose interpreter is named `python`.
PYTHON="${PYTHON:-python3}"

DATASET="kitti"
MAXITER=15       # DE generations
POPSIZE=10       # DE population per generation
WORKERS=-1       # -1 = all CPUs; set to 1 to debug
OUTAGES=1        # outage windows per TRAINING dataset (each one is a filter run / eval)
VAL_OUTAGES=2    # outage windows per VALIDATION dataset (averaged to a single val cost)

# ── Parse arguments ───────────────────────────────────────────────────────────
FILTERS_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET="$2"; shift 2 ;;
        --maxiter)         MAXITER="$2"; shift 2 ;;
        --popsize)         POPSIZE="$2"; shift 2 ;;
        --workers)         WORKERS="$2"; shift 2 ;;
        --outages)         OUTAGES="$2"; shift 2 ;;
        --val-outages)     VAL_OUTAGES="$2"; shift 2 ;;
        --filters)         shift; FILTERS_ARG="$*"; break ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Sequence list (per dataset) ───────────────────────────────────────────────
if [[ "$DATASET" == "cookies" ]]; then
    # Must match data_loader.COOKIES_CLEAN_SEQS (currently c01..c05).
    CLEAN_SEQS=(c01 c02 c03 c04 c05)
elif [[ "$DATASET" == "kitti" ]]; then
    CLEAN_SEQS=(01 04 06 07 08 09 10)
else
    echo "Unknown dataset: '$DATASET'. Use 'kitti' or 'cookies'."
    exit 1
fi

# Map KITTI seq IDs → drive names (must match data_loader.KITTI_SEQ_TO_DRIVE)
declare -A KITTI_DRIVE
KITTI_DRIVE['01']='2011_10_03_drive_0042_extract'
KITTI_DRIVE['04']='2011_09_30_drive_0016_extract'
KITTI_DRIVE['06']='2011_09_30_drive_0020_extract'
KITTI_DRIVE['07']='2011_09_30_drive_0027_extract'
KITTI_DRIVE['08']='2011_09_30_drive_0028_extract'
KITTI_DRIVE['09']='2011_09_30_drive_0033_extract'
KITTI_DRIVE['10']='2011_09_30_drive_0034_extract'

# For cookies the held-out value is the short ID itself (c01..cNN).

TOTAL=${#CLEAN_SEQS[@]}
COUNT=0

echo "════════════════════════════════════════════════════════════════"
echo "  GENETIC LOO PARAMETER TUNING (cost = ins_cost.py three-component normalised)"
echo "  Dataset:   ${DATASET}"
echo "  Sequences: ${CLEAN_SEQS[*]}"
echo "  DE:        maxiter=${MAXITER}  popsize=${POPSIZE}  workers=${WORKERS}"
echo "  Outages:   train=${OUTAGES}, val=${VAL_OUTAGES} (per dataset)"
echo "  Filters:   ${FILTERS_ARG:-all}"
echo "════════════════════════════════════════════════════════════════"
echo ""

for seq in "${CLEAN_SEQS[@]}"; do
    COUNT=$(( COUNT + 1 ))
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  LOO fold [$COUNT/$TOTAL]: held-out seq=${seq}"
    echo "════════════════════════════════════════════════════════════════"

    if [[ "$DATASET" == "kitti" ]]; then
        HELD_OUT="${KITTI_DRIVE[$seq]}"   # full drive name expected by ins_genetic_cv
    else
        HELD_OUT="$seq"                   # cookies: short ID (c01..cNN) is the key
    fi

    "$PYTHON" "$SCRIPT_DIR/ins_genetic_cv.py" $FILTERS_ARG \
        --type "$DATASET" \
        --held-out "$HELD_OUT" --3d \
        --outages "$OUTAGES" \
        --val-outages "$VAL_OUTAGES" \
        --maxiter "$MAXITER" \
        --popsize "$POPSIZE" \
        --workers "$WORKERS"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Genetic LOO complete: $TOTAL folds"
echo "  Cost function: ins_cost.py (journal-grade normalised)"
echo "  Results saved to filter_params.json"
echo "  Next step: ./run_evaluation_grid.sh   (populate the .tex tables)"
echo "════════════════════════════════════════════════════════════════"
