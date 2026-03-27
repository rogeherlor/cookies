#!/usr/bin/env bash
# run_genetic_loo.sh — LOO genetic parameter tuning for all classical filters.
#
# Trains on N-1 sequences per fold using ins_genetic_fast.py.
# Outage used during tuning is fixed and explicit (default: 60s at start=60s).
# DL filters are NOT tuned here — they have their own training scripts.
#
# Strategy: "Train once, evaluate all" — one representative outage during
# tuning produces parameters robust to both GPS-aided and outage conditions.
# The evaluation grid (run_evaluation_grid.sh) then tests across multiple
# outage scenarios with these fixed parameters.
#
# Usage:
#   ./run_genetic_loo.sh                                         # kitti, all folds, default outage
#   ./run_genetic_loo.sh --dataset cookies                       # cookies, all folds
#   ./run_genetic_loo.sh --outage-start 80 --outage-duration 40  # custom outage
#   ./run_genetic_loo.sh --filters eskf_enhanced iekf_enhanced   # specific filters

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="kitti"
OUTAGE_START=60     # [s] — default training outage start
OUTAGE_DURATION=60  # [s] — default training outage duration (literature standard)

# ── Parse arguments ───────────────────────────────────────────────────────────
FILTERS_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET="$2"; shift 2 ;;
        --outage-start)    OUTAGE_START="$2"; shift 2 ;;
        --outage-duration) OUTAGE_DURATION="$2"; shift 2 ;;
        --filters)         shift; FILTERS_ARG="$*"; break ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Sequence list (per dataset) ───────────────────────────────────────────────
if [[ "$DATASET" == "cookies" ]]; then
    CLEAN_SEQS=(c01 c02 c03 c04 c05 c06)
elif [[ "$DATASET" == "kitti" ]]; then
    CLEAN_SEQS=(01 04 06 07 08 09 10)
else
    echo "Unknown dataset: '$DATASET'. Use 'kitti' or 'cookies'."
    exit 1
fi

TOTAL=${#CLEAN_SEQS[@]}
COUNT=0

echo "════════════════════════════════════════════════════════════════"
echo "  GENETIC LOO PARAMETER TUNING"
echo "  Dataset:  ${DATASET}"
echo "  Training outage: start=${OUTAGE_START}s, duration=${OUTAGE_DURATION}s"
echo "  Sequences: ${CLEAN_SEQS[*]}"
echo "  Filters: ${FILTERS_ARG:-all}"
echo "════════════════════════════════════════════════════════════════"
echo ""

for seq in "${CLEAN_SEQS[@]}"; do
    COUNT=$(( COUNT + 1 ))
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  LOO fold [$COUNT/$TOTAL]: held-out seq=${seq}"
    echo "════════════════════════════════════════════════════════════════"
    python "$SCRIPT_DIR/ins_genetic_fast.py" $FILTERS_ARG \
        --dataset "$DATASET" \
        --seq "$seq" --3d \
        --outage-start "$OUTAGE_START" \
        --outage-duration "$OUTAGE_DURATION"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Genetic LOO complete: $TOTAL folds"
echo "  Outage used: start=${OUTAGE_START}s, duration=${OUTAGE_DURATION}s"
echo "  Results saved to filter_params.json"
echo "════════════════════════════════════════════════════════════════"
