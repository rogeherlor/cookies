#!/usr/bin/env bash
# run_dl_training.sh — LOO deep learning training for all DL filters.
#
# Wraps ins_train.py which handles TLIO, Deep KF, Tartan IMU, and AI-IMU.
#
# DL training is OUTAGE-FREE by design — models learn motion dynamics
# on clean data; outages are simulated only at evaluation time
# (run_evaluation_grid.sh).
#
# Usage:
#   ./run_dl_training.sh                                    # kitti, all DL filters, all folds
#   ./run_dl_training.sh --dataset cookies                  # cookies (DL not yet supported — warns)
#   ./run_dl_training.sh tlio deep_kf                       # specific filters
#   ./run_dl_training.sh --kitti-raw-dir /path/to/kitti_raw # needed for ai_imu
#   ./run_dl_training.sh --skip-existing                    # skip already trained folds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Known DL filter names. Anything in PASSTHROUGH_ARGS that matches one of these
# is routed as a positional argument *before* --seqs, so argparse's nargs='+'
# does not swallow it into the sequence list (which is what happened with
# `./run_dl_training.sh deep_kf` interpreting deep_kf as an 8th sequence).
KNOWN_FILTERS=("tlio" "deep_kf" "tartan_imu" "ai_imu")

# ── Parse --dataset and split remaining args into (filters, other) ────────────
DATASET="kitti"
FILTER_ARGS=()
OTHER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        *)
            arg="$1"
            shift
            matched=0
            for f in "${KNOWN_FILTERS[@]}"; do
                if [[ "$arg" == "$f" ]]; then
                    FILTER_ARGS+=("$arg")
                    matched=1
                    break
                fi
            done
            if [[ $matched -eq 0 ]]; then
                OTHER_ARGS+=("$arg")
            fi
            ;;
    esac
done

if [[ "$DATASET" == "cookies" ]]; then
    SEQS="c01 c02 c03 c04 c05 c06"
else
    SEQS="01 04 06 07 08 09 10"
fi

if [[ ${#FILTER_ARGS[@]} -gt 0 ]]; then
    FILTERS_DISPLAY="${FILTER_ARGS[*]}"
else
    FILTERS_DISPLAY="(all four)"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO TRAINING"
echo "  Dataset: ${DATASET}"
echo "  Training strategy: OUTAGE-FREE (clean sequences only)"
echo "  Sequences: ${SEQS}"
echo "  Filters:   ${FILTERS_DISPLAY}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Filter names go before --seqs so argparse treats them as the positional
# `filters` argument; --skip-existing, --val-metric-every K, etc. go after.
python "$SCRIPT_DIR/ins_train.py" "${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"}" \
    --dataset "$DATASET" --seqs $SEQS \
    "${OTHER_ARGS[@]+"${OTHER_ARGS[@]}"}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO training complete"
echo "  Weights saved to artifacts/<filter>/"
echo "════════════════════════════════════════════════════════════════"
