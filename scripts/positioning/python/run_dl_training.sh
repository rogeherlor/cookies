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

# ── Parse --dataset (pass remaining args through to ins_train.py) ─────────────
DATASET="kitti"
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        *)         PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

if [[ "$DATASET" == "cookies" ]]; then
    SEQS="c01 c02 c03 c04 c05 c06"
else
    SEQS="01 04 06 07 08 09 10"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO TRAINING"
echo "  Dataset: ${DATASET}"
echo "  Training strategy: OUTAGE-FREE (clean sequences only)"
echo "  Sequences: ${SEQS}"
echo "════════════════════════════════════════════════════════════════"
echo ""

python "$SCRIPT_DIR/ins_train.py" --dataset "$DATASET" --seqs $SEQS "${PASSTHROUGH_ARGS[@]}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO training complete"
echo "  Weights saved to artifacts/<filter>/"
echo "════════════════════════════════════════════════════════════════"
