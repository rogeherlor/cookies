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
#   ./run_dl_training.sh                                    # all DL filters, all folds
#   ./run_dl_training.sh tlio deep_kf                       # specific filters
#   ./run_dl_training.sh --kitti-raw-dir /path/to/kitti_raw # needed for ai_imu
#   ./run_dl_training.sh --skip-existing                    # skip already trained folds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO TRAINING"
echo "  Training strategy: OUTAGE-FREE (clean sequences only)"
echo "  Sequences: 01 04 06 07 08 09 10"
echo "════════════════════════════════════════════════════════════════"
echo ""

python "$SCRIPT_DIR/ins_train.py" --seqs 01 04 06 07 08 09 10 "$@"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  DL LOO training complete"
echo "  Weights saved to artifacts/<filter>/"
echo "════════════════════════════════════════════════════════════════"
