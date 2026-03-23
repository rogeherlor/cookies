#!/usr/bin/env bash
# run_full_pipeline.sh — Complete LOO training + evaluation pipeline for paper.
#
# Runs the full reproducible pipeline in order:
#   Step 1: Genetic LOO parameter tuning  (classical filters, with outage in fitness)
#   Step 2: DL LOO training               (TLIO, Deep KF, Tartan, AI-IMU — outage-free)
#   Step 3: Evaluation grid per sequence   (no-outage baseline + outage grid)
#
# Training strategy:
#   - Genetic: fixed outage (default start=60s, dur=60s) in fitness function
#   - DL: outage-free (models learn motion dynamics on clean data)
#   - Evaluation: 1 baseline + 6 outage cells (start 40/60/80 × dur 30/60)
#
# Usage:
#   ./run_full_pipeline.sh                                       # full pipeline
#   ./run_full_pipeline.sh --skip-genetic                        # skip step 1
#   ./run_full_pipeline.sh --skip-dl                             # skip step 2
#   ./run_full_pipeline.sh --skip-genetic --skip-dl              # evaluation only
#   ./run_full_pipeline.sh --kitti-raw-dir /path/to/kitti_raw    # needed for AI-IMU
#   ./run_full_pipeline.sh --outage-start 80 --outage-duration 40  # custom genetic outage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLEAN_SEQS=(01 04 06 07 08 09 10)

# ── Defaults ──────────────────────────────────────────────────────────────────
SKIP_GENETIC=false
SKIP_DL=false
SKIP_EVAL=false
OUTAGE_START=60
OUTAGE_DURATION=60
KITTI_RAW_ARG=""
DL_EXTRA_ARGS=""
GENETIC_FILTERS_ARG=""

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-genetic)    SKIP_GENETIC=true; shift ;;
        --skip-dl)         SKIP_DL=true; shift ;;
        --skip-eval)       SKIP_EVAL=true; shift ;;
        --outage-start)    OUTAGE_START="$2"; shift 2 ;;
        --outage-duration) OUTAGE_DURATION="$2"; shift 2 ;;
        --kitti-raw-dir)   KITTI_RAW_ARG="--kitti-raw-dir $2"; shift 2 ;;
        --skip-existing)   DL_EXTRA_ARGS="$DL_EXTRA_ARGS --skip-existing"; shift ;;
        --filters)         shift; GENETIC_FILTERS_ARG="--filters $*"; break ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              FULL LOO PIPELINE FOR PAPER                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Sequences     : ${CLEAN_SEQS[*]}                  ║"
echo "║  Genetic outage: start=${OUTAGE_START}s, dur=${OUTAGE_DURATION}s                    ║"
echo "║  DL training   : outage-free                               ║"
echo "║  Eval grid     : no-outage + (40/60/80 × 30/60)           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Step 1 Genetic: $(if $SKIP_GENETIC; then echo "SKIP"; else echo "RUN "; fi)                                       ║"
echo "║  Step 2 DL     : $(if $SKIP_DL; then echo "SKIP"; else echo "RUN "; fi)                                       ║"
echo "║  Step 3 Eval   : $(if $SKIP_EVAL; then echo "SKIP"; else echo "RUN "; fi)                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Genetic LOO parameter tuning ──────────────────────────────────────
if ! $SKIP_GENETIC; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  STEP 1/3 — GENETIC LOO PARAMETER TUNING                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    "$SCRIPT_DIR/run_genetic_loo.sh" \
        --outage-start "$OUTAGE_START" \
        --outage-duration "$OUTAGE_DURATION" \
        $GENETIC_FILTERS_ARG
else
    echo "── Step 1 skipped (--skip-genetic) ──"
fi

# ── Step 2: DL LOO training ──────────────────────────────────────────────────
if ! $SKIP_DL; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  STEP 2/3 — DL LOO TRAINING (outage-free)                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    "$SCRIPT_DIR/run_dl_training.sh" $KITTI_RAW_ARG $DL_EXTRA_ARGS
else
    echo "── Step 2 skipped (--skip-dl) ──"
fi

# ── Step 3: Evaluation grid per sequence ──────────────────────────────────────
if ! $SKIP_EVAL; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  STEP 3/3 — EVALUATION GRID (all sequences)               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    TOTAL=${#CLEAN_SEQS[@]}
    COUNT=0

    for seq in "${CLEAN_SEQS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  Evaluating seq ${seq} [$COUNT/$TOTAL]"
        echo "════════════════════════════════════════════════════════════════"
        "$SCRIPT_DIR/run_evaluation_grid.sh" --test-seq "$seq"
    done
else
    echo "── Step 3 skipped (--skip-eval) ──"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PIPELINE COMPLETE                                         ║"
echo "║                                                            ║"
echo "║  Genetic params : filter_params.json                       ║"
echo "║  DL weights     : artifacts/<filter>/                      ║"
echo "║  Eval results   : outputs/comparison/<seq>_<outage>/       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
