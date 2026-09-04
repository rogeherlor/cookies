#!/usr/bin/env bash
# run_journal_evaluation.sh — Populate the journal-table cells.
#
# After `run_genetic_loo.sh` has produced LOO-tuned parameters in
# filter_params.json, this script runs `ins_compare.py` once per
# (held-out sequence × regime) so each cell of the no-outage and
# 60 s-outage tables in 3.tex can be read off the result JSONs.
#
# For each held-out KITTI drive {01, 04, 06, 07, 08, 09, 10}:
#   * no-outage   →  ins_compare.py --outage-start 0  --outage-duration 0
#   * 60s outage  →  ins_compare.py --outage-start 40 --outage-duration 60
#
# Each invocation uses the LOO-tuned parameters keyed
# `__loo_held_<drive>__` automatically (see filter_params.get).
#
# Usage:
#   ./run_journal_evaluation.sh                                  # all 7 KITTI seqs
#   ./run_journal_evaluation.sh --dataset cookies                # cookies (5 seqs)
#   ./run_journal_evaluation.sh --seqs 01 08                     # only seqs 01, 08
#
# After this completes, run:
#   python emit_journal_tables.py        # writes outputs/tables/*.tex

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE="$SCRIPT_DIR/ins_compare.py"

# Use python3 by default. Override with PYTHON=python ./run_journal_evaluation.sh
# when running inside a venv whose interpreter is named `python`.
PYTHON="${PYTHON:-python3}"

DATASET="kitti"
declare -a SEQS_ARG=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --seqs)    shift; while [[ $# -gt 0 && ! "$1" == --* ]]; do SEQS_ARG+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ${#SEQS_ARG[@]} -eq 0 ]]; then
    if [[ "$DATASET" == "cookies" ]]; then
        SEQS=(c01 c02 c03 c04 c05)
    else
        SEQS=(01 04 06 07 08 09 10)
    fi
else
    SEQS=("${SEQS_ARG[@]}")
fi

OUTAGE_START_TABLE2=40
OUTAGE_DURATION_TABLE2=60

echo "════════════════════════════════════════════════════════════════"
echo "  JOURNAL-TABLE EVALUATION"
echo "  Dataset:   ${DATASET}"
echo "  Sequences: ${SEQS[*]}"
echo "  Regimes:   no-outage  +  ${OUTAGE_START_TABLE2}s start / ${OUTAGE_DURATION_TABLE2}s duration"
echo "════════════════════════════════════════════════════════════════"
echo ""

TOTAL=$(( ${#SEQS[@]} * 2 ))
COUNT=0

for seq in "${SEQS[@]}"; do
    COUNT=$(( COUNT + 1 ))
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  [$COUNT/$TOTAL]  ${DATASET}/${seq}  — no-outage"
    echo "────────────────────────────────────────────────────────────────"
    "$PYTHON" "$COMPARE" \
        --dataset "$DATASET" \
        --test-seq "$seq" \
        --outage-start 0 \
        --outage-duration 0

    COUNT=$(( COUNT + 1 ))
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  [$COUNT/$TOTAL]  ${DATASET}/${seq}  — outage ${OUTAGE_START_TABLE2}s+${OUTAGE_DURATION_TABLE2}s"
    echo "────────────────────────────────────────────────────────────────"
    "$PYTHON" "$COMPARE" \
        --dataset "$DATASET" \
        --test-seq "$seq" \
        --outage-start "$OUTAGE_START_TABLE2" \
        --outage-duration "$OUTAGE_DURATION_TABLE2"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Journal evaluation complete"
echo "  Per-(filter, seq, regime) JSONs in outputs/<filter>/<seq>_*/"
echo "  Next step: python emit_journal_tables.py --dataset $DATASET"
echo "════════════════════════════════════════════════════════════════"
