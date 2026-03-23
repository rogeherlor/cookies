#!/usr/bin/env bash
# run_evaluation_grid.sh — Systematic outage evaluation grid for paper results.
#
# Runs ins_compare.py across:
#   - Table A: no-outage baseline
#   - Table B: outage grid (start: 40s / 60s / 80s × duration: 30s / 60s)
#
# Usage:
#   ./run_evaluation_grid.sh                     # default dataset (ins_config.py)
#   ./run_evaluation_grid.sh --test-seq 08       # specific KITTI sequence
#   ./run_evaluation_grid.sh --test-seq 08 --no-baseline  # skip Table A
#
# Outputs land in outputs/comparison/<dataset>_<outage_tag>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE="$SCRIPT_DIR/ins_compare.py"

# ── Parse args ────────────────────────────────────────────────────────────────
TEST_SEQ_ARG=""
RUN_BASELINE=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-seq)    TEST_SEQ_ARG="--test-seq $2"; shift 2 ;;
        --no-baseline) RUN_BASELINE=false; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Grid definition ───────────────────────────────────────────────────────────
STARTS=(40 60 80)       # outage start times [s]
DURATIONS=(30 60)       # outage durations [s]  (standard in literature)

# ── Helper ────────────────────────────────────────────────────────────────────
run_case() {
    local label="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $label"
    echo "════════════════════════════════════════════════════════════════"
    python "$COMPARE" "$@"
}

# ── Table A: No-outage baseline ───────────────────────────────────────────────
if $RUN_BASELINE; then
    run_case "TABLE A — No outage (baseline)" \
        $TEST_SEQ_ARG \
        --outage-start 0 --outage-duration 0
fi

# ── Table B: Outage grid ──────────────────────────────────────────────────────
TOTAL=$(( ${#STARTS[@]} * ${#DURATIONS[@]} ))
COUNT=0

for start in "${STARTS[@]}"; do
    for dur in "${DURATIONS[@]}"; do
        COUNT=$(( COUNT + 1 ))
        run_case "TABLE B [$COUNT/$TOTAL] — Outage: start=${start}s, duration=${dur}s" \
            $TEST_SEQ_ARG \
            --outage-start "$start" --outage-duration "$dur"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Evaluation grid complete: 1 baseline + $TOTAL outage scenarios"
echo "════════════════════════════════════════════════════════════════"
