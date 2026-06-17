#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas — Cross-Regional Multimodal Retrieval Experiment
# =============================================================================
#
# Runs the full retrieval evaluation pipeline:
#   1. Main experiment: ALL + FULL + CROSS-REG × {image, text, combined} × 5 VLMs
#   2. Negative control: label-shuffle (3 seeds) to validate signal genuineness
#   3. Result collection: formats tables for paper inclusion
#
# Usage:
#   ./retrieval_experiments.sh                   # Full run + negative control
#   ./retrieval_experiments.sh --main-only       # Skip negative control
#   ./retrieval_experiments.sh --control-only    # Only negative control
#   ./retrieval_experiments.sh --dry-run         # List queries, no retrieval
#
# Output:
#   data/04_evaluation/results/retrieval/
#     retrieval_gemma4-bf16.json       Main results (Gemma-4-31B-IT)
#     retrieval_qwen25vl.json          Main results (Qwen2.5-VL)
#     retrieval_qwen3-fp8.json         Main results (Qwen3-VL)
#     retrieval_qwen35-fp8.json        Main results (Qwen3.5-VL)
#     retrieval_qwen36-fp8.json        Main results (Qwen3.6-VL)
#     retrieval_negative_control.json  Negative control (label shuffle)
#     retrieval_summary.json           Collected summary for paper
#
# Machine:
#   Single-GPU inference (CUDA:0).  ~2 min for main + ~1 min for control.
#
# =============================================================================

set -e

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/retrieval_experiments.py"
DEFAULT_RESULTS_DIR="$PROJECT_ROOT/data/04_evaluation/results/retrieval"
RESULTS_DIR="${RETRIEVAL_RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
LOG_DIR="$RESULTS_DIR"
COLLECTOR="$SCRIPT_DIR/collect_retrieval_results.py"

DEVICE="${RETRIEVAL_DEVICE:-cuda:0}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/retrieval_run_${TIMESTAMP}.log"

# ─── Parse arguments ────────────────────────────────────────────────────────
MAIN=true
CONTROL=true
DRY_RUN=false
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --main-only)    CONTROL=false ;;
        --control-only) MAIN=false ;;
        --dry-run)      DRY_RUN=true ;;
        --vit_dir=*|--vit_checkpoint=*|--n_boot=*|--no_ci)
            EXTRA_ARGS+=("$arg") ;;
        *)              echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ─── Environment ─────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

mkdir -p "$RESULTS_DIR"

# ─── Banner ──────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "  CROSS-REGIONAL MULTIMODAL RETRIEVAL EXPERIMENT"
echo "======================================================================"
echo "  Timestamp:  $TIMESTAMP"
echo "  Device:     $DEVICE"
echo "  Main run:   $MAIN"
echo "  Neg ctrl:   $CONTROL"
echo "  Log:        $LOG_FILE"
echo "======================================================================"
echo ""

# ─── Function: run with logging ─────────────────────────────────────────────
run_logged() {
    local description="$1"
    shift
    echo "──────────────────────────────────────────────────────────────────"
    echo "  $description"
    echo "  $(date +'%Y-%m-%d %H:%M:%S')"
    echo "──────────────────────────────────────────────────────────────────"
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[ERROR] $description failed with exit code $exit_code" | tee -a "$LOG_FILE"
        exit $exit_code
    fi
    echo "" | tee -a "$LOG_FILE"
}

# ─── Dry run ─────────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    run_logged "DRY RUN: listing all queries" \
        env RETRIEVAL_OUTPUT_DIR="$RESULTS_DIR" python "$SCRIPT" --device "$DEVICE" --dry_run "${EXTRA_ARGS[@]}"
    exit 0
fi

# ─── Step 1: Main retrieval experiment ───────────────────────────────────────
if [ "$MAIN" = true ]; then
    run_logged "STEP 1/3: Main retrieval (ALL + FULL + CROSS-REG × image/text/combined × 5 VLMs)" \
        env RETRIEVAL_OUTPUT_DIR="$RESULTS_DIR" python "$SCRIPT" --device "$DEVICE" "${EXTRA_ARGS[@]}"
fi

# ─── Step 2: Negative control (label shuffle) ───────────────────────────────
if [ "$CONTROL" = true ]; then
    if [ "$MAIN" = true ]; then
        # Main was already run — just add negative control
        run_logged "STEP 2/3: Negative control (label-shuffle, seeds 42,123,456)" \
            env RETRIEVAL_OUTPUT_DIR="$RESULTS_DIR" python "$SCRIPT" --device "$DEVICE" --negative_control_only \
            --negative_control_seeds 42 123 456 "${EXTRA_ARGS[@]}"
    else
        # --control-only: skip main, run negative control only
        run_logged "STEP 2/3: Negative control only (label-shuffle, seeds 42,123,456)" \
            env RETRIEVAL_OUTPUT_DIR="$RESULTS_DIR" python "$SCRIPT" --device "$DEVICE" --negative_control_only \
            --negative_control_seeds 42 123 456 "${EXTRA_ARGS[@]}"
    fi
fi

# ─── Step 3: Collect and format results ──────────────────────────────────────
if [ -f "$COLLECTOR" ]; then
    run_logged "STEP 3/3: Collecting results for paper" \
        env RETRIEVAL_RESULTS_DIR="$RESULTS_DIR" python "$COLLECTOR"
else
    echo "[SKIP] Result collector not found: $COLLECTOR"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "  RETRIEVAL EXPERIMENT COMPLETE"
echo "======================================================================"
echo "  Timestamp:  $(date +'%Y-%m-%d %H:%M:%S')"
echo "  Results:    $RESULTS_DIR/"
echo "  Log:        $LOG_FILE"
echo ""

# List output files
echo "  Output files:"
for f in "$RESULTS_DIR"/retrieval_*.json; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "    $(basename "$f")  ($size)"
    fi
done
echo "======================================================================"
