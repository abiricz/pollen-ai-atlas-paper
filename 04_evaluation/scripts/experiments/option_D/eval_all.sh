#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Evaluate Distillation (Option D) Experiment
# =============================================================================
#
# Evaluates the distilled student model using Option A's evaluation pipeline.
# The student model is architecturally identical to Option A (384→46 linear head).
#
# Usage:
#   ./eval_all.sh                           # Evaluate distill_all (intra + cross)
#   ./eval_all.sh distill_all               # Specific experiment
#   ./eval_all.sh --intra-only distill_all  # Skip cross-region
#   ./eval_all.sh --cross-only distill_all  # Cross-region only
#
# Output matches Option A naming:
#   data/04_evaluation/results/exp16_distill_all/
#     - eval_ts1_legacy.json
#     - eval_ts2_expert.json
#     - eval_cross_{source}_to_{target}.json
#     - confusion_matrix_*.png
#     - evaluation_summary.json
#
# Cross-region overlap reference:
#   French → Hungarian (4), Mediterranean (7), Swedish (11)
#   Hungarian → French (4), Swedish (4), Mediterranean (⛔ 0)
#   Swedish → French (11), Hungarian (4), Mediterranean (3)
#   Mediterranean → French (7), Swedish (3), Hungarian (⛔ 0)
#   All → (no cross-region, unified training set)
#
# =============================================================================

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
DATA_ROOT="$PROJECT_ROOT/data"
LOG_DIR="$DATA_ROOT/04_evaluation/results/logs"

cd "$SCRIPT_DIR"
source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$LOG_DIR"

CONFIG="../experiment_config.yaml"
DEVICE="${CUDA_DEVICE:-cuda:0}"

INTRA_ONLY=false
CROSS_ONLY=false
EXPERIMENT=""
SEED=""
SEEDS=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --intra-only) INTRA_ONLY=true; shift ;;
        --cross-only) CROSS_ONLY=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;  # comma-separated: e.g., 41,42,43
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) EXPERIMENT="$1"; shift ;;
    esac
done

eval_experiment() {
    local exp_name="$1"
    local source_ds="$2"
    local targets="$3"  # Space-separated list
    local run_seed="$4"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_suffix=""
    [ -n "$run_seed" ] && seed_suffix="_seed${run_seed}"
    local log_file="$LOG_DIR/eval_${exp_name}${seed_suffix}_${timestamp}.log"
    local extra_args=""
    [ -n "$run_seed" ] && extra_args="$extra_args --seed $run_seed"
    [ -n "$RUN_NAME" ] && extra_args="$extra_args --run_name $RUN_NAME"

    echo ""
    echo "================================================================="
    echo " Evaluating: $exp_name"
    echo " Source: $source_ds"
    [[ -n "$targets" ]] && echo " Targets: $targets"
    [ -n "$run_seed" ] && echo " Seed: $run_seed"
    [ -n "$RUN_NAME" ] && echo " Run name: $RUN_NAME"
    echo " Device: $DEVICE"
    echo " Log: $log_file"
    echo "================================================================="
    echo ""

    {
        echo "================================================================="
        echo "EVALUATING: $exp_name"
        echo "Started: $(date)"
        echo "Source: $source_ds"
        [[ -n "$targets" ]] && echo "Targets: $targets"
        echo "================================================================="
        echo ""

        # Intra-domain evaluation (TS1 + TS2)
        if [[ "$CROSS_ONLY" == "false" ]]; then
            echo "[1/2] Intra-region evaluation (TS1, TS2)..."
            python evaluate_distill.py \
                --config "$CONFIG" \
                --experiment "$exp_name" \
                --device "$DEVICE" \
                $extra_args
            echo ""
        else
            echo "[1/2] Intra-region evaluation SKIPPED (--cross-only)"
        fi

        # Cross-region evaluation
        if [[ "$INTRA_ONLY" == "false" && -n "$targets" ]]; then
            echo "[2/2] Cross-region evaluation..."
            python evaluate_distill.py \
                --config "$CONFIG" \
                --experiment "$exp_name" \
                --source_dataset "$source_ds" \
                --target_datasets $targets \
                --device "$DEVICE" \
                $extra_args
        else
            echo "[2/2] Cross-region evaluation SKIPPED"
        fi

        echo ""
        echo "================================================================="
        echo "EVALUATION COMPLETE: $exp_name"
        echo "Finished: $(date)"
        echo "================================================================="
    } 2>&1 | tee -a "$log_file"
}

# Default experiment
if [ -z "$EXPERIMENT" ]; then
    EXPERIMENT="distill_all"
fi

# All Option D experiments
ALL_EXPERIMENTS="distill_all distill_french distill_hungarian distill_swedish distill_mediterranean"

# Main execution
echo "================================================================="
echo " Pollen AI Atlas — Option D: Distillation Evaluation"
echo " $(date)"
echo " Device: $DEVICE"
echo "================================================================="

run_eval() {
    local exp="$1"
    local run_seed="$2"
    case "${exp}" in
        distill_all)
            eval_experiment "distill_all" "all" "" "$run_seed"
            ;;
        distill_french)
            eval_experiment "distill_french" "french" "hungarian swedish mediterranean" "$run_seed"
            ;;
        distill_hungarian)
            eval_experiment "distill_hungarian" "hungarian" "french swedish" "$run_seed"
            ;;
        distill_swedish)
            eval_experiment "distill_swedish" "swedish" "french hungarian mediterranean" "$run_seed"
            ;;
        distill_mediterranean)
            eval_experiment "distill_mediterranean" "mediterranean" "french swedish" "$run_seed"
            ;;
        *)
            eval_experiment "$exp" "all" "" "$run_seed"
            ;;
    esac
}

run_eval_with_seed_matrix() {
    local exp="$1"
    if [ -n "$SEEDS" ]; then
        IFS=',' read -r -a seed_list <<< "$SEEDS"
        for seed_item in "${seed_list[@]}"; do
            local s="$(echo "$seed_item" | xargs)"
            [ -z "$s" ] && continue
            run_eval "$exp" "$s" || echo "[WARNING] $exp eval failed for seed=$s, continuing..."
        done
    else
        run_eval "$exp" "$SEED"
    fi
}

if [ "$EXPERIMENT" = "all" ]; then
    echo "[Eval All] Running all Option D evaluations: $ALL_EXPERIMENTS"
    for exp in $ALL_EXPERIMENTS; do
        run_eval_with_seed_matrix "$exp"
    done
else
    run_eval_with_seed_matrix "$EXPERIMENT"
fi

echo ""
echo "================================================================="
echo " All evaluations complete!"
echo " Results: $DATA_ROOT/04_evaluation/results/"
echo " Logs: $LOG_DIR/"
echo "================================================================="
