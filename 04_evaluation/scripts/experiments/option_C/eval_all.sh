#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Evaluate All LUPI (Option C) Experiments
# =============================================================================
#
# Evaluates LUPI classifiers using the EXACT same metrics as Option A.
# Uses image-only inference (text embedding = zero vector).
#
# Usage:
#   ./eval_all.sh                      # Evaluate all 5 LUPI experiments
#   ./eval_all.sh lupi_all             # Evaluate combined only
#   ./eval_all.sh --intra-only all     # Skip cross-region
#   ./eval_all.sh --cross-only lupi_french  # Cross-region only
#
# Output matches Option A naming:
#   data/04_evaluation/results/exp11_lupi_all/
#     - eval_ts1_legacy.json
#     - eval_ts2_expert.json
#     - eval_cross_*_to_*.json
#     - confusion_matrix_*.png
#     - evaluation_summary.json
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
        --seeds) SEEDS="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        *) EXPERIMENT="$1"; shift ;;
    esac
done

eval_experiment() {
    local exp_name="$1"
    local source_ds="$2"
    local targets="$3"
    local run_seed="$4"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_suffix=""
    [ -n "$run_seed" ] && seed_suffix="_seed${run_seed}"
    local log_file="$LOG_DIR/eval_${exp_name}${seed_suffix}_${timestamp}.log"
    local extra_args=""
    [ -n "$run_seed" ] && extra_args="$extra_args --seed $run_seed"
    [ -n "$RUN_NAME" ] && extra_args="$extra_args --run_name $RUN_NAME"
    
    echo ""
    echo "============================================================"
    echo "EVALUATING LUPI: $exp_name"
    echo "Source: $source_ds"
    [[ -n "$targets" ]] && echo "Targets: $targets"
    [ -n "$run_seed" ] && echo "Seed: $run_seed"
    [ -n "$RUN_NAME" ] && echo "Run name: $RUN_NAME"
    echo "============================================================"
    
    {
        # Intra-region (TS1, TS2)
        if [[ "$CROSS_ONLY" == "false" ]]; then
            echo "[1/2] Intra-region evaluation..."
            python evaluate_lupi.py --config "$CONFIG" --experiment "$exp_name" --device "$DEVICE" $extra_args
        fi
        
        # Cross-region
        if [[ "$INTRA_ONLY" == "false" && -n "$targets" ]]; then
            echo "[2/2] Cross-region evaluation..."
            python evaluate_lupi.py --config "$CONFIG" --experiment "$exp_name" \
                --source_dataset "$source_ds" --target_datasets $targets --device "$DEVICE" $extra_args
        fi
        
        echo "EVALUATION COMPLETE: $exp_name at $(date)"
    } 2>&1 | tee -a "$log_file"
}

run_eval() {
    local exp="$1"
    local run_seed="$2"
    case "${exp}" in
        lupi_all)
            eval_experiment "lupi_all" "all" "" "$run_seed" ;;
        lupi_french)
            eval_experiment "lupi_french" "french" "hungarian swedish mediterranean" "$run_seed" ;;
        lupi_hungarian)
            eval_experiment "lupi_hungarian" "hungarian" "french swedish" "$run_seed" ;;
        lupi_swedish)
            eval_experiment "lupi_swedish" "swedish" "french hungarian mediterranean" "$run_seed" ;;
        lupi_mediterranean)
            eval_experiment "lupi_mediterranean" "mediterranean" "french swedish" "$run_seed" ;;
        *)
            eval_experiment "$exp" "all" "" "$run_seed" ;;
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

echo ""
echo "============================================================"
echo "POLLEN AI ATLAS - LUPI (Option C) EVALUATION"
echo "============================================================"
echo "Start: $(date)"
echo "Device: $DEVICE"
echo ""

case "${EXPERIMENT:-all}" in
    lupi_all|combined|exp11)
        run_eval_with_seed_matrix "lupi_all"
        ;;
    lupi_french|french|exp12)
        run_eval_with_seed_matrix "lupi_french"
        ;;
    lupi_hungarian|hungarian|exp13)
        run_eval_with_seed_matrix "lupi_hungarian"
        ;;
    lupi_swedish|swedish|exp14)
        run_eval_with_seed_matrix "lupi_swedish"
        ;;
    lupi_mediterranean|mediterranean|exp15)
        run_eval_with_seed_matrix "lupi_mediterranean"
        ;;
    all|11-15)
        echo "Evaluating all 5 LUPI experiments..."
        run_eval_with_seed_matrix "lupi_all"
        run_eval_with_seed_matrix "lupi_french"
        run_eval_with_seed_matrix "lupi_hungarian"
        run_eval_with_seed_matrix "lupi_swedish"
        run_eval_with_seed_matrix "lupi_mediterranean"
        ;;
    *)
        run_eval_with_seed_matrix "$EXPERIMENT"
        ;;
esac

echo ""
echo "============================================================"
echo "ALL LUPI EVALUATION COMPLETE"
echo "============================================================"
echo "End: $(date)"
echo "Results: $DATA_ROOT/04_evaluation/results/"
echo ""
