#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Evaluate All Classifier Experiments
# =============================================================================
#
# Evaluates trained classifiers on expert-validated test sets and cross-region.
# Requires training to be completed first (checkpoints must exist).
#
# Usage:
#   ./eval_all.sh                     # Evaluate all 4 regions (intra + cross)
#   ./eval_all.sh french              # Evaluate French only
#   ./eval_all.sh --intra-only all    # Skip cross-region evaluation
#   ./eval_all.sh --cross-only french # Cross-region only (skip intra)
#
# Evaluation modes:
#   - INTRA-REGION: TS1 (legacy) + TS2 (expert) test sets within same region
#   - CROSS-REGION: Generalization to other regions (overlapping taxa only)
#
# Output:
#   data/04_evaluation/results/exp02_linear_probe_{region}/
#     - eval_ts1_legacy.json
#     - eval_ts2_expert.json
#     - eval_cross_{source}_to_{target}.json
#     - confusion_matrix_*.png
#
# Cross-region overlap reference:
#   French → Hungarian (4), Mediterranean (7), Swedish (11)
#   Hungarian → French (4), Swedish (4), Mediterranean (⛔ 0)
#   Swedish → French (11), Hungarian (4), Mediterranean (3)
#   Mediterranean → French (7), Swedish (3), Hungarian (⛔ 0)
#
# =============================================================================

set -e

# Project root and paths (adjusted for option_A subfolder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
EXPERIMENTS_DIR="$PROJECT_ROOT/04_evaluation/scripts/experiments"
DATA_ROOT="$PROJECT_ROOT/data"
LOG_DIR="$DATA_ROOT/04_evaluation/results/logs"

cd "$SCRIPT_DIR"
source "$PROJECT_ROOT/.venv/bin/activate"
mkdir -p "$LOG_DIR"

# Config is in parent folder (experiments/)
CONFIG="../experiment_config.yaml"
DEVICE="${CUDA_DEVICE:-cuda:0}"

# Parse arguments
INTRA_ONLY=false
CROSS_ONLY=false
EXPERIMENT=""
SEED=""
SEEDS=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --intra-only)
            INTRA_ONLY=true
            shift
            ;;
        --cross-only)
            CROSS_ONLY=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# Evaluation function
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
    echo "============================================================"
    echo "EVALUATING: $exp_name"
    echo "Source: $source_ds"
    [[ -n "$targets" ]] && echo "Targets: $targets"
    [ -n "$run_seed" ] && echo "Seed: $run_seed"
    [ -n "$RUN_NAME" ] && echo "Run name: $RUN_NAME"
    echo "Log: $log_file"
    echo "============================================================"
    
    {
        echo "============================================================"
        echo "EVALUATING: $exp_name"
        echo "Started: $(date)"
        echo "Source: $source_ds"
        [[ -n "$targets" ]] && echo "Targets: $targets"
        echo "============================================================"
        echo ""
        
        # Intra-region evaluation (TS1, TS2)
        if [[ "$CROSS_ONLY" == "false" ]]; then
            echo "[1/2] Intra-region evaluation (TS1, TS2)..."
            python evaluate_classifier.py --config "$CONFIG" --experiment "$exp_name" --device "$DEVICE" $extra_args
            echo ""
        else
            echo "[1/2] Intra-region evaluation SKIPPED (--cross-only)"
        fi
        
        # Cross-region evaluation
        if [[ "$INTRA_ONLY" == "false" && -n "$targets" ]]; then
            echo "[2/2] Cross-region evaluation..."
            python evaluate_classifier.py --config "$CONFIG" --experiment "$exp_name" \
                --source_dataset "$source_ds" --target_datasets $targets --device "$DEVICE" $extra_args
        else
            echo "[2/2] Cross-region evaluation SKIPPED"
        fi
        
        echo ""
        echo "============================================================"
        echo "EVALUATION COMPLETE: $exp_name"
        echo "Finished: $(date)"
        echo "============================================================"
    } 2>&1 | tee -a "$log_file"
}

run_eval() {
    local exp="$1"
    local run_seed="$2"
    case "${exp}" in
        linear_probe_all)
            eval_experiment "linear_probe_all" "all" "" "$run_seed" ;;
        linear_probe_french)
            eval_experiment "linear_probe_french" "french" "hungarian swedish mediterranean" "$run_seed" ;;
        linear_probe_hungarian)
            eval_experiment "linear_probe_hungarian" "hungarian" "french swedish" "$run_seed" ;;
        linear_probe_swedish)
            eval_experiment "linear_probe_swedish" "swedish" "french hungarian mediterranean" "$run_seed" ;;
        linear_probe_mediterranean)
            eval_experiment "linear_probe_mediterranean" "mediterranean" "french swedish" "$run_seed" ;;
        linear_probe_french_stainnorm)
            eval_experiment "linear_probe_french_stainnorm" "french" "hungarian swedish mediterranean" "$run_seed" ;;
        linear_probe_hungarian_stainnorm)
            eval_experiment "linear_probe_hungarian_stainnorm" "hungarian" "french swedish" "$run_seed" ;;
        linear_probe_swedish_stainnorm)
            eval_experiment "linear_probe_swedish_stainnorm" "swedish" "french hungarian mediterranean" "$run_seed" ;;
        linear_probe_mediterranean_stainnorm)
            eval_experiment "linear_probe_mediterranean_stainnorm" "mediterranean" "french swedish" "$run_seed" ;;
        linear_probe_all_stainnorm)
            eval_experiment "linear_probe_all_stainnorm" "all" "" "$run_seed" ;;
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

# Header
echo ""
echo "============================================================"
echo "POLLEN AI ATLAS - CLASSIFIER EVALUATION"
echo "============================================================"
echo "Start: $(date)"
echo "Device: $DEVICE"
echo ""

# =============================================================================
# Cross-region target mapping:
#   French → Hungarian (4 taxa), Swedish (11 taxa), Mediterranean (7 taxa)
#   Hungarian → French (4 taxa), Swedish (4 taxa), Mediterranean (⛔ 0)
#   Swedish → French (11 taxa), Hungarian (4 taxa), Mediterranean (3 taxa)
#   Mediterranean → French (7 taxa), Swedish (3 taxa), Hungarian (⛔ 0)
#   All → (no cross-region, unified training set)
# =============================================================================

# Run based on experiment selection
case "${EXPERIMENT:-all}" in

    # =========================================================================
    # EXPERIMENTS 1-5: ImageNet Normalization
    # =========================================================================
    
    # Exp1: All regions combined (no cross-region needed)
    linear_probe_all|combined|all_regions|exp1)
        run_eval_with_seed_matrix "linear_probe_all"
        ;;
    
    # Exp2: French → cross to all other regions
    linear_probe_french|french|exp2)
        run_eval_with_seed_matrix "linear_probe_french"
        ;;
    
    # Exp3: Hungarian → cross to French + Swedish (no Mediterranean overlap)
    linear_probe_hungarian|hungarian|exp3)
        run_eval_with_seed_matrix "linear_probe_hungarian"
        ;;
    
    # Exp4: Swedish → cross to all other regions
    linear_probe_swedish|swedish|exp4)
        run_eval_with_seed_matrix "linear_probe_swedish"
        ;;
    
    # Exp5: Mediterranean → cross to French + Swedish (no Hungarian overlap)
    linear_probe_mediterranean|mediterranean|exp5)
        run_eval_with_seed_matrix "linear_probe_mediterranean"
        ;;

    # =========================================================================
    # EXPERIMENTS 6-10: Stain Normalization (Macenko)
    # =========================================================================
    
    # Exp6: French + stainnorm → cross to all other regions
    linear_probe_french_stainnorm|exp6)
        run_eval_with_seed_matrix "linear_probe_french_stainnorm"
        ;;
    
    # Exp7: Hungarian + stainnorm → cross to French + Swedish
    linear_probe_hungarian_stainnorm|exp7)
        run_eval_with_seed_matrix "linear_probe_hungarian_stainnorm"
        ;;
    
    # Exp8: Swedish + stainnorm → cross to all other regions
    linear_probe_swedish_stainnorm|exp8)
        run_eval_with_seed_matrix "linear_probe_swedish_stainnorm"
        ;;
    
    # Exp9: Mediterranean + stainnorm → cross to French + Swedish
    linear_probe_mediterranean_stainnorm|exp9)
        run_eval_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        ;;
    
    # Exp10: All + stainnorm (no cross-region needed)
    linear_probe_all_stainnorm|exp10)
        run_eval_with_seed_matrix "linear_probe_all_stainnorm"
        ;;

    # =========================================================================
    # BATCH COMMANDS
    # =========================================================================
    
    # Experiments 1-5 (ImageNet normalization)
    1-5|exp1-5|imagenet)
        echo "Evaluating experiments 1-5 (ImageNet normalization)..."
        echo ""
        run_eval_with_seed_matrix "linear_probe_all"
        run_eval_with_seed_matrix "linear_probe_french"
        run_eval_with_seed_matrix "linear_probe_hungarian"
        run_eval_with_seed_matrix "linear_probe_swedish"
        run_eval_with_seed_matrix "linear_probe_mediterranean"
        ;;
    
    # Experiments 6-10 (stain normalization)
    6-10|exp6-10|stainnorm)
        echo "Evaluating experiments 6-10 (stain normalization)..."
        echo ""
        run_eval_with_seed_matrix "linear_probe_french_stainnorm"
        run_eval_with_seed_matrix "linear_probe_hungarian_stainnorm"
        run_eval_with_seed_matrix "linear_probe_swedish_stainnorm"
        run_eval_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        run_eval_with_seed_matrix "linear_probe_all_stainnorm"
        ;;
    
    # All regional experiments (2-5 + 6-9, skip 1 and 10 which are combined)
    regional)
        echo "Evaluating all regional experiments (skip combined)..."
        echo ""
        echo "=== ImageNet normalization ==="
        run_eval_with_seed_matrix "linear_probe_french"
        run_eval_with_seed_matrix "linear_probe_hungarian"
        run_eval_with_seed_matrix "linear_probe_swedish"
        run_eval_with_seed_matrix "linear_probe_mediterranean"
        echo ""
        echo "=== Stain normalization ==="
        run_eval_with_seed_matrix "linear_probe_french_stainnorm"
        run_eval_with_seed_matrix "linear_probe_hungarian_stainnorm"
        run_eval_with_seed_matrix "linear_probe_swedish_stainnorm"
        run_eval_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        ;;
    
    # Experiments 1-9 (all except exp10)
    1-9|exp1-9)
        echo "Evaluating experiments 1-9 (skip exp10)..."
        echo ""
        echo "=== Experiments 1-5 (ImageNet normalization) ==="
        run_eval_with_seed_matrix "linear_probe_all"
        run_eval_with_seed_matrix "linear_probe_french"
        run_eval_with_seed_matrix "linear_probe_hungarian"
        run_eval_with_seed_matrix "linear_probe_swedish"
        run_eval_with_seed_matrix "linear_probe_mediterranean"
        echo ""
        echo "=== Experiments 6-9 (Stain normalization) ==="
        run_eval_with_seed_matrix "linear_probe_french_stainnorm"
        run_eval_with_seed_matrix "linear_probe_hungarian_stainnorm"
        run_eval_with_seed_matrix "linear_probe_swedish_stainnorm"
        run_eval_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        ;;
    
    # All experiments (1-10)
    all|all-experiments|1-10)
        echo "Evaluating all experiments (1-10)..."
        echo ""
        echo "=== Experiments 1-5 (ImageNet normalization) ==="
        run_eval_with_seed_matrix "linear_probe_all"
        run_eval_with_seed_matrix "linear_probe_french"
        run_eval_with_seed_matrix "linear_probe_hungarian"
        run_eval_with_seed_matrix "linear_probe_swedish"
        run_eval_with_seed_matrix "linear_probe_mediterranean"
        echo ""
        echo "=== Experiments 6-10 (Stain normalization) ==="
        run_eval_with_seed_matrix "linear_probe_french_stainnorm"
        run_eval_with_seed_matrix "linear_probe_hungarian_stainnorm"
        run_eval_with_seed_matrix "linear_probe_swedish_stainnorm"
        run_eval_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        run_eval_with_seed_matrix "linear_probe_all_stainnorm"
        ;;
    
    *)
        echo "ERROR: Unknown experiment '$EXPERIMENT'"
        echo ""
        echo "Usage: ./eval_all.sh [EXPERIMENT]"
        echo ""
        echo "Individual experiments:"
        echo "  exp1, linear_probe_all          - All regions, ImageNet norm"
        echo "  exp2, linear_probe_french       - French, ImageNet norm"
        echo "  exp3, linear_probe_hungarian    - Hungarian, ImageNet norm"
        echo "  exp4, linear_probe_swedish      - Swedish, ImageNet norm"
        echo "  exp5, linear_probe_mediterranean - Mediterranean, ImageNet norm"
        echo "  exp6, linear_probe_french_stainnorm - French, stain norm"
        echo "  exp7, linear_probe_hungarian_stainnorm - Hungarian, stain norm"
        echo "  exp8, linear_probe_swedish_stainnorm - Swedish, stain norm"
        echo "  exp9, linear_probe_mediterranean_stainnorm - Mediterranean, stain norm"
        echo "  exp10, linear_probe_all_stainnorm - All regions, stain norm"
        echo ""
        echo "Batch commands:"
        echo "  1-5, imagenet   - All ImageNet experiments"
        echo "  6-10, stainnorm - All stain norm experiments"
        echo "  regional        - All regional (skip combined)"
        echo "  all, 1-10       - All experiments"
        echo ""
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "============================================================"
echo "End: $(date)"
echo ""
echo "Cross-region overlap summary:"
echo "  French → Hungarian: 4 taxa, Swedish: 11 taxa, Mediterranean: 7 taxa"
echo "  Hungarian → French: 4 taxa, Swedish: 4 taxa, Mediterranean: ⛔ 0"
echo "  Swedish → French: 11 taxa, Hungarian: 4 taxa, Mediterranean: 3 taxa"
echo "  Mediterranean → French: 7 taxa, Swedish: 3 taxa, Hungarian: ⛔ 0"
echo ""
echo "Results: $DATA_ROOT/04_evaluation/results/"
echo "Logs: $LOG_DIR/"
echo ""
