#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Train All Classifier Experiments
# =============================================================================
#
# Trains pollen classifiers for each region. No stain normalization applied.
# Each experiment trains on one region and evaluates on expert-validated test sets.
#
# Usage:
#   ./train_all.sh                    # Train all 4 regions (French, Hungarian, Swedish, Mediterranean)
#   ./train_all.sh french             # Train French only
#   ./train_all.sh --epochs 10 all    # Override epochs for all
#   ./train_all.sh --debug french     # Debug mode (1000 samples, 2 epochs)
#
# Output:
#   data/04_evaluation/results/exp02_linear_probe_{region}/
#     - best_model.pth          (best validation checkpoint)
#     - final_model.pth         (last epoch checkpoint)
#     - checkpoint_epoch*.pth   (periodic checkpoints)
#     - training_log.csv        (per-epoch metrics)
#     - experiment_metadata.json
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

# DDP optimization: 2 OMP threads per process (128 cores / 8 GPUs / 8 workers = 2)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# Config is in parent folder (experiments/)
CONFIG="../experiment_config.yaml"
DEVICE="${CUDA_DEVICE:-cuda:0}"

# Parse arguments
DEBUG=""
EPOCHS=""
MAX_SAMPLES=""
EXPERIMENT=""
USE_DDP=""
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
SEED=""
SEEDS=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG="1"
            MAX_SAMPLES="1000"
            EPOCHS="2"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --ddp)
            USE_DDP="1"
            shift
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
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

# Build extra args
EXTRA_ARGS=""
[[ -n "$EPOCHS" ]] && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[[ -n "$MAX_SAMPLES" ]] && EXTRA_ARGS="$EXTRA_ARGS --max_samples $MAX_SAMPLES"

if [[ -n "$USE_DDP" && -z "$NPROC_PER_NODE" ]]; then
    NPROC_PER_NODE=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
fi

# Training function
train_experiment() {
    local exp_name="$1"
    local run_seed="$2"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_suffix=""
    [ -n "$run_seed" ] && seed_suffix="_seed${run_seed}"
    local log_file="$LOG_DIR/train_${exp_name}${seed_suffix}_${timestamp}.log"
    local extra_seed_args=""
    [ -n "$run_seed" ] && extra_seed_args="$extra_seed_args --seed $run_seed"
    [ -n "$RUN_NAME" ] && extra_seed_args="$extra_seed_args --run_name $RUN_NAME"
    
    echo ""
    echo "============================================================"
    echo "TRAINING: $exp_name"
    echo "Device: $DEVICE"
    echo "Log: $log_file"
    [ -n "$run_seed" ] && echo "Seed: $run_seed"
    [ -n "$RUN_NAME" ] && echo "Run name: $RUN_NAME"
    [[ -n "$DEBUG" ]] && echo "Mode: DEBUG (samples=$MAX_SAMPLES, epochs=$EPOCHS)"
    echo "============================================================"
    
    {
        echo "============================================================"
        echo "TRAINING: $exp_name"
        echo "Started: $(date)"
        echo "Device: $DEVICE"
        [[ -n "$USE_DDP" ]] && echo "DDP: enabled (nproc_per_node=$NPROC_PER_NODE)"
        echo "Extra args: $EXTRA_ARGS $extra_seed_args"
        echo "============================================================"
        echo ""

        if [[ -n "$USE_DDP" ]]; then
            torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
                train_classifier.py --config "$CONFIG" --experiment "$exp_name" --device "$DEVICE" $EXTRA_ARGS $extra_seed_args
        else
            python train_classifier.py --config "$CONFIG" --experiment "$exp_name" --device "$DEVICE" $EXTRA_ARGS $extra_seed_args
        fi
        
        echo ""
        echo "============================================================"
        echo "TRAINING COMPLETE: $exp_name"
        echo "Finished: $(date)"
        echo "============================================================"
    } 2>&1 | tee -a "$log_file"
}

run_with_seed_matrix() {
    local exp_name="$1"
    if [ -n "$SEEDS" ]; then
        IFS=',' read -r -a seed_list <<< "$SEEDS"
        for seed_item in "${seed_list[@]}"; do
            local s="$(echo "$seed_item" | xargs)"
            [ -z "$s" ] && continue
            train_experiment "$exp_name" "$s" || echo "[WARNING] $exp_name (seed=$s) failed, continuing..."
        done
    else
        train_experiment "$exp_name" "$SEED"
    fi
}

# Header
echo ""
echo "============================================================"
echo "POLLEN AI ATLAS - CLASSIFIER TRAINING"
echo "============================================================"
echo "Start: $(date)"
echo "Device: $DEVICE"
echo "Config: $CONFIG"
echo ""

# Run based on experiment selection
case "${EXPERIMENT:-all}" in
    french)
        run_with_seed_matrix "linear_probe_french"
        ;;
    hungarian)
        run_with_seed_matrix "linear_probe_hungarian"
        ;;
    swedish)
        run_with_seed_matrix "linear_probe_swedish"
        ;;
    mediterranean)
        run_with_seed_matrix "linear_probe_mediterranean"
        ;;
    combined|all_regions)
        run_with_seed_matrix "linear_probe_all"
        ;;
    all)
        echo "Training all 4 regional experiments (ImageNet normalization)..."
        run_with_seed_matrix "linear_probe_french"
        run_with_seed_matrix "linear_probe_hungarian"
        run_with_seed_matrix "linear_probe_swedish"
        run_with_seed_matrix "linear_probe_mediterranean"
        ;;
    1-5|exp1-5)
        echo "Training experiments 1-5 (all + 4 regional, ImageNet norm)..."
        run_with_seed_matrix "linear_probe_all"
        run_with_seed_matrix "linear_probe_french"
        run_with_seed_matrix "linear_probe_hungarian"
        run_with_seed_matrix "linear_probe_swedish"
        run_with_seed_matrix "linear_probe_mediterranean"
        ;;
    6-10|exp6-10|stainnorm)
        echo "Training experiments 6-10 (stainnorm variants)..."
        echo "Requires: data/04_evaluation/normalization/*_reference.npy files"
        echo ""
        run_with_seed_matrix "linear_probe_french_stainnorm"
        run_with_seed_matrix "linear_probe_hungarian_stainnorm"
        run_with_seed_matrix "linear_probe_swedish_stainnorm"
        run_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        run_with_seed_matrix "linear_probe_all_stainnorm"
        ;;
    all-experiments|1-10)
        echo "Training all experiments (1-10)..."
        echo ""
        echo "=== Experiments 1-5 (ImageNet normalization) ==="
        run_with_seed_matrix "linear_probe_all"
        run_with_seed_matrix "linear_probe_french"
        run_with_seed_matrix "linear_probe_hungarian"
        run_with_seed_matrix "linear_probe_swedish"
        run_with_seed_matrix "linear_probe_mediterranean"
        echo ""
        echo "=== Experiments 6-10 (Stainnorm) ==="
        run_with_seed_matrix "linear_probe_french_stainnorm"
        run_with_seed_matrix "linear_probe_hungarian_stainnorm"
        run_with_seed_matrix "linear_probe_swedish_stainnorm"
        run_with_seed_matrix "linear_probe_mediterranean_stainnorm"
        run_with_seed_matrix "linear_probe_all_stainnorm"
        ;;
    *)
        # Try as direct experiment name
        run_with_seed_matrix "$EXPERIMENT"
        ;;
esac

echo ""
echo "============================================================"
echo "ALL TRAINING COMPLETE"
echo "============================================================"
echo "End: $(date)"
echo "Results: $DATA_ROOT/04_evaluation/results/"
echo "Logs: $LOG_DIR/"
echo ""
