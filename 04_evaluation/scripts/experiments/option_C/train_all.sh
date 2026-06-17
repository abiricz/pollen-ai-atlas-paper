#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Train All LUPI (Option C) Experiments
# =============================================================================
#
# Trains LUPI classifiers (image + caption → classifier, image-only at test time).
# Mirrors Option A's train_all.sh structure exactly.
#
# Prerequisites:
#   1. Caption embeddings pre-computed:
#      python embed_captions.py
#   2. Option A train/eval complete (for comparison)
#
# Usage:
#   ./train_all.sh                    # Train all 5 LUPI experiments
#   ./train_all.sh lupi_all           # Train combined only
#   ./train_all.sh --debug lupi_all   # Debug (500 samples, 2 epochs)
#
# Output:
#   data/04_evaluation/results/exp11_lupi_all/
#   data/04_evaluation/results/exp12_lupi_french/
#   data/04_evaluation/results/exp13_lupi_hungarian/
#   data/04_evaluation/results/exp14_lupi_swedish/
#   data/04_evaluation/results/exp15_lupi_mediterranean/
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

# DDP optimization: 2 OMP threads per process (128 cores / 8 GPUs / 8 workers = 2)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

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
NUM_WORKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG="1"
            MAX_SAMPLES="500"
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
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

EXTRA_ARGS=""
[[ -n "$EPOCHS" ]] && EXTRA_ARGS="$EXTRA_ARGS --epochs $EPOCHS"
[[ -n "$MAX_SAMPLES" ]] && EXTRA_ARGS="$EXTRA_ARGS --max_samples $MAX_SAMPLES"
[[ -n "$NUM_WORKERS" ]] && EXTRA_ARGS="$EXTRA_ARGS --num_workers $NUM_WORKERS"

if [[ -n "$USE_DDP" && -z "$NPROC_PER_NODE" ]]; then
    NPROC_PER_NODE=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
fi

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
    echo "LUPI TRAINING: $exp_name"
    echo "Device: $DEVICE"
    echo "Log: $log_file"
    [ -n "$run_seed" ] && echo "Seed: $run_seed"
    [ -n "$RUN_NAME" ] && echo "Run name: $RUN_NAME"
    [[ -n "$DEBUG" ]] && echo "Mode: DEBUG (samples=$MAX_SAMPLES, epochs=$EPOCHS)"
    echo "============================================================"
    
    {
        echo "============================================================"
        echo "LUPI TRAINING: $exp_name"
        echo "Started: $(date)"
        echo "Device: $DEVICE"
        [[ -n "$USE_DDP" ]] && echo "DDP: enabled (nproc_per_node=$NPROC_PER_NODE)"
        echo "Extra args: $EXTRA_ARGS $extra_seed_args"
        echo "============================================================"

        if [[ -n "$USE_DDP" ]]; then
            torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
                train_lupi.py --config "$CONFIG" --experiment "$exp_name" --device "$DEVICE" $EXTRA_ARGS $extra_seed_args
        else
            python train_lupi.py --config "$CONFIG" --experiment "$exp_name" --device "$DEVICE" $EXTRA_ARGS $extra_seed_args
        fi
        
        echo ""
        echo "TRAINING COMPLETE: $exp_name at $(date)"
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

echo ""
echo "============================================================"
echo "POLLEN AI ATLAS - LUPI (Option C) TRAINING"
echo "============================================================"
echo "Start: $(date)"
echo "Device: $DEVICE"
echo ""

case "${EXPERIMENT:-all}" in
    lupi_all|combined|exp11)
        run_with_seed_matrix "lupi_all"
        ;;
    lupi_french|french|exp12)
        run_with_seed_matrix "lupi_french"
        ;;
    lupi_hungarian|hungarian|exp13)
        run_with_seed_matrix "lupi_hungarian"
        ;;
    lupi_swedish|swedish|exp14)
        run_with_seed_matrix "lupi_swedish"
        ;;
    lupi_mediterranean|mediterranean|exp15)
        run_with_seed_matrix "lupi_mediterranean"
        ;;
    all|11-15)
        echo "Training all 5 LUPI experiments..."
        run_with_seed_matrix "lupi_all"
        run_with_seed_matrix "lupi_french"
        run_with_seed_matrix "lupi_hungarian"
        run_with_seed_matrix "lupi_swedish"
        run_with_seed_matrix "lupi_mediterranean"
        ;;
    *)
        run_with_seed_matrix "$EXPERIMENT"
        ;;
esac

echo ""
echo "============================================================"
echo "ALL LUPI TRAINING COMPLETE"
echo "============================================================"
echo "End: $(date)"
echo "Results: $DATA_ROOT/04_evaluation/results/"
echo ""
