#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Pollen AI Atlas - Train Distillation (Option D) Experiment
# =============================================================================
#
# Knowledge distillation: teacher→student training.
#
# DEFAULT: Stage 2 only (reuse Option C teacher checkpoint).
#   The script auto-discovers the matching Option C (LUPI) teacher.
#   No need to specify --teacher-checkpoint unless overriding.
#
# FULL TWO-STAGE: Pass --train-teacher to train teacher from scratch.
#   Stage 1: Teacher on [img+caption], Stage 2: Student on [img] with KD.
#
# Prerequisites:
#   1. Caption embeddings pre-computed:
#      python option_C/embed_captions.py
#   2. Option C teacher trained (unless using --train-teacher):
#      cd option_C && ./train_all.sh
#
# Usage:
#   ./train_all.sh                             # Default: Stage 2 only (reuse Option C)
#   ./train_all.sh distill_all                 # Same, explicit name
#   ./train_all.sh --debug distill_all         # Debug (500 samples, 2 epochs)
#   ./train_all.sh --train-teacher distill_all # Full two-stage (train teacher too)
#   ./train_all.sh --teacher-checkpoint /path/to/ckpt.pth distill_all  # Custom teacher
#
# Output:
#   data/04_evaluation/results/exp16_distill_all/
#     ├── teacher/                  # Stage 1 artifacts (only with --train-teacher)
#     │   ├── best_model.pth
#     │   └── training_log.csv
#     ├── student/                  # Stage 2 artifacts
#     │   ├── best_model.pth
#     │   └── training_log.csv
#     ├── best_model.pth            # Student (for evaluation)
#     ├── cached_teacher_logits.pt
#     └── distillation_summary.json
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
TRAIN_TEACHER=""
TEACHER_CKPT=""
CACHE_LOGITS=""
ALLOW_CACHE_WITH_AUG=""
SEED=""
SEEDS=""
RUN_NAME=""
TEACHER_SEED=""
TEACHER_RUN_NAME=""
USE_DDP=""
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

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
        --train-teacher)
            TRAIN_TEACHER="1"
            shift
            ;;
        --teacher-checkpoint)
            TEACHER_CKPT="$2"
            shift 2
            ;;
        --cache-logits)
            CACHE_LOGITS="1"
            shift
            ;;
        --allow-cache-with-augmentation)
            ALLOW_CACHE_WITH_AUG="1"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"  # comma-separated: e.g., 41,42,43
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        --teacher-seed)
            TEACHER_SEED="$2"
            shift 2
            ;;
        --teacher-run-name)
            TEACHER_RUN_NAME="$2"
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
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# Default experiment
if [ -z "$EXPERIMENT" ]; then
    EXPERIMENT="distill_all"
fi

if [[ -n "$USE_DDP" && -z "$NPROC_PER_NODE" ]]; then
    NPROC_PER_NODE=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")
fi

# All Option D experiments (matching Option A Exp1-5 and Option C Exp11-15)
ALL_EXPERIMENTS="distill_all distill_french distill_hungarian distill_swedish distill_mediterranean"

train_experiment() {
    local exp_name="$1"
    local run_seed="$2"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local seed_suffix=""
    [ -n "$run_seed" ] && seed_suffix="_seed${run_seed}"
    local log_file="$LOG_DIR/train_${exp_name}${seed_suffix}_${timestamp}.log"

    echo ""
    echo "================================================================="
    echo " Training: $exp_name"
    echo " Device: $DEVICE"
    [ -n "$run_seed" ] && echo " Seed: $run_seed"
    [ -n "$RUN_NAME" ] && echo " Run name: $RUN_NAME"
    [ -n "$USE_DDP" ] && echo " DDP: enabled (nproc_per_node=$NPROC_PER_NODE)"
    echo " Log: $log_file"
    if [ -n "$DEBUG" ]; then
        echo " MODE: DEBUG (max_samples=$MAX_SAMPLES, epochs=$EPOCHS)"
    fi
    if [ -n "$TRAIN_TEACHER" ]; then
        echo " MODE: Full two-stage (training teacher from scratch)"
    else
        echo " MODE: Stage 2 only (reusing Option C teacher)"
    fi
    echo "================================================================="
    echo ""

    # Build command — stage2_only is the default in train_distill.py
    if [ -n "$USE_DDP" ]; then
        CMD="torchrun --standalone --nproc_per_node $NPROC_PER_NODE train_distill.py --config $CONFIG --experiment $exp_name --device $DEVICE"
    else
        CMD="python train_distill.py --config $CONFIG --experiment $exp_name --device $DEVICE"
    fi

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi
    if [ -n "$EPOCHS" ]; then
        CMD="$CMD --epochs $EPOCHS"
    fi
    if [ -n "$TRAIN_TEACHER" ]; then
        CMD="$CMD --train_teacher"
    fi
    if [ -n "$TEACHER_CKPT" ]; then
        CMD="$CMD --teacher_checkpoint $TEACHER_CKPT"
    fi
    if [ -n "$CACHE_LOGITS" ]; then
        CMD="$CMD --cache_logits"
    fi
    if [ -n "$ALLOW_CACHE_WITH_AUG" ]; then
        CMD="$CMD --allow_cache_with_augmentation"
    fi
    if [ -n "$run_seed" ]; then
        CMD="$CMD --seed $run_seed"
    fi
    if [ -n "$RUN_NAME" ]; then
        CMD="$CMD --run_name $RUN_NAME"
    fi
    if [ -n "$TEACHER_SEED" ]; then
        CMD="$CMD --teacher_seed $TEACHER_SEED"
    fi
    if [ -n "$TEACHER_RUN_NAME" ]; then
        CMD="$CMD --teacher_run_name $TEACHER_RUN_NAME"
    fi

    echo "[$(date)] Starting: $CMD"
    echo ""

    # Run with logging
    $CMD 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "[$(date)] ✓ $exp_name completed successfully"
    else
        echo ""
        echo "[$(date)] ✗ $exp_name FAILED (exit code: $exit_code)"
        echo "  Check log: $log_file"
        return $exit_code
    fi
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

# Main execution
echo "================================================================="
echo " Pollen AI Atlas — Option D: Knowledge Distillation"
echo " $(date)"
echo "================================================================="

if [ "$EXPERIMENT" = "all" ]; then
    echo "[Train All] Running all Option D experiments: $ALL_EXPERIMENTS"
    for exp in $ALL_EXPERIMENTS; do
        run_with_seed_matrix "$exp"
    done
else
    run_with_seed_matrix "$EXPERIMENT"
fi

echo ""
echo "================================================================="
echo " All training complete!"
echo " Evaluate with: ./eval_all.sh $EXPERIMENT"
echo "================================================================="
