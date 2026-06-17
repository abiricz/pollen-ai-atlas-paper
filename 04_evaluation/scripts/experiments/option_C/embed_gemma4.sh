#!/usr/bin/env bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# embed_gemma4.sh — Pre-compute SBERT caption embeddings for Gemma4-BF16
#
# Runs embed_captions.py with the Gemma4 caption model.
# Output: data/04_evaluation/caption_embeddings/gemma4-bf16/
#
# Prerequisites:
#   - Gemma4 re-captioning complete (all 4 affected slides re-captioned)
#   - .venv active or deepl environment available
#
# Usage:
#   bash embed_gemma4.sh                     # All datasets
#   bash embed_gemma4.sh --datasets french   # Single dataset
#   bash embed_gemma4.sh --device cuda:1     # Different GPU
#
# Estimated runtime: ~30-60 min for all 80 slides on a single GPU.
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Activate environment
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv}"
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
fi

DEVICE="${DEVICE:-cuda:0}"
DATASETS="french hungarian mediterranean swedish"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================================"
echo "SBERT EMBEDDING: Gemma4-BF16 captions"
echo "Device:   $DEVICE"
echo "Datasets: $DATASETS"
echo "Output:   data/04_evaluation/caption_embeddings/gemma4-bf16/"
echo "Started:  $(date)"
echo "============================================================"

python "$SCRIPT_DIR/embed_captions.py" \
    --caption_model production_gemma4-bf16_final \
    --datasets $DATASETS \
    --device "$DEVICE" \
    --batch_size 512 \
    $EXTRA_ARGS

echo ""
echo "============================================================"
echo "EMBEDDING COMPLETE: $(date)"
echo "============================================================"
