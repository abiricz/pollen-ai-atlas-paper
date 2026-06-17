#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# Cross-regional UMAP visualization — 15 retrieval species
#
# Creates publication-ready 3-panel UMAP figure:
#   (a) Text,  (b) Image,  (c) Combined (α=0.5 late-fusion)
# For both pretrained and finetuned ViT embedding spaces.
# Plus individual panels for flexible LaTeX layout.
#
# Usage:
#   bash run_umap_cross_regional.sh            # Gemma4-BF16, both ViT modes
#   bash run_umap_cross_regional.sh --both     # Gemma4-BF16 + Qwen2.5-VL, both ViT modes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Activate environment if this release was set up with setup_env.sh.
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

COMMON_ARGS="--max_per_origin 1000 --seed 42 --n_neighbors 15 --min_dist 0.1 --dpi 300 --point_size 12 --individual"

echo "============================================================"
echo "Cross-Regional UMAP Visualization (15 Retrieval Species)"
echo "============================================================"

# Gemma4-BF16 — Pretrained ViT (primary VLM for SR)
echo ""
echo ">>> Gemma4-BF16 · Pretrained ViT..."
python 04_evaluation/scripts/visualization/umap_cross_regional.py \
    --vlm gemma4-bf16 --vit_mode pretrained $COMMON_ARGS

# Gemma4-BF16 — Finetuned ViT
echo ""
echo ">>> Gemma4-BF16 · Finetuned ViT..."
python 04_evaluation/scripts/visualization/umap_cross_regional.py \
    --vlm gemma4-bf16 --vit_mode finetuned $COMMON_ARGS

# Optional: Qwen2.5-VL (legacy/supplementary)
if [[ "${1:-}" == "--both" ]]; then
    echo ""
    echo ">>> Qwen2.5-VL · Pretrained ViT..."
    python 04_evaluation/scripts/visualization/umap_cross_regional.py \
        --vlm qwen25vl --vit_mode pretrained $COMMON_ARGS

    echo ""
    echo ">>> Qwen2.5-VL · Finetuned ViT..."
    python 04_evaluation/scripts/visualization/umap_cross_regional.py \
        --vlm qwen25vl --vit_mode finetuned $COMMON_ARGS
fi

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  data/04_evaluation/results/visualization/pretrained/"
echo "  data/04_evaluation/results/visualization/finetuned/"
echo "============================================================"
