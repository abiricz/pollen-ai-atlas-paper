#!/usr/bin/env bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# Pollen AI Atlas - Caption Statistics Pipeline
#
# Runs the final public-release caption statistics:
#   1. Anchor vocabulary extraction
#   2. Basic compliance statistics
#   3. Morphological vocabulary coverage
#   4. Cross-model Jaccard/SBERT agreement
#   5. Expert audit sample export
#
# Usage:
#   cd 04_evaluation/scripts/statistics && bash run_caption_stats.sh
#   cd 04_evaluation/scripts/statistics && bash run_caption_stats.sh --skip-sbert

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

echo "============================================================"
echo "Caption Statistics Pipeline"
echo "Repo: $REPO_ROOT"
echo "============================================================"

echo ""
echo "[Step 1/2] Extracting anchor vocabulary..."
python "$SCRIPT_DIR/extract_anchor_vocabulary.py"
echo "[Step 1/2] Done."

echo ""
echo "[Step 2/2] Computing caption statistics..."
python "$SCRIPT_DIR/compute_caption_stats.py"     --workers 8     --audit-sample-size 240     --audit-seed 42     "$@"
echo "[Step 2/2] Done."

echo ""
echo "============================================================"
echo "Pipeline complete. Outputs in:"
echo "  data/04_evaluation/results/caption_statistics/"
echo "============================================================"
