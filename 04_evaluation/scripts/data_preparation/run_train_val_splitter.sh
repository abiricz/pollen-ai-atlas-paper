#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Run Train/Val Splitter
# =============================================================================
#
# Creates train/validation splits from captioned JSONL files, excluding
# all test regions (TS1 legacy + TS2 expert).
#
# Output: data/04_evaluation/splits/
#
# Usage:
#   ./run_train_val_splitter.sh              # Standard run
#   ./run_train_val_splitter.sh --dry_run    # Preview only
#
# =============================================================================

set -e

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "Train/Val Splitter for Pollen AI Atlas"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Script: 04_evaluation/scripts/data_preparation/train_val_splitter.py"
echo ""

# Run with all arguments passed through
python 04_evaluation/scripts/data_preparation/train_val_splitter.py "$@"

echo ""
echo "=============================================="
echo "Complete!"
echo "=============================================="
