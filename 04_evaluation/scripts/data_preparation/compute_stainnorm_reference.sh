#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# Compute Stain Normalization Reference Images and Statistics
# =============================================================================
#
# PURPOSE:
#   Computes Macenko stain normalization reference images and per-region
#   channel statistics for experiments 6-10 (stainnorm variants).
#
# WHAT IT DOES:
#   1. Samples 1000 patches from training set (via splits/train/*.json)
#   2. Extracts patches from WSI using bbox coordinates
#   3. Computes pixel-wise median image → saves as {region}_reference.npy
#   4. Fits MacenkoNormalizer to median image
#   5. Applies stain normalization to all sampled patches
#   6. Computes channel mean/std on NORMALIZED patches → saves as stats.json
#
# OUTPUTS (per region):
#   data/04_evaluation/normalization/
#     ├── {region}_reference.npy           # Median reference image (uint8)
#     ├── {region}_reference.png           # Visualization of reference
#     ├── {region}_stainnorm_comparison.png # Before/after visualization
#     └── {region}_stainnorm_stats.json    # Channel mean/std statistics
#
# USAGE:
#   # Run all regions (recommended for reproducibility)
#   ./compute_stainnorm_reference.sh all
#
#   # Run single region
#   ./compute_stainnorm_reference.sh french
#   ./compute_stainnorm_reference.sh hungarian
#   ./compute_stainnorm_reference.sh swedish
#   ./compute_stainnorm_reference.sh mediterranean
#
#   # Run "all" combined (unified reference from all regions)
#   ./compute_stainnorm_reference.sh combined
#
# REQUIREMENTS:
#   - tiatoolbox (pip install tiatoolbox)
#   - tiffslide or openslide-python
#   - Train/val splits must exist (run train_val_splitter.py first)
#
# AFTER RUNNING:
#   1. Update experiment_config.yaml with the exact values from stats.json
#   2. Verify reference.npy files exist at paths referenced in config
#   3. Run experiments 6-10 with stainnorm enabled
#
# =============================================================================

set -e

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$PROJECT_ROOT/04_evaluation/scripts/data_preparation/compute_stainnorm_reference.py"
OUTPUT_DIR="$PROJECT_ROOT/data/04_evaluation/normalization"
LOG_DIR="$OUTPUT_DIR/logs"

# Activate virtual environment
cd "$PROJECT_ROOT"
source .venv/bin/activate

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Default parameters
NUM_SAMPLES=1000
PATCH_SIZE=518
SEED=42

# Parse arguments
REGION="${1:-help}"

print_help() {
    echo ""
    echo "Usage: ./compute_stainnorm_reference.sh <region>"
    echo ""
    echo "Regions:"
    echo "  french        - Compute for French dataset only"
    echo "  hungarian     - Compute for Hungarian dataset only"  
    echo "  swedish       - Compute for Swedish dataset only"
    echo "  mediterranean - Compute for Mediterranean dataset only"
    echo "  combined      - Compute unified reference from ALL regions"
    echo "  all           - Compute for ALL regions sequentially (recommended)"
    echo ""
    echo "Outputs go to: data/04_evaluation/normalization/"
    echo ""
    exit 0
}

run_region() {
    local region="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$LOG_DIR/compute_${region}_${timestamp}.log"
    
    echo ""
    echo "============================================================"
    echo "COMPUTING STAIN NORMALIZATION: $region"
    echo "============================================================"
    echo "Output: $OUTPUT_DIR/${region}_*.{npy,json,png}"
    echo "Log: $log_file"
    echo ""
    
    {
        echo "============================================================"
        echo "COMPUTING STAIN NORMALIZATION: $region"
        echo "Started: $(date)"
        echo "Samples: $NUM_SAMPLES"
        echo "Patch size: $PATCH_SIZE"
        echo "Seed: $SEED"
        echo "============================================================"
        echo ""
        
        python "$SCRIPT" \
            --region "$region" \
            --num_samples "$NUM_SAMPLES" \
            --patch_size "$PATCH_SIZE" \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR"
        
        echo ""
        echo "============================================================"
        echo "COMPLETED: $region"
        echo "Finished: $(date)"
        echo "============================================================"
    } 2>&1 | tee "$log_file"
    
    echo ""
}

verify_outputs() {
    local region="$1"
    
    echo "Verifying outputs for $region..."
    
    local ref_file="$OUTPUT_DIR/${region}_reference.npy"
    local stats_file="$OUTPUT_DIR/${region}_stainnorm_stats.json"
    
    if [[ -f "$ref_file" && -f "$stats_file" ]]; then
        echo "✓ Reference: $ref_file"
        echo "✓ Statistics: $stats_file"
        
        # Print the normalization values for copy-paste into experiment_config.yaml
        echo ""
        echo "=== VALUES FOR experiment_config.yaml ==="
        echo "# $region stainnorm experiment:"
        python3 -c "
import json
with open('$stats_file') as f:
    stats = json.load(f)
print(f\"normalize_mean: [{stats['mean'][0]:.4f}, {stats['mean'][1]:.4f}, {stats['mean'][2]:.4f}]\")
print(f\"normalize_std: [{stats['std'][0]:.4f}, {stats['std'][1]:.4f}, {stats['std'][2]:.4f}]\")
"
        echo "reference_path: \"data/04_evaluation/normalization/${region}_reference.npy\""
        echo "========================================"
    else
        echo "❌ Missing outputs for $region"
        [[ ! -f "$ref_file" ]] && echo "  Missing: $ref_file"
        [[ ! -f "$stats_file" ]] && echo "  Missing: $stats_file"
        return 1
    fi
}

# Main dispatch
case "$REGION" in
    help|--help|-h)
        print_help
        ;;
    french|hungarian|swedish|mediterranean)
        run_region "$REGION"
        verify_outputs "$REGION"
        ;;
    combined)
        run_region "all"  # "all" is the combined reference
        verify_outputs "all"
        ;;
    all)
        echo ""
        echo "============================================================"
        echo "COMPUTING STAIN NORMALIZATION FOR ALL REGIONS"
        echo "============================================================"
        echo "This will compute reference images and statistics for:"
        echo "  - french"
        echo "  - hungarian"
        echo "  - swedish"
        echo "  - mediterranean"
        echo "  - all (combined)"
        echo ""
        echo "Estimated time: ~5-10 minutes total"
        echo ""
        
        for r in french hungarian swedish mediterranean all; do
            run_region "$r"
            verify_outputs "$r"
        done
        
        echo ""
        echo "============================================================"
        echo "ALL REGIONS COMPLETED"
        echo "============================================================"
        echo ""
        echo "Next steps:"
        echo "1. Review the statistics above"
        echo "2. Update experiment_config.yaml experiments 6-10 with exact values"
        echo "3. Run experiments: ./train_all.sh 6-10"
        echo ""
        ;;
    *)
        echo "Unknown region: $REGION"
        print_help
        ;;
esac

echo ""
echo "Done. Outputs saved to: $OUTPUT_DIR/"
echo ""
