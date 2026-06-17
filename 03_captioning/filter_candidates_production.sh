#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# PRODUCTION FILTERING SCRIPT
# =============================================================================
# 
# Filters mining results for multiple slides.
# Machine-agnostic: configure paths at top.
#
# Usage:
#   1. Configure PATHS section below for your machine
#   2. Enable/disable slides by uncommenting in slides array
#   3. Run: bash filter_candidates_production.sh
#
# Outputs per slide:
#   - {slide}_filtered.h5
#   - {slide}_test_region.geojson
#   - {slide}_test_region_metadata.json
# =============================================================================

set -e  # Exit on error

# =============================================================================
# MACHINE CONFIGURATION - EDIT THESE FOR YOUR MACHINE
# =============================================================================

# Project root and external data root. Override DATA_ROOT/WSI_ROOT/SAM2_ROOT as needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
WSI_ROOT="${WSI_ROOT:-$DATA_ROOT/00_raw_wsi}"
SAM2_ROOT="${SAM2_ROOT:-$PROJECT_ROOT/../sam2}"

# Mining results input folder (02_mining/)
MINING_RESULTS="${DATA_ROOT}/02_mining"

# WSI files location (00_raw_wsi/)
WSI_ROOT="${WSI_ROOT}"

# Output folder for filtered results (03_captioning/{dataset}/filtered/)
OUTPUT_ROOT="${DATA_ROOT}/03_captioning"

# Model checkpoints
VIT_CKPT="${PROJECT_ROOT}/01_initialization/weights_vit_small_lvd_20250620_0312.pth"
SAM2_CKPT="${SAM2_CKPT:-${SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt}"
SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"

# Query images folder
QUERY_IMAGES="${PROJECT_ROOT}/01_initialization/query_images"

# GPU device
DEVICE="cuda:0"

# =============================================================================
# SLIDES TO PROCESS - Enable by uncommenting
# =============================================================================

# Dataset: french
dataset="french"

slides=(
    # French dataset slides (uncomment to enable)
    #"acer_edf"
    #"ambrosia_edf"
    #"betula_2_edf"
    #"betula_edf"
    #"brassica_napus_edf"
    #"corylus_2_edf"
    #"corylus_edf"
    #"typha_edf"
    # ... add more as needed
)

# Optionally, process all H5 files in mining results folder:
# slides=($(ls ${MINING_RESULTS}/${dataset}/*_detections.h5 2>/dev/null | xargs -n1 basename | sed 's/_detections.h5//'))

# =============================================================================
# FILTERING PARAMETERS
# =============================================================================
NMS_IOU=0.1
CONF_THR=0.0
CLASSIFIER_THR=0.5
TEST_REGION_SIZE=5000
TARGET_GRAINS=100
LOW_MEMORY="--low_memory"  # Enable for slides >100K detections

# =============================================================================
# PROCESSING LOOP
# =============================================================================

echo "============================================================"
echo "PRODUCTION FILTERING"
echo "============================================================"
echo "Dataset: ${dataset}"
echo "Slides: ${#slides[@]}"
echo "Output: ${OUTPUT_ROOT}/${dataset}/"
echo ""

# Create output directory (logs go alongside H5 files, not separate folder)
mkdir -p "${OUTPUT_ROOT}/${dataset}/filtered"

# Activate venv if exists
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

cd "${PROJECT_ROOT}"

for slide in "${slides[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing: ${slide}"
    echo "============================================================"
    
    # Construct paths
    H5_PATH="${MINING_RESULTS}/${dataset}/${slide}_detections.h5"
    WSI_PATH="${WSI_ROOT}/${dataset}/${slide}.tif"
    QUERY_PATH="${QUERY_IMAGES}/${slide}.png"
    OUTPUT_H5="${OUTPUT_ROOT}/${dataset}/filtered/${slide}_filtered.h5"
    LOG_FILE="${OUTPUT_ROOT}/${dataset}/filtered/${slide}_filter.log"
    
    # Check inputs exist
    if [ ! -f "$H5_PATH" ]; then
        echo "WARNING: Mining H5 not found: $H5_PATH - SKIPPING"
        continue
    fi
    if [ ! -f "$WSI_PATH" ]; then
        echo "WARNING: WSI not found: $WSI_PATH - SKIPPING"
        continue
    fi
    if [ ! -f "$QUERY_PATH" ]; then
        echo "WARNING: Query image not found: $QUERY_PATH - SKIPPING"
        continue
    fi
    
    echo "  Input H5: $H5_PATH"
    echo "  WSI: $WSI_PATH"
    echo "  Query: $QUERY_PATH"
    echo "  Output: $OUTPUT_H5"
    echo "  Log: $LOG_FILE"
    echo ""
    
    # Run filtering
    python3 03_captioning/filter_candidates.py \
        --h5_path "$H5_PATH" \
        --wsi_path "$WSI_PATH" \
        --query_image "$QUERY_PATH" \
        --output "$OUTPUT_H5" \
        --vit_ckpt "$VIT_CKPT" \
        --sam2_ckpt "$SAM2_CKPT" \
        --sam2_cfg "$SAM2_CFG" \
        --nms_iou "$NMS_IOU" \
        --conf_thr "$CONF_THR" \
        --classifier_threshold "$CLASSIFIER_THR" \
        --select_test_region \
        --test_region_size "$TEST_REGION_SIZE" \
        --target_grains "$TARGET_GRAINS" \
        $LOW_MEMORY \
        --device "$DEVICE" \
        2>&1 | tee "$LOG_FILE"
    
    echo ""
    echo "  Completed: ${slide}"
    
done

echo ""
echo "============================================================"
echo "FILTERING COMPLETE"
echo "============================================================"
echo "Outputs in: ${OUTPUT_ROOT}/${dataset}/filtered/"
ls -la "${OUTPUT_ROOT}/${dataset}/filtered/"*.h5 2>/dev/null || echo "(no H5 files)"
