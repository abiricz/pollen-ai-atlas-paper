#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# =============================================================================
# CONCURRENT PRODUCTION CAPTIONING BATCH SCRIPT
# =============================================================================
# 
# Runs high-throughput concurrent production captioning for ALL grains.
# Uses the same prompts as validation but with massive parallelism.
#
# THROUGHPUT:
#   - Single server: ~2.7s/grain (like validation)
#   - 8 servers × 20 concurrency = 160 parallel → ~0.017s/grain effective
#   - 300,000 grains → ~83 minutes total (8 servers)
#
# ⚠️ GPU REQUIREMENTS:
#   - Each vLLM server needs ~48GB VRAM for Qwen2.5-VL-32B-AWQ
#   - 8x A100 80GB: Can run 8 independent servers
#   - 2x RTX 4090: Use tensor-parallel=2 for ONE server
#
# USAGE:
#   ./caption_production_concurrent.sh [OPTIONS]
#
# OPTIONS:
#   -p, --ports       Comma-separated list of vLLM server ports (default: 11446)
#   -c, --concurrency Concurrent requests per server (default: 20)
#   -d, --dataset     Dataset to process: french, hungarian, mediterranean, swedish
#   -n, --dry-run     Show what would be processed without running
#   -h, --help        Show this help message
#
# EXAMPLES:
#   # Single server (2x 24GB GPUs with tensor-parallel)
#   ./caption_production_concurrent.sh --dataset hungarian --concurrency 10
#
#   # 8 servers (8x 80GB GPUs) with maximum throughput
#   ./caption_production_concurrent.sh \
#       --ports 11446,11447,11448,11449,11450,11451,11452,11453 \
#       --concurrency 20 --dataset hungarian
#
#   # Dry run
#   ./caption_production_concurrent.sh --dataset hungarian --dry-run
#
# =============================================================================

set -e

# =============================================================================
# ⚠️ MACHINE CONFIGURATION - EDIT THESE PATHS FOR YOUR MACHINE ⚠️
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"

# =============================================================================
# DERIVED PATHS
# =============================================================================
FILTERED_ROOT="${DATA_ROOT}/03_captioning"
WSI_ROOT="${DATA_ROOT}/00_raw_wsi"
OUTPUT_ROOT="${DATA_ROOT}/03_captioning"
QUERY_IMAGES="${PROJECT_ROOT}/01_initialization/query_images"
ANCHOR_ROOT="${PROJECT_ROOT}/03_captioning/caption_anchors"
EXCLUSIONS_FILE="${PROJECT_ROOT}/03_captioning/slide_exclusions.yaml"

# =============================================================================
# DEFAULT SETTINGS
# =============================================================================
VLLM_PORTS="11446"
CONCURRENCY=16
DATASET="hungarian"
DRY_RUN=false
DEBUG_LIMIT=0  # 0 = no limit, >0 = limit grains per slide (for testing)
TEMPERATURE=0.0
MAX_TOKENS=250
MAX_RETRIES=10
PIXEL_UM=0.242797
PIXEL_CONFIG="${PROJECT_ROOT}/03_captioning/pixel_config.yaml"
OUTPUT_SUFFIX="gemma4-bf16_final"  # Produces production_<suffix>; set to match the active VLM.

# =============================================================================
# SLIDE CONFIGURATIONS - pixel_um per slide
# Now handled by pixel_config.yaml for per-slide control
# Fallback function kept for compatibility
# =============================================================================
get_pixel_um() {
    local slide=$1
    
    # Check if pixel_config.yaml exists and use Python to parse it
    if [ -f "$PIXEL_CONFIG" ]; then
        local result=$(python3 -c "
import yaml
with open('$PIXEL_CONFIG') as f:
    cfg = yaml.safe_load(f)
slides = cfg.get('slides', {})
defaults = cfg.get('defaults', {})
slide = '$slide'
if slide in slides:
    print(slides[slide])
elif slide.startswith('hun_'):
    print(defaults.get('hungarian', 0.242797))
elif slide.startswith('mediterranean_'):
    print(defaults.get('mediterranean', 0.17))
elif any(slide.startswith(p) for p in ['Alnus', 'Betula', 'Corylus', 'Quercus', 'Ulmus', 'Urtica', 'Salix', 'Ambrosia', 'Pinus', 'Picea', 'Dactyl', 'plantago', 'acer', 'aescul', 'calluna']):
    print(defaults.get('swedish', 0.243))
else:
    print(defaults.get('french', 0.242797))
" 2>/dev/null)
        if [ -n "$result" ]; then
            echo "$result"
            return
        fi
    fi
    
    # Fallback if YAML parsing fails
    case "$slide" in
        # Hungarian slides - same microscope as French
        hun_*) echo "0.242797" ;;
        
        # Mediterranean slides (Olympus VS120 @ 40x)
        mediterranean_*) echo "0.17" ;;
        
        # Swedish slides with known 40x effective resolution
        Alnus_*|Betula_cf_*|Betula_sp_01_*|Corylus_*) echo "0.1225" ;;
        
        # Other Swedish slides (default 20x effective)
        Quercus_*|Ulmus_*|Urtica_*|Salix_*|Ambrosia_*|Pinus_*|Picea_*|Dactyl*|plantago_*|acer_*|aescul*|calluna_*|betula_sp_10_*) echo "0.243" ;;
        
        # French slides (default) - 0.242797 µm/pixel
        *) echo "0.242797" ;;
    esac
}

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --ports        Comma-separated list of vLLM server ports (default: 11446)"
    echo "  -c, --concurrency  Concurrent requests per vLLM server (default: 20)"
    echo "  -d, --dataset      Dataset to process: french, hungarian, mediterranean, swedish"
    echo "  -o, --output-suffix Output folder suffix (e.g., 'qwen3' → production_qwen3)"
    echo "  -l, --debug-limit  Limit grains per slide for testing (default: 0 = no limit)"
    echo "  -n, --dry-run      Show what would be processed without running"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --dataset hungarian --concurrency 10"
    echo "  $0 --ports 11446,11447 --concurrency 20 --dataset hungarian"
    echo "  $0 --dataset french --output-suffix gemma4-bf16_final"
    echo "  $0 --dataset hungarian --debug-limit 100  # Test with 100 grains per slide"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--ports)
            VLLM_PORTS="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -o|--output-suffix)
            OUTPUT_SUFFIX="$2"
            shift 2
            ;;
        -l|--debug-limit)
            DEBUG_LIMIT="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# =============================================================================
# CHECK vLLM SERVERS
# =============================================================================
check_vllm_servers() {
    IFS=',' read -ra PORT_ARRAY <<< "$VLLM_PORTS"
    AVAILABLE_PORTS=""
    
    echo "[INFO] Checking vLLM servers on ports: ${VLLM_PORTS}"
    
    for port in "${PORT_ARRAY[@]}"; do
        url="http://localhost:${port}/v1"
        if curl -s "${url}/models" > /dev/null 2>&1; then
            model=$(curl -s "${url}/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
            echo "       Port ${port}: ${model}"
            if [ -z "$AVAILABLE_PORTS" ]; then
                AVAILABLE_PORTS="${port}"
            else
                AVAILABLE_PORTS="${AVAILABLE_PORTS},${port}"
            fi
        else
            echo "       Port ${port}: NOT RESPONDING"
        fi
    done
    
    if [ -z "$AVAILABLE_PORTS" ]; then
        echo ""
        echo "ERROR: No vLLM servers available!"
        echo "       Start vLLM first."
        exit 1
    fi
    
    NUM_SERVERS=$(echo "$AVAILABLE_PORTS" | tr ',' '\n' | wc -l)
    echo "[INFO] ${NUM_SERVERS} server(s) available"
}

# =============================================================================
# GET READY SLIDES (have filtering + anchor files)
# =============================================================================
get_ready_slides() {
    local dataset=$1
    local slides=()
    
    for h5_file in "${FILTERED_ROOT}/${dataset}/filtered"/*_filtered.h5; do
        if [ -f "$h5_file" ]; then
            slide=$(basename "$h5_file" _filtered.h5)
            
            anchor_file="${ANCHOR_ROOT}/${slide}_anchor.txt"
            if [ -f "$anchor_file" ]; then
                if grep -v '^\s*#' "$EXCLUSIONS_FILE" 2>/dev/null | grep -q "slide: ${slide}"; then
                    echo "       [SKIP] ${slide} (excluded)" >&2
                else
                    slides+=("$slide")
                fi
            else
                echo "       [SKIP] ${slide} (no anchor file)" >&2
            fi
        fi
    done
    
    printf '%s\n' "${slides[@]}"
}

# =============================================================================
# GET SPECIES/FAMILY FROM ANCHOR FILES
# =============================================================================
get_species_family() {
    local slide=$1
    
    local species_file="${ANCHOR_ROOT}/${slide}_species.txt"
    local family_file="${ANCHOR_ROOT}/${slide}_family.txt"
    
    if [ -f "$species_file" ]; then
        SPECIES=$(cat "$species_file" | head -1)
    else
        SPECIES=$(echo "$slide" | sed 's/_edf$//' | sed 's/_[0-9]*$//' | sed 's/_/ /g')
    fi
    
    if [ -f "$family_file" ]; then
        FAMILY=$(cat "$family_file" | head -1)
    else
        FAMILY="unknown"
    fi
}

# =============================================================================
# GET TAXON HINT FROM HINT FILE OR FALLBACK TO SPECIES
# =============================================================================
get_taxon_hint() {
    local slide=$1
    local species=$2
    local hint_file="${ANCHOR_ROOT}/${slide}_hint.txt"
    
    if [ -f "$hint_file" ]; then
        # Use the descriptive hint from file
        cat "$hint_file"
    else
        # Fallback: extract genus (first word) from species
        echo "$species" | cut -d' ' -f1
    fi
}

# =============================================================================
# PROCESS SINGLE SLIDE
# =============================================================================
process_slide() {
    local slide=$1
    local dataset=$2
    
    FILTERED_H5="${FILTERED_ROOT}/${dataset}/filtered/${slide}_filtered.h5"
    WSI_PATH="${WSI_ROOT}/${dataset}/${slide}.tif"
    QUERY_PATH="${QUERY_IMAGES}/${slide}.png"
    ANCHOR_PATH="${ANCHOR_ROOT}/${slide}_anchor.txt"
    HINT_PATH="${ANCHOR_ROOT}/${slide}_hint.txt"
    
    # Output directory: production or production_<suffix>
    if [ -n "$OUTPUT_SUFFIX" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/${dataset}/production_${OUTPUT_SUFFIX}"
    else
        OUTPUT_DIR="${OUTPUT_ROOT}/${dataset}/production"
    fi
    LOG_FILE="${OUTPUT_DIR}/${slide}_production.log"
    
    # Validate inputs
    if [ ! -f "$FILTERED_H5" ]; then
        echo "  Missing: $FILTERED_H5"
        return 1
    fi
    if [ ! -f "$WSI_PATH" ]; then
        echo "  Missing: $WSI_PATH"
        return 1
    fi
    if [ ! -f "$QUERY_PATH" ]; then
        echo "  Missing: $QUERY_PATH"
        return 1
    fi
    
    get_species_family "$slide"
    local TAXON_HINT=$(get_taxon_hint "$slide" "$SPECIES")
    local PIXEL=$(get_pixel_um "$slide")
    
    # Count grains
    GRAIN_COUNT=$(python3 -c "import h5py; f = h5py.File('$FILTERED_H5', 'r'); r = f.get('results', f); print(len(r['bbox'][:]) if 'bbox' in r else len(r['boxes'][:]))")
    
    echo "  Species: $SPECIES | Family: $FAMILY"
    echo "  Pixel µm: $PIXEL | Taxon hint: $TAXON_HINT"
    echo "  Grains: $GRAIN_COUNT | Ports: ${AVAILABLE_PORTS}"
    
    # Estimate time (use python instead of bc for portability)
    NUM_SERVERS=$(echo "$AVAILABLE_PORTS" | tr ',' '\n' | wc -l)
    TOTAL_CONCURRENT=$((NUM_SERVERS * CONCURRENCY))
    EST_MINUTES=$(python3 -c "print(f'{${GRAIN_COUNT} * 2.7 / ${TOTAL_CONCURRENT} / 60:.1f}')")
    echo "  Estimated time: ~${EST_MINUTES} minutes (${TOTAL_CONCURRENT} parallel)"
    
    # Build command
    CMD="python3 ${PROJECT_ROOT}/03_captioning/caption_production_concurrent.py \
        --h5_path \"$FILTERED_H5\" \
        --wsi_path \"$WSI_PATH\" \
        --query_image \"$QUERY_PATH\" \
        --vllm_ports \"$AVAILABLE_PORTS\" \
        --output_dir \"$OUTPUT_DIR\" \
        --species \"$SPECIES\" \
        --family \"$FAMILY\" \
        --anchor \"$ANCHOR_PATH\" \
        --taxon_hint \"$TAXON_HINT\" \
        --pixel_um $PIXEL \
        --concurrency $CONCURRENCY \
        --temperature $TEMPERATURE \
        --max_tokens $MAX_TOKENS \
        --max_retries $MAX_RETRIES \
        --resume"
    
    # Add debug limit if specified
    if [ "$DEBUG_LIMIT" -gt 0 ]; then
        CMD="$CMD --max_grains $DEBUG_LIMIT"
        echo "  [DEBUG] Limited to $DEBUG_LIMIT grains"
    fi
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] Would run:"
        echo "    $CMD"
    else
        mkdir -p "$OUTPUT_DIR"
        eval $CMD 2>&1 | tee "$LOG_FILE"
        echo "  Completed: ${slide}"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
echo "============================================================"
echo "CONCURRENT PRODUCTION CAPTIONING"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project root:  ${PROJECT_ROOT}"
echo "  Data root:     ${DATA_ROOT}"
echo "  Dataset:       ${DATASET}"
echo "  vLLM ports:    ${VLLM_PORTS}"
echo "  Concurrency:   ${CONCURRENCY} per server"
echo "  Debug limit:   ${DEBUG_LIMIT:-0}"
echo "  Dry run:       ${DRY_RUN}"
echo ""

# Check vLLM servers (skip in dry-run mode)
if [ "$DRY_RUN" = false ]; then
    check_vllm_servers
    echo ""
fi

# Get ready slides
echo ""
echo "[INFO] Scanning ${DATASET} dataset for ready slides..."
mapfile -t READY_SLIDES < <(get_ready_slides "$DATASET")
echo ""
echo "[INFO] ${#READY_SLIDES[@]} slides ready for production captioning"

if [ ${#READY_SLIDES[@]} -eq 0 ]; then
    echo "ERROR: No slides ready. Check that:"
    echo "       1. Filtering is complete"
    echo "       2. Anchor files exist"
    exit 1
fi

# Compute output directory (mirrors logic in process_slide)
if [ -n "$OUTPUT_SUFFIX" ]; then
    MAIN_OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/production_${OUTPUT_SUFFIX}"
else
    MAIN_OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/production"
fi

# Create output directory
mkdir -p "$MAIN_OUTPUT_DIR"

# Process slides
echo ""
echo "============================================================"
echo "PROCESSING ${#READY_SLIDES[@]} SLIDES"
echo "============================================================"

for slide in "${READY_SLIDES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing: ${slide}"
    echo "------------------------------------------------------------"
    
    process_slide "$slide" "$DATASET"
done

# Summary
echo ""
echo "============================================================"
echo "PRODUCTION CAPTIONING COMPLETE"
echo "============================================================"
echo "Outputs in: ${MAIN_OUTPUT_DIR}/"

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "Caption files:"
    ls -lh "${MAIN_OUTPUT_DIR}/"*_captions.jsonl 2>/dev/null | head -10 || echo "(no caption files)"
fi

echo ""
echo "Next steps:"
echo "  1. Check output JSONL files for completeness"
echo "  2. Run data integrity check"
echo "  3. Compute final statistics"
