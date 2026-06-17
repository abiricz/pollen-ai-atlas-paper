#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# === CONFIGURATION ===
dataset="french"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_home="$(cd "$SCRIPT_DIR/.." && pwd)"
data_home="${DATA_ROOT:-$project_home/data}"
wsi_home="${WSI_ROOT:-$data_home/00_raw_wsi}/${dataset}"
sam2_root="${SAM2_ROOT:-$project_home/../sam2}"

query_img_dir="${project_home}/01_initialization/query_images"
output_base="${data_home}/02_mining/${dataset}"
script_path="miner.py"  # formerly detect.py — renamed when pipeline was completed beyond mining

# ViT + SAM2 configs
SAM2_CKPT="${SAM2_CKPT:-${sam2_root}/checkpoints/sam2.1_hiera_large.pt}"
SAM2_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
VIT_CKPT="${project_home}/01_initialization/weights_vit_small_lvd_20250620_0312.pth"
VIT_NAME="vit_small_patch14_dinov2.lvd142m"
VIT_PATCH_SIZE=14
PATCH_SIZE=518
PERCENTILE=99.5
MIN_MASK_RATIO=0.925
MAX_MASK_RATIO=1.60
N_SHIFTS_AUGMENTS=10
dS_multiplier=100
tile_size_multiplier=2.5
max_nms_objs=25000

# === LOGGING SETUP ===
log_dir="./logs"
mkdir -p "$log_dir"

# === SLIDE LIST ===
# Select one or more slides. Process one dataset at a time when tuning mining parameters.
slides=(
        #"mediterranean_pollen_causarina_reference"
        #"mediterranean_pollen_pinus_reference"
        #"picea_edf"
##      "pinus_2_edf"
        "pinus_edf"
        #"typha_edf"
        #"sambucus_edf"
        #"cedrus_edf"
        #"cedrus_2_edf"
        #"Pinus_sp_10_Pinaceae_14_layers_40x_Blue_colou_ZS017_5_mm_circle"
        #"acer_edf"
        #"rumex_edf"
        #"urtica_edf"
        #corylus_2_edf
        #hun_betula_edf
        #hun_corylus_edf
        #hun_alnus_edf
        #Ambrosia-Iva_reference_10l_1m_Ambrosia_edf
        #Ambrosia-Iva_reference_10l_1m_Iva_edf
        #"acer_platanoides_12_layers_40x_zs015_5_mm_circle"
        #"aesculus_hippocustanum_10_layers_40x_zs015_5_mm_circle"
    #"Alnus_cf_incana_40x_11_steps_merged_reference"
        #"Alnus_glutinosa_01a_26_layers_40x_ZS028_Al"
        #"Ambrosia_artemisifolia_Asteraceae_14_layers_40x_ZS017_5_mm_circle"
    #"Betula_cf_pendula_11_layers_40x_merged_reference"
    #"Betula_cf_pendula_40x_11_steps_merged_reference"
        #"Betula_sp_01_40x_ZS013_Be"
        #"betula_sp_10_betulaceae_14_layers_40x_blue_colou_zs017_5_mm_circle"  
        #"calluna_vulgaris_ericaceae_zs017_5_mm_circle"
    #"Corylus_avellana_40x_11_steps_merged_reference"
        #"Corylus_sp_01_40x_ZS013_Co"
        #"Dactylus_glomerata_Poaceae_14_layers_40x_ZS017_5_mm_circle"  
        #"Picea_abies_01_20x_ZS017_Pc"                                 
        #"Pinus_sp_10_Pinaceae_14_layers_40x_Blue_colou_ZS017_5_mm_circle"
        #"plantago_major_plantaginaceae_14_layers_40x_zs017_5_mm_circle"
        #"Quercus_robur_Fagaceae_14_layers_40x_check_quality_ZS017_5_mm_circle"
        #"Salix_sp_10_Salicaceae_14_layers_40x_ZS017_5_mm_circle"
        #"Ulmus_glabra_12_layers_01_40x_ZS014_Ulmus_01"
        #"Urtica_dioica_40x_12_layers_03_crowded_ZS015_5_mm_circle"
)

# === GPU SETTINGS ===
gpus=(0)   # Set your available GPUs
max_jobs_per_gpu=1       # Max concurrent jobs per GPU
declare -A gpu_jobs
declare -A gpu_pids

# Initialize counters
for gpu in "${gpus[@]}"; do
    gpu_jobs[$gpu]=0
done

# === HELPER FUNCTION ===
cleanup_finished_jobs() {
    for gpu in "${gpus[@]}"; do
        if [[ -n "${gpu_pids[$gpu]}" ]]; then
            if ! kill -0 "${gpu_pids[$gpu]}" 2>/dev/null; then
                ((gpu_jobs[$gpu]--))
                unset gpu_pids[$gpu]
            fi
        fi
    done
}


# === MAIN LOOP ===
for slide in "${slides[@]}"; do
    while true; do
        cleanup_finished_jobs
        for gpu in "${gpus[@]}"; do
            if (( gpu_jobs[$gpu] < max_jobs_per_gpu )); then
                ((gpu_jobs[$gpu]++))

                WSI_PATH="${wsi_home}/${slide}.tif"
                QUERY_IMAGE="${query_img_dir}/${slide}.png"
                OUTPUT_DIR="${output_base}/"
                LOG_FILE="${log_dir}/${slide}_gpu${gpu}.log"
                mkdir -p "$OUTPUT_DIR"

                echo "Launching $slide on GPU $gpu"

                # build the command in an array
                cmd=(
                  python -u "$script_path"
                  --wsi_path "$WSI_PATH"
                  --query_image "$QUERY_IMAGE"
                  --sam2_ckpt "$SAM2_CKPT"
                  --sam2_cfg "$SAM2_CFG"
                  --output "$OUTPUT_DIR"
                  --vit_ckpt "$VIT_CKPT"
                  --vit_name "$VIT_NAME"
                  --vit_patch_size "$VIT_PATCH_SIZE"
                  --patch_size "$PATCH_SIZE"
                  --percentile "$PERCENTILE"
                  --min_mask_ratio "$MIN_MASK_RATIO"
                  --max_mask_ratio "$MAX_MASK_RATIO"
                  --device "cuda:$gpu"
                  --n_shifts_augments "$N_SHIFTS_AUGMENTS"
                  --dS_multiplier "$dS_multiplier"
                  --tile_size_multiplier "$tile_size_multiplier"
                  --max_nms_objs "$max_nms_objs"
                  --sam_multi_point_query
                  --sam_multimask_output
                )

                # write the command line as the first line of the log
                printf '%q ' "${cmd[@]}" > "$LOG_FILE"
                printf '\n' >> "$LOG_FILE"

                # execute and append all output
                "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE" &

                gpu_pids[$gpu]=$!
                break 2  # Break both loops
            fi
        done
        sleep 2
    done
done

# Wait for all background jobs
wait
echo "All slides processed!"
