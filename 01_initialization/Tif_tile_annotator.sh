#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
# Global settings

dataset="${DATASET:-hungarian}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
WSI_ROOT="${WSI_ROOT:-$DATA_ROOT/00_raw_wsi}"

DATA_PARENT_PATH="${WSI_ROOT}/${dataset}/"
OUTPUT_FOLDER="${DATA_ROOT}/01_initialization/one_shot_annotations_${dataset}/"
QUERY_IMAGE_FOLDER="${PROJECT_ROOT}/01_initialization/query_images/"

# tif slides in folder
echo "Searching for .tif files in ${DATA_PARENT_PATH}..."
slides=($(find "${DATA_PARENT_PATH}" -type f -name "*.tif" -exec basename {} \; | sort))

echo "Found ${#slides[@]} .tif files to process:"
printf '%s\n' "${slides[@]}"

# possible models:
#google/owlvit-base-patch32
#google/owlvit-base-patch16
#google/owlvit-large-patch14
#google/owlv2-base-patch16
#google/owlv2-large-patch14

#google/owlv2-base-patch16-ensemble
#google/owlvit-base-patch16
#google/owlv2-base-patch16
#google/owlv2-large-patch14-ensemble
#google/owlvit-base-patch32
#google/owlvit-large-patch14
#google/owlv2-base-patch16-finetuned
#google/owlv2-large-patch14
#google/owlv2-large-patch14-finetuned


# Model list (you can adjust or comment out models as needed)
model_list=(
    "owlvit-base-patch32"
    "owlvit-base-patch16"
    "owlvit-large-patch14"
    "owlv2-base-patch16"
    "owlv2-large-patch14"
)

# Available GPUs
gpus=(0 1)  # Modify as needed
num_gpus=${#gpus[@]}

# Max parallel jobs per GPU
max_jobs_per_gpu=1
declare -A gpu_jobs  # Track jobs per GPU
declare -A gpu_pids  # Track running processes per GPU

# Initialize GPU job counters
for gpu in "${gpus[@]}"; do
    gpu_jobs[$gpu]=0
done

# Function to wait for a specific GPU process to finish and free up the slot
cleanup_finished_jobs() {
    for gpu in "${gpus[@]}"; do
        if [[ -n "${gpu_pids[$gpu]}" ]]; then
            if ! kill -0 "${gpu_pids[$gpu]}" 2>/dev/null; then
                # Process has finished, decrement GPU job count
                ((gpu_jobs[$gpu]--))
                unset gpu_pids[$gpu]  # Remove the process from tracking
            fi
        fi
    done
}

# Iterate over models
for model in "${model_list[@]}"; do
    echo "Processing with Model: $model"

    # Iterate over slide names
    for slide in "${slides[@]}"; do
        available_gpu=-1
        
        # Find an available GPU
        while [ "$available_gpu" -eq -1 ]; do
            cleanup_finished_jobs  # Check and update job counters
            for gpu in "${gpus[@]}"; do
                if (( gpu_jobs[$gpu] < max_jobs_per_gpu )); then
                    available_gpu=$gpu
                    break
                fi
            done
            [[ "$available_gpu" -eq -1 ]] && sleep 2  # If no GPU is free, wait a bit and retry
        done

        # Increment job counter for assigned GPU
        ((gpu_jobs[$available_gpu]++))

        # Print processing information
        echo "Processing slide: $slide on GPU: $available_gpu with Model: $model"

        # Run the command in the background and store its PID
        python Tif_tile_annotator.py \
            --data_parent_path "${DATA_PARENT_PATH}" \
            --current_slide_name "${slide}" \
            --query_image_folder "${QUERY_IMAGE_FOLDER}" \
            --output_folder "${OUTPUT_FOLDER}" \
            --device cuda:${available_gpu} \
            --confidence 0.01 \
            --overlap 224 \
            --checkpoint "${model}" &

        gpu_pids[$available_gpu]=$!  # Store the PID of the background process

        # Ensure no more than `max_jobs_per_gpu * num_gpus` jobs run simultaneously
        while (( $(jobs -r | wc -l) >= (max_jobs_per_gpu * num_gpus) )); do
            cleanup_finished_jobs
            sleep 2
        done
    done
done

# Final cleanup wait
wait
echo "Processing completed for all models and slides!"