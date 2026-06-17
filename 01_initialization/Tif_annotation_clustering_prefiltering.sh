#!/bin/bash
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
dataset="french" # can change the set for other sources

# Paths are intentionally configurable for the public release.
# DATA_ROOT should point at the external processed-data root.
# WSI_ROOT should contain one subdirectory per dataset with the original WSI files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
WSI_ROOT="${WSI_ROOT:-$DATA_ROOT/00_raw_wsi}"

PATH_PREFIX="${DATA_ROOT}/"
WSI_PATH_PREFIX="${WSI_ROOT}/${dataset}/"

# select one or more or all: uncomment
slides=(
    #"acer_edf"
    #"ambrosia_edf"
    #"betula_2_edf"
    #"betula_edf"
    #"brassica_napus_edf"
    #"brassicaceae_2_edf"
    #"brassicaceae_edf"
    #"buxus_edf"
    #"cannabis_edf"
    #"carpinus_edf"
    #"cheno_edf"
    #"corylus_2_edf"
    #"corylus_edf"
    #"cupressus_edf"
    #"ericaceae_edf"
    #"fabaceae_edf"
    #"festuca_edf"
    #"forsythia_edf"
    #"fraxinus_edf"
    #"hedera_edf"
    #"humulus_japonicus_2_edf"
    #"humulus_japonicus_edf"
    #"juglans_edf"
    #"juncaceae_edf"
    #"ligustrum_2_edf"
    #"ligustrum_edf"
    #"mimosa_edf"
    #"morus_alba_edf"
    #"parietaria_edf"
    #"phacelia_edf"
    #"plantago_edf"
    #"platanus_edf"
    #"quercus_edf"
    #"ranunculus_edf"
    #"rumex_edf"
    #"salix_edf"
    #"sambucus_edf"
    #"tilia_edf"
    #"triticum_aestivum_edf"
    #"typha_edf"
    #"ulmus_edf"
    #"urtica_edf"
    #"vitis_edf"
    #"cedrus_2_edf" 
    #"cedrus_edf"
    #"ginkgo_edf"
    #"picea_edf" "pinus_2_edf"
    #"pinus_edf"
)

echo "Slides to be processed: ${slides[@]}"

# Define the list of models to use
model_list=(
    "owlvit-base-patch32"
    # "owlvit-base-patch16"
    # "owlvit-large-patch14"
    # "owlv2-base-patch16"
    # "owlv2-large-patch14"
)

# Available GPUs
gpu=1 
#gpus=(0)  # Adjust based on available GPUs
#num_gpus=${#gpus[@]}

# Function to run annotation for each model and slide
run_annotation_clustering() {
    local model=$1
    local slide=$2
    local gpu=$3

    # Construct the full command string
    CMD="python Tif_annotation_clustering.py \
        --project_root_dir \"${PATH_PREFIX}\" \
        --wsi_root_dir \"${WSI_PATH_PREFIX}\" \
        --hdf5_path \"one_shot_annotations_${dataset}/${model}/${slide}.h5\" \
        --current_wsi \"${slide}.tif\" \
        --cuda_num ${gpu} \
        --query_image_path \"query_images/${slide}.png\" \
        --output_images_path \"images_${dataset}/${model}/${slide}/\" "

    # Print the full command for debugging
    echo "Executing on GPU ${gpu}: $CMD"

    # Execute the command in the background
    eval $CMD
}

# Iterate over models and slides sequentially
for model in "${model_list[@]}"; do
    echo "Processing with Model: $model"

    for slide in "${slides[@]}"; do
        echo "Processing slide: $slide"
        # just execute one-by-one, after each other
        run_annotation_clustering "${model}" "${slide}" "${gpu}"
    done
done

echo "Processing completed for all models and slides!"
