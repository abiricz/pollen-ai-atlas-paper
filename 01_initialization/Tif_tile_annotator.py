# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import requests
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import json
import cv2
import re
from torch.utils.data import Dataset, DataLoader
import openslide
import tiffslide
import h5py
#from argparse import Namespace

## FUNCTIONS 

class WSITileDataset(Dataset):
    """A PyTorch Dataset to process and return preprocessed tiles from Whole Slide Images (WSIs)."""
    def __init__(self, wsi_path, query_image_path_list, processor, tile_size=896, overlap=0, tiffslide_use=False):
        self.wsi_path = wsi_path
        self.query_image_list = [ Image.open(i).convert('RGB') for i in query_image_path_list ] # Preload query image
        self.processor = processor
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiffslide_use = tiffslide_use

        # Load WSI dimensions
        if not tiffslide_use:
            with openslide.OpenSlide(wsi_path) as wsi:
                self.w, self.h = wsi.dimensions
        else:
            with tiffslide.TiffSlide(wsi_path) as wsi:
                self.w, self.h = wsi.dimensions

        print('Init:', self.w, self.h, self.tile_size, self.overlap )
        # Generate tile coordinates
        self.tile_coords = [
            (x, y)
            for y in range(0, self.h, self.tile_size - self.overlap)
            for x in range(0, self.w, self.tile_size - self.overlap)
        ]

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        x, y = self.tile_coords[idx]

        # Read the corresponding tile
        if not self.tiffslide_use:
            with openslide.OpenSlide(self.wsi_path) as wsi:
                tile = wsi.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
        else:
            with tiffslide.TiffSlide(self.wsi_path) as wsi:
                tile = wsi.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')

        # Preprocess the tile
        inputs = self.processor(images=tile, query_images=self.query_image_list, return_tensors="pt")

        return {
            "tile": tile,
            "inputs": {k: v.squeeze(0) for k, v in inputs.items()},  # Remove batch dimension for individual tiles
            "coords": (x, y)
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle preprocessing outputs from the dataset.
    """
    batched_inputs = {key: torch.stack([item["inputs"][key] for item in batch]) for key in batch[0]["inputs"].keys()}
    coords = [item["coords"] for item in batch]
    tiles = [item["tile"] for item in batch]  # Extract PIL images

    return {"inputs": batched_inputs, "coords": coords, "tiles": tiles}


def process_results(results, tile_topleft_coords, confidence, query_image):
    """Filter detection results and prepare annotations."""
    annotations = []
    boxes = results["boxes"].tolist()
    scores = results["scores"].tolist()

    # Filtering based on size
    low_thres = min(query_image.size) * 3 / 4
    up_thres = max(query_image.size) * 4 / 3

    box_widths = np.array([i[3] - i[1] for i in boxes])
    box_heights = np.array([i[2] - i[0] for i in boxes])

    filt_widths = np.logical_and(box_widths > low_thres, box_widths < up_thres)
    filt_heights = np.logical_and(box_heights > low_thres, box_heights < up_thres)

    filt = np.logical_and(filt_widths, filt_heights)

    #print('Annotations found:', filt.shape[0], '\n', 'Annotations after filtering:', filt.sum() )
    
    # Filter boxes and scores
    if filt.sum():
        boxes_filt = np.array(boxes)[filt]
        scores_filt = np.array(scores)[filt]
        #print('scores:', scores_filt)

        for box, score in zip(boxes_filt, scores_filt):
            #print( 'DEBUG:', score, confidence, score > confidence )
            if score > confidence:
                xmin, ymin, xmax, ymax = box
                annotations.append({
                    "Tile_info": {"X": tile_topleft_coords[0], "Y": tile_topleft_coords[1]},
                    "Bounding_boxes": [{
                        'xmin': int(np.round(xmin)),
                        'ymin': int(np.round(ymin)),
                        'xmax': int(np.round(xmax)),
                        'ymax': int(np.round(ymax)),
                        'score': float(np.round(score, 3))
                    }]
                })
    #print('Annotations returned:', len(annotations))

    return annotations

def inference(model, dataloader, device, confidence, debug=False, debug_folder=None):
    """Perform inference on preprocessed tiles, collect annotations, and optionally save debug images."""
    all_annotations = []

    if debug and debug_folder:
        os.makedirs(debug_folder, exist_ok=True)

    for batch in tqdm(dataloader, desc="Processing Tiles"):
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}  # Move inputs to device
        coords = batch["coords"]
        tiles = batch["tiles"]

        with torch.no_grad():
            # Perform inference
            outputs = model.image_guided_detection(
                pixel_values=inputs["pixel_values"],
                query_pixel_values=inputs["query_pixel_values"][:1]
            )

            # Compute target_sizes for post-processing
            target_sizes = torch.tensor(
                [[tiles[0].size[1], tiles[0].size[0]] for _ in range(len(tiles))]
            ).to(device)

            # Post-process the results
            results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)

        for idx, (result, coord) in enumerate(zip(results, coords)):
            # Process results and collect annotations
            annotations = process_results(result, coord, confidence, dataloader.dataset.query_image_list[0])
            all_annotations.extend(annotations)

            # Debugging: Save annotated images
            if debug and debug_folder:
                save_debug_image(tiles[idx], annotations, coord, debug_folder)

    return all_annotations

def save_debug_image(tile, annotations, coord, debug_folder):
    """Save a single debug image with bounding boxes and annotations."""
    # Create a drawer for the tile
    draw_filt = ImageDraw.Draw(tile)
    font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', 16)

    # Iterate over annotations
    for annotation in annotations:
        for bbox in annotation["Bounding_boxes"]:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            score = bbox.get('score', 0)  # Fetch the score from bbox

            # Draw the bounding box
            draw_filt.rectangle((xmin, ymin, xmax, ymax), outline="red", width=5)
            text_position = (xmin, ymin - 20)

            # Add the confidence score text
            draw_filt.text(text_position, f"C: {score:.2f}", fill="black", font=font)

    # Save the debug image with coordinates in the filename
    x, y = coord
    debug_path = os.path.join(debug_folder, f"debug_{x}_{y}.jpg")
    tile.save(debug_path)

def save_annotations_to_hdf5(annotations, output_hdf5_path):
    """Save the annotations to an HDF5 file, including scores."""
    with h5py.File(output_hdf5_path, "w") as hdf5_file:
        for idx, annotation in enumerate(annotations):
            group = hdf5_file.create_group(f"tile_{idx}")
            
            # Save tile information as attributes
            for key, value in annotation["Tile_info"].items():
                group.attrs[key] = value

            # Save bounding boxes with scores
            boxes = group.create_dataset("Bounding_boxes", data=np.array([
                [
                    bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'], bbox['score']
                ] for bbox in annotation.get("Bounding_boxes", [])
            ]))

## MAIN

## Add arguments

parser = argparse.ArgumentParser()
parser.add_argument("--data_parent_path", type=str, required=True) # directory containing input WSI files
parser.add_argument("--query_image_folder", type=str, required=True) 
parser.add_argument("--current_slide_name", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--device", type=str, required=False, default='cpu')
parser.add_argument("--confidence", type=float, required=False, default=0.01) # modify if needed ! 
parser.add_argument("--tile_size", type=int, required=False, default=896) 
parser.add_argument("--overlap", type=int, required=False, default=0) 
parser.add_argument("--debug", type=bool, required=False, default=False) 
parser.add_argument("--debug_folder", type=str, required=False, default=None)
parser.add_argument("--checkpoint", type=str, required=False, default="owlvit-base-patch32",
                    help="Model checkpoint to use for OWL-ViT (e.g., 'owlvit-base-patch32').")


# Parse arguments
args = parser.parse_args()

data_parent_path = args.data_parent_path
current_slide_name = args.current_slide_name
query_image_folder = args.query_image_folder
device = args.device
confidence = args.confidence
tile_size = args.tile_size
overlap = args.overlap
debug = args.debug
debug_folder = args.debug_folder
if debug_folder:
    os.makedirs(debug_folder, exist_ok=True)

checkpoint = args.checkpoint
output_folder = args.output_folder + checkpoint + '/'
checkpoint = 'google/'+checkpoint # fix full reference to model after folder structure created

# Load model and processor dynamically based on the checkpoint
print(f"Using model checkpoint: {checkpoint}", 'output folder:', output_folder)
model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

# Load model
#checkpoint = "google/owlvit-base-patch32"
#model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(device)
#processor = AutoProcessor.from_pretrained(checkpoint)

# Locate slides
slide_files = np.array( sorted([ i for i in os.listdir(data_parent_path) if 'edf.tif' in i ]) )
print( 'Found slides:', slide_files.shape, slide_files, '\n\n\n', 'Processing:', current_slide_name )

query_image_path_list = [ query_image_folder+k for k in os.listdir(query_image_folder) if current_slide_name.replace('.tif', '') in k ]
print( 'Found query images:', query_image_path_list )

# Dataloader 
dataset = WSITileDataset( data_parent_path+current_slide_name, tile_size=tile_size, overlap=overlap, tiffslide_use=False,
                          processor=processor, query_image_path_list=query_image_path_list )
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=16, collate_fn=custom_collate_fn)

# Inference with OWL-ViT
annotations = inference(model, dataloader, device, confidence, debug=debug, debug_folder=debug_folder)

# Saving annotations
os.makedirs(output_folder, exist_ok=True)
save_annotations_to_hdf5( annotations, output_folder+f"{current_slide_name.replace('.tif', '.h5')}" )