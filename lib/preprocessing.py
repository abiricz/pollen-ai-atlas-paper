# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

import os
import h5py
import openslide
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict 

import torch
from torchvision.ops import nms

from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import concurrent.futures

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
import cv2

import time
import gc

def cleanup_multiprocessing():
    active_children = mp.active_children()
    if active_children:
        print(f"Cleaning up {len(active_children)} active child processes...")
        for child in active_children:
            child.terminate()
        time.sleep(2)  # Give OS time to fully release resources
    else:
        print("No active child processes detected.")
        time.sleep(2)
    gc.collect()  # Collect any garbage Python objects lingering

def seed_torch(seed=7, CUDANUM=0):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device=torch.device(f'cuda:{CUDANUM}' if torch.cuda.is_available() else "cpu") 
    if device.type == f'cuda:{CUDANUM}':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return device


def load_and_map_annotations_to_global(hdf5_path):
    """
    Load annotations from an HDF5 file and map them to global WSI coordinates.
    
    Args:
        hdf5_path (str): Path to the HDF5 file containing annotations.
    
    Returns:
        list: A list of annotations with global WSI coordinates.
    """
    global_annotations = []

    with h5py.File(hdf5_path, "r") as hdf5_file:
        for tile_key in hdf5_file.keys():
            tile_group = hdf5_file[tile_key]

            # Extract tile information
            tile_info = {key: tile_group.attrs[key] for key in tile_group.attrs.keys()}
            tile_x = tile_info["X"]
            tile_y = tile_info["Y"]

            # Extract bounding boxes
            bboxes = tile_group["Bounding_boxes"][:]
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = bbox[:4]
                global_bbox = {
                    "xmin": int(xmin + tile_x),
                    "ymin": int(ymin + tile_y),
                    "xmax": int(xmax + tile_x),
                    "ymax": int(ymax + tile_y),
                    "score": float(bbox[4]) if len(bbox) > 4 else None  # Include score if available
                }
                global_annotations.append(global_bbox)

    return global_annotations


def save_global_annotations_to_hdf5(final_centroids_and_patches, output_hdf5_path, tile_size=(896, 896)):
    """
    Save final_centroids_and_patches into HDF5, mapped into the unified format 
    expected by load_and_map_annotations_to_global.

    Args:
        final_centroids_and_patches (list): List of dicts, each containing 'scan_name', 'bounding_box', 'score', etc.
        output_hdf5_path (str): Path to output HDF5 file.
        tile_size (tuple): Tile size (x, y) to compute correct tile coordinates.
    """
    with h5py.File(output_hdf5_path, "w") as hdf5_file:
        grouped_by_tile = defaultdict(list)

        # Group annotations by tile (the WSI grid section they belong to)
        for entry in final_centroids_and_patches:
            x_min, y_min, x_max, y_max = entry['bounding_box']

            tile_x = (x_min // tile_size[0]) * tile_size[0]
            tile_y = (y_min // tile_size[1]) * tile_size[1]
            tile_key = f"tile_{tile_x}_{tile_y}"

            grouped_by_tile[tile_key].append({
                "local_bbox": (x_min - tile_x, y_min - tile_y, x_max - tile_x, y_max - tile_y),
                "global_bbox": entry['bounding_box'],
                "score": entry.get('score', 1.0),  # Default 1.0 if missing
                "cos_sim_dist": entry.get('cos_sim_dist', np.nan), # Add cos_sim_dist!
                "scan_name": entry['scan_name']
            })

        # Write each tile into the HDF5
        for tile_key, annotations in grouped_by_tile.items():
            group = hdf5_file.create_group(tile_key)

            # Store tile metadata (X, Y coordinates in the global WSI)
            tile_x, tile_y = map(int, tile_key.replace("tile_", "").split("_"))
            group.attrs["X"] = tile_x
            group.attrs["Y"] = tile_y

            # Prepare bbox array for HDF5
            bbox_array = []
            for ann in annotations:
                x1, y1, x2, y2 = ann['local_bbox']
                score = ann['score']
                cos_sim_dist = ann['cos_sim_dist']
                bbox_array.append([x1, y1, x2, y2, score, cos_sim_dist])  # Adding cos_sim_dist as well

            group.create_dataset("Bounding_boxes", data=np.array(bbox_array, dtype=np.float32))

    print(f"Saved {len(final_centroids_and_patches)} annotations into HDF5: {output_hdf5_path}")


def pad_image_to_size(img, target_width=224, target_height=224, background_color=(255, 255, 255)):
    """
    Pad an image up to the target size with a specified background color.
    """
    padded_img = Image.new("RGB", (target_width, target_height), background_color)
    left = (target_width - img.width) // 2
    top = (target_height - img.height) // 2
    padded_img.paste(img, (left, top))
    return padded_img

def remove_padding(image):
    """
    Removes padding with value 255 from the image by cropping to the content area.

    Args:
        image (numpy.ndarray): Input image with padding (H, W, C).
    
    Returns:
        numpy.ndarray: Cropped image without padding.
    """
    # Create a mask for pixels not equal to 255
    mask = (image != 255).any(axis=-1)  # Combine along color channels for RGB
    
    # Find the bounding box of the non-padding area
    rows = np.where(mask.any(axis=1))[0]  # Non-padding rows
    cols = np.where(mask.any(axis=0))[0]  # Non-padding columns
    
    if rows.size == 0 or cols.size == 0:
        # If the image is entirely padded, return it as is
        return image

    # Crop the image
    cropped_image = image[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return cropped_image

def extract_patch_from_annotation(wsi_path, annotation):
    """
    Helper function to extract a single patch from a WSI based on an annotation.
    
    Args:
        wsi_path (str): Path to the WSI file.
        annotation (dict): Annotation containing bounding box and metadata.
    
    Returns:
        dict: Metadata and extracted patch. For invalid annotations, a 100x100 white patch with default metadata is returned.
    """
    scan_name = annotation.get("Scan_name", "unknown.tif")
    annotation_id = annotation.get("annotation_id", None)
    category_id = annotation.get("category_id", None)
    
    # Default values for invalid annotations
    default_score = 0.
    default_bbox = (0, 0, 100, 100)  # Invalid bounding box

    try:
        xmin = int(annotation["xmin"])
        ymin = int(annotation["ymin"])
        xmax = int(annotation["xmax"])
        ymax = int(annotation["ymax"])
        score = annotation.get("score", default_score)

        # Open the WSI and extract the bounding box
        with openslide.OpenSlide(wsi_path) as wsi:
            # Ensure coordinates stay within bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(wsi.dimensions[0], xmax)
            ymax = min(wsi.dimensions[1], ymax)

            # Calculate width and height
            patch_width = xmax - xmin
            patch_height = ymax - ymin

            # Sanity check for positive width and height
            if patch_width <= 0 or patch_height <= 0:
                raise ValueError(f"Invalid bounding box dimensions: ({xmin}, {ymin}, {xmax}, {ymax})")

            # Extract region
            patch = wsi.read_region((xmin, ymin), 0, (patch_width, patch_height)).convert("RGB")
            bounding_box = (xmin, ymin, xmax, ymax)
    except (ValueError, openslide.OpenSlideError) as e:
        # Handle invalid bounding boxes or OpenSlide errors
        #print(f"Warning: {e}. Returning default values for annotation {annotation_id}.")
        patch = Image.new("RGB", (100, 100), (255, 255, 255))  # Create a 100x100 white patch
        bounding_box = default_bbox
        score = default_score

    return {
        'scan_name': scan_name,
        'annotation_id': annotation_id,
        'category_id': category_id,
        'bounding_box': bounding_box,
        'score': score,
        'patch': patch
    }

def collect_bounding_boxes_from_wsi(wsi_path, annotations):
    """
    Extract bounding box regions from a WSI sequentially based on annotations.

    Args:
        wsi_path (str): Path to the WSI file.
        annotations (list): List of global annotations (e.g., from HDF5).

    Returns:
        list: A list containing extracted patches and their metadata.
    """
    print("Extracting bounding box regions sequentially...")
    bounding_boxes_and_patches = []

    for annotation in tqdm(annotations, total=len(annotations)):
        bounding_boxes_and_patches.append(extract_patch_from_annotation(wsi_path, annotation))

    return bounding_boxes_and_patches


def collect_bounding_boxes_from_wsi_parallel(wsi_path, annotations, num_workers=4):
    """
    Extract bounding box regions from a WSI in parallel based on annotations.
    
    Args:
        wsi_path (str): Path to the WSI file.
        annotations (list): List of global annotations (e.g., from HDF5).
        num_workers (int): Number of parallel processes to use.
    
    Returns:
        list: A list containing extracted patches and their metadata.
    """
    print("Extracting bounding box regions in parallel...")
    bounding_boxes_and_patches = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks for each annotation
        futures = [
            executor.submit(extract_patch_from_annotation, wsi_path, annotation)
            for annotation in annotations
        ]

        # Collect results as they are completed
        for future in tqdm(futures, total=len(futures)):
            bounding_boxes_and_patches.append(future.result())

    return bounding_boxes_and_patches


def process_patch(item, max_width, max_height):
    """
    Helper function to process a single patch: pad it and prepare it for further processing.
    
    Args:
        item (dict): Contains patch and metadata.
        max_width (int): Target width for padding.
        max_height (int): Target height for padding.
    
    Returns:
        tuple: Processed image (as numpy array) and its label.
    """
    img = item['patch']
    label = item.get('category_id', None)
    score = item.get('score', None)

    # Pad the image if it is smaller than the target size
    if img.width < max_width or img.height < max_height:
        img = pad_image_to_size(img, target_height=max_height, target_width=max_width)

    # If image is larger, resize it
    else:
        img = img.resize((max_width, max_height), Image.LANCZOS)
        
    return np.array(img), label, score

def give_back_padded_dataset(centroids_and_patches, max_width=224, max_height=224):
    """
    Process the extracted patches sequentially, pad them if needed.

    Args:
        centroids_and_patches (list): List of patches and their metadata.
        max_width (int): Target width for padding.
        max_height (int): Target height for padding.

    Returns:
        tuple: Arrays of padded patches and labels.
    """
    print('Padding and processing patches sequentially...')

    patches_all = []
    labels_all = []
    scores_all = []

    for item in tqdm(centroids_and_patches, total=len(centroids_and_patches)):
        patch, label, score = process_patch(item, max_width, max_height)
        patches_all.append(patch)
        labels_all.append(label)
        scores_all.append(score)

    return np.array(patches_all), np.array(labels_all), np.array(scores_all)

def give_back_padded_dataset_parallel(centroids_and_patches, max_width=224, max_height=224, num_workers=4):
    """
    Process the extracted patches in parallel, pad them if needed.
    
    Args:
        centroids_and_patches (list): List of patches and their metadata.
        max_width (int): Target width for padding.
        max_height (int): Target height for padding.
        num_workers (int): Number of parallel workers.
    
    Returns:
        tuple: Arrays of padded patches and labels.
    """
    print('Padding and processing patches in parallel...')
    patches_all = []
    labels_all = []
    scores_all = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = [
            executor.submit(process_patch, item, max_width, max_height)
            for item in centroids_and_patches
        ]

        # Collect results
        for future in tqdm(futures, total=len(futures)):
            patch, label, score = future.result()
            patches_all.append(patch)
            labels_all.append(label)
            scores_all.append(score)

    return np.array(patches_all), np.array(labels_all), np.array(scores_all)

def apply_nms(annotations, iou_threshold=0.5, score_threshold=0.0):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping annotations with an optional score threshold.
    
    Args:
        annotations (list): List of annotations in the format:
                            [{'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2, 'score': score}, ...]
        iou_threshold (float): IoU threshold for NMS.
        score_threshold (float): Minimum score required to keep an annotation.
    
    Returns:
        list: Filtered annotations after applying NMS and score thresholding.
    """
    # Filter annotations based on the score threshold
    annotations = [a for a in annotations if a['score'] >= score_threshold]
    if not annotations:
        return []  # Return empty list if no annotations meet the score threshold

    # Convert annotations to tensors
    boxes = torch.tensor([[a['xmin'], a['ymin'], a['xmax'], a['ymax']] for a in annotations], dtype=torch.float32)
    scores = torch.tensor([a['score'] for a in annotations], dtype=torch.float32)

    # Apply NMS
    keep_indices = nms(boxes, scores, iou_threshold)

    # Filter annotations based on NMS results
    filtered_annotations = [annotations[i] for i in keep_indices]

    return filtered_annotations

def apply_nms_vectorized(annotations, iou_threshold=0.5, score_threshold=0.0):
    if len(annotations) == 0:
        return []

    # Convert once to numpy arrays (faster than list-comp every time)
    boxes = np.array([[a['xmin'], a['ymin'], a['xmax'], a['ymax']] for a in annotations], dtype=np.float32)
    scores = np.array([a['score'] for a in annotations], dtype=np.float32)

    # Filter by score threshold directly
    valid_mask = scores >= score_threshold
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]

    if len(boxes) == 0:
        return []

    # Move to tensor for NMS
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)

    # Apply NMS
    keep_indices = nms(boxes, scores, iou_threshold)

    # Map back to annotations (only those kept by NMS)
    filtered_annotations = [annotations[i] for i in np.where(valid_mask)[0][keep_indices]]

    return filtered_annotations

def apply_nms_cuda(annotations, iou_threshold=0.5, score_threshold=0.0, device='cuda:0'):
    """
    Optimized, vectorized NMS using GPU (CUDA) for large-scale annotation lists.

    Args:
        annotations (list): List of annotation dicts (each with bbox & score).
        iou_threshold (float): IoU threshold for NMS.
        score_threshold (float): Minimum score to keep.
        device (str): Device to run NMS on, e.g., 'cuda' or 'cpu'.

    Returns:
        list: Filtered annotations after NMS.
    """
    if len(annotations) == 0:
        return []

    # Convert to numpy arrays at once (faster than looping)
    boxes = np.array([[a['xmin'], a['ymin'], a['xmax'], a['ymax']] for a in annotations], dtype=np.float32)
    scores = np.array([a['score'] for a in annotations], dtype=np.float32)

    # Apply initial score threshold filtering (before moving to GPU)
    keep_mask = scores >= score_threshold
    if not np.any(keep_mask):
        return []

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    kept_annotations = np.array(annotations)[keep_mask]

    # Move to GPU (CUDA)
    boxes = torch.from_numpy(boxes).to(device)
    scores = torch.from_numpy(scores).to(device)

    # Apply NMS directly on GPU
    keep_indices = nms(boxes, scores, iou_threshold)

    # Move indices back to CPU for final selection
    keep_indices = keep_indices.cpu().numpy()

    # Final selection
    filtered_annotations = [kept_annotations[i] for i in keep_indices]

    return filtered_annotations


def calculate_stats_batched(images, batch_size=128, device='cuda'):
    """
    Calculate mean and standard deviation for a NumPy dataset in batches using PyTorch,
    avoiding extra CPU memory allocation.
    
    Args:
        images (np.ndarray): Dataset array of shape (N, H, W, C), where N is the number of images.
        batch_size (int): Number of images to process per batch.
        device (str): Device to perform the calculations on ('cuda' or 'cpu').
    
    Returns:
        tuple: (mean, std) for each channel (R, G, B).
    """
    # Ensure the input is a NumPy array
    if not isinstance(images, np.ndarray):
        raise TypeError("Input images must be a NumPy array")
    
    # Validate input dimensions
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Input array must have shape (N, H, W, C) with C=3 (RGB) channels")

    # Initialize variables for aggregation
    num_images, height, width, channels = images.shape
    total_sum = torch.zeros(channels, dtype=torch.float64, device=device)
    total_sum_sq = torch.zeros(channels, dtype=torch.float64, device=device)
    total_pixels = 0
    
    # Process in batches
    for i in range(0, num_images, batch_size):
        # Load a batch from NumPy and move it directly to GPU
        batch = torch.as_tensor(images[i:i + batch_size], device=device, dtype=torch.float32)  # Shape: (B, H, W, C)
        
        # Compute batch statistics along the spatial dimensions (H, W)
        batch_pixels = batch.size(0) * batch.size(1) * batch.size(2)  # B * H * W
        total_pixels += batch_pixels
        total_sum += batch.sum(dim=(0, 1, 2))  # Sum over (B, H, W)
        total_sum_sq += (batch ** 2).sum(dim=(0, 1, 2))  # Sum of squares over (B, H, W)
    
    # Calculate global mean and variance
    mean = total_sum / total_pixels
    variance = (total_sum_sq / total_pixels) - (mean ** 2)
    variance = torch.clamp(variance, min=0)  # Avoid negative variance due to precision issues
    std = torch.sqrt(variance)
    
    # Return results as NumPy arrays
    return mean.cpu().numpy(), std.cpu().numpy()


def gaussian(x, a, mu, sigma):
    """Gaussian function to fit the histogram."""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def smooth_histogram(data, bins=None, smooth_sigma=None):
    """
    Compute and smooth a histogram from input data using adaptive binning.

    Args:
        data (array-like): Input data for the histogram.
        bins (int or None): Number of bins (adaptive if None).
        smooth_sigma (float or None): Gaussian smoothing factor (adaptive if None).

    Returns:
        tuple: (bin_centers, smoothed_hist, raw_hist)
    """
    # Adaptive binning using Freedman-Diaconis rule
    if bins is None:
        iqr = np.percentile(data, 75) - np.percentile(data, 25)  # Interquartile range
        bin_width = 2 * iqr / np.cbrt(len(data))  # Freedman-Diaconis rule
        bins = max(100, int((max(data) - min(data)) / bin_width))

    # Compute histogram
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Adaptive smoothing
    if smooth_sigma is None:
        smooth_sigma = max(1, bins / 50)

    print('Number of bins used:', bins, 'smoothing sigma value:', smooth_sigma)
    
    smoothed_hist = gaussian_filter1d(hist, sigma=smooth_sigma)

    return bin_centers, smoothed_hist, hist


def find_first_peak(bin_centers, smoothed_hist):
    """
    Detect the first peak in a smoothed histogram.

    Args:
        bin_centers (array): X-axis values of histogram bins.
        smoothed_hist (array): Smoothed histogram values.

    Returns:
        float: X-coordinate of the first detected peak, or None if no peak is found.
    """
    # Find peaks, filtering out noise
    peaks, _ = find_peaks(smoothed_hist, height=np.max(smoothed_hist) * 0.1, distance=5)

    if len(peaks) > 0:
        return bin_centers[peaks[0]]
    return None


def fit_gaussian_to_rising_edge(bin_centers, smoothed_hist, first_peak_x):
    """
    Fit a Gaussian to the rising edge of the histogram near the first peak.

    Args:
        bin_centers (array): X-axis values of histogram bins.
        smoothed_hist (array): Smoothed histogram values.
        first_peak_x (float): Initial guess for the peak location.

    Returns:
        tuple: (Gaussian parameters: amplitude, mu (peak), sigma)
    """
    if first_peak_x is None:
        print("No first peak found. Cannot fit Gaussian.")
        return None

    # Select points to the left of the first peak (rising edge)
    rising_mask = bin_centers < first_peak_x
    fit_x = bin_centers[rising_mask]
    fit_y = smoothed_hist[rising_mask]

    if len(fit_x) < 5:  # Not enough points to fit a Gaussian
        print("Not enough data on the rising edge for fitting.")
        return None

    # Initial guess for Gaussian parameters
    initial_guess = [max(fit_y), first_peak_x, np.std(fit_x)]

    try:
        popt, _ = curve_fit(
            lambda x, a, mu, sigma: a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)), 
            fit_x, fit_y, p0=initial_guess,
            bounds=([0, first_peak_x * 0.8, 0.001], [np.inf, first_peak_x * 1.2, np.inf])  # Constrain fit
        )
        return popt  # (amplitude, mu, sigma)
    except RuntimeError:
        print("Gaussian fitting failed!")
        return None


def analyze_distribution(bin_centers, raw_hist, smoothed_hist, gaussian_params, data_values, to_plot=False, NUMSIGMA=1):
    """
    Analyze the distribution, fit a Gaussian, and apply thresholding based on mu ± sigma.

    Args:
        bin_centers (array): X-axis values of histogram bins.
        raw_hist (array): Raw histogram values.
        smoothed_hist (array): Smoothed histogram values.
        gaussian_params (tuple): Parameters of the fitted Gaussian (a, mu, sigma).
        data_values (array-like): Original data points (e.g., cosine similarity distances).

    Returns:
        np.ndarray: Boolean mask indicating which samples are inside the selected cluster.
    """
    if gaussian_params is None or len(gaussian_params) < 3:
        print("Gaussian fitting failed or insufficient data.")
        return np.zeros_like(data_values, dtype=bool)  # Return all False mask

    # Extract Gaussian parameters
    a, mu, sigma = gaussian_params

    # Define filtering thresholds: μ ± σ
    lower_threshold = max(0, mu - NUMSIGMA*sigma)  # Ensure non-negative similarity
    upper_threshold = mu + NUMSIGMA*sigma

    print(f"Thresholding Range: [{lower_threshold:.4f}, {upper_threshold:.4f}] (mu={mu:.4f}, sigma={sigma:.4f})")

    # Generate a logical mask for filtering
    mask = (data_values >= lower_threshold) & (data_values <= upper_threshold)

    if to_plot:
        # Step 4: Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(bin_centers, raw_hist, label="Original Histogram", alpha=0.5)
        plt.plot(bin_centers, smoothed_hist, label="Smoothed Histogram")

        # Plot detected first peak
        plt.axvline(mu, color="green", linestyle="--", label=f"Gaussian Peak (mu): {mu:.4f}")

        # Plot Gaussian fit
        fit_x_dense = np.linspace(0, 2 * mu, 100)
        plt.plot(fit_x_dense, a * np.exp(-((fit_x_dense - mu) ** 2) / (2 * sigma ** 2)),
                label="Gaussian Fit", color="midnightblue")

        # Plot threshold range
        plt.axvline(lower_threshold, color="orange", linestyle="--", label=f"Lower Threshold: {lower_threshold:.4f}")
        plt.axvline(upper_threshold, color="red", linestyle="--", label=f"Upper Threshold: {upper_threshold:.4f}")

        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.title("Gaussian Fit and Thresholding")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return mask  # Logical mask for filtering


def calculate_distances(features, query_embedding, metric="cosine"):
    """
    Calculate distances between the query image's embedding and all other samples in the feature space.

    Args:
        features (numpy.ndarray): Feature embeddings (n_samples, n_features).
        query_embedding (numpy.ndarray): Query image embedding (1, n_features).
        metric (str): Distance metric to use ("cosine" or "euclidean").

    Returns:
        numpy.ndarray: Array of distances (n_samples,).
    """
    # Reshape query_embedding to ensure it has the correct shape
    query_embedding = query_embedding.reshape(1, -1)
    
    if metric == "cosine":
        distances = cosine_distances(features, query_embedding)
    elif metric == "euclidean":
        distances = euclidean_distances(features, query_embedding)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    # Flatten the distance array to get a 1D array of distances
    return distances.flatten()

## ADDED functions for filtering with conventional image processing

def is_centered(image, tolerance=0.2):
    """
    Check if the pollen grain is centered within the crop.
    Args:
        image (np.array): Input image (RGB).
        tolerance (float): Acceptable offset ratio from center (0.1 = 10%).
    Returns:
        bool: True if centered, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    com = center_of_mass(binary)
    h, w = binary.shape
    center_x, center_y = w / 2, h / 2

    # Distance from center normalized by image size
    offset_x = abs(com[1] - center_x) / w
    offset_y = abs(com[0] - center_y) / h

    return offset_x < tolerance and offset_y < tolerance

def is_regular_shape(image, aspect_ratio_tolerance=0.1, min_coverage=0.01):
    """
    Filter based on shape regularity and pollen coverage.
    Args:
        image (np.array): Input image.
        aspect_ratio_tolerance (float): Tolerance for aspect ratio deviation from 1.
        min_coverage (float): Minimum coverage ratio of pollen region.
    Returns:
        bool: True if image passes, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h
        coverage = cv2.contourArea(largest_contour) / (image.shape[0] * image.shape[1])

        return (abs(aspect_ratio - 1) < aspect_ratio_tolerance) and (coverage >= min_coverage)
    return False

def refine_bbox_with_global_coords(image, global_bbox):
    """
    Refines the bounding box around the pollen grain for better alignment and updates global coordinates.
    
    Args:
        image (PIL.Image.Image): Input patch image (RGB).
        global_bbox (tuple): Original global bounding box (x_min, y_min, x_max, y_max).
        
    Returns:
        dict: Updated global bounding box and refined image.
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Threshold for contour detection
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Update global coordinates
        global_x_min = global_bbox[0] + x
        global_y_min = global_bbox[1] + y
        global_x_max = global_x_min + w
        global_y_max = global_y_min + h

        refined_crop = image_np[y:y + h, x:x + w, :]
        refined_crop_pil = Image.fromarray(refined_crop)

        return {
            "updated_bbox": (global_x_min, global_y_min, global_x_max, global_y_max),
            "refined_patch": refined_crop_pil
        }

    # If no contour is found, return the original
    return {
        "updated_bbox": global_bbox,
        "refined_patch": image
    }
    
def robust_circularity_filter(annotations, center_tolerance=0.1, lower_ellipse_axis_ratio=0.85, upper_ellipse_axis_ratio=1.15,
                              min_coverage=0.2, max_coverage=0.8, min_contour_area=500):
    """
    Enhanced filtering with morphological fixes, contour aggregation, and ellipse-based circularity.

    Args:
        annotations (list): List of dicts with 'patch' and 'bounding_box'.
        center_tolerance (float): Max deviation of centroid from patch center.
        ellipse_axis_ratio (float): Minimum ratio between ellipse axes to ensure circularity.
        min_coverage (float): Minimum pollen area coverage.
        max_coverage (float): Maximum pollen area coverage.
        min_contour_area (int): Minimum area to consider a contour valid.

    Returns:
        list: Filtered high-quality annotations.
    """
    filtered_annotations = []

    for entry in annotations:
        patch = np.array(entry['patch'])
        img_w, img_h, _ = patch.shape
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        #block_size = (img_w // 20) * 2 + 1  # Ensures it's odd
        #binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                               cv2.THRESH_BINARY_INV, block_size, 2)
        
        # Morphological closing to connect edges
        #kernel = np.ones((5, 5), np.uint8)
        #binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_size = max(3, min(img_w, img_h) // 50)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Aggregate relevant contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]
        if not valid_contours:
            continue

        combined_contour = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(combined_contour)
        img_h, img_w = binary.shape

        # Center alignment
        cx, cy = center_of_mass(binary_closed)
        if abs(cx - img_w / 2) / img_w > center_tolerance or abs(cy - img_h / 2) / img_h > center_tolerance:
            continue

        # Ellipse-based circularity (axis ratio)
        if len(combined_contour) >= 5:  # fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(combined_contour)
            (center_x, center_y), (major_axis, minor_axis), angle = ellipse
            axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
            if np.logical_or( axis_ratio < lower_ellipse_axis_ratio, upper_ellipse_axis_ratio < axis_ratio ):
                continue

        # Coverage ratio check
        area = cv2.contourArea(combined_contour)
        coverage = area / (img_w * img_h)
        if coverage < min_coverage or coverage > max_coverage:
            continue

        # If all checks pass
        filtered_annotations.append(entry)

    print(f"{len(filtered_annotations)} images retained (from {len(annotations)} total).")
    return filtered_annotations


def filter_by_patch_size(entries, min_size=10, iqr_multiplier=1.0):
    """
    Filters a list of patch entries by removing extreme size outliers based on IQR.

    Args:
        entries (list): List of dicts containing 'patch' (PIL.Image).
        min_size (int): Minimum allowed width/height (hard lower bound).
        iqr_multiplier (float): Controls how aggressively to filter outliers.

    Returns:
        list: Filtered list of entries.
    """
    widths = np.array([e['patch'].size[0] for e in entries])
    heights = np.array([e['patch'].size[1] for e in entries])

    w_q1, w_q3 = np.percentile(widths, [25, 75]) 
    h_q1, h_q3 = np.percentile(heights, [25, 75]) 

    w_lower, w_upper = max(min_size, w_q1 - iqr_multiplier * (w_q3 - w_q1)), w_q3 + iqr_multiplier * (w_q3 - w_q1)
    h_lower, h_upper = max(min_size, h_q1 - iqr_multiplier * (h_q3 - h_q1)), h_q3 + iqr_multiplier * (h_q3 - h_q1)

    return [
        e for e in entries
        if (w_lower <= e['patch'].size[0] <= w_upper) and (h_lower <= e['patch'].size[1] <= h_upper)
    ]