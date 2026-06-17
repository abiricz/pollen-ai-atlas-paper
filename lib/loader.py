# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

## HINT: apply this with Macenko norm to minimize CPU usage !
#import os
# Restrict CPU usage 
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset
from PIL import Image
import openslide
import tiffslide
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# Optional: tiatoolbox for Macenko stain normalization
# Requires Python <3.12 due to numba dependency
try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    HAS_TIATOOLBOX = True
except ImportError:
    HAS_TIATOOLBOX = False
    MacenkoNormalizer = None


class WSITileDataset(Dataset):
    """
    A generic PyTorch Dataset to process and return preprocessed tiles
    from Whole Slide Images (WSIs) (TIF files) for inference.

    The dataset:
      - Reads a WSI from disk.
      - Generates tile coordinates based on the specified tile size and overlap.
      - Optionally applies stain normalization and histogram matching.
      - Applies a generic transformation (provided externally) to each tile.

    Args:
        wsi_path (str): Path to the whole slide image.
        transform (callable): A function or torchvision.transforms.Compose that takes a PIL image
                              as input and returns a transformed tensor.
        tile_size (int): The size (in pixels) of each square tile.
        overlap (int): The number of pixels that consecutive tiles overlap.
        tiffslide_use (bool): If True, uses tiffslide to read the image; otherwise, uses openslide.
        stainnorm (bool): Whether to apply stain normalization.
        stainnorm_func (callable, optional): Function that takes a PIL image and returns a stain-normalized PIL image.
        hist_match (bool): Whether to apply histogram matching after stain normalization.
        hist_match_func (callable, optional): Function that takes a PIL image and returns a histogram-matched PIL image.
    """
    def __init__(self, wsi_path, transform=None, tile_size=512, overlap=64, tiffslide_use=False,
                 stainnorm=False, stainnorm_func=None, hist_match=False, hist_match_func=None,
                 rescaling=None, region_bounds=None):
        self.wsi_path = wsi_path
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.tile_size = tile_size
        self.overlap = overlap
        self.tiffslide_use = tiffslide_use

        # Optional processing functions.
        self.stainnorm = stainnorm
        self.stainnorm_func = stainnorm_func
        self.hist_match = hist_match
        self.hist_match_func = hist_match_func
        
        self.rescaling = rescaling

        # Load WSI dimensions.
        if not self.tiffslide_use:
            with openslide.OpenSlide(wsi_path) as wsi:
                self.w, self.h = wsi.dimensions
        else:
            with tiffslide.TiffSlide(wsi_path) as wsi:
                self.w, self.h = wsi.dimensions

        # Restrict region bounds if provided
        if region_bounds:
            x_min, y_min, x_max, y_max = region_bounds
            self.x_min = max(0, x_min)
            self.y_min = max(0, y_min)
            self.x_max = min(self.w, x_max)
            self.y_max = min(self.h, y_max)
        else:
            self.x_min, self.y_min, self.x_max, self.y_max = 0, 0, self.w, self.h  # Full WSI

        # Generate tile coordinates **within the restricted region**, ensuring full tile coverage
        self.tile_coords = []
        for y in range(self.y_min, self.y_max, self.tile_size - self.overlap):
            for x in range(self.x_min, self.x_max, self.tile_size - self.overlap):

                ## COMMENTED OUT FOR NOW ! FOR VIT MAP SIM CALCS !! 
                # Ensure the last tile at the right and bottom edges is fully within bounds
                #if x + self.tile_size > self.x_max:
                #    x = self.x_max - self.tile_size  # Shift left to fit within bounds
                #if y + self.tile_size > self.y_max:
                #    y = self.y_max - self.tile_size  # Shift up to fit within bounds
                
                # Append adjusted coordinates
                self.tile_coords.append((x, y))

    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        x, y = self.tile_coords[idx]

        # Read the tile.
        if not self.tiffslide_use:
            with openslide.OpenSlide(self.wsi_path) as wsi:
                tile = wsi.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
        else:
            with tiffslide.TiffSlide(self.wsi_path) as wsi:
                tile = wsi.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
        
        if self.rescaling:
                tile = tile.resize(( int(tile.size[0]*self.rescaling), int(tile.size[1]*self.rescaling)))

        # Optionally apply stain normalization.
        if self.stainnorm and self.stainnorm_func is not None:
            tile = self.stainnorm_func(tile) # input is PIL image, output is np array !
        
        # Optionally apply histogram matching after stain normalization.
        if self.hist_match and self.hist_match_func is not None:
            tile = self.hist_match_func(tile)

        # Apply the external transform to the tile or just comvert to torch tensor as is.
        #if self.transform is not None:
        transformed_tile = self.transform(tile)
        #else:
        #    transformed_tile = torch.from_numpy( np.array(tile).astype(np.float32) )

        return {
            "tile": tile,                # Original PIL tile
            "inputs": transformed_tile,  # Processed tensor ready for inference
            "coords": (x, y)
        }
        
    def custom_collate_fn(self, batch):
        """
        Custom collate function to handle PIL images and tensors properly.
        
        Args:
            batch (list of dicts): Each dict contains "tile" (PIL image), "tile_tensor" (Tensor), and "coords".
        
        Returns:
            dict: A dictionary with batched tensor tiles and lists of PIL images and coordinates.
        """
        tiles = [item["tile"] for item in batch]  # Keep PIL images as a list (not batched)
        tiles_tensors = torch.stack([item["inputs"] for item in batch])  # Stack tensors into a batch
        coords = [item["coords"] for item in batch]  # Keep coordinates as a list

        return {"tiles": tiles, "tiles_tensors": tiles_tensors, "coords": coords}


class Normalization():

    def __init__(self, stain_normalizer, stainnorm_ref_img_path, 
                 content_mean=None, content_std=None,
                 training_mean=None, training_std=None):

        self.stain_normalizer = stain_normalizer
        self.normalizer = None
        self.stainnorm_ref_img_path = stainnorm_ref_img_path
        
        if self.stainnorm_ref_img_path:
            median_image = np.load(stainnorm_ref_img_path)
            self.normalizer = stain_normalizer() # Initialize the Macenko Normalizer
            self.normalizer.fit(np.array(median_image))
            print('Stain normalizer fitted, ready to transform images.')
        
        self.content_mean = content_mean
        self.content_std = content_std
        self.training_mean = training_mean
        self.training_std = training_std
        print('Initialized: ',
                self.content_mean,
                self.content_std,
                self.training_mean,
                self.training_std)
        
        self.transform = None
        if self.training_mean is not None and self.training_std is not None:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=self.training_mean / 255,
                                                     std=self.training_std / 255)
                                ])

    def stain_norm(self, image):
        try:
            image = self.normalizer.transform(np.array(image))
        except (ValueError, RuntimeWarning) as e:
            print(f"Skipping stain normalization: {e}")
            return image  # Return original image if stain normalization fails
        return image

    def hist_norm(self, content_image, alpha=1.0):
        """
        Match the mean and variance of a single content image to training statistics using NumPy.
    
        Args:
            content_image (np.ndarray): Input image (H, W, C), expected in uint8 format.
            content_mean (np.ndarray or list): Mean of the input image (C,).
            content_std (np.ndarray or list): Standard deviation of the input image (C,).
            train_mean (np.ndarray or list): Mean of the training dataset (C,).
            train_std (np.ndarray or list): Standard deviation of the training dataset (C,).
            alpha (float): Degree of alignment (0 = no adjustment, 1 = full alignment).
    
        Returns:
            np.ndarray: Aligned image with values clamped to 0-255.
        """
    
        # Convert to numpy and float32 for computation
        content_image = np.array(content_image).astype(np.float32)
    
        # Reshape means and stds for broadcasting (1, 1, C)
        content_mean = np.array(self.content_mean, dtype=np.float32).reshape(1, 1, -1)
        content_std = np.array(self.content_std, dtype=np.float32).reshape(1, 1, -1)
        train_mean = np.array(self.training_mean, dtype=np.float32).reshape(1, 1, -1)
        train_std = np.array(self.training_std, dtype=np.float32).reshape(1, 1, -1)
    
        # Prevent division by zero by adding a small constant
        adjusted_image = (content_image - content_mean) / (content_std + 1e-8)
        adjusted_image = adjusted_image * train_std + train_mean
    
        # Blend with the original image
        blended_image = alpha * adjusted_image + (1 - alpha) * content_image
    
        # Clip values to ensure they remain in valid range [0, 255]
        blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    
        return blended_image
    

## SOME HELPER FUNCTION

def calculate_stats_batched(images, batch_size=128, device='cpu'):
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


def calc_dataset_stats(DATALOADER, num_batches_to_sample=10, tile_shape=(896, 896, 3), device='cpu' ):
    
    # Adjust if needed
    batch_size = DATALOADER.batch_size
    num_samples_needed = num_batches_to_sample * batch_size  # Total number of tiles
    
    # Storage for NumPy arrays directly
    sampled_tiles_np = np.zeros((num_samples_needed, *tile_shape), dtype=np.uint8)
    
    # Convert the DataLoader into an iterable
    dataloader_iter = iter(DATALOADER)
    
    # Track the index
    sampled_idx = 0
    
    # Randomly sample `num_batches_to_sample` batches
    for _ in tqdm(range(num_batches_to_sample), desc="Sampling and Converting Tiles"):
        try:
            batch = next(dataloader_iter)  # Get next batch
        except StopIteration:
            # Restart the iterator if exhausted
            dataloader_iter = iter(DATALOADER)
            batch = next(dataloader_iter)
    
        # Convert tiles (PIL) to NumPy arrays and assign directly
        for tile in batch["tiles"]:
            if sampled_idx < num_samples_needed:
                sampled_tiles_np[sampled_idx] = np.asarray(tile, dtype=np.uint8)
                sampled_idx += 1
            else:
                break  # Stop once the target number is reached
    
    print(f"Stored {sampled_idx} tiles in memory with shape {sampled_tiles_np.shape}")
    
    mean, std = calculate_stats_batched(sampled_tiles_np, batch_size=4, device=device)
    
    return mean, std