#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Compute stain normalization reference images and dataset statistics.

This script implements the proper pipeline:
1. Sample patches from training set (load from WSI using bbox)
2. Compute pixel-wise median across patches → save as reference.npy  
3. Fit MacenkoNormalizer to median image
4. Apply stain normalization to all sampled patches
5. Compute channel mean/std on NORMALIZED patches → save as stats.json

The stats.json replaces ImageNet normalization defaults for stainnorm experiments.

Usage:
    python compute_stainnorm_reference.py --region french --num_samples 1000
    python compute_stainnorm_reference.py --region all --num_samples 500
"""

import os

# Restrict CPU threads to minimize resource usage (before other imports)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add repo root to path for imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent.parent  # data_preparation -> scripts -> 04_evaluation -> repo
sys.path.insert(0, str(repo_root))

# Import from lib
from lib.loader import calculate_stats_batched

# Import stain normalization
try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    HAS_TIATOOLBOX = True
except ImportError:
    HAS_TIATOOLBOX = False
    print("Warning: tiatoolbox not available, falling back to simple implementation")


# Region → dataset folder mapping
REGION_TO_DATASETS = {
    "french": ["french"],
    "hungarian": ["hungarian"],
    "swedish": ["swedish"],
    "mediterranean": ["mediterranean"],
    "all": ["french", "hungarian", "swedish", "mediterranean"],
}


def find_repo_root() -> Path:
    """Find repository root by looking for AGENTS.md or .git."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "AGENTS.md").exists() or (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def load_samples_from_splits(
    splits_dir: Path,
    caption_dir: Path,
    wsi_dir: Path,
    region: str,
    caption_model: str = "production_qwen25vl_final",
) -> list:
    """
    Load training samples from splits, matching train_classifier.py logic.
    
    Returns list of dicts with: id, slide, bbox, wsi_path
    """
    samples = []
    train_dir = splits_dir / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train split directory not found: {train_dir}")
    
    datasets = REGION_TO_DATASETS.get(region, [region])
    
    # Find split files for the target datasets
    split_files = list(train_dir.glob("*_train.json"))
    
    for split_file in tqdm(split_files, desc=f"Loading {region} splits"):
        with open(split_file) as f:
            split_data = json.load(f)
        
        slide_name = split_data["slide"]
        sample_ids = set(split_data["sample_ids"])
        
        # Find corresponding caption file (determines dataset)
        caption_file = None
        slide_dataset = None
        for dataset in datasets:
            caption_path = caption_dir / dataset / caption_model / f"{slide_name}_captions.jsonl"
            if caption_path.exists():
                caption_file = caption_path
                slide_dataset = dataset
                break
        
        if caption_file is None:
            continue  # Slide not in target region
        
        # Find WSI path
        wsi_path = None
        for ext in [".tif", ".tiff", ".svs", ".ndpi"]:
            candidate = wsi_dir / slide_dataset / f"{slide_name}{ext}"
            if candidate.exists():
                wsi_path = str(candidate)
                break
        
        if wsi_path is None:
            continue
        
        # Load JSONL and filter by sample_ids
        with open(caption_file) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if record["id"] in sample_ids:
                        bbox = record["bbox"]
                        samples.append({
                            "id": record["id"],
                            "slide": slide_name,
                            "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                            "wsi_path": wsi_path,
                        })
                except json.JSONDecodeError:
                    continue
    
    return samples


def sample_records(samples: list, num_samples: int, seed: int = 42) -> list:
    """Randomly sample records."""
    np.random.seed(seed)
    if num_samples >= len(samples):
        return samples
    indices = np.random.choice(len(samples), num_samples, replace=False)
    return [samples[i] for i in indices]


def extract_patches_from_wsi(samples: list, target_size: int = 518) -> np.ndarray:
    """Extract patches from WSI files using bbox coordinates."""
    try:
        import tiffslide
        use_tiffslide = True
    except ImportError:
        import openslide
        use_tiffslide = False
    
    images = []
    wsi_cache = {}
    
    for sample in tqdm(samples, desc="Extracting patches"):
        wsi_path = sample["wsi_path"]
        
        try:
            if wsi_path not in wsi_cache:
                if use_tiffslide:
                    wsi_cache[wsi_path] = tiffslide.TiffSlide(wsi_path)
                else:
                    wsi_cache[wsi_path] = openslide.OpenSlide(wsi_path)
            
            wsi = wsi_cache[wsi_path]
            x1, y1, x2, y2 = sample["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            region = wsi.read_region((x1, y1), 0, (width, height)).convert("RGB")
            region = region.resize((target_size, target_size), Image.LANCZOS)
            images.append(np.array(region))
        except Exception as e:
            print(f"Warning: Failed to extract {sample['id']}: {e}")
            continue
    
    for wsi in wsi_cache.values():
        try:
            wsi.close()
        except:
            pass
    
    return np.array(images, dtype=np.uint8)


def compute_median_image(images: np.ndarray) -> np.ndarray:
    """Compute pixel-wise median across images."""
    print(f"Computing median image from {len(images)} patches...")
    n, h, w, c = images.shape
    reshaped = images.reshape(n, -1, c)
    median_flat = np.median(reshaped, axis=0)
    return median_flat.reshape(h, w, c).astype(np.uint8)


def apply_stain_normalization(images: np.ndarray, normalizer) -> np.ndarray:
    """Apply stain normalization to all images."""
    normalized = []
    failed_count = 0
    
    for img in tqdm(images, desc="Applying stain normalization"):
        try:
            norm_img = normalizer.transform(img)
            normalized.append(norm_img)
        except Exception:
            normalized.append(img)
            failed_count += 1
    
    if failed_count > 0:
        print(f"Warning: Stain normalization failed for {failed_count}/{len(images)} images")
    
    return np.array(normalized, dtype=np.uint8)


def compute_channel_stats(images: np.ndarray) -> dict:
    """Compute per-channel mean and std using GPU-optimized method."""
    print(f"Computing channel statistics from {len(images)} images...")
    n, h, w, c = images.shape
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    mean, std = calculate_stats_batched(images, batch_size=32, device=device)
    
    return {
        "mean": (mean / 255.0).tolist(),
        "std": (std / 255.0).tolist(),
        "mean_raw": mean.tolist(),
        "std_raw": std.tolist(),
        "num_samples": len(images),
        "num_pixels": n * h * w
    }


def save_visualization(images: np.ndarray, normalized_images: np.ndarray, output_path: Path):
    """Save before/after comparison visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_viz = min(16, len(images))
    np.random.seed(42)
    viz_indices = np.random.choice(len(images), n_viz, replace=False)
    
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for i, idx in enumerate(viz_indices[:16]):
        row = i // 4
        col_orig = (i % 4) * 2
        col_norm = col_orig + 1
        
        axes[row, col_orig].imshow(images[idx])
        axes[row, col_orig].axis('off')
        axes[row, col_orig].set_title('Original', fontsize=8)
        
        axes[row, col_norm].imshow(normalized_images[idx])
        axes[row, col_norm].axis('off')
        axes[row, col_norm].set_title('Normalized', fontsize=8)
    
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis('off')
    
    plt.suptitle(f'Stain Normalization - Before/After', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute stain normalization reference and statistics"
    )
    parser.add_argument('--region', type=str, required=True,
                        choices=['french', 'hungarian', 'swedish', 'mediterranean', 'all'],
                        help="Region to compute reference for")
    parser.add_argument('--num_samples', type=int, default=1000,
                        help="Number of patches to sample (default: 1000)")
    parser.add_argument('--patch_size', type=int, default=518,
                        help="Patch size for extraction (default: 518)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument('--caption_model', type=str, default="production_qwen25vl_final",
                        help="Caption model directory")
    parser.add_argument('--splits_dir', type=str, default=None,
                        help="Path to splits directory")
    parser.add_argument('--caption_dir', type=str, default=None,
                        help="Path to caption directory")
    parser.add_argument('--wsi_dir', type=str, default=None,
                        help="Path to WSI directory")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Path to output directory")
    args = parser.parse_args()
    
    # Find paths
    repo_root = find_repo_root()
    
    splits_dir = Path(args.splits_dir) if args.splits_dir else repo_root / "data" / "04_evaluation" / "splits"
    caption_dir = Path(args.caption_dir) if args.caption_dir else repo_root / "data" / "03_captioning"
    wsi_dir = Path(args.wsi_dir) if args.wsi_dir else repo_root / "data" / "00_raw_wsi"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data" / "04_evaluation" / "normalization"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Computing stain normalization reference for: {args.region} ===")
    print(f"Splits directory: {splits_dir}")
    print(f"Caption directory: {caption_dir}")
    print(f"WSI directory: {wsi_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Patch size: {args.patch_size}")
    
    # Step 1: Load and sample training patches
    print("\n[Step 1/5] Loading training samples from splits...")
    all_samples = load_samples_from_splits(
        splits_dir=splits_dir,
        caption_dir=caption_dir,
        wsi_dir=wsi_dir,
        region=args.region,
        caption_model=args.caption_model,
    )
    print(f"Total samples available: {len(all_samples)}")
    
    samples = sample_records(all_samples, args.num_samples, seed=args.seed)
    print(f"Sampled: {len(samples)}")
    
    # Step 2: Extract patches from WSI
    print("\n[Step 2/5] Extracting patches from WSI...")
    images = extract_patches_from_wsi(samples, target_size=args.patch_size)
    print(f"Extracted {len(images)} patches with shape {images.shape}")
    
    # Step 3: Compute median image and save as reference
    print("\n[Step 3/5] Computing median reference image...")
    median_image = compute_median_image(images)
    
    reference_path = output_dir / f"{args.region}_reference.npy"
    np.save(reference_path, median_image)
    print(f"Saved reference image: {reference_path}")
    
    reference_png = output_dir / f"{args.region}_reference.png"
    Image.fromarray(median_image).save(reference_png)
    print(f"Saved reference visualization: {reference_png}")
    
    # Step 4: Fit normalizer and apply stain normalization
    print("\n[Step 4/5] Applying stain normalization...")
    
    if HAS_TIATOOLBOX:
        normalizer = MacenkoNormalizer()
        normalizer.fit(median_image)
        normalizer_name = "tiatoolbox"
        print("Using tiatoolbox MacenkoNormalizer")
    else:
        raise RuntimeError("tiatoolbox required for stain normalization")
    
    normalized_images = apply_stain_normalization(images, normalizer)
    
    # Save visualization
    print("\nGenerating comparison visualization...")
    viz_path = output_dir / f"{args.region}_stainnorm_comparison.png"
    save_visualization(images, normalized_images, viz_path)
    print(f"Saved comparison visualization: {viz_path}")
    
    # Step 5: Compute statistics on NORMALIZED images
    print("\n[Step 5/5] Computing channel statistics on normalized images...")
    stats = compute_channel_stats(normalized_images)
    
    stats["region"] = args.region
    stats["reference_file"] = str(reference_path)
    stats["patch_size"] = args.patch_size
    stats["normalizer"] = normalizer_name
    
    stats_path = output_dir / f"{args.region}_stainnorm_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Reference image: {reference_path}")
    print(f"Statistics file: {stats_path}")
    print(f"Channel means (0-1): {[f'{v:.4f}' for v in stats['mean']]}")
    print(f"Channel stds (0-1):  {[f'{v:.4f}' for v in stats['std']]}")
    print(f"\nFor comparison, ImageNet stats:")
    print(f"  Mean: [0.485, 0.456, 0.406]")
    print(f"  Std:  [0.229, 0.224, 0.225]")


if __name__ == "__main__":
    main()
