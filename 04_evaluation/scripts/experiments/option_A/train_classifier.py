#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Pollen Classification Training Script
======================================

Train a classifier on the Pollen AI Atlas dataset using frozen or finetuned backbone.
Supports linear probing and full finetuning with configurable experiments via YAML.

Features:
- Full experiment tracking (metadata, hyperparams, results in JSON)
- Incremental CSV logging (per-epoch, survives interruption)
- Periodic checkpoint saving (configurable interval)
- Downsampled training for quick validation (--max_samples, --epochs)
- ImageNet normalization by default
- Class balancing (sqrt/balanced/none)
- Checkpoint saving with best model tracking

Usage:
    # Full training
    python train_classifier.py --config experiment_config.yaml --experiment linear_probe_all
    
    # Quick validation (downsampled)
    python train_classifier.py --config experiment_config.yaml --experiment linear_probe_all \\
        --max_samples 5000 --epochs 3
    
    # Different device
    python train_classifier.py --config experiment_config.yaml --experiment linear_probe_all --device cuda:1

"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import csv
import gc
import json
import math
import yaml
import time
import platform
import subprocess
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm

# Note: We use fork mode (default) for faster worker startup.
# Memory is managed via:
# 1. LRU cache for WSI handles (max 20 per worker)
# 2. gc.collect() between train/val epochs
# 3. Minimal sample dict (no captions stored)

# Add project root to path (4 levels up from option_A/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Use HITL-validated species mapping (not legacy lib.classifier)
from lib.species_mapping import load_caption_anchors, build_class_mappings, get_slide_class_id, get_trainable_slides

# Optional: Macenko stain normalization
try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    HAS_TIATOOLBOX = True
except ImportError:
    HAS_TIATOOLBOX = False
    MacenkoNormalizer = None


# =============================================================================
# STAIN/HISTOGRAM NORMALIZATION PREPROCESSING
# =============================================================================

class StainNormPreprocessor:
    """Apply Macenko stain normalization to images.
    
    Reference image should be computed from training set:
        - Sample 1000 random images
        - Compute median LAB image
        - Save as .npy file
    
    Usage:
        # Create preprocessor
        preprocessor = StainNormPreprocessor(reference_path='data/.../french_reference.npy')
        
        # Apply in dataset __getitem__
        img = preprocessor(img)  # PIL Image in, PIL Image out
    """
    
    def __init__(self, reference_path: str):
        if not HAS_TIATOOLBOX:
            raise ImportError("tiatoolbox required for stain normalization. "
                              "Install: pip install tiatoolbox")
        
        reference_path = Path(reference_path)
        if not reference_path.exists():
            raise FileNotFoundError(
                f"Stain reference not found: {reference_path}\n"
                f"Run: python compute_stainnorm_reference.py --region {reference_path.stem.split('_')[0]}"
            )
        
        # Load reference image
        self.reference_img = np.load(reference_path)
        if self.reference_img.dtype != np.uint8:
            self.reference_img = (self.reference_img * 255).astype(np.uint8)
        
        # Initialize normalizer
        self.normalizer = MacenkoNormalizer()
        self.normalizer.fit(self.reference_img)
        print(f"[StainNorm] Initialized with reference: {reference_path.name}")
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Normalize a PIL Image."""
        img_np = np.array(img)
        try:
            normalized = self.normalizer.transform(img_np)
            return Image.fromarray(normalized)
        except Exception as e:
            # Some images fail stain extraction (too light, artifacts)
            # Return original in these cases
            return img


class HistogramNormPreprocessor:
    """Apply histogram normalization (mean/std alignment) to images.
    
    Aligns image histogram to training set statistics.
    Lighter weight than Macenko stain normalization.
    
    Usage:
        preprocessor = HistogramNormPreprocessor(mean=[r,g,b], std=[r,g,b])
        img = preprocessor(img)  # PIL Image in, PIL Image out
    """
    
    def __init__(self, mean: List[float], std: List[float]):
        """
        Args:
            mean: Target mean per channel [R, G, B] in [0, 255]
            std: Target std per channel [R, G, B] in [0, 255]
        """
        self.target_mean = np.array(mean)
        self.target_std = np.array(std)
        print(f"[HistNorm] Target mean={mean}, std={std}")
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Normalize a PIL Image to target distribution."""
        img_np = np.array(img).astype(np.float32)
        
        # Per-channel normalization
        for c in range(3):
            channel = img_np[:, :, c]
            ch_mean = channel.mean()
            ch_std = channel.std() + 1e-6
            
            # Z-score and rescale to target
            normalized = (channel - ch_mean) / ch_std
            normalized = normalized * self.target_std[c] + self.target_mean[c]
            
            # Clip to valid range
            img_np[:, :, c] = np.clip(normalized, 0, 255)
        
        return Image.fromarray(img_np.astype(np.uint8))


def get_preprocessor(config: Dict) -> Optional[callable]:
    """Create preprocessor from config.
    
    Args:
        config: Preprocessing config with stainnorm/hist_match settings
        
    Returns:
        Preprocessor callable or None
    """
    preprocessing = config.get("preprocessing", {})
    
    # Stain normalization takes priority
    stainnorm = preprocessing.get("stainnorm", {})
    if stainnorm.get("enabled", False):
        reference_path = stainnorm.get("reference_path")
        if reference_path:
            return StainNormPreprocessor(str(PROJECT_ROOT / reference_path))
    
    # Histogram normalization as fallback
    hist_match = preprocessing.get("hist_match", {})
    if hist_match.get("enabled", False):
        mean = hist_match.get("training_mean") or hist_match.get("reference_mean")
        std = hist_match.get("training_std") or hist_match.get("reference_std")
        if mean and std:
            return HistogramNormPreprocessor(mean, std)
    
    return None


# =============================================================================
# INCREMENTAL CSV LOGGER
# =============================================================================

class CSVLogger:
    """Incremental CSV logger that writes per-epoch and survives interruption."""
    
    def __init__(self, filepath: Path, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.file = None
        self.writer = None
        
        # Open file and write header if new
        is_new = not filepath.exists()
        self.file = open(filepath, 'a', newline='', buffering=1)  # Line buffering
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if is_new:
            self.writer.writeheader()
            self.file.flush()
    
    def log(self, row: Dict[str, Any]):
        """Write a row and flush immediately."""
        self.writer.writerow(row)
        self.file.flush()
        os.fsync(self.file.fileno())  # Force disk write
    
    def close(self):
        if self.file:
            self.file.close()


# =============================================================================
# EXPERIMENT TRACKING
# =============================================================================

def get_git_info() -> Dict[str, str]:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "unknown", "branch": "unknown"}


def get_system_info() -> Dict[str, Any]:
    """Get system and GPU info for experiment tracking."""
    info = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "timm_version": timm.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def create_experiment_metadata(
    config: Dict,
    experiment_name: str,
    train_dataset_size: int,
    val_dataset_size: int,
    class_counts: Dict[int, int],
    int_to_species: Dict[int, str],
    max_samples: Optional[int] = None,
    epochs_override: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Create comprehensive experiment metadata for tracking."""
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    train_config = exp_config["training"]
    
    metadata = {
        "experiment": {
            "name": experiment_name,
            "display_name": exp_config.get("name", experiment_name),
            "description": exp_config.get("description", ""),
            "started_at": datetime.now().isoformat(),
            "status": "running",
        },
        "git": get_git_info(),
        "system": get_system_info(),
        "model": {
            "backbone_name": global_config["backbone_name"],
            "backbone_checkpoint": global_config["backbone_checkpoint"],
            "num_classes": global_config["num_classes"],
            "embedding_dim": global_config["embedding_dim"],
            "img_size": global_config["img_size"],
            "freeze_backbone": train_config.get("freeze_backbone", True),
        },
        "data": {
            "train_region": train_config["train_region"],
            "train_samples": train_dataset_size,
            "val_samples": val_dataset_size,
            "class_distribution": {int_to_species.get(k, f"cls_{k}"): v 
                                   for k, v in sorted(class_counts.items())},
            "num_active_classes": len([v for v in class_counts.values() if v > 0]),
            "max_samples_override": max_samples,
        },
        "training": {
            "epochs": epochs_override or train_config.get("epochs", 30),
            "batch_size": train_config.get("batch_size", 128),
            "learning_rate": train_config.get("learning_rate", 0.001),
            "weight_decay": train_config.get("weight_decay", 0.0001),
            "optimizer": train_config.get("optimizer", "adamw"),
            "scheduler": train_config.get("scheduler", "cosine"),
            "warmup_epochs": train_config.get("warmup_epochs", 2),
            "class_balancing": train_config.get("class_balancing", "sqrt"),
            "seed": global_config.get("seed", 42) if seed is None else seed,
        },
        "preprocessing": {
            "normalization_mode": exp_config.get("preprocessing", {}).get("normalization_mode", "imagenet"),
            # Use experiment-specific values if custom mode, otherwise global defaults
            "normalize_mean": (
                exp_config.get("preprocessing", {}).get("normalize_mean")
                if exp_config.get("preprocessing", {}).get("normalization_mode") == "custom"
                else global_config.get("normalize_mean", [0.485, 0.456, 0.406])
            ),
            "normalize_std": (
                exp_config.get("preprocessing", {}).get("normalize_std")
                if exp_config.get("preprocessing", {}).get("normalization_mode") == "custom"
                else global_config.get("normalize_std", [0.229, 0.224, 0.225])
            ),
            "stainnorm_enabled": exp_config.get("preprocessing", {}).get("stainnorm", {}).get("enabled", False),
            "stainnorm_reference": exp_config.get("preprocessing", {}).get("stainnorm", {}).get("reference_path"),
        },
        "augmentation": train_config.get("augmentation", {}),
        "eval_test_sets": exp_config.get("eval_test_sets", []),
        "output_dir": exp_config.get("output_dir", ""),
    }
    
    return metadata


def save_experiment_results(
    output_dir: Path,
    metadata: Dict[str, Any],
    history: Dict[str, List],
    best_val_acc: float,
    training_time: float,
) -> None:
    """Save complete experiment results with full traceability."""
    
    # Update metadata with final results
    metadata["experiment"]["status"] = "completed"
    metadata["experiment"]["completed_at"] = datetime.now().isoformat()
    
    results = {
        "best_val_accuracy": best_val_acc,
        "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else 0,
        "final_train_accuracy": history["train_acc"][-1] if history["train_acc"] else 0,
        "final_val_top5_accuracy": history["val_top5_acc"][-1] if history["val_top5_acc"] else 0,
        "training_time_seconds": training_time,
        "training_time_hours": training_time / 3600,
        "epochs_completed": len(history["train_loss"]),
    }
    
    # Only main process writes files
    if not is_main_process():
        return

    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save results summary
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Tracking] Saved experiment metadata to {output_dir}")
    print(f"[Tracking] Files: experiment_metadata.json, training_history.json, results_summary.json")


# =============================================================================
# DATASET
# =============================================================================

class PollenClassificationDataset(Dataset):
    """
    Dataset for pollen classification training.
    
    Loads patches from caption JSONL files using sample_ids from splits.
    Each sample has: image crop, species label, caption (optional).
    
    Uses HITL-validated species from caption_anchors (stored in JSONL "species" field).
    """
    
    def __init__(
        self,
        split_dir: Path,
        caption_dir: Path,
        wsi_dir: Path,
        split: str = "train",
        transform: transforms.Compose = None,
        datasets: List[str] = None,
        caption_model: str = "production_qwen25vl_final",
        stainnorm_func: callable = None,
        max_samples: int = None,
        species_to_int: Dict[str, int] = None,
        num_classes: int = 46,
    ):
        """
        Args:
            split_dir: Path to splits directory (contains train/, val/)
            caption_dir: Path to 03_captioning directory
            wsi_dir: Path to 00_raw_wsi directory  
            split: "train" or "val"
            transform: Image transforms (includes resize to 518x518)
            datasets: List of dataset names to include (e.g., ["french", "hungarian"])
            caption_model: Which caption model folder to use
            stainnorm_func: Optional stain normalization function
            max_samples: Limit total samples (for debugging)
            species_to_int: HITL-validated species → class_id mapping
            num_classes: Total number of classes
        """
        self.split_dir = Path(split_dir) / split
        self.caption_dir = Path(caption_dir)
        self.wsi_dir = Path(wsi_dir)
        self.split = split
        self.transform = transform
        self.datasets = datasets or ["french", "hungarian", "mediterranean", "swedish"]
        self.caption_model = caption_model
        self.stainnorm_func = stainnorm_func
        
        # HITL-validated species mapping
        caption_anchors = load_caption_anchors()
        if species_to_int is None:
            species_to_int, _, num_classes = build_class_mappings(caption_anchors)
        self.caption_anchors = caption_anchors
        self.species_to_int = species_to_int
        self.num_classes = num_classes
        
        # Build sample index
        self.samples = []
        self._build_sample_index()
        
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        print(f"[Dataset] Loaded {len(self.samples)} samples for {split}")
        
    def _build_sample_index(self):
        """Load all sample IDs from split files and index to JSONL lines."""
        
        # First pass: load all split files
        split_files = list(self.split_dir.glob(f"*_{self.split}.json"))
        
        for split_file in tqdm(split_files, desc=f"Loading {self.split} splits"):
            with open(split_file) as f:
                split_data = json.load(f)
            
            slide_name = split_data["slide"]
            sample_ids = set(split_data["sample_ids"])
            
            # Find corresponding caption file
            caption_file = self._find_caption_file(slide_name)
            if caption_file is None:
                continue
            
            # Cache WSI path for this slide (avoid repeated lookups)
            wsi_path = self._find_wsi_path(slide_name)
            
            # Use HITL-validated species from _species.txt (genus level)
            # NOT from JSONL records which may have binomial names like "Betula sp."
            species = self.caption_anchors.get(slide_name, "Unknown")
            if species.lower() == "unknown":
                continue  # Skip slides with no valid taxonomy
            
            # Load JSONL and filter by sample_ids
            with open(caption_file) as f:
                for line_num, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                        if record["id"] in sample_ids:
                            bbox = record["bbox"]
                            self.samples.append({
                                "id": record["id"],
                                "slide": slide_name,
                                "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                                "species": species,
                                "wsi_path": wsi_path,
                            })
                    except json.JSONDecodeError:
                        continue
    
    def _find_caption_file(self, slide_name: str) -> Optional[Path]:
        """Find caption JSONL for a slide."""
        for dataset in self.datasets:
            caption_path = self.caption_dir / dataset / self.caption_model / f"{slide_name}_captions.jsonl"
            if caption_path.exists():
                return caption_path
        return None
    
    def _find_wsi_path(self, slide_name: str) -> Optional[str]:
        """Find WSI file path for a slide."""
        # Check common extensions and dataset subdirectories
        for dataset in ["french", "hungarian", "mediterranean", "swedish"]:
            for ext in [".tif", ".tiff", ".svs", ".ndpi"]:
                wsi_path = self.wsi_dir / dataset / f"{slide_name}{ext}"
                if wsi_path.exists():
                    return str(wsi_path)
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Extract patch from WSI
        image = self._extract_patch(sample)
        
        # Apply stain/histogram normalization if enabled (PIL → PIL)
        if self.stainnorm_func is not None and image is not None:
            try:
                image = self.stainnorm_func(image)
            except Exception:
                pass  # Keep original on failure
        
        # Apply transforms
        if self.transform is not None and image is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image) if image is not None else torch.zeros(3, 224, 224)
        
        # Get label from HITL-validated species field (NOT class_id or class_name)
        species = sample["species"]
        label = self.species_to_int.get(species.lower(), self.num_classes - 1)  # default to Unknown
        
        return {
            "image": image,
            "label": label,
            "sample_id": sample["id"],
            "species": sample["species"],
        }
    
    def _get_wsi_handle(self, wsi_path: str):
        """Get or create cached WSI handle with LRU eviction (max 20 slides per worker)."""
        MAX_CACHE_SIZE = 20  # Limit memory per worker
        
        if not hasattr(self, '_wsi_cache'):
            self._wsi_cache = {}
            self._wsi_access_order = []
        
        if wsi_path in self._wsi_cache:
            # Move to end (most recently used)
            if wsi_path in self._wsi_access_order:
                self._wsi_access_order.remove(wsi_path)
            self._wsi_access_order.append(wsi_path)
            return self._wsi_cache[wsi_path]
        
        # Evict oldest if cache is full
        while len(self._wsi_cache) >= MAX_CACHE_SIZE and self._wsi_access_order:
            oldest = self._wsi_access_order.pop(0)
            if oldest in self._wsi_cache:
                try:
                    self._wsi_cache[oldest].close()
                except:
                    pass
                del self._wsi_cache[oldest]
        
        # Open new WSI
        try:
            import tiffslide
            self._wsi_cache[wsi_path] = tiffslide.TiffSlide(wsi_path)
        except Exception:
            import openslide
            self._wsi_cache[wsi_path] = openslide.OpenSlide(wsi_path)
        
        self._wsi_access_order.append(wsi_path)
        return self._wsi_cache[wsi_path]
    
    def _extract_patch(self, sample: Dict) -> Optional[Image.Image]:
        """
        Extract patch from WSI using the exact bounding box.
        
        The bbox already defines the annotation region precisely.
        We extract the full bbox and resize to img_size (518) during transform.
        No fixed-size cropping - each grain may have different native size.
        Uses per-worker WSI handle cache to avoid repeated file opens.
        """
        wsi_path = sample.get("wsi_path")
        if wsi_path is None or not os.path.exists(wsi_path):
            return Image.new("RGB", (224, 224), (255, 255, 255))
        
        try:
            wsi = self._get_wsi_handle(wsi_path)
            
            x1, y1, x2, y2 = sample["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            # Read the exact bbox region (no fixed-size cropping)
            region = wsi.read_region((x1, y1), 0, (width, height))
            # Don't close - keep handle cached for reuse
            
            return region.convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to extract patch for {sample['id']}: {e}")
            return Image.new("RGB", (224, 224), (255, 255, 255))
    
    def __del__(self):
        """Clean up cached WSI handles."""
        if hasattr(self, '_wsi_cache'):
            for wsi in self._wsi_cache.values():
                try:
                    wsi.close()
                except:
                    pass
            self._wsi_cache.clear()
    
    def get_class_counts(self) -> Dict[int, int]:
        """Return count per class for class balancing (using HITL-validated species)."""
        counts = Counter()
        for s in self.samples:
            species = s["species"]
            label = self.species_to_int.get(species.lower(), self.num_classes - 1)
            counts[label] += 1
        return dict(counts)


# =============================================================================
# MODEL
# =============================================================================

class PollenClassifierModel(nn.Module):
    """
    Classifier model with ViT backbone and linear head.
    Supports frozen backbone (linear probing) and full finetuning.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        num_classes: int = 46,  # HITL-validated 45 species + Unknown
        img_size: int = 518,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        # Determine embedding dimension
        if 'small' in backbone_name:
            self.embed_dim = 384
        elif 'base' in backbone_name:
            self.embed_dim = 768
        elif 'large' in backbone_name:
            self.embed_dim = 1024
        else:
            self.embed_dim = 384
        
        # Build backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=(checkpoint_path is None),
            img_size=img_size,
            init_values=1e-5,
            num_classes=0,  # Remove default head
        )
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Freeze backbone if requested
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
        
        # Classification head
        self.head = nn.Linear(self.embed_dim, num_classes)
        
    def _load_checkpoint(self, checkpoint_path: str):
        """Load pretrained weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter out head weights (we're reinitializing the head)
        backbone_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"[Model] Loaded backbone from {checkpoint_path}")
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for finetuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("[Model] Backbone unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match Option D: autocast only frozen backbone, keep head/loss in FP32.
        with torch.amp.autocast('cuda', enabled=self.freeze_backbone and x.is_cuda):
            features = self.backbone(x)
        features = features.float()
        logits = self.head(features)
        return logits


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_run_name(run_name: Optional[str]) -> Optional[str]:
    """Normalize run names for filesystem-safe subfolder naming."""
    if run_name is None:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in run_name.strip())
    return cleaned or None


def build_run_subdir(seed: Optional[int] = None, run_name: Optional[str] = None) -> Optional[str]:
    """Build deterministic run subfolder name from seed and optional tag."""
    clean_name = sanitize_run_name(run_name)
    if seed is None and clean_name is None:
        return None
    parts = []
    if seed is not None:
        parts.append(f"seed_{seed}")
    if clean_name is not None:
        parts.append(clean_name)
    return "__".join(parts)


def resolve_output_dir(base_output_dir: Path, seed: Optional[int] = None, run_name: Optional[str] = None) -> Path:
    """Resolve output dir, optionally creating seed/run subfolders."""
    run_subdir = build_run_subdir(seed, run_name)
    if run_subdir is None:
        return base_output_dir
    return base_output_dir / run_subdir


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(device_str: str, use_multi_gpu: bool) -> Tuple[torch.device, bool, int, int]:
    """Initialize DDP when launched with torchrun (WORLD_SIZE>1)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1

    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return device, True, rank, world_size

    return torch.device(device_str), False, 0, 1


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def reduce_metrics_sums(values: List[float], device: torch.device) -> List[float]:
    """All-reduce metric sums across DDP ranks."""
    if not is_dist_initialized():
        return values
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


class DistributedWeightedSampler(Sampler[int]):
    """
    Distributed equivalent of WeightedRandomSampler.

    It draws weighted samples globally, then shards by rank so each process
    receives a disjoint stride slice.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 42,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples_global = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = replacement
        self.seed = int(seed)
        self.epoch = 0

        self.num_samples_per_rank = int(math.ceil(self.num_samples_global / self.num_replicas))
        self.total_size = self.num_samples_per_rank * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=g,
        ).tolist()
        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def get_train_transform(img_size: int, config: Dict, global_config: Dict = None) -> transforms.Compose:
    """Build training transforms with augmentation.
    
    Augmentations based on Tif_clustering_embedder_finetuning.ipynb:
    - RandomHorizontalFlip (p=0.5)
    - RandomVerticalFlip (p=0.5)
    - RandomRotation (degrees=30)
    - RandomGrayscale (p=0.2) - removes color info to improve robustness
    - ColorJitter (brightness, contrast, saturation, hue)
    - RandomAffine (translate, scale)
    - GaussianBlur (optional)
    
    Args:
        img_size: Target image size
        config: Training config with augmentation settings
        global_config: Global config with normalization settings
    """
    aug_config = config.get("augmentation", {})
    
    # Get normalization values from global config or use ImageNet defaults
    if global_config is not None:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    
    # Flips
    if aug_config.get("random_horizontal_flip", True):
        p = aug_config.get("horizontal_flip_p", 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=p))
    
    if aug_config.get("random_vertical_flip", True):
        p = aug_config.get("vertical_flip_p", 0.5)
        transform_list.append(transforms.RandomVerticalFlip(p=p))
    
    # Rotation
    rotation = aug_config.get("random_rotation", 30)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    
    # Random grayscale (removes color, improves robustness to staining variations)
    grayscale_p = aug_config.get("random_grayscale", 0.2)
    if grayscale_p > 0:
        transform_list.append(transforms.RandomGrayscale(p=grayscale_p))
    
    # Color jitter (brightness, contrast, saturation, hue)
    jitter = aug_config.get("color_jitter", {})
    if isinstance(jitter, (int, float)) and jitter > 0:
        # Legacy support: single value for all except hue
        transform_list.append(transforms.ColorJitter(
            brightness=jitter, contrast=jitter, saturation=jitter, hue=min(jitter, 0.1)
        ))
    elif isinstance(jitter, dict) and any(jitter.values()):
        transform_list.append(transforms.ColorJitter(
            brightness=jitter.get("brightness", 0.3),
            contrast=jitter.get("contrast", 0.3),
            saturation=jitter.get("saturation", 0.3),
            hue=jitter.get("hue", 0.1),
        ))
    
    # Random affine (translate + scale)
    affine = aug_config.get("random_affine", {})
    if isinstance(affine, dict) and affine.get("enabled", False):
        transform_list.append(transforms.RandomAffine(
            degrees=affine.get("degrees", 0),
            translate=tuple(affine.get("translate", [0.1, 0.1])),
            scale=tuple(affine.get("scale", [0.9, 1.1])),
        ))
    
    # Gaussian blur
    blur = aug_config.get("gaussian_blur", {})
    if isinstance(blur, dict) and blur.get("enabled", False):
        kernel_size = blur.get("kernel_size", [3, 5])
        sigma = blur.get("sigma", [0.1, 2.0])
        transform_list.append(transforms.GaussianBlur(
            kernel_size=tuple(kernel_size), sigma=tuple(sigma)
        ))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    return transforms.Compose(transform_list)


def get_val_transform(img_size: int, global_config: Dict = None) -> transforms.Compose:
    """Build validation transforms (no augmentation).
    
    Args:
        img_size: Target image size
        global_config: Global config with normalization settings
    """
    # Get normalization values from global config or use ImageNet defaults
    if global_config is not None:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def compute_class_weights(class_counts: Dict[int, int], num_classes: int, method: str = "sqrt") -> torch.Tensor:
    """Compute class weights for balanced training.
    
    Args:
        class_counts: Dict mapping class_id -> count
        num_classes: Total number of classes
        method: Balancing method:
            - 'sqrt': inverse sqrt of counts (default, recommended)
            - 'inverse': inverse of counts (strong balancing)
            - 'balanced': same as inverse
            - 'none': uniform weights
    """
    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)])
    
    if method == "sqrt":
        weights = 1.0 / np.sqrt(counts + 1)
    elif method in ("inverse", "balanced"):
        weights = 1.0 / (counts + 1)
    else:  # 'none' or unknown
        weights = np.ones(num_classes)
    
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def create_balanced_sampler(
    dataset: Dataset, 
    balancing_method: str = "sqrt",
    samples_per_epoch: Optional[int] = None
) -> torch.utils.data.Sampler:
    """Create a balanced sampler for class-imbalanced data.
    
    Uses sqrt-based sampling where:
    - Minority class uses all its samples
    - Other classes are downsampled proportionally using sqrt ratio
    
    Args:
        dataset: Dataset with samples
        balancing_method: "sqrt" (default), "uniform", or "none"
            - sqrt: sqrt(count) weighting, minority determines epoch size
            - uniform: equal samples per class (all get minority count)
            - none: use class frequencies as-is
        samples_per_epoch: Optional override for epoch size.
            If None, computed from class distribution.
    """
    # Get labels using species-based mapping
    labels = []
    for s in dataset.samples:
        species = s["species"]
        label = dataset.species_to_int.get(species.lower(), dataset.num_classes - 1)
        labels.append(label)
    
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    
    # Compute target samples per class based on balancing method
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    if balancing_method == "sqrt":
        # Sqrt-based: minority uses all samples, others scaled by sqrt ratio
        # Target per class = min_count * sqrt(class_count / min_count)
        # This gives minority all its samples, and reduces large classes by sqrt
        target_per_class = {}
        for label, count in class_counts.items():
            ratio = count / min_count
            target = int(min_count * np.sqrt(ratio))
            # Cap at actual count (can't sample more than available)
            target_per_class[label] = min(target, count)
        
    elif balancing_method == "uniform":
        # Uniform: all classes get same number (minority count)
        target_per_class = {label: min_count for label in class_counts}
        
    else:  # none
        # Use natural distribution
        target_per_class = dict(class_counts)
    
    # Compute total samples per epoch
    computed_epoch_size = sum(target_per_class.values())
    
    if samples_per_epoch is not None:
        num_samples = samples_per_epoch
    else:
        num_samples = computed_epoch_size
    
    # Compute sample weights based on target distribution
    # Weight = target_count / actual_count for that class
    weights = []
    for label in labels:
        actual = class_counts[label]
        target = target_per_class[label]
        # Higher weight for underrepresented samples
        weights.append(target / actual)
    weights = torch.FloatTensor(weights)
    
    # Print stats
    if is_main_process():
        print(f"[Sampler] Balancing: {balancing_method}")
        print(f"[Sampler] Class range: {min_count:,} - {max_count:,} ({max_count/min_count:.0f}x imbalance)")
        print(f"[Sampler] Target epoch: {computed_epoch_size:,} samples")
        if samples_per_epoch:
            print(f"[Sampler] Override epoch: {num_samples:,} samples")

        # Show a few target counts
        sorted_targets = sorted(target_per_class.items(), key=lambda x: x[1])
        print(f"[Sampler] Min target: class {sorted_targets[0][0]} → {sorted_targets[0][1]:,}")
        print(f"[Sampler] Max target: class {sorted_targets[-1][0]} → {sorted_targets[-1][1]:,}")
    
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)
    
    return sampler


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=not is_main_process())
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.1f}%"
        })
        
        # Periodic GC to prevent system RAM accumulation from DataLoader workers
        if (batch_idx + 1) % 50 == 0:
            import gc
            gc.collect()
    
    running_loss, correct, total = reduce_metrics_sums(
        [running_loss, float(correct), float(total)],
        device,
    )

    epoch_loss = running_loss / max(total, 1.0)
    epoch_acc = 100 * correct / max(total, 1.0)
    
    return {"train_loss": epoch_loss, "train_acc": epoch_acc}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    correct_top5 = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", disable=not is_main_process())
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # Top-1
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Top-5
            _, top5_pred = torch.topk(logits, 5, dim=1)
            for i, label in enumerate(labels):
                if label in top5_pred[i]:
                    correct_top5 += 1
            
            # Periodic GC to prevent system RAM accumulation
            if (batch_idx + 1) % 50 == 0:
                import gc
                gc.collect()
    
    running_loss, correct, total, correct_top5 = reduce_metrics_sums(
        [running_loss, float(correct), float(total), float(correct_top5)],
        device,
    )

    epoch_loss = running_loss / max(total, 1.0)
    epoch_acc = 100 * correct / max(total, 1.0)
    epoch_top5 = 100 * correct_top5 / max(total, 1.0)
    
    return {
        "val_loss": epoch_loss,
        "val_acc": epoch_acc,
        "val_top5_acc": epoch_top5
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(config: Dict, experiment_name: str, device: str = "cuda:0",
          max_samples: int = None, epochs_override: int = None,
          seed_override: Optional[int] = None, run_name: Optional[str] = None):
    """Main training function.
    
    Args:
        config: Full experiment configuration
        experiment_name: Name of experiment in config
        device: Device string (cuda:0, cpu, etc.)
        max_samples: Limit samples per dataset split (for quick testing)
        epochs_override: Override number of epochs (for testing)
        seed_override: Override random seed (outputs saved to seed subfolder)
        run_name: Optional run tag appended to seed subfolder
    """
    
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    train_config = exp_config["training"]
    
    # Set seed
    seed = global_config.get("seed", 42) if seed_override is None else seed_override
    set_seed(seed)

    use_multi_gpu = train_config.get("multi_gpu", False)
    device, use_ddp, rank, world_size = setup_distributed(device, use_multi_gpu)
    main_process = is_main_process()

    if main_process:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_config['name']}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        if use_ddp:
            print(f"DDP: enabled (rank {rank}/{world_size})")
        print(f"{'='*60}\n")
    
    # Paths
    data_root = PROJECT_ROOT / global_config["data_root"]
    splits_dir = PROJECT_ROOT / global_config["splits_dir"]
    base_output_dir = PROJECT_ROOT / exp_config["output_dir"]
    output_dir = resolve_output_dir(base_output_dir, seed_override, run_name)
    run_subdir = build_run_subdir(seed_override, run_name)
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if run_subdir is not None:
            print(f"[Run] Using output subfolder: {run_subdir}")
        print(f"[Run] Output directory: {output_dir}")
    
    # Get datasets for this experiment
    train_region = train_config["train_region"]
    region_config = config["regions"][train_region]
    datasets = region_config["datasets"]
    
    # Build transforms
    # For experiments with custom normalization (stainnorm), use experiment-specific values
    # Otherwise fall back to global (ImageNet) defaults
    img_size = global_config["img_size"]
    preprocessing_config = exp_config.get("preprocessing", {})
    
    # Merge preprocessing config into global config for transforms
    # This ensures experiment-specific normalize_mean/std are used when defined
    transform_config = global_config.copy()
    if preprocessing_config.get("normalization_mode") == "custom":
        if "normalize_mean" in preprocessing_config:
            transform_config["normalize_mean"] = preprocessing_config["normalize_mean"]
        if "normalize_std" in preprocessing_config:
            transform_config["normalize_std"] = preprocessing_config["normalize_std"]
        print(f"[Preprocessing] Using custom normalization:")
        print(f"  Mean: {transform_config.get('normalize_mean')}")
        print(f"  Std: {transform_config.get('normalize_std')}")
    else:
        print(f"[Preprocessing] Using ImageNet normalization (default)")
    
    train_transform = get_train_transform(img_size, train_config, transform_config)
    val_transform = get_val_transform(img_size, transform_config)
    
    # Setup stain/histogram normalization if enabled
    stainnorm_func = None
    try:
        stainnorm_func = get_preprocessor(exp_config)
        if stainnorm_func is not None:
            print(f"[Preprocessing] Enabled: {type(stainnorm_func).__name__}")
    except FileNotFoundError as e:
        print(f"[Warning] Preprocessing skipped: {e}")
    except ImportError as e:
        print(f"[Warning] Preprocessing skipped (missing dependency): {e}")
    
    # Load HITL-validated species mappings
    all_anchors = load_caption_anchors()
    caption_anchors, excluded_slides = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    
    print(f"[Setup] Loaded {len(all_anchors)} slides from caption_anchors/")
    print(f"[Setup] Using {len(caption_anchors)} trainable slides ({len(excluded_slides)} Unknown excluded)")
    if excluded_slides:
        print(f"[Setup] Excluded slides: {', '.join(excluded_slides)}")
    
    # Verify num_classes matches config
    config_num_classes = global_config.get("num_classes", 46)
    if config_num_classes != num_classes:
        print(f"[WARNING] Config num_classes ({config_num_classes}) doesn't match caption_anchors ({num_classes})!")
        print(f"[WARNING] Using {num_classes} from caption_anchors (source of truth)")
    
    # Build datasets (load all samples, let sampler control epoch size)
    # max_samples is for quick debugging - limits samples per split
    print("\nLoading datasets...")
    if max_samples is not None:
        print(f"[DEBUG] Using max_samples={max_samples} for quick testing")
    
    train_dataset = PollenClassificationDataset(
        split_dir=splits_dir,
        caption_dir=data_root / "03_captioning",
        wsi_dir=data_root / "00_raw_wsi",
        split="train",
        transform=train_transform,
        datasets=datasets,
        stainnorm_func=stainnorm_func,
        max_samples=max_samples,  # For quick testing; None loads all
        species_to_int=species_to_int,
        num_classes=num_classes,
    )
    
    val_dataset = PollenClassificationDataset(
        split_dir=splits_dir,
        caption_dir=data_root / "03_captioning",
        wsi_dir=data_root / "00_raw_wsi",
        split="val",
        transform=val_transform,
        datasets=datasets,
        stainnorm_func=stainnorm_func,
        max_samples=max_samples,  # For quick testing; None loads all
        species_to_int=species_to_int,
        num_classes=num_classes,
    )
    
    # Class balancing (loss weights)
    class_counts = train_dataset.get_class_counts()
    balancing = train_config.get("class_balancing", "none")
    use_balanced_sampling = train_config.get("balanced_sampling", False)
    
    if use_balanced_sampling and balancing != "none":
        print(f"[Note] Using BOTH balanced sampling ({train_config.get('sampling_method', 'sqrt')}) "
              f"AND loss weighting ({balancing}). This is intentional for strong imbalance correction.")
    
    class_weights = compute_class_weights(class_counts, num_classes, balancing).to(device)
    
    # Create experiment metadata for tracking
    metadata = create_experiment_metadata(
        config=config,
        experiment_name=experiment_name,
        train_dataset_size=len(train_dataset),
        val_dataset_size=len(val_dataset),
        class_counts=class_counts,
        int_to_species=int_to_species,
        max_samples=max_samples,
        epochs_override=epochs_override,
        seed=seed,
    )
    
    # Save initial metadata (in case training crashes)
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Create dataloaders with simple settings (NVMe RAID is fast)
    batch_size = train_config.get("batch_size", 64)
    # Per-experiment num_workers overrides global default (useful for RAM-heavy stainnorm)
    num_workers = train_config.get("num_workers", global_config.get("num_workers", 8))
    
    dataloader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }
    train_sampler = None
    
    # Optionally use balanced sampling
    use_balanced_sampling = train_config.get("balanced_sampling", False)
    if use_balanced_sampling:
        balancing_method = train_config.get("sampling_method", "sqrt")  # sqrt, uniform, none
        samples_per_epoch = train_config.get("samples_per_epoch")  # None = computed from class dist
        base_sampler = create_balanced_sampler(train_dataset, balancing_method, samples_per_epoch)
        if use_ddp:
            train_sampler = DistributedWeightedSampler(
                weights=base_sampler.weights,
                num_samples=base_sampler.num_samples,
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=seed,
            )
            effective_drop_last = (len(train_sampler) >= batch_size)
        else:
            train_sampler = base_sampler
            effective_drop_last = True
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,  # Don't shuffle when using sampler
            drop_last=effective_drop_last,
            **dataloader_kwargs,
        )
    else:
        if use_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            effective_drop_last = (len(train_sampler) >= batch_size)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                drop_last=effective_drop_last,
                **dataloader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                **dataloader_kwargs,
            )
    
    # Validation uses same batch size as training for consistency
    if use_ddp:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,  # Same as training
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    if main_process:
        print(f"[DataLoader] Train batch={batch_size}, {len(train_loader)} batches/epoch")
        print(f"[DataLoader] Val batch={batch_size}, {len(val_loader)} batches")
    
    # Build model
    # If backbone_checkpoint is null/None, use pretrained=True from timm
    backbone_ckpt = global_config.get("backbone_checkpoint")
    if backbone_ckpt:
        checkpoint_path = PROJECT_ROOT / backbone_ckpt
        if checkpoint_path.exists():
            checkpoint_path = str(checkpoint_path)
            print(f"[Model] Using custom checkpoint: {backbone_ckpt}")
        else:
            print(f"[WARNING] Backbone checkpoint not found: {checkpoint_path}")
            print(f"[WARNING] Falling back to pretrained LVD weights from timm")
            checkpoint_path = None
    else:
        checkpoint_path = None
        print("[Model] Using original pretrained LVD weights from timm")
    
    model = PollenClassifierModel(
        backbone_name=global_config["backbone_name"],
        num_classes=num_classes,  # Use caption_anchors count (source of truth), not config
        img_size=img_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=train_config.get("freeze_backbone", True),
    )
    
    # Multi-GPU support
    num_gpus = torch.cuda.device_count()
    if use_ddp:
        model = model.to(device)
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index], output_device=device.index)
        else:
            model = DDP(model)
        if main_process:
            print(f"[Model] Using DDP with world_size={world_size}")
            print(f"[Model] Effective global batch size: {batch_size * world_size} ({batch_size} per rank)")
    elif use_multi_gpu and num_gpus > 1:
        model = model.to(device)
        model = nn.DataParallel(model)
        if main_process:
            print(f"[Model] Using DataParallel with {num_gpus} GPUs")
            print(f"[Model] Effective global batch size: {batch_size}")
    else:
        model = model.to(device)
        if use_multi_gpu and num_gpus <= 1 and main_process:
            print(f"[Model] multi_gpu=True but only {num_gpus} GPU(s) available, using single GPU")
    
    # Optimizer - handle parallel wrappers
    lr = train_config.get("learning_rate", 0.001)
    weight_decay = train_config.get("weight_decay", 0.0001)
    
    # Get underlying model if wrapped in DataParallel/DDP
    base_model = unwrap_model(model)
    
    if train_config.get("freeze_backbone", True):
        # Only optimize head
        optimizer = torch.optim.AdamW(base_model.head.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    epochs = epochs_override if epochs_override else train_config.get("epochs", 30)
    warmup_epochs = train_config.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    
    # Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top5_acc": []}
    best_val_acc = 0.0
    unfreeze_epoch = train_config.get("unfreeze_after_epoch")
    checkpoint_every = train_config.get("checkpoint_every", 5)  # Save checkpoint every N epochs
    
    # Initialize incremental CSV logger
    csv_logger = None
    if main_process:
        csv_logger = CSVLogger(
            output_dir / "training_log.csv",
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5_acc",
                        "lr", "elapsed_sec", "timestamp"]
        )
        print(f"\nTraining for {epochs} epochs...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}, LR: {lr}")
        print(f"Backbone frozen: {train_config.get('freeze_backbone', True)}")
        print(f"Checkpoint every: {checkpoint_every} epochs")
        if unfreeze_epoch:
            print(f"Will unfreeze backbone after epoch {unfreeze_epoch}")
        print()
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # Check if we should unfreeze backbone
        if unfreeze_epoch and epoch == unfreeze_epoch + 1:
            base_model.unfreeze_backbone()
            # Add backbone params to optimizer with lower LR
            backbone_lr = lr * train_config.get("backbone_lr_multiplier", 0.1)
            optimizer = torch.optim.AdamW([
                {"params": base_model.backbone.parameters(), "lr": backbone_lr},
                {"params": base_model.head.parameters(), "lr": lr},
            ], weight_decay=weight_decay)
            # Recreate scheduler with new optimizer for remaining epochs
            # Note: No warmup subtraction - warmup already completed in frozen phase
            remaining_epochs = epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, remaining_epochs)
            )
            if main_process:
                print(f"[Epoch {epoch}] Unfroze backbone, backbone_lr={backbone_lr}, new T_max={remaining_epochs}")
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Force garbage collection between train and val to release any cached objects
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Skip validation during warmup epochs (save time)
        skip_val_until = train_config.get("skip_val_until_epoch", 0)
        if epoch < skip_val_until:
            if main_process:
                print(f"[Epoch {epoch}] Skipping validation (warmup, will start at epoch {skip_val_until})")
            val_metrics = {"val_loss": float('inf'), "val_acc": 0.0, "val_top5_acc": 0.0}
        else:
            # Validate
            val_metrics = validate(model, val_loader, criterion, device, epoch)
            
            # Force garbage collection after validation
            gc.collect()
            torch.cuda.empty_cache()
        
        # Update scheduler
        if epoch > warmup_epochs:
            scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        
        # Log to history
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_acc"].append(train_metrics["train_acc"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["val_top5_acc"].append(val_metrics["val_top5_acc"])
        
        # Log to CSV (incremental, survives interruption)
        if main_process and csv_logger is not None:
            csv_logger.log({
                "epoch": epoch,
                "train_loss": f"{train_metrics['train_loss']:.4f}",
                "train_acc": f"{train_metrics['train_acc']:.2f}",
                "val_loss": f"{val_metrics['val_loss']:.4f}",
                "val_acc": f"{val_metrics['val_acc']:.2f}",
                "val_top5_acc": f"{val_metrics['val_top5_acc']:.2f}",
                "lr": f"{current_lr:.6f}",
                "elapsed_sec": f"{elapsed:.1f}",
                "timestamp": datetime.now().isoformat(),
            })

        if main_process:
            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                  f"train_acc={train_metrics['train_acc']:.1f}%, "
                  f"val_acc={val_metrics['val_acc']:.1f}%, "
                  f"val_top5={val_metrics['val_top5_acc']:.1f}%")
        
        # Save best model (use base_model to avoid DataParallel wrapper in state_dict)
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            if main_process:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "num_classes": num_classes,  # Store for eval compatibility
                    "int_to_species": int_to_species,  # Store for label decoding
                    "config": exp_config,
                }, output_dir / "best_model.pth")
                print(f"  → Saved best model (val_acc={best_val_acc:.1f}%)")
        
        # Periodic checkpoint (every N epochs, and always on last epoch)
        if (epoch % checkpoint_every == 0 or epoch == epochs) and main_process:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch:03d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_metrics["val_acc"],
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "int_to_species": int_to_species,
                "config": exp_config,
                "history": history,
            }, checkpoint_path)
            print(f"  → Saved checkpoint: {checkpoint_path.name}")
    
    # Close CSV logger
    if csv_logger is not None:
        csv_logger.close()
    
    # Save final model
    if main_process:
        torch.save({
            "epoch": epochs,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_metrics["val_acc"],
            "num_classes": num_classes,  # Store for eval compatibility
            "int_to_species": int_to_species,  # Store for label decoding
            "config": exp_config,
        }, output_dir / "final_model.pth")
    
    # Save complete experiment results with full traceability
    training_time = time.time() - start_time
    history["training_time_seconds"] = training_time
    history["best_val_acc"] = best_val_acc
    
    if main_process:
        save_experiment_results(
            output_dir=output_dir,
            metadata=metadata,
            history=history,
            best_val_acc=best_val_acc,
            training_time=training_time,
        )

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Training time: {training_time/3600:.1f} hours")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}\n")

    cleanup_distributed()
    
    return model, history


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train pollen classifier")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name from config")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split for quick testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed. Outputs saved to seed subfolder.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run tag appended to seed subfolder.")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Validate experiment exists
    if args.experiment not in config["experiments"]:
        available = list(config["experiments"].keys())
        print(f"Error: Experiment '{args.experiment}' not found.")
        print(f"Available experiments: {available}")
        sys.exit(1)
    
    # Run training
    train(config, args.experiment, args.device, 
          max_samples=args.max_samples, 
          epochs_override=args.epochs,
          seed_override=args.seed,
          run_name=args.run_name)


if __name__ == "__main__":
    main()
