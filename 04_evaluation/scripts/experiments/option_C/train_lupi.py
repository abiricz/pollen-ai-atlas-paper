#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
LUPI (Learning Using Privileged Information) Training Script
=============================================================

Train a classifier with image + caption (privileged information).
Evaluate with image only (captions NOT available at test time).

Architecture:
    TRAINING:  ViT(image) → [img_embed(384) ; sbert_embed(384)] → Linear(768→46)
    TESTING:   ViT(image) → [img_embed(384) ; zeros(384)]       → Linear(768→46)

This mirrors Option A's training script exactly, with two additions:
1. Dataset loads pre-computed SBERT caption embeddings
2. Model concatenates image + text embeddings before the linear head

The evaluation is IDENTICAL to Option A — same evaluate_classifier.py, same metrics.
The LUPI model's checkpoint stores the model in a format compatible with Option A's
evaluation pipeline (backbone + head, with head input dimension = 768).

Usage:
    # Full training
    python train_lupi.py --config ../experiment_config.yaml --experiment lupi_all

    # Quick smoke test
    python train_lupi.py --config ../experiment_config.yaml --experiment lupi_all \\
        --max_samples 500 --epochs 3

Dependencies:
    pip install sentence-transformers h5py
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
import h5py
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

# Add project root (4 levels up from option_C/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Use HITL-validated species mapping
from lib.species_mapping import load_caption_anchors, build_class_mappings, get_slide_class_id, get_trainable_slides

# Optional: Macenko stain normalization
try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    HAS_TIATOOLBOX = True
except ImportError:
    HAS_TIATOOLBOX = False
    MacenkoNormalizer = None


# =============================================================================
# STAIN/HISTOGRAM NORMALIZATION PREPROCESSING (identical to Option A)
# =============================================================================

class StainNormPreprocessor:
    """Apply Macenko stain normalization to images."""
    
    def __init__(self, reference_path: str):
        if not HAS_TIATOOLBOX:
            raise ImportError("tiatoolbox required for stain normalization.")
        reference_path = Path(reference_path)
        if not reference_path.exists():
            raise FileNotFoundError(f"Stain reference not found: {reference_path}")
        self.reference_img = np.load(reference_path)
        if self.reference_img.dtype != np.uint8:
            self.reference_img = (self.reference_img * 255).astype(np.uint8)
        self.normalizer = MacenkoNormalizer()
        self.normalizer.fit(self.reference_img)
        print(f"[StainNorm] Initialized with reference: {reference_path.name}")
    
    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        try:
            normalized = self.normalizer.transform(img_np)
            return Image.fromarray(normalized)
        except Exception:
            return img


class HistogramNormPreprocessor:
    """Apply histogram normalization to images."""
    
    def __init__(self, mean: List[float], std: List[float]):
        self.target_mean = np.array(mean)
        self.target_std = np.array(std)
        print(f"[HistNorm] Target mean={mean}, std={std}")
    
    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img).astype(np.float32)
        for c in range(3):
            channel = img_np[:, :, c]
            ch_mean = channel.mean()
            ch_std = channel.std() + 1e-6
            normalized = (channel - ch_mean) / ch_std
            normalized = normalized * self.target_std[c] + self.target_mean[c]
            img_np[:, :, c] = np.clip(normalized, 0, 255)
        return Image.fromarray(img_np.astype(np.uint8))


def get_preprocessor(config: Dict) -> Optional[callable]:
    """Create preprocessor from config."""
    preprocessing = config.get("preprocessing", {})
    stainnorm = preprocessing.get("stainnorm", {})
    if stainnorm.get("enabled", False):
        reference_path = stainnorm.get("reference_path")
        if reference_path:
            return StainNormPreprocessor(str(PROJECT_ROOT / reference_path))
    hist_match = preprocessing.get("hist_match", {})
    if hist_match.get("enabled", False):
        mean = hist_match.get("training_mean") or hist_match.get("reference_mean")
        std = hist_match.get("training_std") or hist_match.get("reference_std")
        if mean and std:
            return HistogramNormPreprocessor(mean, std)
    return None


# =============================================================================
# CSV LOGGER (identical to Option A)
# =============================================================================

class CSVLogger:
    """Incremental CSV logger that writes per-epoch and survives interruption."""
    
    def __init__(self, filepath: Path, fieldnames: List[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        is_new = not filepath.exists()
        self.file = open(filepath, 'a', newline='', buffering=1)
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if is_new:
            self.writer.writeheader()
            self.file.flush()
    
    def log(self, row: Dict[str, Any]):
        self.writer.writerow(row)
        self.file.flush()
        os.fsync(self.file.fileno())
    
    def close(self):
        if self.file:
            self.file.close()


# =============================================================================
# EXPERIMENT TRACKING (identical to Option A)
# =============================================================================

def get_git_info() -> Dict[str, str]:
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
    config: Dict, experiment_name: str,
    train_dataset_size: int, val_dataset_size: int,
    class_counts: Dict[int, int], int_to_species: Dict[int, str],
    max_samples: Optional[int] = None, epochs_override: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    train_config = exp_config["training"]
    lupi_config = exp_config.get("lupi", {})
    
    metadata = {
        "experiment": {
            "name": experiment_name,
            "display_name": exp_config.get("name", experiment_name),
            "description": exp_config.get("description", ""),
            "type": "LUPI (Option C)",
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
            "fusion_method": lupi_config.get("fusion_method", "concat"),
            "text_embed_dim": lupi_config.get("text_embed_dim", 384),
            "caption_dropout": lupi_config.get("caption_dropout", 0.3),
        },
        "data": {
            "train_region": train_config["train_region"],
            "train_samples": train_dataset_size,
            "val_samples": val_dataset_size,
            "class_distribution": {int_to_species.get(k, f"cls_{k}"): v 
                                   for k, v in sorted(class_counts.items())},
            "num_active_classes": len([v for v in class_counts.values() if v > 0]),
            "max_samples_override": max_samples,
            "caption_model": lupi_config.get("caption_model", "qwen25vl"),
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
        "lupi": lupi_config,
        "preprocessing": {
            "normalization_mode": exp_config.get("preprocessing", {}).get("normalization_mode", "imagenet"),
        },
        "augmentation": train_config.get("augmentation", {}),
        "eval_test_sets": exp_config.get("eval_test_sets", []),
        "output_dir": exp_config.get("output_dir", ""),
    }
    return metadata


def save_experiment_results(output_dir, metadata, history, best_val_acc, training_time):
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
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Tracking] Saved experiment metadata to {output_dir}")


# =============================================================================
# CAPTION EMBEDDING CACHE
# =============================================================================

class CaptionEmbeddingCache:
    """Load pre-computed SBERT embeddings from H5 files into a lookup dict.
    
    Each H5 file contains:
        - sample_ids: string array
        - embeddings: float32 array (N, 384)
    
    Provides O(1) lookup by sample_id.
    """
    
    def __init__(self, embedding_dir: Path, embed_dim: int = 384):
        self.embedding_dir = Path(embedding_dir)
        self.embed_dim = embed_dim
        self.cache = {}  # sample_id -> np.array(384,)
        self._loaded_slides = set()
    
    def load_all(self):
        """Load all H5 files into memory."""
        if not self.embedding_dir.exists():
            print(f"[CaptionCache] WARNING: Embedding dir not found: {self.embedding_dir}")
            return
        
        h5_files = list(self.embedding_dir.glob("*_embeddings.h5"))
        print(f"[CaptionCache] Loading {len(h5_files)} embedding files...")
        
        for h5_path in tqdm(h5_files, desc="Loading caption embeddings"):
            self._load_h5(h5_path)
        
        print(f"[CaptionCache] Loaded {len(self.cache):,} embeddings ({len(self._loaded_slides)} slides)")
    
    def load_for_slides(self, slide_names: set):
        """Load embeddings only for specific slides (memory efficient)."""
        for slide_name in slide_names:
            if slide_name in self._loaded_slides:
                continue
            h5_path = self.embedding_dir / f"{slide_name}_embeddings.h5"
            if h5_path.exists():
                self._load_h5(h5_path)
    
    def _load_h5(self, h5_path: Path):
        """Load a single H5 file."""
        try:
            with h5py.File(h5_path, "r") as hf:
                sample_ids = [s.decode() if isinstance(s, bytes) else s 
                              for s in hf["sample_ids"][:]]
                embeddings = hf["embeddings"][:]
                
                for sid, emb in zip(sample_ids, embeddings):
                    self.cache[sid] = emb
                
                slide_name = h5_path.stem.replace("_embeddings", "")
                self._loaded_slides.add(slide_name)
        except Exception as e:
            print(f"[CaptionCache] Failed to load {h5_path.name}: {e}")
    
    def get(self, sample_id: str) -> np.ndarray:
        """Get embedding for a sample. Returns zero vector if not found."""
        return self.cache.get(sample_id, np.zeros(self.embed_dim, dtype=np.float32))
    
    def __len__(self):
        return len(self.cache)
    
    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self.cache


# =============================================================================
# DATASET (extends Option A with caption embeddings)
# =============================================================================

class LUPIPollenDataset(Dataset):
    """
    Dataset for LUPI pollen classification.
    
    Same as Option A's PollenClassificationDataset but additionally loads
    pre-computed SBERT caption embeddings for each sample.
    """
    
    def __init__(
        self,
        split_dir: Path,
        caption_dir: Path,
        wsi_dir: Path,
        caption_cache: CaptionEmbeddingCache,
        split: str = "train",
        transform: transforms.Compose = None,
        datasets: List[str] = None,
        caption_model: str = "production_qwen25vl_final",
        stainnorm_func: callable = None,
        max_samples: int = None,
        species_to_int: Dict[str, int] = None,
        num_classes: int = 46,
        caption_dropout: float = 0.0,
        text_embed_dim: int = 384,
    ):
        self.split_dir = Path(split_dir) / split
        self.caption_dir = Path(caption_dir)
        self.wsi_dir = Path(wsi_dir)
        self.caption_cache = caption_cache
        self.split = split
        self.transform = transform
        self.datasets = datasets or ["french", "hungarian", "mediterranean", "swedish"]
        self.caption_model = caption_model
        self.stainnorm_func = stainnorm_func
        self.caption_dropout = caption_dropout
        self.text_embed_dim = text_embed_dim
        
        # HITL-validated species mapping
        caption_anchors = load_caption_anchors()
        if species_to_int is None:
            species_to_int, _, num_classes = build_class_mappings(caption_anchors)
        self.caption_anchors = caption_anchors
        self.species_to_int = species_to_int
        self.num_classes = num_classes
        
        # Build sample index (identical to Option A)
        self.samples = []
        self._build_sample_index()
        
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        # Count how many samples have embeddings
        n_with_emb = sum(1 for s in self.samples if s["id"] in self.caption_cache)
        print(f"[LUPIDataset] {split}: {len(self.samples)} samples, "
              f"{n_with_emb} with caption embeddings ({100*n_with_emb/max(1,len(self.samples)):.1f}%)")
        if self.caption_dropout > 0:
            print(f"[LUPIDataset] Caption dropout: {self.caption_dropout:.0%}")
        
        # Coverage gate: fail if <95% of training samples have embeddings
        coverage_pct = 100 * n_with_emb / max(1, len(self.samples))
        min_coverage = 95.0
        n_missing = len(self.samples) - n_with_emb
        if n_missing > 0:
            print(f"[WARNING] {n_missing} samples missing caption embeddings "
                  f"(will use zero vectors)")
        if split == "train" and coverage_pct < min_coverage and len(self.samples) > 100:
            raise RuntimeError(
                f"Caption embedding coverage too low: {coverage_pct:.1f}% < {min_coverage}% "
                f"({n_with_emb}/{len(self.samples)} samples). "
                f"Run embed_captions.py first or check embedding directory."
            )
    
    def _build_sample_index(self):
        """Load all sample IDs from split files and index to JSONL lines."""
        split_files = list(self.split_dir.glob(f"*_{self.split}.json"))
        
        for split_file in tqdm(split_files, desc=f"Loading {self.split} splits"):
            with open(split_file) as f:
                split_data = json.load(f)
            
            slide_name = split_data["slide"]
            sample_ids = set(split_data["sample_ids"])
            
            caption_file = self._find_caption_file(slide_name)
            if caption_file is None:
                continue
            
            wsi_path = self._find_wsi_path(slide_name)
            
            # Use HITL-validated species from _species.txt (genus level)
            # NOT from JSONL records which may have binomial names like "Betula sp."
            species = self.caption_anchors.get(slide_name, "Unknown")
            if species.lower() == "unknown":
                continue  # Skip slides with no valid taxonomy
            
            with open(caption_file) as f:
                for line in f:
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
        for dataset in self.datasets:
            caption_path = self.caption_dir / dataset / self.caption_model / f"{slide_name}_captions.jsonl"
            if caption_path.exists():
                return caption_path
        return None
    
    def _find_wsi_path(self, slide_name: str) -> Optional[str]:
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
        
        # Extract patch from WSI (identical to Option A)
        image = self._extract_patch(sample)
        
        if self.stainnorm_func is not None and image is not None:
            try:
                image = self.stainnorm_func(image)
            except Exception:
                pass
        
        if self.transform is not None and image is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image) if image is not None else torch.zeros(3, 224, 224)
        
        species = sample["species"]
        label = self.species_to_int.get(species.lower(), self.num_classes - 1)
        
        # Get caption embedding
        caption_emb = self.caption_cache.get(sample["id"])
        caption_emb = torch.from_numpy(caption_emb).float()
        
        # Apply caption dropout during training (zero out embedding)
        if self.split == "train" and self.caption_dropout > 0:
            if random.random() < self.caption_dropout:
                caption_emb = torch.zeros(self.text_embed_dim)
        
        return {
            "image": image,
            "label": label,
            "caption_embedding": caption_emb,
            "sample_id": sample["id"],
            "species": sample["species"],
        }
    
    def _get_wsi_handle(self, wsi_path: str):
        MAX_CACHE_SIZE = 20
        if not hasattr(self, '_wsi_cache'):
            self._wsi_cache = {}
            self._wsi_access_order = []
        if wsi_path in self._wsi_cache:
            if wsi_path in self._wsi_access_order:
                self._wsi_access_order.remove(wsi_path)
            self._wsi_access_order.append(wsi_path)
            return self._wsi_cache[wsi_path]
        while len(self._wsi_cache) >= MAX_CACHE_SIZE and self._wsi_access_order:
            oldest = self._wsi_access_order.pop(0)
            if oldest in self._wsi_cache:
                try:
                    self._wsi_cache[oldest].close()
                except:
                    pass
                del self._wsi_cache[oldest]
        try:
            import tiffslide
            self._wsi_cache[wsi_path] = tiffslide.TiffSlide(wsi_path)
        except Exception:
            import openslide
            self._wsi_cache[wsi_path] = openslide.OpenSlide(wsi_path)
        self._wsi_access_order.append(wsi_path)
        return self._wsi_cache[wsi_path]
    
    def _extract_patch(self, sample: Dict) -> Optional[Image.Image]:
        wsi_path = sample.get("wsi_path")
        if wsi_path is None or not os.path.exists(wsi_path):
            return Image.new("RGB", (224, 224), (255, 255, 255))
        try:
            wsi = self._get_wsi_handle(wsi_path)
            x1, y1, x2, y2 = sample["bbox"]
            width = x2 - x1
            height = y2 - y1
            region = wsi.read_region((x1, y1), 0, (width, height))
            return region.convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to extract patch for {sample['id']}: {e}")
            return Image.new("RGB", (224, 224), (255, 255, 255))
    
    def __del__(self):
        if hasattr(self, '_wsi_cache'):
            for wsi in self._wsi_cache.values():
                try:
                    wsi.close()
                except:
                    pass
            self._wsi_cache.clear()
    
    def get_class_counts(self) -> Dict[int, int]:
        counts = Counter()
        for s in self.samples:
            species = s["species"]
            label = self.species_to_int.get(species.lower(), self.num_classes - 1)
            counts[label] += 1
        return dict(counts)


# =============================================================================
# MODEL (LUPI variant — concat fusion)
# =============================================================================

class LUPIClassifierModel(nn.Module):
    """
    LUPI Classifier: ViT backbone + caption embedding → Linear head.
    
    Training:  [img_embed(384) ; caption_embed(384)] → Linear(768→46) → logits
    Inference: [img_embed(384) ; zeros(384)]          → Linear(768→46) → logits
    
    The head is 768→46 so that caption information influences training.
    At test time, the zero text embedding still allows image-only prediction.
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        num_classes: int = 46,
        img_size: int = 518,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
        text_embed_dim: int = 384,
    ):
        super().__init__()
        
        # Determine image embedding dimension
        if 'small' in backbone_name:
            self.img_embed_dim = 384
        elif 'base' in backbone_name:
            self.img_embed_dim = 768
        elif 'large' in backbone_name:
            self.img_embed_dim = 1024
        else:
            self.img_embed_dim = 384
        
        self.text_embed_dim = text_embed_dim
        self.fused_dim = self.img_embed_dim + self.text_embed_dim
        
        # Build backbone (identical to Option A)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=(checkpoint_path is None),
            img_size=img_size,
            init_values=1e-5,
            num_classes=0,
        )
        
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
        
        # LUPI head: takes concatenated [img_embed ; text_embed]
        self.head = nn.Linear(self.fused_dim, num_classes)
    
    def _load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        backbone_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"[Model] Loaded backbone from {checkpoint_path}")
    
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Model] Backbone frozen")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("[Model] Backbone unfrozen")
    
    def forward(self, x: torch.Tensor, text_embed: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional text embedding.
        
        Args:
            x: Image tensor (B, 3, 518, 518)
            text_embed: Caption embedding (B, 384), or None for inference
            
        Returns:
            logits: (B, num_classes)
        """
        # Match Option D: autocast only frozen backbone, keep head/loss in FP32.
        with torch.amp.autocast('cuda', enabled=self.freeze_backbone and x.is_cuda):
            img_features = self.backbone(x)  # (B, 384)
        img_features = img_features.float()
        
        if text_embed is None:
            # Inference mode: zero text embedding
            text_embed = torch.zeros(
                img_features.size(0), self.text_embed_dim,
                device=img_features.device, dtype=img_features.dtype
            )
        
        # Concatenate image + text features
        fused = torch.cat([img_features, text_embed], dim=1)  # (B, 768)
        logits = self.head(fused)  # (B, 46)
        
        return logits
    
    def forward_image_only(self, x: torch.Tensor) -> torch.Tensor:
        """Image-only forward pass (for evaluation compatibility)."""
        return self.forward(x, text_embed=None)


# =============================================================================
# TRAINING UTILITIES (identical to Option A)
# =============================================================================

def set_seed(seed: int):
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
    if not is_dist_initialized():
        return values
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


class DistributedWeightedSampler(Sampler[int]):
    """Distributed equivalent of WeightedRandomSampler."""

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
    aug_config = config.get("augmentation", {})
    if global_config is not None:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    transform_list = [transforms.Resize((img_size, img_size))]
    
    if aug_config.get("random_horizontal_flip", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)))
    if aug_config.get("random_vertical_flip", True):
        transform_list.append(transforms.RandomVerticalFlip(p=aug_config.get("vertical_flip_p", 0.5)))
    rotation = aug_config.get("random_rotation", 30)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    grayscale_p = aug_config.get("random_grayscale", 0.2)
    if grayscale_p > 0:
        transform_list.append(transforms.RandomGrayscale(p=grayscale_p))
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
    affine = aug_config.get("random_affine", {})
    if isinstance(affine, dict) and affine.get("enabled", False):
        transform_list.append(transforms.RandomAffine(
            degrees=affine.get("degrees", 0),
            translate=tuple(affine.get("translate", [0.1, 0.1])),
            scale=tuple(affine.get("scale", [0.9, 1.1])),
        ))
    blur = aug_config.get("gaussian_blur", {})
    if isinstance(blur, dict) and blur.get("enabled", False):
        transform_list.append(transforms.GaussianBlur(
            kernel_size=tuple(blur.get("kernel_size", [3, 5])),
            sigma=tuple(blur.get("sigma", [0.1, 2.0])),
        ))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(transform_list)


def get_val_transform(img_size: int, global_config: Dict = None) -> transforms.Compose:
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


def compute_class_weights(class_counts, num_classes, method="sqrt"):
    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)])
    if method == "sqrt":
        weights = 1.0 / np.sqrt(counts + 1)
    elif method in ("inverse", "balanced"):
        weights = 1.0 / (counts + 1)
    else:
        weights = np.ones(num_classes)
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def create_balanced_sampler(dataset, balancing_method="sqrt", samples_per_epoch=None):
    labels = []
    for s in dataset.samples:
        species = s["species"]
        label = dataset.species_to_int.get(species.lower(), dataset.num_classes - 1)
        labels.append(label)
    
    class_counts = Counter(labels)
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    if balancing_method == "sqrt":
        target_per_class = {}
        for label, count in class_counts.items():
            ratio = count / min_count
            target = int(min_count * np.sqrt(ratio))
            target_per_class[label] = min(target, count)
    elif balancing_method == "uniform":
        target_per_class = {label: min_count for label in class_counts}
    else:
        target_per_class = dict(class_counts)
    
    computed_epoch_size = sum(target_per_class.values())
    num_samples = samples_per_epoch if samples_per_epoch else computed_epoch_size
    
    weights = []
    for label in labels:
        actual = class_counts[label]
        target = target_per_class[label]
        weights.append(target / actual)
    weights = torch.FloatTensor(weights)
    
    if is_main_process():
        print(f"[Sampler] Balancing: {balancing_method}")
        print(f"[Sampler] Class range: {min_count:,} - {max_count:,} ({max_count/min_count:.0f}x imbalance)")
        print(f"[Sampler] Target epoch: {computed_epoch_size:,} samples")
    
    return torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)


# =============================================================================
# TRAINING LOOP (adapted for LUPI — text embedding passed to model)
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with LUPI (text embeddings passed during training)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", disable=not is_main_process())
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        caption_emb = batch["caption_embedding"].to(device)
        
        optimizer.zero_grad()
        
        # LUPI forward: pass caption embedding during training
        logits = model(images, text_embed=caption_emb)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.1f}%"})
        
        # Explicit tensor cleanup to reduce peak RSS
        del images, labels, caption_emb, logits, loss
        
        # Aggressive GC to prevent system RAM accumulation from DataLoader workers
        # (French region: 1051 batches/epoch — OOM-killed at batch ~999 with every-50)
        if (batch_idx + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    running_loss, correct, total = reduce_metrics_sums(
        [running_loss, float(correct), float(total)],
        device,
    )

    if total <= 0:
        return {"train_loss": 0.0, "train_acc": 0.0}
    return {"train_loss": running_loss / total, "train_acc": 100 * correct / total}


def validate(model, dataloader, criterion, device, epoch, text_embed_dim=384):
    """Validate with image-only (no caption — simulates test-time conditions)."""
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
            
            # Explicit zero text at validation (matches image-only test conditions).
            text_zeros = torch.zeros(images.size(0), text_embed_dim,
                                     device=device, dtype=images.dtype)
            logits = model(images, text_embed=text_zeros)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            _, top5_pred = torch.topk(logits, 5, dim=1)
            for i, label in enumerate(labels):
                if label in top5_pred[i]:
                    correct_top5 += 1
            
            # Explicit tensor cleanup
            del images, labels, text_zeros, logits, loss, top5_pred
            
            # Aggressive GC to prevent system RAM accumulation
            if (batch_idx + 1) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    
    running_loss, correct, total, correct_top5 = reduce_metrics_sums(
        [running_loss, float(correct), float(total), float(correct_top5)],
        device,
    )

    total = max(total, 1.0)
    return {
        "val_loss": running_loss / total,
        "val_acc": 100 * correct / total,
        "val_top5_acc": 100 * correct_top5 / total,
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(config: Dict, experiment_name: str, device: str = "cuda:0",
          max_samples: int = None, epochs_override: int = None,
          seed_override: Optional[int] = None, run_name: Optional[str] = None,
          num_workers_override: Optional[int] = None):
    """Main LUPI training function."""
    
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    train_config = exp_config["training"]
    lupi_config = exp_config.get("lupi", {})
    
    seed = global_config.get("seed", 42) if seed_override is None else seed_override
    set_seed(seed)

    use_multi_gpu = train_config.get("multi_gpu", False)
    device, use_ddp, rank, world_size = setup_distributed(device, use_multi_gpu)
    main_process = is_main_process()

    if main_process:
        print(f"\n{'='*60}")
        print(f"LUPI Experiment: {exp_config['name']}")
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
    
    # LUPI-specific: caption embedding directory
    caption_model_short = lupi_config.get("caption_model", "qwen25vl")
    # Normalize: accept both "qwen25vl" and "production_qwen25vl_final"
    if caption_model_short.startswith("production_") and caption_model_short.endswith("_final"):
        caption_model_short = caption_model_short.replace("production_", "").replace("_final", "")
    embedding_dir = data_root / "04_evaluation" / "caption_embeddings" / caption_model_short
    text_embed_dim = lupi_config.get("text_embed_dim", 384)
    caption_dropout = lupi_config.get("caption_dropout", 0.3)
    
    if main_process:
        print(f"[LUPI] Caption model: {caption_model_short}")
        print(f"[LUPI] Embedding dir: {embedding_dir}")
        print(f"[LUPI] Caption dropout: {caption_dropout}")
        print(f"[LUPI] Text embed dim: {text_embed_dim}")
    
    # Load caption embedding cache
    # For regional experiments, only load relevant slides to save RAM
    caption_cache = CaptionEmbeddingCache(embedding_dir, embed_dim=text_embed_dim)
    
    # Get datasets for this experiment
    train_region = train_config["train_region"]
    region_config = config["regions"][train_region]
    datasets = region_config["datasets"]
    
    # Caption model folder name for JSONL lookup
    caption_model_folder = f"production_{caption_model_short}_final"
    
    if train_region == "all":
        caption_cache.load_all()
    else:
        # Discover slides from caption dirs for this region's datasets only
        slide_names = set()
        for dataset in datasets:
            caption_path = data_root / "03_captioning" / dataset / caption_model_folder
            if caption_path.exists():
                for jsonl in caption_path.glob("*_captions.jsonl"):
                    slide_names.add(jsonl.stem.replace("_captions", ""))
        if slide_names:
            if main_process:
                print(f"[LUPI] Loading embeddings for {len(slide_names)} slides (region: {train_region})")
            caption_cache.load_for_slides(slide_names)
        else:
            caption_cache.load_all()  # Fallback
    
    if len(caption_cache) == 0:
        if main_process:
            print("\n[ERROR] No caption embeddings found!")
            print(f"Run: python embed_captions.py --caption_model production_{caption_model_short}_final")
        sys.exit(1)
    
    # Build transforms (identical to Option A)
    img_size = global_config["img_size"]
    preprocessing_config = exp_config.get("preprocessing", {})
    transform_config = global_config.copy()
    if preprocessing_config.get("normalization_mode") == "custom":
        if "normalize_mean" in preprocessing_config:
            transform_config["normalize_mean"] = preprocessing_config["normalize_mean"]
        if "normalize_std" in preprocessing_config:
            transform_config["normalize_std"] = preprocessing_config["normalize_std"]
        if main_process:
            print(f"[Preprocessing] Using custom normalization")
    else:
        if main_process:
            print(f"[Preprocessing] Using ImageNet normalization (default)")
    
    train_transform = get_train_transform(img_size, train_config, transform_config)
    val_transform = get_val_transform(img_size, transform_config)
    
    # Stain normalization
    stainnorm_func = None
    try:
        stainnorm_func = get_preprocessor(exp_config)
    except (FileNotFoundError, ImportError) as e:
        if main_process:
            print(f"[Warning] Preprocessing skipped: {e}")
    
    # HITL-validated species mappings
    all_anchors = load_caption_anchors()
    caption_anchors, excluded_slides = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    
    if main_process:
        print(f"[Setup] {len(all_anchors)} slides, {len(caption_anchors)} trainable, {num_classes} classes")
    
    # Build LUPI datasets
    if main_process:
        print("\nLoading datasets...")
    
    train_dataset = LUPIPollenDataset(
        split_dir=splits_dir,
        caption_dir=data_root / "03_captioning",
        wsi_dir=data_root / "00_raw_wsi",
        caption_cache=caption_cache,
        split="train",
        transform=train_transform,
        datasets=datasets,
        caption_model=caption_model_folder,
        stainnorm_func=stainnorm_func,
        max_samples=max_samples,
        species_to_int=species_to_int,
        num_classes=num_classes,
        caption_dropout=caption_dropout,
        text_embed_dim=text_embed_dim,
    )
    
    val_dataset = LUPIPollenDataset(
        split_dir=splits_dir,
        caption_dir=data_root / "03_captioning",
        wsi_dir=data_root / "00_raw_wsi",
        caption_cache=caption_cache,
        split="val",
        transform=val_transform,
        datasets=datasets,
        caption_model=caption_model_folder,
        stainnorm_func=stainnorm_func,
        max_samples=max_samples,
        species_to_int=species_to_int,
        num_classes=num_classes,
        caption_dropout=0.0,  # No dropout at validation
        text_embed_dim=text_embed_dim,
    )
    
    # Class balancing
    class_counts = train_dataset.get_class_counts()
    balancing = train_config.get("class_balancing", "none")
    class_weights = compute_class_weights(class_counts, num_classes, balancing).to(device)
    
    # Experiment metadata
    metadata = create_experiment_metadata(
        config, experiment_name, len(train_dataset), len(val_dataset),
        class_counts, int_to_species, max_samples, epochs_override,
        seed=seed,
    )
    if main_process:
        with open(output_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    
    # DataLoaders
    batch_size = train_config.get("batch_size", 64)
    num_workers = num_workers_override if num_workers_override is not None else global_config.get("num_workers", 8)
    loader_extra = {
        "prefetch_factor": 2 if num_workers > 0 else None,
    }
    if main_process:
        print(f"[DataLoader] num_workers={num_workers}, batch_size={batch_size}, prefetch_factor=2")
    train_sampler = None
    
    use_balanced_sampling = train_config.get("balanced_sampling", False)
    if use_balanced_sampling:
        base_sampler = create_balanced_sampler(
            train_dataset, train_config.get("sampling_method", "sqrt"),
            train_config.get("samples_per_epoch"),
        )
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
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
            )
        else:
            train_sampler = base_sampler
            # Don't drop_last if sampler epoch < batch_size (e.g., smoke tests)
            effective_drop_last = (len(train_sampler) >= batch_size)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
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
                train_dataset, batch_size=batch_size, sampler=train_sampler,
                drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                drop_last=True, num_workers=num_workers, pin_memory=True, **loader_extra,
            )

    if use_ddp:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler,
            num_workers=num_workers, pin_memory=True, **loader_extra,
        )
    else:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, **loader_extra,
        )
    
    if main_process:
        print(f"[DataLoader] Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Build LUPI model
    backbone_ckpt = global_config.get("backbone_checkpoint")
    if backbone_ckpt:
        ckpt_path = PROJECT_ROOT / backbone_ckpt
        checkpoint_path = str(ckpt_path) if ckpt_path.exists() else None
    else:
        checkpoint_path = None
        print("[Model] Using pretrained LVD weights from timm")
    
    model = LUPIClassifierModel(
        backbone_name=global_config["backbone_name"],
        num_classes=num_classes,
        img_size=img_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=train_config.get("freeze_backbone", True),
        text_embed_dim=text_embed_dim,
    )
    
    # Multi-GPU
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
            print(f"[Model] DataParallel with {num_gpus} GPUs")
            print(f"[Model] Effective global batch size: {batch_size}")
    else:
        model = model.to(device)

    base_model = unwrap_model(model)
    
    # Optimizer
    lr = train_config.get("learning_rate", 0.001)
    weight_decay = train_config.get("weight_decay", 0.0001)
    
    if train_config.get("freeze_backbone", True):
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
    best_val_acc = 0.0  # Match Option A: only save when val_acc > 0
    checkpoint_every = train_config.get("checkpoint_every", 5)
    
    csv_logger = None
    if main_process:
        csv_logger = CSVLogger(
            output_dir / "training_log.csv",
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                        "val_top5_acc", "lr", "elapsed_sec", "timestamp"]
        )
        print(f"\nTraining for {epochs} epochs...")
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"Batch: {batch_size}, LR: {lr}, Caption dropout: {caption_dropout}")
        print()
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # Unfreeze backbone if configured
        unfreeze_epoch = train_config.get("unfreeze_after_epoch")
        if unfreeze_epoch and epoch == unfreeze_epoch + 1:
            base_model.unfreeze_backbone()
            backbone_lr = lr * train_config.get("backbone_lr_multiplier", 0.1)
            optimizer = torch.optim.AdamW([
                {"params": base_model.backbone.parameters(), "lr": backbone_lr},
                {"params": base_model.head.parameters(), "lr": lr},
            ], weight_decay=weight_decay)
            remaining = epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, remaining))
        
        # Train (with captions)
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Validate (WITHOUT captions — simulates test time)
        skip_val_until = train_config.get("skip_val_until_epoch", 0)
        if epoch < skip_val_until:
            if main_process:
                print(f"[Epoch {epoch}] Skipping validation (warmup)")
            val_metrics = {"val_loss": float('inf'), "val_acc": 0.0, "val_top5_acc": 0.0}
        else:
            val_metrics = validate(model, val_loader, criterion, device, epoch,
                                   text_embed_dim=text_embed_dim)
            gc.collect()
            torch.cuda.empty_cache()
        
        if epoch > warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        
        history["train_loss"].append(train_metrics["train_loss"])
        history["train_acc"].append(train_metrics["train_acc"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["val_top5_acc"].append(val_metrics["val_top5_acc"])
        
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
        
        # Save best model
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            if main_process:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "num_classes": num_classes,
                    "int_to_species": int_to_species,
                    "config": exp_config,
                    "model_type": "lupi",
                    "img_embed_dim": base_model.img_embed_dim,
                    "text_embed_dim": text_embed_dim,
                }, output_dir / "best_model.pth")
                print(f"  -> Saved best model (val_acc={best_val_acc:.1f}%)")
        
        # Periodic checkpoint
        if (epoch % checkpoint_every == 0 or epoch == epochs) and main_process:
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
                "model_type": "lupi",
                "img_embed_dim": base_model.img_embed_dim,
                "text_embed_dim": text_embed_dim,
                "history": history,
            }, output_dir / f"checkpoint_epoch{epoch:03d}.pth")
            print(f"  -> Saved checkpoint: checkpoint_epoch{epoch:03d}.pth")
    
    if csv_logger is not None:
        csv_logger.close()
    
    # Save final model
    if main_process:
        torch.save({
            "epoch": epochs,
            "model_state_dict": base_model.state_dict(),
            "val_acc": val_metrics["val_acc"],
            "num_classes": num_classes,
            "int_to_species": int_to_species,
            "config": exp_config,
            "model_type": "lupi",
            "img_embed_dim": base_model.img_embed_dim,
            "text_embed_dim": text_embed_dim,
        }, output_dir / "final_model.pth")
    
    # Save results
    training_time = time.time() - start_time
    history["training_time_seconds"] = training_time
    history["best_val_acc"] = best_val_acc
    if main_process:
        save_experiment_results(output_dir, metadata, history, best_val_acc, training_time)
        
        print(f"\n{'='*60}")
        print(f"LUPI Training complete!")
        print(f"Best val accuracy (image-only): {best_val_acc:.2f}%")
        print(f"Training time: {training_time/3600:.1f} hours")
        print(f"Results: {output_dir}")
        print(f"{'='*60}\n")

    cleanup_distributed()
    
    return model, history


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train LUPI pollen classifier")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name from config")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed. Outputs saved to seed subfolder.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run tag appended to seed subfolder.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override num_workers (default: from config). Reduce for OOM.")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.experiment not in config["experiments"]:
        available = list(config["experiments"].keys())
        print(f"Error: Experiment '{args.experiment}' not found.")
        print(f"Available: {available}")
        sys.exit(1)
    
    train(config, args.experiment, args.device,
          max_samples=args.max_samples, epochs_override=args.epochs,
          seed_override=args.seed, run_name=args.run_name,
          num_workers_override=args.num_workers)


if __name__ == "__main__":
    main()
