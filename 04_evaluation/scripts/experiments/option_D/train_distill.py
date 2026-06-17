#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Knowledge Distillation Training Script (Option D)
====================================================

Two-stage teacher→student distillation with privileged information.

Stage 1 (TEACHER):
    Train a teacher on [img_embed(384) ; sbert_embed(384)] → Linear(768→46)
    This is identical to Option C (LUPI) training.

Stage 2 (STUDENT):
    Train a student on img_embed(384) → Linear(384→46) using:
      - (1-α) × CE(hard_labels) + α × T² × KL(teacher_soft, student_soft)
    The student is architecturally IDENTICAL to Option A.

Evaluation:
    The student model is a standard 384→46 linear head on frozen ViT.
    We reuse Option A's evaluate_classifier.py EXACTLY — no adapter needed.

Key design:
    - Frozen ViT backbone (shared, loaded once)
    - Frozen SBERT embeddings (pre-computed H5)
    - Teacher logits cached after Stage 1 → near-zero overhead in Stage 2
    - Student checkpoint is drop-in compatible with Option A evaluation

References:
    - Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
    - Lopez-Paz et al. (2016) "Unifying Distillation and Privileged Information"

Usage:
    # Default: Stage 2 only, auto-discovers Option C teacher checkpoint
    python train_distill.py --config ../experiment_config.yaml --experiment distill_all

    # Quick smoke test
    python train_distill.py --config ../experiment_config.yaml --experiment distill_all \\
        --max_samples 500 --epochs 3

    # Stage 2 only with explicit teacher checkpoint
    python train_distill.py --config ../experiment_config.yaml --experiment distill_all \\
        --teacher_checkpoint path/to/teacher_best_model.pth

    # Full two-stage training (train teacher from scratch instead of reusing Option C)
    python train_distill.py --config ../experiment_config.yaml --experiment distill_all \\
        --train_teacher

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

# Add project root (4 levels up from option_D/)
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
# STAIN/HISTOGRAM NORMALIZATION PREPROCESSING (identical to Option A/C)
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
# CSV LOGGER (identical to Option A/C)
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
# EXPERIMENT TRACKING (identical to Option A/C)
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
    stage: str = "full",
    seed: Optional[int] = None,
    output_dir_resolved: Optional[Path] = None,
    run_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    train_config = exp_config["training"]
    distill_config = exp_config.get("distillation", {})

    metadata = {
        "experiment": {
            "name": experiment_name,
            "display_name": exp_config.get("name", experiment_name),
            "description": exp_config.get("description", ""),
            "type": "Distillation (Option D)",
            "stage": stage,
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
            "teacher_fusion": "concat",
            "teacher_text_embed_dim": distill_config.get("text_embed_dim", 384),
            "student_architecture": "linear_384_to_C (identical to Option A)",
        },
        "data": {
            "train_region": train_config["train_region"],
            "train_samples": train_dataset_size,
            "val_samples": val_dataset_size,
            "class_distribution": {int_to_species.get(k, f"cls_{k}"): v
                                   for k, v in sorted(class_counts.items())},
            "num_active_classes": len([v for v in class_counts.values() if v > 0]),
            "max_samples_override": max_samples,
            "caption_model": distill_config.get("caption_model", "qwen25vl"),
        },
        "training": {
            "epochs": epochs_override or train_config.get("epochs", 30),
            "batch_size": train_config.get("batch_size", 128),
            "learning_rate": train_config.get("learning_rate", 0.001),
            "weight_decay": train_config.get("weight_decay", 0.0001),
            "optimizer": train_config.get("optimizer", "adamw"),
            "scheduler": train_config.get("scheduler", "cosine"),
            "warmup_epochs": train_config.get("warmup_epochs", 2),
            "class_balancing": train_config.get("class_balancing", "none"),
            "seed": global_config.get("seed", 42) if seed is None else seed,
        },
        "distillation": distill_config,
        "preprocessing": {
            "normalization_mode": exp_config.get("preprocessing", {}).get("normalization_mode", "imagenet"),
        },
        "augmentation": train_config.get("augmentation", {}),
        "eval_test_sets": exp_config.get("eval_test_sets", []),
        "output_dir": exp_config.get("output_dir", ""),
        "output_dir_resolved": str(output_dir_resolved) if output_dir_resolved is not None else None,
        "run_subdir": run_subdir,
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
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Tracking] Saved experiment metadata to {output_dir}")


# =============================================================================
# CAPTION EMBEDDING CACHE (identical to Option C)
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
# DATASET (extends Option A with caption embeddings for teacher training)
# =============================================================================

class DistillPollenDataset(Dataset):
    """
    Dataset for distillation pollen classification.

    Same as Option C's LUPIPollenDataset — provides both image and caption
    embedding. Used for Stage 1 (teacher training) and Stage 2 (student
    training with cached teacher logits).

    In Stage 2, the dataset additionally stores a pre-computed teacher logit
    index so the DataLoader can return the teacher soft targets alongside
    each sample.
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
        # Stage 2: teacher logits
        teacher_logits: Optional[torch.Tensor] = None,
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
        self.teacher_logits = teacher_logits  # (N, C) cached teacher logits for Stage 2

        # HITL-validated species mapping
        caption_anchors = load_caption_anchors()
        if species_to_int is None:
            species_to_int, _, num_classes = build_class_mappings(caption_anchors)
        self.caption_anchors = caption_anchors
        self.species_to_int = species_to_int
        self.num_classes = num_classes

        # Build sample index (identical to Option A/C)
        self.samples = []
        self._build_sample_index()

        # ALWAYS sort by ID for deterministic ordering.
        # This is CRITICAL for teacher logit caching: the cache_dataset and
        # train_dataset_s2 must have identical sample ordering so that
        # teacher_logits[idx] corresponds to the correct sample.
        self.samples.sort(key=lambda s: s["id"])

        if max_samples and len(self.samples) > max_samples:
            # Use fixed seed for deterministic subset selection
            # This ensures cache dataset and Stage 2 dataset get identical subsets
            rng = random.Random(42)
            rng.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
            # Re-sort after subsetting to maintain stable index order
            self.samples.sort(key=lambda s: s["id"])

        # Count how many samples have embeddings
        n_with_emb = sum(1 for s in self.samples if s["id"] in self.caption_cache)
        print(f"[DistillDataset] {split}: {len(self.samples)} samples, "
              f"{n_with_emb} with caption embeddings ({100 * n_with_emb / max(1, len(self.samples)):.1f}%)")
        if self.caption_dropout > 0:
            print(f"[DistillDataset] Caption dropout: {self.caption_dropout:.0%}")

        # Coverage gate (same as Option C)
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
                f"Run embed_captions.py first."
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

        # Extract patch from WSI (identical to Option A/C)
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

        result = {
            "image": image,
            "label": label,
            "caption_embedding": caption_emb,
            "sample_id": sample["id"],
            "species": sample["species"],
            "index": idx,
        }

        # Stage 2: include cached teacher logits
        if self.teacher_logits is not None:
            result["teacher_logits"] = self.teacher_logits[idx]

        return result

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
# TEACHER MODEL (identical to Option C's LUPIClassifierModel)
# =============================================================================

class TeacherModel(nn.Module):
    """
    Teacher: ViT backbone + [img_embed ; text_embed] → Linear(768→46).

    Architecturally identical to Option C's LUPIClassifierModel.
    Trained with CE loss on hard labels using both image and caption.
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

        # Build backbone (identical to Option A/C)
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

        # Teacher head: [img ; text] → classes
        self.head = nn.Linear(self.fused_dim, num_classes)

    def _load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        backbone_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"[Teacher] Loaded backbone from {checkpoint_path}")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Teacher] Backbone frozen")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("[Teacher] Backbone unfrozen")

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional text embedding.

        Args:
            x: Image tensor (B, 3, 518, 518)
            text_embed: Caption embedding (B, 384), or None for zeros

        Returns:
            logits: (B, num_classes)
        """
        # FP16 autocast for frozen backbone only (head + loss stay FP32)
        with torch.amp.autocast('cuda', enabled=self.freeze_backbone and x.is_cuda):
            img_features = self.backbone(x)  # (B, 384)
        img_features = img_features.float()  # ensure FP32 for head

        if text_embed is None:
            text_embed = torch.zeros(
                img_features.size(0), self.text_embed_dim,
                device=img_features.device, dtype=img_features.dtype
            )

        fused = torch.cat([img_features, text_embed], dim=1)  # (B, 768)
        logits = self.head(fused)  # (B, C)
        return logits


# =============================================================================
# STUDENT MODEL (identical to Option A's PollenClassifierModel)
# =============================================================================

class StudentModel(nn.Module):
    """
    Student: ViT backbone → img_embed(384) → Linear(384→46).

    Architecturally IDENTICAL to Option A's PollenClassifierModel.
    The checkpoint can be evaluated directly with Option A's evaluate_classifier.py.
    """

    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        num_classes: int = 46,
        img_size: int = 518,
        checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        if 'small' in backbone_name:
            self.embed_dim = 384
        elif 'base' in backbone_name:
            self.embed_dim = 768
        elif 'large' in backbone_name:
            self.embed_dim = 1024
        else:
            self.embed_dim = 384

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

        # Student head: img_embed → classes (IDENTICAL to Option A)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def _load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        backbone_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        self.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"[Student] Loaded backbone from {checkpoint_path}")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[Student] Backbone frozen")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("[Student] Backbone unfrozen")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP16 autocast for frozen backbone only (head + loss stay FP32)
        with torch.amp.autocast('cuda', enabled=self.freeze_backbone and x.is_cuda):
            features = self.backbone(x)
        features = features.float()  # ensure FP32 for head
        logits = self.head(features)
        return logits


# =============================================================================
# TRAINING UTILITIES (identical to Option A/C)
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        print(f"[Sampler] Class range: {min_count:,} - {max_count:,} ({max_count / min_count:.0f}x imbalance)")
        print(f"[Sampler] Target epoch: {computed_epoch_size:,} samples")

    return torch.utils.data.WeightedRandomSampler(weights, num_samples, replacement=True)


# =============================================================================
# STAGE 1: TRAIN TEACHER (identical logic to Option C training loop)
# =============================================================================

def train_teacher_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train teacher for one epoch (image + text → logits)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"S1 Epoch {epoch} [Teacher Train]", disable=not is_main_process())
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        caption_emb = batch["caption_embedding"].to(device)

        optimizer.zero_grad()
        logits = model(images, text_embed=caption_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.1f}%"})

        # Periodic GC to prevent system RAM accumulation from DataLoader workers
        if (batch_idx + 1) % 50 == 0:
            import gc
            gc.collect()

    running_loss, correct, total = reduce_metrics_sums(
        [running_loss, float(correct), float(total)],
        device,
    )
    total = max(total, 1.0)
    return {"train_loss": running_loss / total, "train_acc": 100 * correct / total}


def validate_teacher(model, dataloader, criterion, device, epoch, text_embed_dim=384):
    """Validate teacher with image-only (zero text) to gauge test-time proxy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    correct_top5 = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"S1 Epoch {epoch} [Teacher Val]", disable=not is_main_process())
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            # Zero text for validation (like Option C)
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

            # Periodic GC to prevent system RAM accumulation
            if (batch_idx + 1) % 50 == 0:
                import gc
                gc.collect()

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
# CACHE TEACHER LOGITS (key efficiency step)
# =============================================================================

def cache_teacher_logits(
    teacher: nn.Module,
    dataset: Dataset,
    caption_cache: CaptionEmbeddingCache,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
    text_embed_dim: int = 384,
) -> torch.Tensor:
    """
    Run teacher forward pass on all training samples and cache the logits.

    Teacher sees [img_embed ; text_embed] to produce logits.
    These are stored as float16 for memory efficiency.

    Storage: N × C × 2 bytes = 1.27M × 46 × 2 ≈ 117 MB

    Returns:
        logits_cache: (N, C) float16 tensor
    """
    teacher.eval()

    # Use val transform (no augmentation) for deterministic logits
    val_transform = dataset.transform  # Will be set to val transform before calling

    num_samples = len(dataset)
    num_classes = dataset.num_classes
    logits_cache = torch.zeros(num_samples, num_classes, dtype=torch.float16)

    # Create a DataLoader that returns index so we can map back
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    print(f"\n[Cache] Computing teacher logits for {num_samples:,} samples...")
    with torch.no_grad():
        sample_idx = 0
        for batch in tqdm(loader, desc="Caching teacher logits"):
            images = batch["image"].to(device)
            caption_emb = batch["caption_embedding"].to(device)
            indices = batch["index"]

            logits = teacher(images, text_embed=caption_emb)  # (B, C)
            logits_cache[indices] = logits.cpu().half()

            sample_idx += images.size(0)

    mem_mb = logits_cache.element_size() * logits_cache.nelement() / (1024 * 1024)
    print(f"[Cache] Stored {num_samples:,} × {num_classes} logits ({mem_mb:.1f} MB)")

    return logits_cache


# =============================================================================
# STAGE 2: TRAIN STUDENT WITH DISTILLATION
# =============================================================================

def distill_one_epoch(
    student: nn.Module,
    dataloader: DataLoader,
    criterion_hard: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    temperature: float = 3.0,
    alpha: float = 0.5,
    teacher: nn.Module = None,
) -> Dict[str, float]:
    """
    Train student for one epoch with knowledge distillation.

    Loss = (1 - α) × CE(hard_labels, student_logits)
         + α × T² × KL(teacher_soft || student_soft)

    where soft = softmax(logits / T)

    The T² factor scales KL gradients to be temperature-independent,
    following Hinton et al. (2015).

    Two modes:
      - On-the-fly (teacher is not None): compute teacher logits each batch.
        Slightly slower per-epoch but avoids 1-2h upfront caching.
      - Cached (teacher is None): read pre-cached logits from batch["teacher_logits"].
    """
    student.train()
    if teacher is not None:
        teacher.eval()
    running_loss = 0.0
    running_loss_hard = 0.0
    running_loss_kd = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"S2 Epoch {epoch} [Student Train]", disable=not is_main_process())
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Get teacher logits: on-the-fly or from cache
        if teacher is not None:
            caption_emb = batch["caption_embedding"].to(device)
            with torch.no_grad():
                teacher_logits = teacher(images, text_embed=caption_emb)  # (B, C)
        else:
            teacher_logits = batch["teacher_logits"].float().to(device)  # (B, C)

        optimizer.zero_grad()

        # Student forward (image only)
        student_logits = student(images)  # (B, C)

        # Hard label loss
        loss_hard = criterion_hard(student_logits, labels)

        # Soft target KD loss
        teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
        student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
        loss_kd = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

        # Combined loss
        loss = (1. - alpha) * loss_hard + alpha * loss_kd

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_loss_hard += loss_hard.item() * images.size(0)
        running_loss_kd += loss_kd.item() * images.size(0)
        _, predicted = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "CE": f"{loss_hard.item():.4f}",
            "KD": f"{loss_kd.item():.4f}",
            "acc": f"{100 * correct / total:.1f}%",
        })

        # Periodic GC to prevent system RAM accumulation from DataLoader workers
        if (batch_idx + 1) % 50 == 0:
            import gc
            gc.collect()

    running_loss, running_loss_hard, running_loss_kd, correct, total = reduce_metrics_sums(
        [running_loss, running_loss_hard, running_loss_kd, float(correct), float(total)],
        device,
    )
    total = max(total, 1.0)
    return {
        "train_loss": running_loss / total,
        "train_loss_hard": running_loss_hard / total,
        "train_loss_kd": running_loss_kd / total,
        "train_acc": 100 * correct / total,
    }


def validate_student(student, dataloader, criterion, device, epoch):
    """Validate student (image-only, identical to Option A's validate)."""
    student.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    correct_top5 = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"S2 Epoch {epoch} [Student Val]", disable=not is_main_process())
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = student(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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
    total = max(total, 1.0)
    return {
        "val_loss": running_loss / total,
        "val_acc": 100 * correct / total,
        "val_top5_acc": 100 * correct_top5 / total,
    }


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

# =============================================================================
# OPTION C TEACHER CHECKPOINT AUTO-DISCOVERY
# =============================================================================

# Mapping from Option D train_region → Option C (LUPI) experiment output dir
OPTION_C_TEACHER_MAP = {
    "all": "data/04_evaluation/results/exp11_lupi_all",
    "french": "data/04_evaluation/results/exp12_lupi_french",
    "hungarian": "data/04_evaluation/results/exp13_lupi_hungarian",
    "swedish": "data/04_evaluation/results/exp14_lupi_swedish",
    "mediterranean": "data/04_evaluation/results/exp15_lupi_mediterranean",
}


def sanitize_run_name(run_name: Optional[str]) -> Optional[str]:
    """Normalize run names for filesystem-safe subfolder naming."""
    if run_name is None:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in run_name.strip())
    return cleaned or None


def build_run_subdir(seed: Optional[int], run_name: Optional[str]) -> Optional[str]:
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


def resolve_output_dir(base_output_dir: Path, seed: Optional[int], run_name: Optional[str]) -> Path:
    """Resolve output dir, optionally creating seed/run subfolders."""
    run_subdir = build_run_subdir(seed, run_name)
    if run_subdir is None:
        return base_output_dir
    return base_output_dir / run_subdir


def has_train_augmentation(train_config: Dict[str, Any]) -> bool:
    """Detect whether stochastic train-time augmentation is enabled."""
    aug = train_config.get("augmentation", {})
    if not isinstance(aug, dict):
        return False

    if aug.get("random_horizontal_flip", False):
        return True
    if aug.get("random_vertical_flip", False):
        return True
    if aug.get("random_rotation", 0) > 0:
        return True
    if aug.get("random_grayscale", 0) > 0:
        return True

    jitter = aug.get("color_jitter", {})
    if isinstance(jitter, (int, float)) and jitter > 0:
        return True
    if isinstance(jitter, dict):
        for value in jitter.values():
            if isinstance(value, (int, float)) and value > 0:
                return True

    affine = aug.get("random_affine", {})
    if isinstance(affine, dict) and affine.get("enabled", False):
        return True

    blur = aug.get("gaussian_blur", {})
    if isinstance(blur, dict) and blur.get("enabled", False):
        return True

    return False


def find_option_c_teacher(
    config: Dict,
    experiment_name: str,
    teacher_seed: Optional[int] = None,
    teacher_run_name: Optional[str] = None,
) -> Optional[str]:
    """Auto-discover the matching Option C (LUPI) teacher checkpoint.

    Returns the path if found, None otherwise.
    """
    exp_config = config["experiments"][experiment_name]
    train_region = exp_config["training"]["train_region"]

    lupi_dir = OPTION_C_TEACHER_MAP.get(train_region)
    if lupi_dir is None:
        return None

    base_dir = PROJECT_ROOT / lupi_dir
    candidate_dirs = []

    run_subdir = build_run_subdir(teacher_seed, teacher_run_name)
    if run_subdir is not None:
        candidate_dirs.append(base_dir / run_subdir)

    if teacher_seed is not None and teacher_run_name is None:
        candidate_dirs.append(base_dir / f"seed_{teacher_seed}")
    if teacher_seed is None and teacher_run_name is not None:
        clean_name = sanitize_run_name(teacher_run_name)
        if clean_name is not None:
            candidate_dirs.append(base_dir / clean_name)

    # Default Option C location (non-seeded runs)
    candidate_dirs.append(base_dir)

    for ckpt_dir in candidate_dirs:
        best_path = ckpt_dir / "best_model.pth"
        if best_path.exists():
            return str(best_path)
        final_path = ckpt_dir / "final_model.pth"
        if final_path.exists():
            return str(final_path)

    # Last fallback: if no explicit seed/run requested, pick latest seed_* run.
    if teacher_seed is None and teacher_run_name is None and base_dir.exists():
        seed_dirs = sorted(
            [p for p in base_dir.glob("seed_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for seed_dir in seed_dirs:
            best_path = seed_dir / "best_model.pth"
            if best_path.exists():
                return str(best_path)
            final_path = seed_dir / "final_model.pth"
            if final_path.exists():
                return str(final_path)

    return None


def train(config: Dict, experiment_name: str, device: str = "cuda:0",
          max_samples: int = None, epochs_override: int = None,
          stage2_only: bool = True, teacher_checkpoint: str = None,
          train_teacher: bool = False, cache_logits: bool = False,
          seed_override: Optional[int] = None, run_name: Optional[str] = None,
          teacher_seed: Optional[int] = None, teacher_run_name: Optional[str] = None,
          allow_cache_with_augmentation: bool = False):
    """
    Main distillation training function.

    Two stages:
        Stage 1: Train teacher on [img ; text] → Linear(768 → C)
        Stage 2: Distill into student on img → Linear(384 → C) using KD

    By default, Stage 1 is skipped and the Option C (LUPI) teacher checkpoint
    is auto-discovered. Pass --train_teacher to do full two-stage training.

    Teacher logit modes:
        - On-the-fly (default): teacher stays in GPU memory, logits computed
          per-batch during Stage 2. Avoids 1-2h caching overhead.
        - Cached (--cache_logits): pre-compute all logits upfront, store as
          float16 tensor. Better for repeated runs / fixed augmentations.
    """

    # Resolve stage2_only vs train_teacher
    if train_teacher:
        stage2_only = False

    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    train_config = exp_config["training"]
    distill_config = exp_config.get("distillation", {})

    seed = global_config.get("seed", 42) if seed_override is None else seed_override
    set_seed(seed)
    use_multi_gpu = train_config.get("multi_gpu", False)
    device, use_ddp, rank, world_size = setup_distributed(device, use_multi_gpu)
    main_process = is_main_process()

    if main_process:
        print(f"\n{'=' * 80}")
        print(f"Option D: Knowledge Distillation Experiment")
        print(f"Experiment: {exp_config['name']}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        if use_ddp:
            print(f"DDP: enabled (rank {rank}/{world_size})")
        print(f"{'=' * 80}\n")

    # ============================
    # PATHS & CONFIG
    # ============================
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

    # Sub-directories for Stage 1 / Stage 2 artifacts
    teacher_dir = output_dir / "teacher"
    student_dir = output_dir / "student"
    if main_process:
        teacher_dir.mkdir(parents=True, exist_ok=True)
        student_dir.mkdir(parents=True, exist_ok=True)

    # KD hyperparameters
    temperature = distill_config.get("temperature", 3.0)
    alpha = distill_config.get("alpha", 0.5)
    caption_model_short = distill_config.get("caption_model", "qwen25vl")
    text_embed_dim = distill_config.get("text_embed_dim", 384)
    caption_dropout = distill_config.get("caption_dropout", 0.3)
    teacher_epochs = distill_config.get("teacher_epochs", None)  # Defaults to training.epochs
    student_epochs = distill_config.get("student_epochs", None)  # Defaults to training.epochs

    # Epochs
    base_epochs = epochs_override or train_config.get("epochs", 30)
    if teacher_epochs is None:
        teacher_epochs = base_epochs
    if student_epochs is None:
        student_epochs = base_epochs
    if epochs_override:
        teacher_epochs = epochs_override
        student_epochs = epochs_override

    if main_process:
        print(f"[KD Config]")
        print(f"  Temperature (T): {temperature}")
        print(f"  Alpha (α):       {alpha}")
        print(f"  Loss: (1-{alpha}) × CE + {alpha} × T² × KL")
        print(f"  Teacher epochs:  {teacher_epochs}")
        print(f"  Student epochs:  {student_epochs}")
        print(f"  Caption model:   {caption_model_short}")
        print(f"  Caption dropout: {caption_dropout}")
        print(f"  Logit mode:      {'cached' if cache_logits else 'on-the-fly'}")
        if seed_override is not None:
            print(f"  Seed override:   {seed_override}")
        if teacher_seed is not None:
            print(f"  Teacher seed:    {teacher_seed}")
        if run_subdir is not None:
            print(f"  Run subdir:      {run_subdir}")
        print()

    # ============================
    # CAPTION EMBEDDINGS
    # ============================
    if caption_model_short.startswith("production_") and caption_model_short.endswith("_final"):
        caption_model_short_clean = caption_model_short.replace("production_", "").replace("_final", "")
    else:
        caption_model_short_clean = caption_model_short
    embedding_dir = data_root / "04_evaluation" / "caption_embeddings" / caption_model_short_clean

    caption_cache = CaptionEmbeddingCache(embedding_dir, embed_dim=text_embed_dim)

    train_region = train_config["train_region"]
    region_config = config["regions"][train_region]
    datasets = region_config["datasets"]
    caption_model_folder = f"production_{caption_model_short_clean}_final"

    if train_region == "all":
        caption_cache.load_all()
    else:
        slide_names = set()
        for dataset in datasets:
            caption_path = data_root / "03_captioning" / dataset / caption_model_folder
            if caption_path.exists():
                for jsonl in caption_path.glob("*_captions.jsonl"):
                    slide_names.add(jsonl.stem.replace("_captions", ""))
        if slide_names:
            caption_cache.load_for_slides(slide_names)
        else:
            caption_cache.load_all()

    if len(caption_cache) == 0:
        if main_process:
            print(f"\n[ERROR] No caption embeddings found at {embedding_dir}!")
            print(f"Run: python option_C/embed_captions.py --caption_model {caption_model_folder}")
        sys.exit(1)

    # ============================
    # TRANSFORMS & PREPROCESSING (identical to Option A/C)
    # ============================
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

    stainnorm_func = None
    try:
        stainnorm_func = get_preprocessor(exp_config)
    except (FileNotFoundError, ImportError) as e:
        if main_process:
            print(f"[Warning] Preprocessing skipped: {e}")

    # ============================
    # SPECIES MAPPINGS (identical to Option A/C)
    # ============================
    all_anchors = load_caption_anchors()
    caption_anchors, excluded_slides = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    if main_process:
        print(f"[Setup] {len(all_anchors)} slides, {len(caption_anchors)} trainable, {num_classes} classes")

    # ============================
    # BACKBONE PATH (shared by teacher & student)
    # ============================
    backbone_ckpt = global_config.get("backbone_checkpoint")
    if backbone_ckpt:
        ckpt_path = PROJECT_ROOT / backbone_ckpt
        if ckpt_path.exists():
            checkpoint_path = str(ckpt_path)
            if main_process:
                print(f"[Model] Using custom checkpoint: {backbone_ckpt}")
        else:
            if main_process:
                print(f"[WARNING] Backbone checkpoint not found: {ckpt_path}")
                print(f"[WARNING] Falling back to pretrained LVD weights from timm")
            checkpoint_path = None
    else:
        checkpoint_path = None
        if main_process:
            print("[Model] Using original pretrained LVD weights from timm")

    # ============================
    # STAGE 1: TRAIN TEACHER
    # ============================
    if not stage2_only:
        if main_process:
            print(f"\n{'=' * 80}")
            print(f"STAGE 1: Training Teacher (image + text) → Linear(768→{num_classes})")
            print(f"{'=' * 80}\n")

        # Build datasets for teacher training
        train_dataset_s1 = DistillPollenDataset(
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

        val_dataset_s1 = DistillPollenDataset(
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
            caption_dropout=0.0,
            text_embed_dim=text_embed_dim,
        )

        class_counts = train_dataset_s1.get_class_counts()
        class_weights = compute_class_weights(class_counts, num_classes,
                                               train_config.get("class_balancing", "none")).to(device)

        # Metadata (Stage 1)
        metadata_s1 = create_experiment_metadata(
            config, experiment_name, len(train_dataset_s1), len(val_dataset_s1),
            class_counts, int_to_species, max_samples, teacher_epochs, stage="stage1_teacher",
            seed=seed, output_dir_resolved=teacher_dir, run_subdir=run_subdir,
        )
        if main_process:
            with open(teacher_dir / "experiment_metadata.json", "w") as f:
                json.dump(metadata_s1, f, indent=2, default=str)

        # DataLoaders
        batch_size = train_config.get("batch_size", 64)
        num_workers = global_config.get("num_workers", 8)
        loader_extra = {
            "prefetch_factor": 2 if num_workers > 0 else None,
        }
        train_sampler_s1 = None

        use_balanced_sampling = train_config.get("balanced_sampling", False)
        if use_balanced_sampling:
            base_sampler_s1 = create_balanced_sampler(
                train_dataset_s1, train_config.get("sampling_method", "sqrt"),
                train_config.get("samples_per_epoch"),
            )
            if use_ddp:
                train_sampler_s1 = DistributedWeightedSampler(
                    weights=base_sampler_s1.weights,
                    num_samples=base_sampler_s1.num_samples,
                    num_replicas=world_size,
                    rank=rank,
                    replacement=True,
                    seed=seed,
                )
                effective_drop_last = (len(train_sampler_s1) >= batch_size)
                train_loader_s1 = DataLoader(
                    train_dataset_s1, batch_size=batch_size, sampler=train_sampler_s1,
                    drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
                )
            else:
                train_sampler_s1 = base_sampler_s1
                effective_drop_last = (len(train_sampler_s1) >= batch_size)
                train_loader_s1 = DataLoader(
                    train_dataset_s1, batch_size=batch_size, sampler=train_sampler_s1,
                    drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
                )
        else:
            if use_ddp:
                train_sampler_s1 = DistributedSampler(
                    train_dataset_s1,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    drop_last=True,
                )
                effective_drop_last = (len(train_sampler_s1) >= batch_size)
                train_loader_s1 = DataLoader(
                    train_dataset_s1, batch_size=batch_size, sampler=train_sampler_s1,
                    drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
                )
            else:
                train_loader_s1 = DataLoader(
                    train_dataset_s1, batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=num_workers, pin_memory=True, **loader_extra,
                )

        if use_ddp:
            val_sampler_s1 = DistributedSampler(
                val_dataset_s1,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            val_loader_s1 = DataLoader(
                val_dataset_s1, batch_size=batch_size, sampler=val_sampler_s1,
                num_workers=num_workers, pin_memory=True, **loader_extra,
            )
        else:
            val_loader_s1 = DataLoader(
                val_dataset_s1, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, **loader_extra,
            )

        # Build teacher model
        teacher = TeacherModel(
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
            teacher = teacher.to(device)
            if device.type == "cuda":
                teacher = DDP(teacher, device_ids=[device.index], output_device=device.index)
            else:
                teacher = DDP(teacher)
            if main_process:
                print(f"[Teacher] DDP with world_size={world_size}")
        elif use_multi_gpu and num_gpus > 1:
            teacher = teacher.to(device)
            teacher = nn.DataParallel(teacher)
            if main_process:
                print(f"[Teacher] DataParallel with {num_gpus} GPUs")
        else:
            teacher = teacher.to(device)

        base_teacher = unwrap_model(teacher)

        # Optimizer
        lr = train_config.get("learning_rate", 0.001)
        weight_decay = train_config.get("weight_decay", 0.0001)

        if train_config.get("freeze_backbone", True):
            optimizer = torch.optim.AdamW(base_teacher.head.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(base_teacher.parameters(), lr=lr, weight_decay=weight_decay)

        warmup_epochs = train_config.get("warmup_epochs", 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, teacher_epochs - warmup_epochs))

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training loop for teacher
        history_s1 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top5_acc": []}
        best_teacher_val_acc = 0.0
        saved_best_teacher = False
        checkpoint_every = train_config.get("checkpoint_every", 5)

        csv_logger_s1 = None
        if main_process:
            csv_logger_s1 = CSVLogger(
                teacher_dir / "training_log.csv",
                fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                            "val_top5_acc", "lr", "elapsed_sec", "timestamp"]
            )

            print(f"\n[Stage 1] Training teacher for {teacher_epochs} epochs...")
            print(f"  Train: {len(train_dataset_s1)}, Val: {len(val_dataset_s1)}")
            print(f"  Batch: {batch_size}, LR: {lr}, Caption dropout: {caption_dropout}")
            print()

        start_time_s1 = time.time()

        for epoch in range(1, teacher_epochs + 1):
            if train_sampler_s1 is not None and hasattr(train_sampler_s1, "set_epoch"):
                train_sampler_s1.set_epoch(epoch)

            # Unfreeze backbone if configured
            unfreeze_epoch = train_config.get("unfreeze_after_epoch")
            if unfreeze_epoch and epoch == unfreeze_epoch + 1:
                base_teacher.unfreeze_backbone()
                backbone_lr = lr * train_config.get("backbone_lr_multiplier", 0.1)
                optimizer = torch.optim.AdamW([
                    {"params": base_teacher.backbone.parameters(), "lr": backbone_lr},
                    {"params": base_teacher.head.parameters(), "lr": lr},
                ], weight_decay=weight_decay)
                remaining = teacher_epochs - epoch + 1
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, remaining))

            train_metrics = train_teacher_one_epoch(
                teacher, train_loader_s1, criterion, optimizer, device, epoch)

            gc.collect()
            torch.cuda.empty_cache()

            skip_val_until = train_config.get("skip_val_until_epoch", 0)
            if epoch < skip_val_until:
                val_metrics = {"val_loss": float('inf'), "val_acc": 0.0, "val_top5_acc": 0.0}
            else:
                val_metrics = validate_teacher(
                    teacher, val_loader_s1, criterion, device, epoch,
                    text_embed_dim=text_embed_dim)
                gc.collect()
                torch.cuda.empty_cache()

            if epoch > warmup_epochs:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time_s1

            history_s1["train_loss"].append(train_metrics["train_loss"])
            history_s1["train_acc"].append(train_metrics["train_acc"])
            history_s1["val_loss"].append(val_metrics["val_loss"])
            history_s1["val_acc"].append(val_metrics["val_acc"])
            history_s1["val_top5_acc"].append(val_metrics["val_top5_acc"])

            if main_process and csv_logger_s1 is not None:
                csv_logger_s1.log({
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
                print(f"[S1] Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                      f"train_acc={train_metrics['train_acc']:.1f}%, "
                      f"val_acc(img-only)={val_metrics['val_acc']:.1f}%, "
                      f"val_top5={val_metrics['val_top5_acc']:.1f}%")

            # Save best teacher
            if val_metrics["val_acc"] > best_teacher_val_acc:
                best_teacher_val_acc = val_metrics["val_acc"]
                if main_process:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": base_teacher.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_teacher_val_acc,
                        "num_classes": num_classes,
                        "int_to_species": int_to_species,
                        "config": exp_config,
                        "model_type": "teacher",
                        "img_embed_dim": base_teacher.img_embed_dim,
                        "text_embed_dim": text_embed_dim,
                    }, teacher_dir / "best_model.pth")
                saved_best_teacher = True
                if main_process:
                    print(f"  -> Saved best teacher (val_acc_img_only={best_teacher_val_acc:.1f}%)")

            # Periodic checkpoint
            if (epoch % checkpoint_every == 0 or epoch == teacher_epochs) and main_process:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base_teacher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_metrics["val_acc"],
                    "best_val_acc": best_teacher_val_acc,
                    "num_classes": num_classes,
                    "int_to_species": int_to_species,
                    "config": exp_config,
                    "model_type": "teacher",
                    "img_embed_dim": base_teacher.img_embed_dim,
                    "text_embed_dim": text_embed_dim,
                    "history": history_s1,
                }, teacher_dir / f"checkpoint_epoch{epoch:03d}.pth")

        if csv_logger_s1 is not None:
            csv_logger_s1.close()

        # Always save final teacher checkpoint (prevents missing-best edge cases)
        final_teacher_path = teacher_dir / "final_model.pth"
        if main_process:
            torch.save({
                "epoch": teacher_epochs,
                "model_state_dict": base_teacher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["val_acc"],
                "best_val_acc": best_teacher_val_acc,
                "num_classes": num_classes,
                "int_to_species": int_to_species,
                "config": exp_config,
                "model_type": "teacher",
                "img_embed_dim": base_teacher.img_embed_dim,
                "text_embed_dim": text_embed_dim,
                "seed": seed,
            }, final_teacher_path)

        # If best checkpoint was never written (e.g., all val_acc==0), promote final.
        teacher_best_path = teacher_dir / "best_model.pth"
        if not saved_best_teacher:
            if main_process:
                torch.save({
                    "epoch": teacher_epochs,
                    "model_state_dict": base_teacher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_metrics["val_acc"],
                    "num_classes": num_classes,
                    "int_to_species": int_to_species,
                    "config": exp_config,
                    "model_type": "teacher",
                    "img_embed_dim": base_teacher.img_embed_dim,
                    "text_embed_dim": text_embed_dim,
                    "seed": seed,
                }, teacher_best_path)
            best_teacher_val_acc = val_metrics["val_acc"]
            if main_process:
                print("[Stage 1] No best checkpoint captured; promoted final_model.pth to best_model.pth")

        training_time_s1 = time.time() - start_time_s1
        history_s1["training_time_seconds"] = training_time_s1
        history_s1["best_val_acc"] = best_teacher_val_acc
        if main_process:
            save_experiment_results(teacher_dir, metadata_s1, history_s1,
                                    best_teacher_val_acc, training_time_s1)

            print(f"\n[Stage 1] Teacher training complete!")
            print(f"  Best val accuracy (image-only proxy): {best_teacher_val_acc:.2f}%")
            print(f"  Training time: {training_time_s1 / 3600:.1f} hours")

        if use_ddp:
            dist.barrier()

        # Load best teacher for logit caching
        teacher_best_path = teacher_dir / "best_model.pth"
        if not teacher_best_path.exists():
            teacher_best_path = teacher_dir / "final_model.pth"
        best_ckpt = torch.load(teacher_best_path, map_location="cpu")
        base_teacher.load_state_dict(best_ckpt["model_state_dict"])
        teacher = teacher.to(device)
        teacher.eval()

        # Free Stage 1 data loaders
        del train_loader_s1, val_loader_s1
        gc.collect()
        torch.cuda.empty_cache()

    else:
        # Stage 2 only: load existing teacher checkpoint
        if teacher_checkpoint is None:
            # Auto-discover Option C (LUPI) teacher checkpoint
            resolved_teacher_seed = teacher_seed if teacher_seed is not None else seed_override
            option_c_ckpt = find_option_c_teacher(
                config, experiment_name,
                teacher_seed=resolved_teacher_seed,
                teacher_run_name=teacher_run_name,
            )
            if option_c_ckpt is not None:
                teacher_checkpoint = option_c_ckpt
                if main_process:
                    print(f"\n[Stage 2] Auto-discovered Option C teacher: {teacher_checkpoint}")
            else:
                # Fallback: check Option D's own teacher directory
                fallback_best = teacher_dir / "best_model.pth"
                fallback_final = teacher_dir / "final_model.pth"
                teacher_checkpoint = str(fallback_best if fallback_best.exists() else fallback_final)

        if not Path(teacher_checkpoint).exists():
            if main_process:
                print(f"[ERROR] Teacher checkpoint not found: {teacher_checkpoint}")
                print(f"  Options:")
                print(f"    1. Train Option C first:  cd option_C && ./train_all.sh")
                print(f"    2. Train teacher here:    --train_teacher")
                print(f"    3. Specify path:          --teacher_checkpoint /path/to/best_model.pth")
            sys.exit(1)

        if main_process:
            print(f"\n[Stage 2 only] Loading teacher from: {teacher_checkpoint}")

        teacher = TeacherModel(
            backbone_name=global_config["backbone_name"],
            num_classes=num_classes,
            img_size=img_size,
            checkpoint_path=checkpoint_path,
            freeze_backbone=True,
            text_embed_dim=text_embed_dim,
        )

        ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"])
        teacher = teacher.to(device)
        teacher.eval()

        base_teacher = teacher

        class_counts = None  # Will be computed from Stage 2 dataset
        best_teacher_val_acc = ckpt.get("val_acc", 0)
        training_time_s1 = 0

    # ============================
    # TEACHER LOGITS MODE
    # ============================
    batch_size = train_config.get("batch_size", 64)
    num_workers = global_config.get("num_workers", 8)
    loader_extra = {
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    teacher_logits = None  # Will be set only if caching is used
    teacher_for_s2 = None  # Will be set only for on-the-fly mode

    if cache_logits and has_train_augmentation(train_config) and not allow_cache_with_augmentation:
        raise ValueError(
            "cache_logits with stochastic training augmentation can create teacher/student "
            "distribution mismatch. Use default on-the-fly mode, disable augmentations, "
            "or pass --allow_cache_with_augmentation to proceed explicitly."
        )

    if cache_logits:
        if main_process:
            print(f"\n{'=' * 80}")
            print(f"Caching teacher logits for Stage 2")
            print(f"{'=' * 80}\n")

        cache_path = output_dir / "cached_teacher_logits.pt"
        cache_dataset = None

        if not use_ddp or main_process:
            # Build a dataset with val_transform (no augmentation) for deterministic caching
            cache_dataset = DistillPollenDataset(
                split_dir=splits_dir,
                caption_dir=data_root / "03_captioning",
                wsi_dir=data_root / "00_raw_wsi",
                caption_cache=caption_cache,
                split="train",
                transform=val_transform,  # No augmentation for deterministic logits
                datasets=datasets,
                caption_model=caption_model_folder,
                stainnorm_func=stainnorm_func,
                max_samples=max_samples,
                species_to_int=species_to_int,
                num_classes=num_classes,
                caption_dropout=0.0,  # No dropout for caching
                text_embed_dim=text_embed_dim,
            )

            teacher_logits = cache_teacher_logits(
                teacher=teacher,
                dataset=cache_dataset,
                caption_cache=caption_cache,
                device=device,
                batch_size=batch_size * 2,  # Can use larger batch since no backprop
                num_workers=num_workers,
                text_embed_dim=text_embed_dim,
            )

            # Save cached logits for reproducibility
            torch.save(teacher_logits, cache_path)
            if main_process:
                print(f"[Cache] Saved to {cache_path}")

        if use_ddp:
            dist.barrier()
            if not main_process:
                teacher_logits = torch.load(cache_path, map_location="cpu")

        # Free teacher model memory (cached mode doesn't need teacher anymore)
        del teacher
        if cache_dataset is not None:
            del cache_dataset
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # On-the-fly mode: keep teacher in GPU memory for Stage 2
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Wrap teacher only for DataParallel fallback (DDP keeps per-rank teacher unwrapped).
        num_gpus = torch.cuda.device_count()
        if (not use_ddp) and use_multi_gpu and num_gpus > 1:
            teacher = nn.DataParallel(teacher)
            if main_process:
                print(f"[Teacher] On-the-fly mode — DataParallel with {num_gpus} GPUs")
        else:
            if main_process:
                mode = "DDP per-rank teacher" if use_ddp else "single GPU"
                print(f"[Teacher] On-the-fly mode — {mode}")
        teacher_for_s2 = teacher

    if not stage2_only and 'train_dataset_s1' in dir():
        del train_dataset_s1, val_dataset_s1
        gc.collect()

    # ============================
    # STAGE 2: TRAIN STUDENT WITH DISTILLATION
    # ============================
    if main_process:
        print(f"\n{'=' * 80}")
        print(f"STAGE 2: Training Student (image only) → Linear(384→{num_classes})")
        print(f"  Loss: (1-{alpha}) × CE + {alpha} × T²×KL  |  T={temperature}")
        print(f"{'=' * 80}\n")

    # Reset seed for reproducibility of Stage 2
    # Must use the effective run seed (including --seed override).
    set_seed(seed)

    # Build datasets for student training
    train_dataset_s2 = DistillPollenDataset(
        split_dir=splits_dir,
        caption_dir=data_root / "03_captioning",
        wsi_dir=data_root / "00_raw_wsi",
        caption_cache=caption_cache,
        split="train",
        transform=train_transform,  # With augmentation for student
        datasets=datasets,
        caption_model=caption_model_folder,
        stainnorm_func=stainnorm_func,
        max_samples=max_samples,
        species_to_int=species_to_int,
        num_classes=num_classes,
        caption_dropout=0.0,  # No caption dropout for student (it doesn't use captions)
        text_embed_dim=text_embed_dim,
        teacher_logits=teacher_logits,  # None in on-the-fly mode, tensor in cached mode
    )

    val_dataset_s2 = DistillPollenDataset(
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
        caption_dropout=0.0,
        text_embed_dim=text_embed_dim,
    )

    if class_counts is None:
        class_counts = train_dataset_s2.get_class_counts()
    class_weights = compute_class_weights(class_counts, num_classes,
                                           train_config.get("class_balancing", "none")).to(device)

    # Metadata (Stage 2)
    metadata_s2 = create_experiment_metadata(
        config, experiment_name, len(train_dataset_s2), len(val_dataset_s2),
        class_counts, int_to_species, max_samples, student_epochs, stage="stage2_student",
        seed=seed, output_dir_resolved=student_dir, run_subdir=run_subdir,
    )
    metadata_s2["distillation"]["teacher_val_acc_img_only"] = best_teacher_val_acc
    metadata_s2["distillation"]["teacher_training_time_s"] = training_time_s1
    if main_process:
        with open(student_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata_s2, f, indent=2, default=str)

    # DataLoaders
    use_balanced_sampling = train_config.get("balanced_sampling", False)
    train_sampler_s2 = None
    if use_balanced_sampling:
        base_sampler_s2 = create_balanced_sampler(
            train_dataset_s2, train_config.get("sampling_method", "sqrt"),
            train_config.get("samples_per_epoch"),
        )
        if use_ddp:
            train_sampler_s2 = DistributedWeightedSampler(
                weights=base_sampler_s2.weights,
                num_samples=base_sampler_s2.num_samples,
                num_replicas=world_size,
                rank=rank,
                replacement=True,
                seed=seed,
            )
            effective_drop_last = (len(train_sampler_s2) >= batch_size)
            train_loader_s2 = DataLoader(
                train_dataset_s2, batch_size=batch_size, sampler=train_sampler_s2,
                drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
            )
        else:
            train_sampler_s2 = base_sampler_s2
            effective_drop_last = (len(train_sampler_s2) >= batch_size)
            train_loader_s2 = DataLoader(
                train_dataset_s2, batch_size=batch_size, sampler=train_sampler_s2,
                drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
            )
    else:
        if use_ddp:
            train_sampler_s2 = DistributedSampler(
                train_dataset_s2,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            effective_drop_last = (len(train_sampler_s2) >= batch_size)
            train_loader_s2 = DataLoader(
                train_dataset_s2, batch_size=batch_size, sampler=train_sampler_s2,
                drop_last=effective_drop_last, num_workers=num_workers, pin_memory=True, **loader_extra,
            )
        else:
            train_loader_s2 = DataLoader(
                train_dataset_s2, batch_size=batch_size, shuffle=True,
                drop_last=True, num_workers=num_workers, pin_memory=True, **loader_extra,
            )

    if use_ddp:
        val_sampler_s2 = DistributedSampler(
            val_dataset_s2,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        val_loader_s2 = DataLoader(
            val_dataset_s2, batch_size=batch_size, sampler=val_sampler_s2,
            num_workers=num_workers, pin_memory=True, **loader_extra,
        )
    else:
        val_loader_s2 = DataLoader(
            val_dataset_s2, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, **loader_extra,
        )

    # Build student model (IDENTICAL architecture to Option A)
    student = StudentModel(
        backbone_name=global_config["backbone_name"],
        num_classes=num_classes,
        img_size=img_size,
        checkpoint_path=checkpoint_path,
        freeze_backbone=train_config.get("freeze_backbone", True),
    )

    # Multi-GPU
    num_gpus = torch.cuda.device_count()
    if use_ddp:
        student = student.to(device)
        if device.type == "cuda":
            student = DDP(student, device_ids=[device.index], output_device=device.index)
        else:
            student = DDP(student)
        if main_process:
            print(f"[Student] DDP with world_size={world_size}")
            print(f"[Student] Effective global batch size: {batch_size * world_size} ({batch_size} per rank)")
    elif use_multi_gpu and num_gpus > 1:
        student = student.to(device)
        student = nn.DataParallel(student)
        if main_process:
            print(f"[Student] DataParallel with {num_gpus} GPUs")
    else:
        student = student.to(device)

    base_student = unwrap_model(student)

    # Optimizer (student head only when backbone frozen)
    lr = train_config.get("learning_rate", 0.001)
    weight_decay = train_config.get("weight_decay", 0.0001)

    if train_config.get("freeze_backbone", True):
        optimizer = torch.optim.AdamW(base_student.head.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(base_student.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_epochs = train_config.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, student_epochs - warmup_epochs))

    # Hard label criterion (weighted CE like Option A)
    criterion_hard = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop for student
    history_s2 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
                  "val_top5_acc": [], "train_loss_hard": [], "train_loss_kd": []}
    best_student_val_acc = 0.0
    saved_best_student = False
    checkpoint_every = train_config.get("checkpoint_every", 5)

    csv_logger_s2 = None
    if main_process:
        csv_logger_s2 = CSVLogger(
            student_dir / "training_log.csv",
            fieldnames=["epoch", "train_loss", "train_loss_hard", "train_loss_kd",
                        "train_acc", "val_loss", "val_acc", "val_top5_acc",
                        "lr", "elapsed_sec", "timestamp"]
        )

        print(f"\n[Stage 2] Training student for {student_epochs} epochs...")
        print(f"  Train: {len(train_dataset_s2)}, Val: {len(val_dataset_s2)}")
        print(f"  Batch: {batch_size}, LR: {lr}")
        print(f"  KD: T={temperature}, α={alpha}")
        print()

    start_time_s2 = time.time()

    for epoch in range(1, student_epochs + 1):
        if train_sampler_s2 is not None and hasattr(train_sampler_s2, "set_epoch"):
            train_sampler_s2.set_epoch(epoch)

        # Unfreeze backbone if configured
        unfreeze_epoch = train_config.get("unfreeze_after_epoch")
        if unfreeze_epoch and epoch == unfreeze_epoch + 1:
            base_student.unfreeze_backbone()
            backbone_lr = lr * train_config.get("backbone_lr_multiplier", 0.1)
            optimizer = torch.optim.AdamW([
                {"params": base_student.backbone.parameters(), "lr": backbone_lr},
                {"params": base_student.head.parameters(), "lr": lr},
            ], weight_decay=weight_decay)
            remaining = student_epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, remaining))

        train_metrics = distill_one_epoch(
            student, train_loader_s2, criterion_hard, optimizer, device, epoch,
            temperature=temperature, alpha=alpha, teacher=teacher_for_s2,
        )

        gc.collect()
        torch.cuda.empty_cache()

        skip_val_until = train_config.get("skip_val_until_epoch", 0)
        if epoch < skip_val_until:
            val_metrics = {"val_loss": float('inf'), "val_acc": 0.0, "val_top5_acc": 0.0}
        else:
            val_metrics = validate_student(
                student, val_loader_s2, criterion_hard, device, epoch)
            gc.collect()
            torch.cuda.empty_cache()

        if epoch > warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time_s2

        history_s2["train_loss"].append(train_metrics["train_loss"])
        history_s2["train_loss_hard"].append(train_metrics["train_loss_hard"])
        history_s2["train_loss_kd"].append(train_metrics["train_loss_kd"])
        history_s2["train_acc"].append(train_metrics["train_acc"])
        history_s2["val_loss"].append(val_metrics["val_loss"])
        history_s2["val_acc"].append(val_metrics["val_acc"])
        history_s2["val_top5_acc"].append(val_metrics["val_top5_acc"])

        if main_process and csv_logger_s2 is not None:
            csv_logger_s2.log({
                "epoch": epoch,
                "train_loss": f"{train_metrics['train_loss']:.4f}",
                "train_loss_hard": f"{train_metrics['train_loss_hard']:.4f}",
                "train_loss_kd": f"{train_metrics['train_loss_kd']:.4f}",
                "train_acc": f"{train_metrics['train_acc']:.2f}",
                "val_loss": f"{val_metrics['val_loss']:.4f}",
                "val_acc": f"{val_metrics['val_acc']:.2f}",
                "val_top5_acc": f"{val_metrics['val_top5_acc']:.2f}",
                "lr": f"{current_lr:.6f}",
                "elapsed_sec": f"{elapsed:.1f}",
                "timestamp": datetime.now().isoformat(),
            })

        if main_process:
            print(f"[S2] Epoch {epoch}: loss={train_metrics['train_loss']:.4f} "
                  f"(CE={train_metrics['train_loss_hard']:.4f}, "
                  f"KD={train_metrics['train_loss_kd']:.4f}), "
                  f"train_acc={train_metrics['train_acc']:.1f}%, "
                  f"val_acc={val_metrics['val_acc']:.1f}%, "
                  f"val_top5={val_metrics['val_top5_acc']:.1f}%")

        # Save best student
        if val_metrics["val_acc"] > best_student_val_acc:
            best_student_val_acc = val_metrics["val_acc"]
            if main_process:
                # Save in Option A compatible format!
                # model_state_dict keys: backbone.* + head.* (384→C)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base_student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_student_val_acc,
                    "num_classes": num_classes,
                    "int_to_species": int_to_species,
                    "config": exp_config,
                    "model_type": "student_distilled",
                    "distillation": {
                        "temperature": temperature,
                        "alpha": alpha,
                        "teacher_val_acc": best_teacher_val_acc,
                    },
                }, student_dir / "best_model.pth")
            saved_best_student = True
            if main_process:
                print(f"  -> Saved best student (val_acc={best_student_val_acc:.1f}%)")

                # ALSO save to output_dir/best_model.pth for evaluation compatibility
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": base_student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_student_val_acc,
                    "num_classes": num_classes,
                    "int_to_species": int_to_species,
                    "config": exp_config,
                    "model_type": "student_distilled",
                }, output_dir / "best_model.pth")

        # Periodic checkpoint
        if (epoch % checkpoint_every == 0 or epoch == student_epochs) and main_process:
            torch.save({
                "epoch": epoch,
                "model_state_dict": base_student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_metrics["val_acc"],
                "best_val_acc": best_student_val_acc,
                "num_classes": num_classes,
                "int_to_species": int_to_species,
                "config": exp_config,
                "model_type": "student_distilled",
                "distillation": {
                    "temperature": temperature,
                    "alpha": alpha,
                },
                "history": history_s2,
            }, student_dir / f"checkpoint_epoch{epoch:03d}.pth")

    if csv_logger_s2 is not None:
        csv_logger_s2.close()

    # Always save final student checkpoint
    final_student_payload = {
        "epoch": student_epochs,
        "model_state_dict": base_student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_metrics["val_acc"],
        "best_val_acc": best_student_val_acc,
        "num_classes": num_classes,
        "int_to_species": int_to_species,
        "config": exp_config,
        "model_type": "student_distilled",
        "distillation": {
            "temperature": temperature,
            "alpha": alpha,
            "teacher_val_acc": best_teacher_val_acc,
        },
        "seed": seed,
    }
    if main_process:
        torch.save(final_student_payload, student_dir / "final_model.pth")
        torch.save({
            "epoch": student_epochs,
            "model_state_dict": base_student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_metrics["val_acc"],
            "num_classes": num_classes,
            "int_to_species": int_to_species,
            "config": exp_config,
            "model_type": "student_distilled",
            "seed": seed,
        }, output_dir / "final_model.pth")

    # If best checkpoint was never written (e.g., all val_acc==0), promote final.
    student_best_path = student_dir / "best_model.pth"
    if not saved_best_student:
        if main_process:
            torch.save(final_student_payload, student_best_path)
            torch.save({
                "epoch": student_epochs,
                "model_state_dict": base_student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["val_acc"],
                "num_classes": num_classes,
                "int_to_species": int_to_species,
                "config": exp_config,
                "model_type": "student_distilled",
                "seed": seed,
            }, output_dir / "best_model.pth")
        best_student_val_acc = val_metrics["val_acc"]
        if main_process:
            print("[Stage 2] No best checkpoint captured; promoted final_model.pth to best_model.pth")

    training_time_s2 = time.time() - start_time_s2
    history_s2["training_time_seconds"] = training_time_s2
    history_s2["best_val_acc"] = best_student_val_acc
    if main_process:
        save_experiment_results(student_dir, metadata_s2, history_s2,
                                best_student_val_acc, training_time_s2)

    # ============================
    # FINAL SUMMARY
    # ============================
    total_time = training_time_s1 + training_time_s2

    # Save overall summary
    overall_summary = {
        "experiment": experiment_name,
        "type": "Option D: Knowledge Distillation",
        "stage1_teacher": {
            "best_val_acc_img_only": best_teacher_val_acc,
            "training_time_hours": training_time_s1 / 3600,
        },
        "stage2_student": {
            "best_val_acc": best_student_val_acc,
            "training_time_hours": training_time_s2 / 3600,
        },
        "distillation_params": {
            "temperature": temperature,
            "alpha": alpha,
            "caption_model": caption_model_short,
        },
        "seed": seed,
        "run_subdir": run_subdir,
        "base_output_dir": str(base_output_dir),
        "resolved_output_dir": str(output_dir),
        "total_time_hours": total_time / 3600,
        "student_model_path": str(output_dir / "best_model.pth"),
        "note": "Student model is architecturally identical to Option A. "
                "Evaluate with Option A's evaluate_classifier.py.",
    }
    if main_process:
        with open(output_dir / "distillation_summary.json", "w") as f:
            json.dump(overall_summary, f, indent=2, default=str)

        print(f"\n{'=' * 80}")
        print(f"Option D: Knowledge Distillation Complete!")
        print(f"{'=' * 80}")
        print(f"  Teacher (Stage 1) val_acc (img-only): {best_teacher_val_acc:.2f}%")
        print(f"  Student (Stage 2) val_acc:            {best_student_val_acc:.2f}%")
        print(f"  Stage 1 time: {training_time_s1 / 3600:.1f}h")
        print(f"  Stage 2 time: {training_time_s2 / 3600:.1f}h")
        print(f"  Total time:   {total_time / 3600:.1f}h")
        print(f"\n  Student model: {output_dir / 'best_model.pth'}")
        print(f"  Evaluate with: python option_A/evaluate_classifier.py \\")
        print(f"    --config experiment_config.yaml --experiment {experiment_name}")
        print(f"{'=' * 80}\n")

    cleanup_distributed()
    return student, history_s2


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train pollen classifier via knowledge distillation (Option D)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name from config")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (applies to both stages)")
    parser.add_argument("--stage2_only", action="store_true",
                        help="Explicitly force Stage 2 only. "
                             "Default behavior is already Stage 2 only unless --train_teacher is set.")
    parser.add_argument("--train_teacher", action="store_true",
                        help="Train teacher from scratch (Stage 1 + Stage 2). "
                             "By default, the Option C (LUPI) teacher is reused.")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Explicit path to teacher checkpoint. "
                             "If unset, auto-discovers matching Option C checkpoint.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed for this run. "
                             "When provided, outputs are saved to a seed subfolder.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run tag appended to seed subfolder "
                             "(e.g., seed_42__ablationA).")
    parser.add_argument("--teacher_seed", type=int, default=None,
                        help="Optional seed used when auto-discovering Option C teacher "
                             "(defaults to --seed if provided).")
    parser.add_argument("--teacher_run_name", type=str, default=None,
                        help="Optional run tag used when auto-discovering Option C teacher.")
    parser.add_argument("--cache_logits", action="store_true",
                        help="Pre-cache all teacher logits before Stage 2 (slow but "
                             "avoids teacher in GPU memory). Default: on-the-fly.")
    parser.add_argument("--allow_cache_with_augmentation", action="store_true",
                        help="Allow --cache_logits even when stochastic train augmentation is enabled.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.experiment not in config["experiments"]:
        available = list(config["experiments"].keys())
        print(f"Error: Experiment '{args.experiment}' not found.")
        print(f"Available: {available}")
        sys.exit(1)

    stage2_only = not args.train_teacher
    if args.stage2_only:
        stage2_only = True

    train(config, args.experiment, args.device,
          max_samples=args.max_samples, epochs_override=args.epochs,
          stage2_only=stage2_only, teacher_checkpoint=args.teacher_checkpoint,
          train_teacher=args.train_teacher, cache_logits=args.cache_logits,
          seed_override=args.seed, run_name=args.run_name,
          teacher_seed=args.teacher_seed, teacher_run_name=args.teacher_run_name,
          allow_cache_with_augmentation=args.allow_cache_with_augmentation)


if __name__ == "__main__":
    main()
