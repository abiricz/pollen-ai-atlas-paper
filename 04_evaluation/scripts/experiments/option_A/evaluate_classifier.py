#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Pollen Classification Evaluation Script
========================================

Evaluate a trained classifier on test sets (TS1 legacy, TS2 expert).
Computes comprehensive metrics and generates confusion matrices.

Usage:
    python evaluate_classifier.py --config experiment_config.yaml --experiment linear_probe_all
    python evaluate_classifier.py --config experiment_config.yaml --experiment linear_probe_all --test_set ts2_expert

"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import json
import yaml
import time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix,
)

# Add project root to path (4 levels up from option_A/)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Use HITL-validated species mapping (not legacy lib.classifier)
from lib.species_mapping import load_caption_anchors, build_class_mappings, get_slide_class_id, get_trainable_slides

# Optional: visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

# Optional: Macenko stain normalization
try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    HAS_TIATOOLBOX = True
except ImportError:
    HAS_TIATOOLBOX = False
    MacenkoNormalizer = None


# =============================================================================
# SEED / RUN-NAME SUBFOLDER UTILITIES
# =============================================================================

def sanitize_run_name(run_name):
    if run_name is None:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in run_name.strip())
    return cleaned or None


def build_run_subdir(seed=None, run_name=None):
    clean_name = sanitize_run_name(run_name)
    if seed is None and clean_name is None:
        return None
    parts = []
    if seed is not None:
        parts.append(f"seed_{seed}")
    if clean_name is not None:
        parts.append(clean_name)
    return "__".join(parts)


def resolve_output_dir(base_output_dir, seed=None, run_name=None):
    run_subdir = build_run_subdir(seed, run_name)
    if run_subdir is None:
        return base_output_dir
    return base_output_dir / run_subdir


# =============================================================================
# TEST DATASET
# =============================================================================

class PollenTestDataset(Dataset):
    """
    Dataset for pollen classification evaluation on curated test sets.
    
    Loads ground truth annotations from curated GeoJSON files.
    Uses HITL-validated caption_anchors for species labels.
    Optionally filters to only species in training set.
    """
    
    def __init__(
        self,
        annotations_dir: Path,
        wsi_dir: Path,
        transform: transforms.Compose = None,
        stainnorm_func: callable = None,
        caption_anchors: Dict[str, str] = None,
        species_to_int: Dict[str, int] = None,
        training_species: set = None,
    ):
        """
        Args:
            annotations_dir: Path to test set annotations (ts1_legacy or ts2_expert)
            wsi_dir: Path to 00_raw_wsi directory
            transform: Image transforms (includes resize to 518x518)
            stainnorm_func: Optional stain normalization function
            caption_anchors: Slide → species mapping from caption_anchors
            species_to_int: Species → class_id mapping
            training_species: Optional set of species (lowercase) to filter to
        """
        self.annotations_dir = Path(annotations_dir)
        self.wsi_dir = Path(wsi_dir)
        self.transform = transform
        self.stainnorm_func = stainnorm_func
        self.training_species = training_species  # If set, only include these species
        
        # Load caption anchors for HITL-validated species labels
        if caption_anchors is None:
            caption_anchors = load_caption_anchors()
        if species_to_int is None:
            species_to_int, _, _ = build_class_mappings(caption_anchors)
        
        self.caption_anchors = caption_anchors
        self.species_to_int = species_to_int
        self.num_classes = len(set(species_to_int.values()))
        
        # Build sample index from GeoJSON files
        self.samples = []
        self._load_annotations()
        
        print(f"[TestDataset] Loaded {len(self.samples)} samples from {annotations_dir.name}")
    
    def _load_annotations(self):
        """Load all curated GeoJSON annotations, optionally filtered by training species."""
        geojson_files = list(self.annotations_dir.glob("*_curated.geojson"))
        
        skipped_slides = []
        skipped_species_count = 0
        
        for geojson_path in tqdm(geojson_files, desc="Loading annotations"):
            slide_name = geojson_path.stem.replace("_curated", "")
            
            # Skip slides without HITL-validated species
            if slide_name not in self.caption_anchors:
                skipped_slides.append(slide_name)
                continue
            
            # Get species from HITL-validated caption_anchors
            species = self.caption_anchors[slide_name]
            
            # Filter by training species if specified
            if self.training_species is not None:
                if species.lower() not in self.training_species:
                    skipped_species_count += 1
                    continue
            
            wsi_path = self._find_wsi_path(slide_name)
            
            with open(geojson_path) as f:
                data = json.load(f)
            
            for feature in data["features"]:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                
                # Skip non-pollen annotations (test regions, etc.)
                classification = props.get("classification", {})
                if classification.get("name") != "Pollen":
                    continue
                
                # Get bounding box from polygon
                coords = geom.get("coordinates", [[]])[0]
                if len(coords) < 4:
                    continue
                
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                
                self.samples.append({
                    "id": feature.get("id", "unknown"),
                    "slide": slide_name,
                    "bbox": bbox,
                    "species": species,
                    "wsi_path": wsi_path,
                    "annotation_name": props.get("name", ""),
                })
        
        # Report skipped slides
        if skipped_slides:
            print(f"  [Note] Skipped {len(skipped_slides)} slides without HITL-validated species:")
            for slide in skipped_slides[:5]:  # Only show first 5
                print(f"    - {slide}")
            if len(skipped_slides) > 5:
                print(f"    ... and {len(skipped_slides) - 5} more")
        
        if skipped_species_count > 0:
            print(f"  [Filter] Excluded {skipped_species_count} slides with species not in training set")
    
    def _find_wsi_path(self, slide_name: str) -> Optional[str]:
        """Find WSI file path for a slide."""
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
        
        # Apply stain normalization if enabled
        if self.stainnorm_func is not None and image is not None:
            try:
                image = Image.fromarray(self.stainnorm_func(np.array(image)))
            except Exception:
                pass
        
        # Apply transforms
        if self.transform is not None and image is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image) if image is not None else torch.zeros(3, 224, 224)
        
        # Get label from HITL-validated species
        species = sample["species"]
        label = self.species_to_int[species.lower()]  # No fallback needed - slides without anchors are skipped
        
        return {
            "image": image,
            "label": label,
            "sample_id": sample["id"],
            "species": sample["species"],
            "slide": sample["slide"],
        }
    
    def _extract_patch(self, sample: Dict) -> Optional[Image.Image]:
        """
        Extract patch from WSI using the exact bounding box.
        
        The bbox already defines the annotation region precisely.
        We extract the full bbox and resize to img_size (518) during transform.
        """
        wsi_path = sample.get("wsi_path")
        if wsi_path is None or not os.path.exists(wsi_path):
            return Image.new("RGB", (224, 224), (255, 255, 255))
        
        try:
            try:
                import tiffslide
                wsi = tiffslide.TiffSlide(wsi_path)
            except Exception:
                import openslide
                wsi = openslide.OpenSlide(wsi_path)
            
            x1, y1, x2, y2 = sample["bbox"]
            width = x2 - x1
            height = y2 - y1
            
            # Read the exact bbox region (no fixed-size cropping)
            region = wsi.read_region((x1, y1), 0, (width, height))
            wsi.close()
            
            return region.convert("RGB")
        except Exception as e:
            return Image.new("RGB", (224, 224), (255, 255, 255))


# =============================================================================
# MODEL LOADING
# =============================================================================

class PollenClassifierModel(nn.Module):
    """Classifier model (same as training script)."""
    
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        num_classes: int = 46,
        img_size: int = 518,
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
            pretrained=False,
            img_size=img_size,
            init_values=1e-5,
            num_classes=0,
        )
        
        self.head = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def load_model(checkpoint_path: Path, config: Dict, device: torch.device, num_classes: int = None) -> nn.Module:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Global config dict
        device: Device to load model to
        num_classes: Override number of classes (if None, uses checkpoint's saved value or config)
    """
    # Use num_classes from checkpoint metadata if available, else from arg, else config
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Try to get num_classes from checkpoint metadata (newer checkpoints have this)
    if num_classes is None:
        if "num_classes" in checkpoint:
            num_classes = checkpoint["num_classes"]
        else:
            num_classes = config.get("num_classes", 46)
            print(f"[WARNING] Checkpoint lacks num_classes metadata, using config value: {num_classes}")
    
    model = PollenClassifierModel(
        backbone_name=config["backbone_name"],
        num_classes=num_classes,
        img_size=config["img_size"],
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    
    print(f"[Model] Loaded from {checkpoint_path}")
    print(f"[Model] num_classes={num_classes}, best val_acc: {checkpoint.get('val_acc', 'N/A')}")
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 46,
    int_to_species: Dict[int, str] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation on test set.
    
    Returns:
        Dict with all metrics and predictions
    """
    if int_to_species is None:
        raise ValueError("int_to_species mapping must be provided (from lib.species_mapping)")
    
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_top5_correct = []
    all_samples = []
    
    inference_times = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"]
            
            start_time = time.time()
            logits = model(images)
            inference_times.append((time.time() - start_time) / images.size(0))
            
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            _, top5_pred = torch.topk(logits, min(5, logits.size(1)), dim=1)
            
            for i in range(len(labels)):
                all_labels.append(labels[i].item())
                all_preds.append(predicted[i].cpu().item())
                all_probs.append(probs[i].cpu().numpy())
                all_top5_correct.append(labels[i].item() in top5_pred[i].cpu().tolist())
                all_samples.append({
                    "sample_id": batch["sample_id"][i],
                    "species": batch["species"][i],
                    "slide": batch["slide"][i],
                    "label": labels[i].item(),
                    "pred": predicted[i].cpu().item(),
                })
    
    # Compute metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_top5_correct = np.array(all_top5_correct)
    
    # Filter out unknown/background for class-specific metrics
    valid_mask = all_labels < num_classes - 1  # Exclude Unknown/background
    num_unknown = (~valid_mask).sum()
    
    if num_unknown > 0:
        print(f"  [Note] {num_unknown} samples have Unknown label (will compute separate metrics)")
    
    # Get unique labels present in the test set (for proper macro averaging)
    unique_labels_in_test = sorted(set(all_labels))
    num_classes_in_test = len(unique_labels_in_test)
    
    # Compute overall metrics - macro averages ONLY over classes present in test set
    metrics = {
        "top1_accuracy": 100 * accuracy_score(all_labels, all_preds),
        "top5_accuracy": 100 * np.mean(all_top5_correct),
        "macro_f1": 100 * f1_score(all_labels, all_preds, labels=unique_labels_in_test, average="macro", zero_division=0),
        "weighted_f1": 100 * f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_precision": 100 * precision_score(all_labels, all_preds, labels=unique_labels_in_test, average="macro", zero_division=0),
        "macro_recall": 100 * recall_score(all_labels, all_preds, labels=unique_labels_in_test, average="macro", zero_division=0),
        "mean_inference_time_ms": 1000 * np.mean(inference_times),
        "total_samples": len(all_labels),
        "num_classes_in_test": num_classes_in_test,
        "num_unknown_labels": int(num_unknown),
    }
    
    # Compute metrics for known species only (excluding Unknown)
    if valid_mask.sum() > 0:
        valid_labels = all_labels[valid_mask]
        valid_preds = all_preds[valid_mask]
        valid_top5 = all_top5_correct[valid_mask]
        valid_unique_labels = sorted(set(valid_labels))
        
        metrics["valid_top1_accuracy"] = 100 * accuracy_score(valid_labels, valid_preds)
        metrics["valid_top5_accuracy"] = 100 * np.mean(valid_top5)
        metrics["valid_macro_f1"] = 100 * f1_score(valid_labels, valid_preds, labels=valid_unique_labels, average="macro", zero_division=0)
        metrics["valid_weighted_f1"] = 100 * f1_score(valid_labels, valid_preds, average="weighted", zero_division=0)
        metrics["valid_samples"] = int(valid_mask.sum())
    
    # Per-class metrics
    unique_labels = np.unique(all_labels)
    per_class = {}
    for label in unique_labels:
        label_name = int_to_species.get(label, f"class_{label}")
        mask = all_labels == label
        if mask.sum() > 0:
            per_class[label_name] = {
                "count": int(mask.sum()),
                "accuracy": 100 * (all_preds[mask] == label).mean(),
                "predicted_as": dict(Counter(all_preds[mask].tolist())),
            }
    metrics["per_class"] = per_class
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    metrics["confusion_matrix"] = cm.tolist()
    
    # Full classification report (only for classes in test set)
    report = classification_report(
        all_labels, all_preds,
        labels=unique_labels_in_test,
        target_names=[int_to_species.get(i, f"class_{i}") for i in unique_labels_in_test],
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    
    # Sample-level predictions
    metrics["predictions"] = all_samples
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (20, 16),
):
    """Plot and save confusion matrix."""
    if not HAS_VIZ:
        print("Warning: matplotlib/seaborn not available, skipping confusion matrix plot")
        return
    
    # Filter to only classes with samples
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    active_classes = (row_sums > 0) | (col_sums > 0)
    
    cm_filtered = cm[active_classes][:, active_classes]
    names_filtered = [n for i, n in enumerate(class_names) if active_classes[i]]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_filtered,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names_filtered,
        yticklabels=names_filtered,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"[Plot] Saved confusion matrix to {output_path}")


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def run_evaluation(config: Dict, experiment_name: str, test_set: str = None, device: str = "cuda:0",
                   seed: int = None, run_name: str = None):
    """Main evaluation function."""
    
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    
    device = torch.device(device)
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_config['name']}")
    print(f"Device: {device}")
    if seed is not None:
        print(f"Seed: {seed}")
    if run_name is not None:
        print(f"Run name: {run_name}")
    print(f"{'='*60}\n")
    
    # Paths
    data_root = PROJECT_ROOT / global_config["data_root"]
    base_output_dir = PROJECT_ROOT / exp_config["output_dir"]
    output_dir = resolve_output_dir(base_output_dir, seed=seed, run_name=run_name)
    checkpoint_path = output_dir / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Run training first with: python train_classifier.py --config ... --experiment ...")
        return None
    
    # Load HITL-validated class mappings from caption_anchors FIRST
    all_anchors = load_caption_anchors()
    caption_anchors, excluded_slides = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    
    print(f"[Eval] Loaded {len(all_anchors)} slides from caption_anchors/")
    print(f"[Eval] Using {num_classes} classes from caption_anchors (HITL-validated)")
    if excluded_slides:
        print(f"[Eval] Excluded slides: {', '.join(excluded_slides)}")
    
    # Get training region and its species for filtering test samples
    train_region = exp_config.get("training", {}).get("train_region", "all")
    training_species = get_training_species(train_region)
    print(f"[Eval] Training region: {train_region} ({len(training_species)} species)")
    
    # Load model with correct num_classes
    model = load_model(checkpoint_path, global_config, device, num_classes=num_classes)
    
    # Get test sets to evaluate
    test_sets = [test_set] if test_set else exp_config.get("eval_test_sets", ["ts2_expert"])
    
    # Transforms - use normalization from experiment config (supports custom stainnorm stats)
    img_size = global_config["img_size"]
    preproc_config = exp_config.get("preprocessing", {})
    
    # Check for experiment-level normalization overrides (for stainnorm experiments)
    if preproc_config.get("normalize_mean") and preproc_config.get("normalize_std"):
        mean = preproc_config["normalize_mean"]
        std = preproc_config["normalize_std"]
        print(f"[Eval] Using custom normalization from experiment config: mean={mean}, std={std}")
    else:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Setup stain normalization if enabled
    stainnorm_func = None
    preproc_config = exp_config.get("preprocessing", {})
    if preproc_config.get("stainnorm", {}).get("enabled", False) and HAS_TIATOOLBOX:
        ref_path = preproc_config["stainnorm"].get("reference_path")
        if ref_path:
            ref_image = np.load(PROJECT_ROOT / ref_path)
            normalizer = MacenkoNormalizer()
            normalizer.fit(ref_image)
            stainnorm_func = normalizer.transform
    
    all_results = {}
    
    for ts_name in test_sets:
        print(f"\n--- Evaluating on {ts_name} ---")
        
        ts_config = config["test_sets"].get(ts_name)
        if ts_config is None:
            print(f"Warning: Test set '{ts_name}' not found in config")
            continue
        
        annotations_dir = PROJECT_ROOT / ts_config["path"]
        
        # Build test dataset with HITL-validated class mappings
        # Filter to training species (fair evaluation - only species the model was trained on)
        test_dataset = PollenTestDataset(
            annotations_dir=annotations_dir,
            wsi_dir=data_root / "00_raw_wsi",
            transform=transform,
            stainnorm_func=stainnorm_func,
            caption_anchors=caption_anchors,
            species_to_int=species_to_int,
            training_species=training_species,  # Filter to species in training set
        )
        
        if len(test_dataset) == 0:
            print(f"Warning: No samples found in {ts_name}")
            continue
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=global_config.get("num_workers", 8),
            pin_memory=True,
        )
        
        # Evaluate with HITL-validated class mappings
        metrics = evaluate(
            model, test_loader, device,
            num_classes=num_classes,
            int_to_species=int_to_species,
        )
        
        # Print summary (focus on Top-1, Precision, Recall, F1 - skip Top-5)
        print(f"\n{ts_name} Results:")
        print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        print(f"  Macro Precision: {metrics['macro_precision']:.2f}%")
        print(f"  Macro Recall: {metrics['macro_recall']:.2f}%")
        print(f"  Macro F1: {metrics['macro_f1']:.2f}%")
        print(f"  Samples: {metrics['total_samples']} ({metrics['num_classes_in_test']} taxa)")
        
        # Save results
        results_path = output_dir / f"eval_{ts_name}.json"
        with open(results_path, "w") as f:
            # Remove confusion matrix from JSON (too large)
            metrics_json = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            json.dump(metrics_json, f, indent=2, default=str)
        print(f"  Results saved to: {results_path}")
        
        # Plot confusion matrix with HITL-validated class names
        cm = np.array(metrics["confusion_matrix"])
        class_names = [int_to_species.get(i, f"class_{i}") for i in range(num_classes)]
        plot_confusion_matrix(
            cm, class_names,
            output_dir / f"confusion_matrix_{ts_name}.png",
            title=f"{exp_config['name']} - {ts_name}",
        )
        
        all_results[ts_name] = metrics
    
    # Save combined summary (keep all metrics for JSON, just simplified console output)
    summary = {
        "experiment": experiment_name,
        "train_region": train_region,
        "training_species_count": len(training_species),
        "timestamp": datetime.now().isoformat(),
        "results": {
            ts: {
                "top1_accuracy": r["top1_accuracy"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "macro_f1": r["macro_f1"],
                "total_samples": r["total_samples"],
                "num_classes": r["num_classes_in_test"],
            }
            for ts, r in all_results.items()
        }
    }
    
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return all_results


# =============================================================================
# CROSS-DATASET EVALUATION (Inter-region)
# =============================================================================

def load_cross_dataset_matrix() -> Dict:
    """Load pre-computed cross-dataset overlap matrix."""
    candidates = [
        PROJECT_ROOT / "data" / "04_evaluation" / "results" / "cross_dataset_matrix.json",
        PROJECT_ROOT / "04_evaluation" / "results" / "cross_dataset_matrix.json",
    ]
    for matrix_path in candidates:
        if matrix_path.exists():
            with open(matrix_path) as f:
                return json.load(f)
    raise FileNotFoundError(
        "Cross-dataset matrix not found. Expected data/04_evaluation/results/"
        "cross_dataset_matrix.json. Run: python 04_evaluation/scripts/analysis/"
        "create_cross_dataset_matrix.py"
    )


def get_training_species(train_region: str) -> set:
    """Get species available in the training region(s)."""
    matrix = load_cross_dataset_matrix()
    
    if train_region == "all":
        # All species from all regions
        all_species = set()
        for region_data in matrix["datasets"].values():
            all_species.update(s.lower() for s in region_data.get("species", []))
        return all_species
    
    region_data = matrix["datasets"].get(train_region)
    if region_data is None:
        print(f"[WARNING] Unknown training region '{train_region}', returning all species")
        return set()
    
    return set(s.lower() for s in region_data.get("species", []))


def get_dataset_from_slide(slide_name: str) -> str:
    """Determine dataset from slide name."""
    if slide_name.startswith("mediterranean_"):
        return "mediterranean"
    elif slide_name.startswith("hun_") or slide_name.startswith("Ambrosia-Iva"):
        return "hungarian"
    
    swedish_patterns = ["_40x_ZS", "_20x_ZS", "layers_40x", "merged_reference", "mm_circle"]
    if any(pattern in slide_name for pattern in swedish_patterns):
        return "swedish"
    
    return "french"


def get_overlapping_species(source_dataset: str, target_dataset: str) -> set:
    """Get species that overlap between source and target datasets."""
    matrix = load_cross_dataset_matrix()
    overlap = matrix["overlap_matrix"].get(source_dataset, {}).get(target_dataset, [])
    return set(s.lower() for s in overlap)


class CrossDatasetTestDataset(Dataset):
    """
    Dataset for cross-region evaluation using expert-validated ground truth.
    
    Filters PollenTestDataset samples by target dataset and tracks overlap
    with source dataset species.
    """
    
    def __init__(
        self,
        source_dataset: str,
        target_dataset: str,
        annotations_dir: Path,
        wsi_dir: Path,
        transform: transforms.Compose = None,
        stainnorm_func: callable = None,
        caption_anchors: Dict[str, str] = None,
        species_to_int: Dict[str, int] = None,
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.annotations_dir = Path(annotations_dir)
        self.wsi_dir = Path(wsi_dir)
        self.transform = transform
        self.stainnorm_func = stainnorm_func
        
        if caption_anchors is None:
            caption_anchors = load_caption_anchors()
        if species_to_int is None:
            species_to_int, _, _ = build_class_mappings(caption_anchors)
        
        self.caption_anchors = caption_anchors
        self.species_to_int = species_to_int
        self.num_classes = len(set(species_to_int.values()))
        
        # Get overlapping species (for non-self evaluation)
        if source_dataset == target_dataset:
            # For self-evaluation, all species are considered "in overlap"
            self.overlapping_species = set(s.lower() for s in species_to_int.keys())
        else:
            self.overlapping_species = get_overlapping_species(source_dataset, target_dataset)
        
        self.samples = []
        self._load_samples()
        
        overlapping_count = sum(1 for s in self.samples if s["in_overlap"])
        print(f"[CrossDataset] {source_dataset} → {target_dataset}")
        print(f"  Loaded {len(self.samples)} expert-validated samples")
        print(f"  Overlapping taxa: {len(self.overlapping_species)}")
        print(f"  Samples in overlap: {overlapping_count} ({100*overlapping_count/max(1,len(self.samples)):.1f}%)")
    
    def _load_samples(self):
        """Load samples from ground truth GeoJSON, filtered by target dataset."""
        geojson_files = list(self.annotations_dir.glob("*_curated.geojson"))
        
        for geojson_path in geojson_files:
            slide_name = geojson_path.stem.replace("_curated", "")
            
            # Filter by target dataset
            if get_dataset_from_slide(slide_name) != self.target_dataset:
                continue
            
            # Skip slides without HITL-validated species
            if slide_name not in self.caption_anchors:
                continue
            
            wsi_path = self._find_wsi_path(slide_name)
            species = self.caption_anchors[slide_name]
            
            with open(geojson_path) as f:
                data = json.load(f)
            
            for feature in data["features"]:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                
                # Skip non-pollen annotations
                classification = props.get("classification", {})
                if classification.get("name") != "Pollen":
                    continue
                
                # Get bounding box from polygon
                coords = geom.get("coordinates", [[]])[0]
                if len(coords) < 4:
                    continue
                
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                
                self.samples.append({
                    "id": feature.get("id", "unknown"),
                    "slide": slide_name,
                    "bbox": bbox,
                    "species": species,
                    "wsi_path": wsi_path,
                    "in_overlap": species.lower() in self.overlapping_species,
                })
    
    def _find_wsi_path(self, slide_name: str) -> Optional[str]:
        for dataset in ["french", "hungarian", "mediterranean", "swedish"]:
            for ext in [".tif", ".tiff", ".svs", ".ndpi"]:
                p = self.wsi_dir / dataset / f"{slide_name}{ext}"
                if p.exists():
                    return str(p)
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = self._extract_patch(sample)
        
        # Apply stain normalization if enabled (using training reference)
        if self.stainnorm_func is not None and image is not None:
            try:
                image = Image.fromarray(self.stainnorm_func(np.array(image)))
            except Exception:
                pass  # Keep original if stainnorm fails
        
        if self.transform is not None and image is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image) if image is not None else torch.zeros(3, 224, 224)
        
        species = sample["species"]
        label = self.species_to_int.get(species.lower(), self.num_classes - 1)
        
        return {
            "image": image,
            "label": label,
            "sample_id": sample["id"],
            "species": sample["species"],
            "slide": sample["slide"],
            "in_overlap": sample["in_overlap"],
        }
    
    def _extract_patch(self, sample: Dict) -> Optional[Image.Image]:
        wsi_path = sample.get("wsi_path")
        if wsi_path is None or not os.path.exists(wsi_path):
            return Image.new("RGB", (224, 224), (255, 255, 255))
        
        try:
            try:
                import tiffslide
                wsi = tiffslide.TiffSlide(wsi_path)
            except Exception:
                import openslide
                wsi = openslide.OpenSlide(wsi_path)
            
            x1, y1, x2, y2 = sample["bbox"]
            region = wsi.read_region((x1, y1), 0, (x2-x1, y2-y1))
            wsi.close()
            return region.convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (255, 255, 255))


def run_cross_dataset_evaluation(
    config: Dict,
    experiment_name: str,
    source_dataset: str,
    target_datasets: List[str],
    device: str = "cuda:0",
    seed: int = None,
    run_name: str = None,
) -> Dict[str, Any]:
    """
    Run cross-dataset evaluation with overlapping taxa filtering.
    
    Evaluates model trained on source_dataset against:
    1. source_dataset itself (in-domain baseline)
    2. Each target_dataset (cross-domain generalization)
    
    For each pair, reports:
    - All samples: metrics on all test samples from target
    - Overlap only: metrics restricted to taxa present in both source and target
    """
    
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    
    device = torch.device(device)
    print(f"\n{'='*70}")
    print(f"CROSS-DATASET EVALUATION")
    print(f"{'='*70}")
    print(f"Source (training): {source_dataset.upper()}")
    print(f"Targets: {', '.join(t.upper() for t in target_datasets)}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"{'='*70}\n")
    
    data_root = PROJECT_ROOT / global_config["data_root"]
    base_output_dir = PROJECT_ROOT / exp_config["output_dir"]
    output_dir = resolve_output_dir(base_output_dir, seed=seed, run_name=run_name)
    checkpoint_path = output_dir / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    all_anchors = load_caption_anchors()
    caption_anchors, _ = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    
    model = load_model(checkpoint_path, global_config, device, num_classes=num_classes)
    
    # Transforms - use normalization from experiment config (supports custom stainnorm stats)
    img_size = global_config["img_size"]
    preproc_config = exp_config.get("preprocessing", {})
    
    # Check for experiment-level normalization overrides (for stainnorm experiments)
    if preproc_config.get("normalize_mean") and preproc_config.get("normalize_std"):
        mean = preproc_config["normalize_mean"]
        std = preproc_config["normalize_std"]
        print(f"[Cross-Eval] Using custom normalization: mean={mean}, std={std}")
    else:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
        
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Setup stain normalization if enabled (uses training reference for ALL targets!)
    stainnorm_func = None
    if preproc_config.get("stainnorm", {}).get("enabled", False) and HAS_TIATOOLBOX:
        ref_path = preproc_config["stainnorm"].get("reference_path")
        if ref_path:
            ref_image = np.load(PROJECT_ROOT / ref_path)
            normalizer = MacenkoNormalizer()
            normalizer.fit(ref_image)
            stainnorm_func = normalizer.transform
            print(f"[Cross-Eval] Using stain normalization with reference: {ref_path}")
    
    all_results = {}
    
    # Evaluate on all targets including source (for in-domain baseline)
    all_targets = [source_dataset] + [t for t in target_datasets if t != source_dataset]
    
    # Use ts2_expert (expert-validated ground truth) for cross-dataset evaluation
    annotations_dir = PROJECT_ROOT / "04_evaluation" / "annotations" / "ts2_expert"
    
    for target in all_targets:
        is_self = (target == source_dataset)
        
        # For self-evaluation, all species overlap
        if is_self:
            overlap = set(s.lower() for s in species_to_int.keys())
            overlap_count = "all"
        else:
            overlap = get_overlapping_species(source_dataset, target)
            overlap_count = len(overlap)
            if not overlap:
                print(f"\n[SKIP] {source_dataset} → {target}: No overlapping taxa")
                continue
        
        label = f"{source_dataset} → {target}" + (" (IN-DOMAIN)" if is_self else "")
        print(f"\n{'─'*60}")
        print(f"{label}")
        print(f"Overlapping taxa: {overlap_count}")
        print(f"{'─'*60}")
        
        test_dataset = CrossDatasetTestDataset(
            source_dataset=source_dataset,
            target_dataset=target,
            annotations_dir=annotations_dir,
            wsi_dir=data_root / "00_raw_wsi",
            transform=transform,
            stainnorm_func=stainnorm_func,  # Use training reference for all targets!
            caption_anchors=caption_anchors,
            species_to_int=species_to_int,
        )
        
        if len(test_dataset) == 0:
            print(f"[SKIP] No samples found")
            continue
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=global_config.get("num_workers", 8),
            pin_memory=True,
        )
        
        # Evaluate with overlap filtering
        metrics = evaluate_with_overlap_filter(
            model, test_loader, device, num_classes, int_to_species, species_to_int, overlap
        )
        
        # Print results for this pair (Top-1, Precision, Recall, F1 - skip Top-5)
        print(f"\n  All samples ({metrics['all'].get('num_samples', 0):,} samples, "
              f"{metrics['all'].get('num_classes', 0)} taxa):")
        print(f"    Top-1: {metrics['all'].get('top1_accuracy', 0):.1f}%")
        print(f"    Prec: {metrics['all'].get('macro_precision', 0):.1f}%  Recall: {metrics['all'].get('macro_recall', 0):.1f}%  F1: {metrics['all'].get('macro_f1', 0):.1f}%")
        
        if not is_self:
            print(f"\n  Overlap only ({metrics['overlap'].get('num_samples', 0):,} samples, "
                  f"{len(overlap)} taxa):")
            print(f"    Top-1: {metrics['overlap'].get('top1_accuracy', 0):.1f}%")
            print(f"    Prec: {metrics['overlap'].get('macro_precision', 0):.1f}%  Recall: {metrics['overlap'].get('macro_recall', 0):.1f}%  F1: {metrics['overlap'].get('macro_f1', 0):.1f}%")
            print(f"    Taxa: {', '.join(sorted(overlap)[:10])}{'...' if len(overlap) > 10 else ''}")
        
        result_key = f"{source_dataset}_to_{target}"
        all_results[result_key] = {
            "source": source_dataset,
            "target": target,
            "is_in_domain": is_self,
            "overlap_taxa_count": len(overlap) if not is_self else "all",
            "all": metrics["all"],
            "overlap": metrics["overlap"],
            "overlap_species": sorted(overlap) if not is_self else [],
        }
        
        # Save individual result file
        result_path = output_dir / f"eval_cross_{result_key}.json"
        with open(result_path, "w") as f:
            json.dump(all_results[result_key], f, indent=2, default=str)
    
    # Print summary table and save to markdown
    print_cross_summary_table(all_results, source_dataset, output_dir)
    
    # Save summary
    summary_path = output_dir / f"eval_cross_{source_dataset}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "source": source_dataset,
            "targets": target_datasets,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }, f, indent=2, default=str)
    
    print(f"\n[Saved] Results to {output_dir}/eval_cross_*.json")
    
    return all_results


def print_cross_summary_table(results: Dict, source: str, output_dir: Path = None):
    """Print clean summary table for cross-dataset evaluation and save to disk."""
    lines = []
    lines.append(f"# Cross-Region Evaluation Summary: {source.upper()}")
    lines.append("")
    lines.append(f"| Target | Type | #Taxa | #Samples | Top-1 | Prec | Recall | F1 |")
    lines.append("|--------|------|-------|----------|-------|------|--------|-----|")
    
    # Print header
    print(f"\n{'='*80}")
    print(f"SUMMARY: {source.upper()} → ALL TARGETS")
    print(f"{'='*80}")
    
    print(f"\n{'Target':<15} {'Type':<12} {'#Taxa':>6} {'#Samples':>9} {'Top-1':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("─" * 80)
    
    for key, data in results.items():
        target = data["target"].capitalize()
        is_self = data.get("is_in_domain", False)
        type_str = "IN-DOMAIN" if is_self else "CROSS"
        
        # Show overlap metrics for cross, all metrics for self
        metrics = data["all"] if is_self else data["overlap"]
        n_taxa = data.get("overlap_taxa_count", 0)
        if n_taxa == "all":
            n_taxa = metrics.get("num_classes", 0)
        
        top1 = metrics.get('top1_accuracy', 0)
        prec = metrics.get('macro_precision', 0)
        rec = metrics.get('macro_recall', 0)
        f1 = metrics.get('macro_f1', 0)
        samples = metrics.get('num_samples', 0)
        
        print(f"{target:<15} {type_str:<12} {n_taxa:>6} {samples:>9,} "
              f"{top1:>7.1f}% {prec:>7.1f}% {rec:>7.1f}% {f1:>7.1f}%")
        
        lines.append(f"| {target} | {type_str} | {n_taxa} | {samples:,} | {top1:.1f}% | {prec:.1f}% | {rec:.1f}% | {f1:.1f}% |")
    
    print("─" * 80)
    print()
    
    # Save to markdown file
    if output_dir:
        md_path = output_dir / f"eval_cross_{source}_summary.md"
        md_path.write_text("\n".join(lines) + "\n")
        print(f"[Saved] Summary table to {md_path}")


def evaluate_with_overlap_filter(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    int_to_species: Dict[int, str],
    species_to_int: Dict[str, int],
    overlapping_species: set,
) -> Dict[str, Any]:
    """Evaluate with separate metrics for all samples and overlapping taxa only."""
    
    model.eval()
    all_labels, all_preds, all_top5, all_in_overlap = [], [], [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"]
            in_overlap = batch["in_overlap"]
            
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            _, top5_pred = torch.topk(logits, min(5, logits.size(1)), dim=1)
            
            for i in range(len(labels)):
                all_labels.append(labels[i].item())
                all_preds.append(predicted[i].cpu().item())
                all_top5.append(labels[i].item() in top5_pred[i].cpu().tolist())
                all_in_overlap.append(bool(in_overlap[i]))
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_top5 = np.array(all_top5)
    all_in_overlap = np.array(all_in_overlap)
    
    def compute_subset_metrics(mask, name):
        if mask.sum() == 0:
            return {"note": f"No {name} samples"}
        labels = all_labels[mask]
        preds = all_preds[mask]
        top5 = all_top5[mask]
        unique = sorted(set(labels))
        return {
            "top1_accuracy": 100 * accuracy_score(labels, preds),
            "top5_accuracy": 100 * np.mean(top5),
            "macro_f1": 100 * f1_score(labels, preds, labels=unique, average="macro", zero_division=0),
            "macro_precision": 100 * precision_score(labels, preds, labels=unique, average="macro", zero_division=0),
            "macro_recall": 100 * recall_score(labels, preds, labels=unique, average="macro", zero_division=0),
            "num_samples": int(mask.sum()),
            "num_classes": len(unique),
        }
    
    return {
        "all": compute_subset_metrics(np.ones(len(all_labels), dtype=bool), "all"),
        "overlap": compute_subset_metrics(all_in_overlap, "overlap"),
        "overlap_species": sorted(overlapping_species),
    }


def print_paper_table(results: Dict, experiment: str, source: str, targets: List[str]):
    """Print paper-ready LaTeX table (deprecated, use print_cross_summary_table)."""
    # Now handled by print_cross_summary_table
    pass



# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pollen classifier (test sets + cross-region)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard test set evaluation
  python evaluate_classifier.py --config experiment_config.yaml --experiment linear_probe_all
  
  # Cross-dataset evaluation
  python evaluate_classifier.py --config experiment_config.yaml --experiment linear_probe_french \\
      --source_dataset french --target_datasets hungarian mediterranean swedish
        """
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Experiment name from config")
    parser.add_argument("--test_set", type=str, default=None,
                        help="Specific test set to evaluate (default: all from config)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    
    # Cross-dataset options
    parser.add_argument("--source_dataset", type=str, default=None,
                        help="Source (training) dataset for cross-region eval")
    parser.add_argument("--target_datasets", type=str, nargs="+", default=None,
                        help="Target datasets for cross-region eval")
    parser.add_argument("--seed", type=int, default=None,
                        help="Evaluate a specific seeded run subfolder (seed_<N>)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run tag used at training time (seed_<N>__<run_name>)")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.experiment not in config["experiments"]:
        available = list(config["experiments"].keys())
        print(f"Error: Experiment '{args.experiment}' not found.")
        print(f"Available: {available}")
        sys.exit(1)
    
    # Cross-dataset evaluation
    if args.source_dataset and args.target_datasets:
        run_cross_dataset_evaluation(
            config, args.experiment, args.source_dataset, args.target_datasets, args.device,
            seed=args.seed, run_name=args.run_name,
        )
    else:
        # Standard test set evaluation
        run_evaluation(config, args.experiment, args.test_set, args.device,
                       seed=args.seed, run_name=args.run_name)


if __name__ == "__main__":
    main()
