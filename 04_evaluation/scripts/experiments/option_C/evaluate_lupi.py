#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
LUPI Evaluation Script — Wraps Option A's evaluate_classifier.py
================================================================

Evaluates LUPI (Option C) models using the EXACT same evaluation pipeline
as Option A. The only difference is model loading — LUPI models have a
768-dim head (384 image + 384 text) vs Option A's 384-dim head.

At test time, the text embedding is always zero, so the evaluation
measures image-only performance of a model trained with privileged info.

This script:
1. Loads a LUPI checkpoint (best_model.pth)
2. Wraps it in an adapter that zeros out the text branch
3. Runs the EXACT same evaluate() function from Option A

Usage:
    python evaluate_lupi.py --config ../experiment_config.yaml --experiment lupi_all
    python evaluate_lupi.py --config ../experiment_config.yaml --experiment lupi_all \\
        --source_dataset french --target_datasets hungarian swedish mediterranean
"""

import os
import sys
import argparse
import json
import yaml
import torch
import torch.nn as nn
import timm
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import EVERYTHING from Option A's evaluation (reuse completely)
sys.path.insert(0, str(PROJECT_ROOT / "04_evaluation" / "scripts" / "experiments" / "option_A"))
from evaluate_classifier import (
    PollenTestDataset,
    CrossDatasetTestDataset,
    evaluate,
    evaluate_with_overlap_filter,
    plot_confusion_matrix,
    load_cross_dataset_matrix,
    get_training_species,
    get_dataset_from_slide,
    get_overlapping_species,
    print_cross_summary_table,
)

from torchvision import transforms
from torch.utils.data import DataLoader

from lib.species_mapping import load_caption_anchors, build_class_mappings, get_trainable_slides

# Optional stain normalization
try:
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer
    HAS_TIATOOLBOX = True
except ImportError:
    HAS_TIATOOLBOX = False


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
# LUPI MODEL FOR INFERENCE (image-only, zero text)
# =============================================================================

class LUPIInferenceModel(nn.Module):
    """
    LUPI model adapted for image-only inference.
    
    Loads the full LUPI checkpoint (backbone + 768-dim head) and
    automatically zeros the text embedding portion during forward pass.
    
    The forward(x) signature is IDENTICAL to Option A's PollenClassifierModel,
    making it drop-in compatible with evaluate_classifier.py's evaluate().
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
        num_classes: int = 46,
        img_size: int = 518,
        img_embed_dim: int = 384,
        text_embed_dim: int = 384,
    ):
        super().__init__()
        self.img_embed_dim = img_embed_dim
        self.text_embed_dim = text_embed_dim
        
        # Build backbone (same as training)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            img_size=img_size,
            init_values=1e-5,
            num_classes=0,
        )
        
        # LUPI head: 768 → num_classes
        self.head = nn.Linear(img_embed_dim + text_embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image-only forward (zero text embedding)."""
        img_features = self.backbone(x)  # (B, 384)
        
        # Zero text embedding
        text_zeros = torch.zeros(
            img_features.size(0), self.text_embed_dim,
            device=img_features.device, dtype=img_features.dtype,
        )
        
        fused = torch.cat([img_features, text_zeros], dim=1)  # (B, 768)
        logits = self.head(fused)  # (B, num_classes)
        return logits


def load_lupi_model(checkpoint_path: str, config: dict, device: torch.device,
                    num_classes: int = None) -> nn.Module:
    """Load LUPI model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pth
        config: Global config dict
        device: Device to load to
        num_classes: Override num_classes
        
    Returns:
        LUPIInferenceModel ready for evaluation
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Get dimensions from checkpoint
    ckpt_num_classes = checkpoint.get("num_classes", num_classes or 46)
    img_embed_dim = checkpoint.get("img_embed_dim", 384)
    text_embed_dim = checkpoint.get("text_embed_dim", 384)
    
    if num_classes and num_classes != ckpt_num_classes:
        print(f"[WARNING] Config num_classes ({num_classes}) != checkpoint ({ckpt_num_classes})")
    
    model = LUPIInferenceModel(
        backbone_name=config.get("backbone_name", "vit_small_patch14_dinov2.lvd142m"),
        num_classes=ckpt_num_classes,
        img_size=config.get("img_size", 518),
        img_embed_dim=img_embed_dim,
        text_embed_dim=text_embed_dim,
    )
    
    # Load state dict
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    
    val_acc = checkpoint.get("val_acc", 0)
    epoch = checkpoint.get("epoch", "?")
    print(f"[Model] Loaded LUPI checkpoint (epoch {epoch}, val_acc={val_acc:.1f}%)")
    print(f"[Model] Architecture: backbone → [{img_embed_dim}+{text_embed_dim}] → {ckpt_num_classes}")
    print(f"[Model] Inference: text embedding zeroed (image-only)")
    
    model = model.to(device)
    model.eval()
    return model


# =============================================================================
# EVALUATION FUNCTIONS (mirrors Option A exactly)
# =============================================================================

def run_evaluation(config, experiment_name, test_set=None, device="cuda:0",
                   seed=None, run_name=None):
    """Run standard evaluation on TS1/TS2 (identical logic to Option A)."""
    
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    
    device = torch.device(device)
    
    # Load species mappings (use trainable anchors for dataset filtering, like Option A)
    all_anchors = load_caption_anchors()
    caption_anchors, excluded_slides = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    
    # Get training species for fair filtering
    train_region = exp_config["training"]["train_region"]
    training_species = get_training_species(train_region)
    num_workers = global_config.get("num_workers", 8)
    
    # Load model
    base_output_dir = PROJECT_ROOT / exp_config["output_dir"]
    output_dir = resolve_output_dir(base_output_dir, seed=seed, run_name=run_name)
    model_path = output_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("Train first: python train_lupi.py --config ... --experiment ...")
        sys.exit(1)
    
    model = load_lupi_model(str(model_path), global_config, device, num_classes)
    
    # Build transforms (identical to Option A)
    img_size = global_config["img_size"]
    preprocessing_config = exp_config.get("preprocessing", {})
    
    if preprocessing_config.get("normalization_mode") == "custom":
        mean = preprocessing_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = preprocessing_config.get("normalize_std", [0.229, 0.224, 0.225])
    else:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Stain normalization (must match Option A's contract: np.ndarray → np.ndarray)
    stainnorm_func = None
    stainnorm_config = preprocessing_config.get("stainnorm", {})
    if stainnorm_config.get("enabled", False) and HAS_TIATOOLBOX:
        ref_path = stainnorm_config.get("reference_path")
        if ref_path:
            import numpy as np
            ref_full = PROJECT_ROOT / ref_path
            if ref_full.exists():
                reference_img = np.load(ref_full)
                if reference_img.dtype != np.uint8:
                    reference_img = (reference_img * 255).astype(np.uint8)
                normalizer = MacenkoNormalizer()
                normalizer.fit(reference_img)
                stainnorm_func = normalizer.transform  # np.ndarray → np.ndarray
    
    # Evaluate on test sets
    test_sets_to_eval = [test_set] if test_set else exp_config.get("eval_test_sets", ["ts2_expert"])
    
    all_results = {}
    
    for ts_name in test_sets_to_eval:
        if ts_name not in config.get("test_sets", {}):
            print(f"[WARNING] Test set '{ts_name}' not found in config, skipping")
            print(f"  Available: {list(config.get('test_sets', {}).keys())}")
            continue
        ts_config = config["test_sets"][ts_name]
        annotations_dir = PROJECT_ROOT / ts_config["path"]
        wsi_dir = PROJECT_ROOT / global_config["data_root"] / "00_raw_wsi"
        
        print(f"\n{'='*60}")
        print(f"Evaluating on: {ts_name}")
        print(f"Annotations: {annotations_dir}")
        print(f"{'='*60}")
        
        test_dataset = PollenTestDataset(
            annotations_dir=annotations_dir,
            wsi_dir=wsi_dir,
            transform=transform,
            stainnorm_func=stainnorm_func,
            caption_anchors=caption_anchors,
            species_to_int=species_to_int,
            training_species=training_species,
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        
        print(f"[Test] {len(test_dataset)} samples, {len(test_loader)} batches")
        
        # Use Option A's evaluate() — model.forward(x) returns logits
        results = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            num_classes=num_classes,
            int_to_species=int_to_species,
        )
        
        all_results[ts_name] = results
        
        print(f"\n--- {ts_name} Results ---")
        print(f"Top-1: {results['top1_accuracy']:.1f}%")
        print(f"Top-5: {results['top5_accuracy']:.1f}%")
        print(f"Macro F1: {results['macro_f1']:.1f}%")
        
        # Save results
        results_save = {k: v for k, v in results.items() if k != "confusion_matrix"}
        with open(output_dir / f"eval_{ts_name}.json", "w") as f:
            json.dump(results_save, f, indent=2, default=str)
        
        # Confusion matrix plot
        if "confusion_matrix" in results:
            import numpy as np
            cm = np.array(results["confusion_matrix"])
            active_classes = [i for i in range(num_classes) if cm[i].sum() > 0 or cm[:, i].sum() > 0]
            if active_classes:
                cm_active = cm[np.ix_(active_classes, active_classes)]
                class_names = [int_to_species.get(i, f"cls_{i}") for i in active_classes]
                plot_confusion_matrix(
                    cm_active, class_names,
                    output_dir / f"confusion_matrix_{ts_name}.png",
                    title=f"LUPI {experiment_name} - {ts_name}",
                )
    
    # Save summary (match Option A's schema exactly)
    from datetime import datetime
    summary = {
        "experiment": experiment_name,
        "train_region": train_region,
        "training_species_count": len(training_species),
        "timestamp": datetime.now().isoformat(),
        "results": {
            ts: {
                "top1_accuracy": r.get("top1_accuracy", 0),
                "macro_precision": r.get("macro_precision", 0),
                "macro_recall": r.get("macro_recall", 0),
                "macro_f1": r.get("macro_f1", 0),
                "total_samples": r.get("total_samples", 0),
                "num_classes": r.get("num_classes_in_test", 0),
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


def run_cross_dataset_evaluation(config, experiment_name, source_dataset,
                                  target_datasets, device="cuda:0",
                                  seed=None, run_name=None):
    """Cross-region evaluation (identical logic to Option A)."""
    
    global_config = config["global"]
    exp_config = config["experiments"][experiment_name]
    device = torch.device(device)
    
    all_anchors = load_caption_anchors()
    caption_anchors, excluded_slides = get_trainable_slides(all_anchors)
    species_to_int, int_to_species, num_classes = build_class_mappings(all_anchors)
    num_workers = global_config.get("num_workers", 8)
    
    base_output_dir = PROJECT_ROOT / exp_config["output_dir"]
    output_dir = resolve_output_dir(base_output_dir, seed=seed, run_name=run_name)
    model_path = output_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"Error: Checkpoint not found at {model_path}")
        return None
    
    model = load_lupi_model(str(model_path), global_config, device, num_classes)
    
    img_size = global_config["img_size"]
    preprocessing_config = exp_config.get("preprocessing", {})
    if preprocessing_config.get("normalization_mode") == "custom":
        mean = preprocessing_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = preprocessing_config.get("normalize_std", [0.229, 0.224, 0.225])
    else:
        mean = global_config.get("normalize_mean", [0.485, 0.456, 0.406])
        std = global_config.get("normalize_std", [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Stain normalization for cross-region (same contract as run_evaluation)
    stainnorm_func = None
    stainnorm_config = preprocessing_config.get("stainnorm", {})
    if stainnorm_config.get("enabled", False) and HAS_TIATOOLBOX:
        ref_path = stainnorm_config.get("reference_path")
        if ref_path:
            import numpy as np
            ref_full = PROJECT_ROOT / ref_path
            if ref_full.exists():
                reference_img = np.load(ref_full)
                if reference_img.dtype != np.uint8:
                    reference_img = (reference_img * 255).astype(np.uint8)
                normalizer = MacenkoNormalizer()
                normalizer.fit(reference_img)
                stainnorm_func = normalizer.transform  # np.ndarray → np.ndarray
    
    # All targets including self
    all_targets = [source_dataset] + [t for t in target_datasets if t != source_dataset]
    
    annotations_dir = PROJECT_ROOT / "04_evaluation" / "annotations" / "ts2_expert"
    wsi_dir = PROJECT_ROOT / global_config["data_root"] / "00_raw_wsi"
    
    cross_results = {}
    
    for target in all_targets:
        overlapping = get_overlapping_species(source_dataset, target)
        
        if not overlapping and source_dataset != target:
            print(f"\n[Cross] {source_dataset} → {target}: NO overlapping species, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Cross-region: {source_dataset} → {target}")
        print(f"Overlapping species: {len(overlapping)}")
        print(f"{'='*60}")
        
        test_dataset = CrossDatasetTestDataset(
            source_dataset=source_dataset,
            target_dataset=target,
            annotations_dir=annotations_dir,
            wsi_dir=wsi_dir,
            transform=transform,
            stainnorm_func=stainnorm_func,
            caption_anchors=caption_anchors,
            species_to_int=species_to_int,
        )
        
        if len(test_dataset) == 0:
            print(f"[Cross] No samples for {target}, skipping")
            continue
        
        test_loader = DataLoader(
            test_dataset, batch_size=64, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        
        results = evaluate_with_overlap_filter(
            model=model,
            dataloader=test_loader,
            device=device,
            num_classes=num_classes,
            int_to_species=int_to_species,
            species_to_int=species_to_int,
            overlapping_species=overlapping,
        )
        
        is_in_domain = (source_dataset == target)
        result_key = f"{source_dataset}_to_{target}"
        cross_results[result_key] = {
            "source": source_dataset,
            "target": target,
            "is_in_domain": is_in_domain,
            "overlap_taxa_count": "all" if is_in_domain else len(overlapping),
            **results,
        }
        
        # Save per-pair results
        with open(output_dir / f"eval_cross_{source_dataset}_to_{target}.json", "w") as f:
            json.dump(cross_results[result_key], f, indent=2, default=str)
    
    # Save summary (match Option A's schema exactly)
    from datetime import datetime
    summary_path = output_dir / f"eval_cross_{source_dataset}_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "source": source_dataset,
            "targets": target_datasets,
            "timestamp": datetime.now().isoformat(),
            "results": cross_results,
        }, f, indent=2, default=str)
    
    # Print summary table
    print_cross_summary_table(cross_results, source_dataset, output_dir)
    
    print(f"\n[Saved] Results to {output_dir}/eval_cross_*.json")
    
    return cross_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate LUPI classifier")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--test_set", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--source_dataset", type=str, default=None)
    parser.add_argument("--target_datasets", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Evaluate a specific seeded run subfolder (seed_<N>)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional run tag used at training time (seed_<N>__<run_name>)")
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.experiment not in config["experiments"]:
        print(f"Error: Experiment '{args.experiment}' not found.")
        sys.exit(1)
    
    if args.source_dataset and args.target_datasets:
        run_cross_dataset_evaluation(
            config, args.experiment, args.source_dataset,
            args.target_datasets, args.device,
            seed=args.seed, run_name=args.run_name,
        )
    else:
        run_evaluation(config, args.experiment, args.test_set, args.device,
                       seed=args.seed, run_name=args.run_name)


if __name__ == "__main__":
    main()
