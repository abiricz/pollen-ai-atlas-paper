#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Extract ViT-Small-LVD Image Embeddings for All Captioned Crops
================================================================

Extracts frozen DINOv2 ViT-Small-LVD features (384-dim) for every
captioned pollen grain and saves per-slide H5 files in the same format
as the pre-computed SBERT caption embeddings.

IMPORTANT: Uses the plain pretrained LVD-142M weights from timm
(pretrained=True), NOT the finetuned 01_initialization checkpoint.
This matches Option A (experiment_config.yaml: backbone_checkpoint: null)
and ensures the retrieval test uses the same feature space as the
classification evaluation backbone.

Optimized for throughput:
  - DataLoader with multiple workers (parallel crop loading)
  - Multi-GPU via DataParallel (2× RTX 4090)
  - FP16 autocast for frozen backbone inference
  - Per-worker WSI handle caching (LRU, max 5 per worker)

The output enables three retrieval modalities:
  1. Text-only    (SBERT caption embeddings)
  2. Image-only   (this script's ViT embeddings)
  3. Combined     (late-fusion: α·sim_image + (1−α)·sim_text)

Usage:
    python extract_vit_embeddings.py                        # All slides, 2 GPUs
    python extract_vit_embeddings.py --batch_size 512       # Larger batches
    python extract_vit_embeddings.py --resume               # Skip existing H5
    python extract_vit_embeddings.py --gpus 0               # Single GPU

Output:
    data/04_evaluation/vit_embeddings/{slide}_embeddings.h5
    Each H5 contains:
        embeddings: (N, 384) float32, L2-normalized
        sample_ids: (N,)     strings
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Project root (4 levels up: scripts/retrieval/ → scripts/ → 04_evaluation/ → root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Default checkpoint: None = use plain pretrained LVD weights from timm
# This matches Option A (experiment_config.yaml: backbone_checkpoint: null)
# The finetuned 01_initialization checkpoint is NOT used here.
DEFAULT_CHECKPOINT = None

# WSI extensions to search
WSI_EXTENSIONS = [".tif", ".tiff", ".svs", ".ndpi"]
WSI_DATASETS = ["french", "hungarian", "mediterranean", "swedish"]


# ═════════════════════════════════════════════════════════════════════════
# TRANSFORMS (exact match to Option A validation pipeline)
# ═════════════════════════════════════════════════════════════════════════

def get_val_transform(img_size: int = 518) -> transforms.Compose:
    """Exact validation transforms from Option A (train_classifier.py L972-L991).

    No augmentation — deterministic pipeline matching the classification
    evaluation to ensure embedding consistency.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ═════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════

def build_vit_model(
    checkpoint_path: str,
    gpus: list = [0, 1],
    backbone_name: str = "vit_small_patch14_dinov2.lvd142m",
    img_size: int = 518,
) -> tuple:
    """Load frozen ViT backbone, optionally wrapped in DataParallel.

    Returns (model, primary_device).
    """
    import timm

    model = timm.create_model(
        backbone_name,
        pretrained=(checkpoint_path is None),
        img_size=img_size,
        init_values=1e-5,
        num_classes=0,  # feature extractor only — no head
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[ViT] Loading checkpoint: {os.path.basename(checkpoint_path)}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # Strip classification head keys
        state = {
            k: v for k, v in state.items()
            if not k.startswith("head.") and not k.startswith("classifier.")
        }
        msg = model.load_state_dict(state, strict=False)
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys[:5]}...")
    else:
        print(f"[ViT] Using timm pretrained weights (no custom checkpoint)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    primary_device = torch.device(f"cuda:{gpus[0]}")
    model = model.to(primary_device)

    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)
        print(f"[ViT] DataParallel on GPUs: {gpus}")

    embed_dim = model.module.num_features if isinstance(model, nn.DataParallel) else model.num_features
    print(f"[ViT] Model ready: {backbone_name}, embed_dim={embed_dim}, "
          f"GPUs={gpus}, FP16=True")
    return model, primary_device


# ═════════════════════════════════════════════════════════════════════════
# DATASET (for DataLoader with workers)
# ═════════════════════════════════════════════════════════════════════════

class SlideDataset(Dataset):
    """Dataset for a single slide's crops. Used with DataLoader for
    parallel loading via multiple workers.

    Each worker gets its own WSI handle via per-worker init.
    """

    def __init__(
        self,
        sample_ids: list,
        bboxes: list,
        wsi_path: str,
        transform: transforms.Compose,
    ):
        self.sample_ids = sample_ids
        self.bboxes = bboxes
        self.wsi_path = wsi_path
        self.transform = transform
        self._wsi = None  # lazy init per worker

    def _get_wsi(self):
        if self._wsi is None:
            try:
                import tiffslide
                self._wsi = tiffslide.TiffSlide(self.wsi_path)
            except (ImportError, Exception):
                import openslide
                self._wsi = openslide.OpenSlide(self.wsi_path)
        return self._wsi

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        bbox = self.bboxes[idx]
        try:
            wsi = self._get_wsi()
            x1, y1, x2, y2 = bbox
            region = wsi.read_region((x1, y1), 0, (x2 - x1, y2 - y1))
            img = region.convert("RGB")
            tensor = self.transform(img)
            return tensor, idx, True  # tensor, original_index, success
        except Exception:
            # Return a dummy tensor on failure
            return torch.zeros(3, 518, 518), idx, False


def find_wsi_path(slide_name: str, wsi_dir: Path) -> str:
    """Find WSI file by searching dataset directories."""
    for dataset in WSI_DATASETS:
        for ext in WSI_EXTENSIONS:
            path = wsi_dir / dataset / f"{slide_name}{ext}"
            if path.exists():
                return str(path)
    return None


def load_slide_samples(slide_name: str, caption_dir: Path, caption_model: str) -> dict:
    """Load sample ID → bbox mapping from JSONL for a slide."""
    id_to_bbox = {}
    for dataset in WSI_DATASETS:
        jsonl_path = caption_dir / dataset / caption_model / f"{slide_name}_captions.jsonl"
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    id_to_bbox[record["id"]] = [int(c) for c in record["bbox"]]
                except (json.JSONDecodeError, KeyError):
                    continue
        break  # each slide is in one dataset only
    return id_to_bbox


def get_all_slide_names(sbert_dir: Path) -> list:
    """Get slide names from existing SBERT embedding H5 files."""
    slides = []
    for h5_path in sorted(sbert_dir.glob("*_embeddings.h5")):
        slide = h5_path.stem.replace("_embeddings", "")
        slides.append(slide)
    return slides


# ═════════════════════════════════════════════════════════════════════════
# PER-SLIDE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════

def extract_slide_embeddings(
    slide_name: str,
    sbert_h5_path: Path,
    wsi_path: str,
    caption_dir: Path,
    caption_model: str,
    model: nn.Module,
    transform: transforms.Compose,
    primary_device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
) -> tuple:
    """Extract ViT features for one slide using DataLoader.

    Uses the SBERT H5 sample_ids as the authoritative ordering so that
    the ViT and SBERT embeddings are index-aligned.

    Returns:
        embeddings: np.ndarray (N, 384), L2-normalized
        sample_ids: list[str]
        n_failed:   int
    """
    # Load authoritative sample ordering from SBERT H5
    with h5py.File(sbert_h5_path, "r") as hf:
        sbert_ids = [
            s.decode() if isinstance(s, bytes) else s
            for s in hf["sample_ids"][:]
        ]

    # Load bbox info from JSONL
    id_to_bbox = load_slide_samples(slide_name, caption_dir, caption_model)

    # Build ordered bbox list (aligned to sbert_ids)
    bboxes = []
    valid_mask = []
    for sid in sbert_ids:
        bbox = id_to_bbox.get(sid)
        if bbox is not None:
            bboxes.append(bbox)
            valid_mask.append(True)
        else:
            bboxes.append([0, 0, 1, 1])  # dummy
            valid_mask.append(False)

    n_missing_bbox = sum(1 for v in valid_mask if not v)

    # Create dataset and dataloader
    dataset = SlideDataset(sbert_ids, bboxes, wsi_path, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )

    # Infer embed dim
    if isinstance(model, nn.DataParallel):
        embed_dim = model.module.num_features
    else:
        embed_dim = model.num_features

    embeddings = np.zeros((len(sbert_ids), embed_dim), dtype=np.float32)
    n_failed = n_missing_bbox

    # Batch inference with FP16
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch_tensors, batch_indices, batch_success in loader:
            # Filter successful crops
            success_mask = batch_success.bool()
            if not success_mask.any():
                n_failed += (~success_mask).sum().item()
                continue

            n_failed += (~success_mask).sum().item()

            valid_tensors = batch_tensors[success_mask].to(primary_device)
            valid_indices = batch_indices[success_mask].numpy()

            feats = model(valid_tensors)
            feats = feats.float().cpu().numpy()

            for j, idx in enumerate(valid_indices):
                embeddings[idx] = feats[j]

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    return embeddings, sbert_ids, n_failed


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract ViT-Small-LVD embeddings for all captioned crops")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data/ directory (default: PROJECT_ROOT/data)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="ViT checkpoint path (default: None = pretrained from timm)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for H5 files (default: data/04_evaluation/vit_embeddings)")
    parser.add_argument("--caption_model", type=str,
                        default="production_qwen25vl_final",
                        help="Caption model (for JSONL bbox lookup)")
    parser.add_argument("--sbert_model_tag", type=str, default="qwen25vl",
                        help="SBERT embedding subfolder (qwen25vl or qwen3vl)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Inference batch size (per forward pass, split across GPUs)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers for parallel crop loading")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1],
                        help="GPU indices to use (default: 0 1 for DataParallel)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip slides that already have H5 files")
    parser.add_argument("--slides", type=str, nargs="*", default=None,
                        help="Process only these slides (default: all)")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    wsi_dir = data_root / "00_raw_wsi"
    caption_dir = data_root / "03_captioning"
    sbert_dir = data_root / "04_evaluation" / "caption_embeddings" / args.sbert_model_tag
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_root / "04_evaluation" / "vit_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to available GPUs
    available_gpus = [g for g in args.gpus if g < torch.cuda.device_count()]
    if not available_gpus:
        available_gpus = [0]

    print("=" * 70)
    print("ViT-Small-LVD EMBEDDING EXTRACTION (optimized)")
    print("=" * 70)
    print(f"Checkpoint:        {args.checkpoint if args.checkpoint else 'pretrained from timm (LVD-142M)'}")
    print(f"SBERT reference:   {sbert_dir}")
    print(f"Caption model:     {args.caption_model}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Num workers:       {args.num_workers}")
    print(f"GPUs:              {available_gpus}")
    print(f"FP16:              True (frozen backbone)")
    print(f"Output dir:        {output_dir}")
    print(f"Resume:            {args.resume}")
    print()

    # Get slide list from SBERT H5 files
    all_slides = get_all_slide_names(sbert_dir)
    if args.slides:
        all_slides = [s for s in all_slides if s in args.slides]

    # Filter for resume
    if args.resume:
        existing = {
            p.stem.replace("_embeddings", "")
            for p in output_dir.glob("*_embeddings.h5")
        }
        before = len(all_slides)
        all_slides = [s for s in all_slides if s not in existing]
        print(f"[RESUME] Skipping {before - len(all_slides)} existing slides")

    print(f"Slides to process: {len(all_slides)}")

    # Build model (multi-GPU + frozen)
    model, primary_device = build_vit_model(
        args.checkpoint, gpus=available_gpus,
    )
    transform = get_val_transform(img_size=518)

    # Process each slide
    total_samples = 0
    total_failed = 0
    start_time = datetime.now()

    for slide_idx, slide_name in enumerate(all_slides):
        sbert_h5 = sbert_dir / f"{slide_name}_embeddings.h5"
        wsi_path = find_wsi_path(slide_name, wsi_dir)
        output_h5 = output_dir / f"{slide_name}_embeddings.h5"

        if wsi_path is None:
            print(f"  [{slide_idx+1}/{len(all_slides)}] SKIP {slide_name}: WSI not found")
            continue

        # Get expected count from SBERT
        with h5py.File(sbert_h5, "r") as hf:
            n_expected = hf["embeddings"].shape[0]

        print(f"  [{slide_idx+1}/{len(all_slides)}] {slide_name} "
              f"({n_expected:,} samples)...", end=" ", flush=True)

        embeddings, sample_ids, n_failed = extract_slide_embeddings(
            slide_name=slide_name,
            sbert_h5_path=sbert_h5,
            wsi_path=wsi_path,
            caption_dir=caption_dir,
            caption_model=args.caption_model,
            model=model,
            transform=transform,
            primary_device=primary_device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Save H5
        with h5py.File(output_h5, "w") as hf:
            hf.create_dataset("embeddings", data=embeddings, dtype="float32")
            hf.create_dataset(
                "sample_ids",
                data=np.array(sample_ids, dtype="S"),
            )

        total_samples += len(sample_ids)
        total_failed += n_failed
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = total_samples / elapsed if elapsed > 0 else 0

        print(f"done ({n_failed} failed) "
              f"[{total_samples:,} total, {rate:.0f} samples/s]")

    # Summary
    elapsed_total = (datetime.now() - start_time).total_seconds()
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Total slides:   {len(all_slides)}")
    print(f"Total samples:  {total_samples:,}")
    print(f"Total failed:   {total_failed:,}")
    print(f"Elapsed:        {elapsed_total / 60:.1f} min")
    print(f"Output dir:     {output_dir}")

    # Save metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint if args.checkpoint else None,
        "checkpoint_note": (
            f"Using finetuned checkpoint: {os.path.basename(args.checkpoint)}"
            if args.checkpoint else
            "Using plain pretrained LVD-142M weights from timm, "
            "NOT the finetuned 01_initialization checkpoint. "
            "Matches Option A (experiment_config.yaml: backbone_checkpoint: null)."
        ),
        "backbone": "vit_small_patch14_dinov2.lvd142m",
        "embed_dim": 384,
        "img_size": 518,
        "transforms": "Resize(518)→ToTensor→Normalize(ImageNet)",
        "normalization": "L2",
        "fp16": True,
        "gpus": available_gpus,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "total_slides": len(all_slides),
        "total_samples": total_samples,
        "total_failed": total_failed,
        "elapsed_seconds": elapsed_total,
        "caption_model_for_bbox": args.caption_model,
        "sbert_reference": args.sbert_model_tag,
    }
    with open(output_dir / "extraction_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {output_dir / 'extraction_metadata.json'}")


if __name__ == "__main__":
    main()
