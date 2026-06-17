#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Candidate Filtering Script
==========================

Filters mining candidates using the finetuned ViT classifier.
Inserts between mining (miner.py) and captioning (caption_single.py).

Pipeline position:
    Mining → [Attention Rerank → Prototype Refine → NMS → CLASSIFIER FILTER] → Captioning

The script:
1. Loads mining results (H5)
2. Applies attention-based reranking (query token attention pooling)
3. Refines prototype via iterative medoid computation
4. Runs NMS with refined scores
5. Applies classifier filtering (pollen vs background)
6. Saves filtered results (H5) with class labels

Usage:
    python filter_candidates.py \\
        --h5_path mining_detections.h5 \\
        --wsi_path slide.tif \\
        --query_image query.png \\
        --output filtered_detections.h5 \\
        --classifier_threshold 0.5

Output H5 contains all original fields plus:
    - confidence_refined: After attention reranking
    - class_id: Predicted class (background=51)
    - class_prob: Classification probability
    - class_name: Human-readable class name
"""

import os
import sys
import json
import argparse
import h5py
import numpy as np
import psutil
from tqdm import tqdm
from PIL import Image
from datetime import datetime

import torch
from torchvision.ops import nms
import torch.nn.functional as F

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import (
    WSIPipeline,
    refine_medoid_proto_with_history,
    multi_query_attention_confs,
)
from lib.classifier import (
    PollenClassifier,
    load_classifier,
    BACKGROUND_CLASS_ID,
    INT_TO_CLASSES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter mining candidates using classifier"
    )
    
    # Input/Output paths
    parser.add_argument('--h5_path', type=str, required=True,
                        help="Input H5 from mining (miner.py, formerly detect.py)")
    parser.add_argument('--wsi_path', type=str, required=True,
                        help="Path to WSI file")
    parser.add_argument('--query_image', type=str, required=True,
                        help="Query image path")
    parser.add_argument('--output', type=str, required=True,
                        help="Output H5 path for filtered results")
    
    # Model paths
    parser.add_argument('--vit_ckpt', type=str,
                        default="../01_initialization/weights_vit_small_lvd_20250620_0312.pth",
                        help="Path to finetuned ViT checkpoint")
    parser.add_argument('--vit_name', type=str,
                        default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument('--sam2_ckpt', type=str, required=True)
    parser.add_argument('--sam2_cfg', type=str, required=True)
    
    # Device
    parser.add_argument('--device', type=str, default="cuda:0")
    
    # Filtering thresholds
    parser.add_argument('--nms_iou', type=float, default=0.10,
                        help="IoU threshold for NMS")
    parser.add_argument('--conf_thr', type=float, default=0.0,
                        help="Minimum refined confidence to keep (before classifier)")
    parser.add_argument('--classifier_threshold', type=float, default=0.5,
                        help="Minimum classifier probability to keep")
    parser.add_argument('--keep_background', action='store_true',
                        help="Keep background predictions (for analysis)")
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for classifier inference")
    parser.add_argument('--crop_size', type=int, default=518,
                        help="Crop size for classification")
    parser.add_argument('--crop_context', type=float, default=1.2,
                        help="Context multiplier for crops (1.2 = 20%% padding)")
    
    # Pre-filtering for very large datasets
    parser.add_argument('--max_candidates', type=int, default=None,
                        help="Max candidates to process (pre-filter by confidence). Default: no limit")
    parser.add_argument('--pre_filter_conf', type=float, default=None,
                        help="Pre-filter by confidence before reranking")
    parser.add_argument('--low_memory', action='store_true',
                        help="Use memory-efficient processing (smaller batches, explicit gc)")
    
    # SAM query options
    parser.add_argument('--sam_multi_point_query', action='store_true',
                        help="Use multi-point SAM query for query image segmentation (helps with pollen grains that have internal structures)")
    
    # Test region selection (optional)
    parser.add_argument('--select_test_region', action='store_true',
                        help="Select origin-based test region and export GeoJSON")
    parser.add_argument('--target_grains', type=int, default=100,
                        help="Target number of grains in test region (default: 100)")
    parser.add_argument('--test_region_size', type=int, default=5000,
                        help="Fixed size of test region in pixels (default: 5000)")
    parser.add_argument('--max_fraction', type=float, default=0.25,
                        help="Max fraction of total grains to use (default: 0.25)")
    parser.add_argument('--min_grains', type=int, default=10,
                        help="Minimum grains in test region (default: 10)")
    
    return parser.parse_args()


def load_mining_results(h5_path: str) -> dict:
    """Load all arrays from mining H5 file."""
    data = {}
    
    with h5py.File(h5_path, "r") as hf:
        grp = hf["results"] if "results" in hf else hf["metadata"]
        
        # Object-level arrays
        data['masks'] = grp["mask"][:]
        data['points'] = grp["point"][:]
        data['toplefts'] = grp["topleft"][:]
        data['bboxes'] = grp["bbox"][:]
        data['confidences'] = grp["confidence"][:]
        data['indices_flat'] = grp["indices_flat"][:]
        data['indices_offs'] = grp["indices_offs"][:]
        data['timestamps'] = grp["timestamp"][:]
        
        # Run-level metrics
        data['entropy_list'] = grp["entropy_list"][:]
        data['delta_entropy'] = grp["delta_entropy"][:]
        data['smoothed_delta'] = grp["smoothed_delta"][:]
        data['confidence_list'] = grp["confidence_list"][:]
        data['smoothed_conf'] = grp["smoothed_conf"][:]
        data['nms_class_indices'] = grp["nms_class_indices"][:]
        raw_status = grp["detection_status"][:]
        data['detection_status'] = [s.decode("utf-8") for s in raw_status]
    
    print(f"[Filter] Loaded {len(data['masks'])} objects from {h5_path}")
    return data


def extract_crop(wsi, bbox, mask, crop_size=518, context=1.2):
    """
    Extract crop from WSI using bounding box.
    Adds context padding and resizes to crop_size.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    
    # Add context padding
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h = w * context, h * context
    
    # Make square (use larger dimension)
    size = max(new_w, new_h)
    
    x1_new = int(cx - size / 2)
    y1_new = int(cy - size / 2)
    size_int = int(size)
    
    # Clamp to WSI bounds
    wsi_w, wsi_h = wsi.dimensions
    x1_new = max(0, min(x1_new, wsi_w - size_int))
    y1_new = max(0, min(y1_new, wsi_h - size_int))
    
    # Read region
    region = wsi.read_region((x1_new, y1_new), 0, (size_int, size_int))
    region = region.convert('RGB')
    
    # Resize to target
    if size_int != crop_size:
        region = region.resize((crop_size, crop_size), Image.BILINEAR)
    
    return region


def save_filtered_results(
    output_path: str,
    data: dict,
    keep_idx: np.ndarray,
    confidences_refined: np.ndarray,
    class_ids: np.ndarray,
    class_probs: np.ndarray,
    class_names: list,
):
    """Save filtered results to H5 file."""
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with h5py.File(output_path, "w") as hf:
        grp = hf.create_group("results")
        
        # Filtered object arrays
        grp.create_dataset("mask", data=data['masks'][keep_idx], compression="gzip")
        grp.create_dataset("point", data=data['points'][keep_idx])
        grp.create_dataset("topleft", data=data['toplefts'][keep_idx])
        grp.create_dataset("bbox", data=data['bboxes'][keep_idx])
        grp.create_dataset("confidence", data=data['confidences'][keep_idx])
        grp.create_dataset("confidence_refined", data=confidences_refined[keep_idx])
        grp.create_dataset("timestamp", data=data['timestamps'][keep_idx])
        
        # Classification results
        grp.create_dataset("class_id", data=class_ids[keep_idx])
        grp.create_dataset("class_prob", data=class_probs[keep_idx])
        
        # Store class names as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        names_arr = np.array([class_names[i] for i in keep_idx], dtype=object)
        grp.create_dataset("class_name", data=names_arr, dtype=dt)
        
        # Reconstruct indices for kept objects
        # This is more complex - need to rebuild offsets
        kept_indices = []
        kept_offsets = [0]
        for i in keep_idx:
            start = data['indices_offs'][i]
            end = data['indices_offs'][i + 1]
            kept_indices.extend(data['indices_flat'][start:end])
            kept_offsets.append(len(kept_indices))
        
        grp.create_dataset("indices_flat", data=np.array(kept_indices))
        grp.create_dataset("indices_offs", data=np.array(kept_offsets))
        
        # Run-level metrics (unchanged)
        grp.create_dataset("entropy_list", data=data['entropy_list'])
        grp.create_dataset("delta_entropy", data=data['delta_entropy'])
        grp.create_dataset("smoothed_delta", data=data['smoothed_delta'])
        grp.create_dataset("confidence_list", data=data['confidence_list'])
        grp.create_dataset("smoothed_conf", data=data['smoothed_conf'])
        grp.create_dataset("nms_class_indices", data=data['nms_class_indices'])
        
        status_bytes = np.array([s.encode('utf-8') for s in data['detection_status']])
        grp.create_dataset("detection_status", data=status_bytes)
        
        # Metadata
        grp.attrs['n_objects_original'] = len(data['masks'])
        grp.attrs['n_objects_filtered'] = len(keep_idx)
        grp.attrs['filter_timestamp'] = datetime.now().isoformat()
    
    print(f"[Filter] Saved {len(keep_idx)} objects to {output_path}")


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    
    print("=" * 60)
    print("CANDIDATE FILTERING")
    print("=" * 60)
    print(f"[Config] Input H5: {args.h5_path}")
    print(f"[Config] WSI: {args.wsi_path}")
    print(f"[Config] Output: {args.output}")
    print(f"[Config] NMS IoU: {args.nms_iou}")
    print(f"[Config] Classifier threshold: {args.classifier_threshold}")
    if args.low_memory:
        print(f"[Config] Low-memory mode: ENABLED")
    print()
    
    # Load mining results
    data = load_mining_results(args.h5_path)
    n_original = len(data['masks'])
    
    # PRE-FILTERING for very large datasets (memory protection)
    if args.max_candidates is not None and n_original > args.max_candidates:
        print(f"[Pre-filter] Too many candidates ({n_original:,}), limiting to {args.max_candidates:,}")
        
        # Determine confidence threshold to get ~max_candidates
        confs = data['confidences']
        if args.pre_filter_conf is not None:
            pre_thr = args.pre_filter_conf
        else:
            # Auto-determine threshold
            sorted_confs = np.sort(confs)[::-1]  # Descending
            pre_thr = sorted_confs[min(args.max_candidates, len(sorted_confs)-1)]
        
        keep_pre = confs >= pre_thr
        n_kept = np.sum(keep_pre)
        print(f"[Pre-filter] Using threshold {pre_thr:.3f}, keeping {n_kept:,} candidates")
        
        # Subsample all arrays
        for key in ['masks', 'points', 'toplefts', 'bboxes', 'confidences', 'timestamps']:
            if key in data:
                data[key] = data[key][keep_pre]
        
        # Rebuild indices (more complex)
        old_offs = data['indices_offs']
        old_flat = data['indices_flat']
        
        # Extract per-object indices for kept objects only
        keep_indices = np.where(keep_pre)[0]
        new_flat_list = []
        new_offs = [0]
        for i in keep_indices:
            start, end = old_offs[i], old_offs[i+1]
            new_flat_list.append(old_flat[start:end])
            new_offs.append(new_offs[-1] + (end - start))
        
        data['indices_flat'] = np.concatenate(new_flat_list) if new_flat_list else np.array([], dtype=old_flat.dtype)
        data['indices_offs'] = np.array(new_offs, dtype=old_offs.dtype)
        
        n_original = len(data['masks'])  # Update
        print(f"[Pre-filter] Now processing {n_original:,} candidates")
    
    # Explicit memory reporting
    if args.low_memory:
        import gc
        gc.collect()
        print(f"[Memory] Low-memory mode enabled")
    
    # Reconstruct per-object token indices
    n_objs = len(data['indices_offs']) - 1
    indices_per_obj = [
        data['indices_flat'][data['indices_offs'][i]:data['indices_offs'][i+1]]
        for i in range(n_objs)
    ]
    
    # Initialize pipeline for attention reranking
    print("[Pipeline] Initializing WSI pipeline...")
    pipeline = WSIPipeline(
        args.wsi_path,
        vit_ckpt=args.vit_ckpt,
        sam2_ckpt=args.sam2_ckpt,
        sam2_cfg=args.sam2_cfg,
        device=device,
        sam_multi_point_query=args.sam_multi_point_query,
    )
    pipeline.compute_vit_embeddings()
    pipeline.compute_token_level_similarity_augmented(
        args.query_image,
        sim_map_creation_mode='gpu_pca',
        pca_comps=10,
        n_shifts=1
    )
    
    # Attention-based reranking
    print(f"[Rerank] Computing attention-based confidence scores for {n_objs:,} objects...")
    confs_attn, vecs_attn = multi_query_attention_confs(
        query_tokens=pipeline.query_tokens,
        all_inds=indices_per_obj,
        embeddings=pipeline.embeddings,
        top_k=1024,
        batch_size=512,  # Large batch for GPU utilization (uses ~2-3GB VRAM)
        device=device,
        low_memory=args.low_memory
    )
    
    # CRITICAL: Free large structures after reranking
    if args.low_memory:
        import gc
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1e9
        print(f"[Memory] Before cleanup: {mem_before:.1f} GB RAM")
        
        print("[Memory] Releasing embeddings and indices...")
        del pipeline.embeddings
        del indices_per_obj
        
        # Release query tokens and similarity maps after reranking to reduce memory pressure.
        if hasattr(pipeline, 'query_tokens'):
            del pipeline.query_tokens
        if hasattr(pipeline, 'sim_map'):
            del pipeline.sim_map
        if hasattr(pipeline, 'dataloader'):
            del pipeline.dataloader
        if hasattr(pipeline, 'dataset'):
            del pipeline.dataset
        
        gc.collect()
        torch.cuda.empty_cache()
        
        mem_after = process.memory_info().rss / 1e9
        print(f"[Memory] After cleanup: {mem_after:.1f} GB RAM (freed {mem_before - mem_after:.1f} GB)")
    
    # Refine prototype
    final_proto, proto_history = refine_medoid_proto_with_history(
        vecs_attn, vecs_attn[0],
        top_frac=0.5,
        max_iter=100,
        tol=1e-4,
        low_memory=args.low_memory
    )
    
    # Compute refined confidence scores
    scores_cos = vecs_attn.dot(final_proto)
    confidences_refined = (scores_cos + 1.0) / 2.0  # Scale to [0, 1]
    
    # Free vecs_attn after computing scores (no longer needed)
    if args.low_memory:
        del vecs_attn, final_proto, proto_history
        gc.collect()
        mem_now = process.memory_info().rss / 1e9
        print(f"[Memory] After proto refinement cleanup: {mem_now:.1f} GB RAM")
    
    # NMS with refined scores
    print("[NMS] Running non-maximum suppression...")
    boxes_tensor = torch.tensor(data['bboxes'], dtype=torch.float32)
    scores_tensor = torch.tensor(confidences_refined, dtype=torch.float32)
    keep_idx_nms = nms(boxes_tensor, scores_tensor, iou_threshold=args.nms_iou)
    keep_idx_nms = keep_idx_nms.cpu().numpy()
    
    # Confidence threshold filter
    keep_idx_conf = keep_idx_nms[confidences_refined[keep_idx_nms] >= args.conf_thr]
    
    print(f"[NMS] {n_original} → {len(keep_idx_nms)} after NMS")
    print(f"[NMS] {len(keep_idx_nms)} → {len(keep_idx_conf)} after conf threshold")
    
    # Load classifier
    print("[Classifier] Loading pollen classifier...")
    classifier = load_classifier(checkpoint_path=args.vit_ckpt, device=str(device))
    
    # Classify all NMS-filtered candidates
    n_to_classify = len(keep_idx_conf)
    print(f"[Classifier] Classifying {n_to_classify:,} candidates...")
    
    import openslide
    from concurrent.futures import ThreadPoolExecutor
    
    wsi = openslide.OpenSlide(args.wsi_path)
    
    class_ids = np.zeros(n_original, dtype=np.int32)
    class_probs = np.zeros(n_original, dtype=np.float32)
    class_names = [''] * n_original
    
    # Helper function for crop extraction (can be parallelized)
    def extract_single_crop(idx):
        bbox = data['bboxes'][idx]
        mask = data['masks'][idx]
        crop = extract_crop(
            wsi, bbox, mask,
            crop_size=args.crop_size,
            context=args.crop_context
        )
        return idx, np.array(crop)
    
    # Optimize batch size for GPU utilization
    # For ~3GB VRAM, batch=128 should be good for 518x518 images
    classify_batch_size = min(128, args.batch_size * 4)  # Larger batches for GPU
    
    # Process in larger chunks to reduce overhead
    chunk_size = min(5000, n_to_classify)  # Process in chunks of 5000
    n_chunks = (n_to_classify + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_to_classify)
        chunk_indices = keep_idx_conf[chunk_start:chunk_end]
        
        # Extract crops in parallel using thread pool
        print(f"[Classifier] Chunk {chunk_idx+1}/{n_chunks}: extracting {len(chunk_indices):,} crops...")
        
        batch_images = []
        batch_indices = []
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results_iter = executor.map(extract_single_crop, chunk_indices)
            
            for idx, crop_arr in results_iter:
                batch_images.append(crop_arr)
                batch_indices.append(idx)
                
                # Process batch when full
                if len(batch_images) >= classify_batch_size:
                    preds = classifier.predict_batch(batch_images, batch_size=classify_batch_size)
                    for bi, res in zip(batch_indices, preds):
                        class_ids[bi] = res['class_id']
                        class_probs[bi] = res['probability']
                        class_names[bi] = res['class_name']
                    batch_images = []
                    batch_indices = []
        
        # Process remaining in chunk
        if batch_images:
            preds = classifier.predict_batch(batch_images, batch_size=len(batch_images))
            for bi, res in zip(batch_indices, preds):
                class_ids[bi] = res['class_id']
                class_probs[bi] = res['probability']
                class_names[bi] = res['class_name']
        
        # Progress update
        classified_so_far = chunk_end
        print(f"[Classifier] Progress: {classified_so_far:,}/{n_to_classify:,} ({100*classified_so_far/n_to_classify:.1f}%)")
    
    wsi.close()
    
    # Apply classifier filter
    is_pollen = class_ids[keep_idx_conf] != BACKGROUND_CLASS_ID
    above_threshold = class_probs[keep_idx_conf] >= args.classifier_threshold
    
    if args.keep_background:
        classifier_mask = above_threshold
    else:
        classifier_mask = is_pollen & above_threshold
    
    keep_idx_final = keep_idx_conf[classifier_mask]
    
    # Statistics
    n_pollen = np.sum(is_pollen)
    n_background = len(keep_idx_conf) - n_pollen
    
    print()
    print("=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Original objects:        {n_original}")
    print(f"After NMS:               {len(keep_idx_nms)}")
    print(f"After conf threshold:    {len(keep_idx_conf)}")
    print(f"  - Classified pollen:   {n_pollen}")
    print(f"  - Classified background: {n_background}")
    print(f"After classifier filter: {len(keep_idx_final)}")
    print(f"Keep ratio:              {100*len(keep_idx_final)/n_original:.1f}%")
    print()
    
    # Class distribution
    final_classes = class_ids[keep_idx_final]
    unique, counts = np.unique(final_classes, return_counts=True)
    print("Class distribution (top 10):")
    sorted_idx = np.argsort(-counts)[:10]
    for i in sorted_idx:
        cname = INT_TO_CLASSES.get(unique[i], f"class_{unique[i]}")
        print(f"  {cname:20s}: {counts[i]:5d} ({100*counts[i]/len(keep_idx_final):5.1f}%)")
    print()
    
    # Save results
    save_filtered_results(
        args.output,
        data,
        keep_idx_final,
        confidences_refined,
        class_ids,
        class_probs,
        class_names,
    )
    
    # Test region selection (optional)
    if args.select_test_region:
        select_and_export_test_region(
            args,
            data,
            keep_idx_final,
            confidences_refined,
            class_ids,
            class_probs,
            class_names,
        )
    
    print("[Done]")


def select_and_export_test_region(
    args,
    data: dict,
    keep_idx: np.ndarray,
    confidences_refined: np.ndarray,
    class_ids: np.ndarray,
    class_probs: np.ndarray,
    class_names: list,
):
    """
    Select a median-density test region from FILTERED candidates and export GeoJSON.
    
    Algorithm:
    1. Grid search all possible region positions
    2. Rank regions by grain count (density)
    3. Select the region at median density (not too dense, not too sparse)
    4. Select up to target_grains from that region (prioritized by confidence)
    5. Export GeoJSON for QuPath visualization
    
    Output:
    - test_region.geojson: QuPath-compatible annotation file
    - test_region_metadata.json: Statistics and configuration
    """
    print()
    print("=" * 60)
    print("TEST REGION SELECTION (from filtered candidates)")
    print("=" * 60)
    
    # Get filtered data
    bboxes_filtered = data['bboxes'][keep_idx]
    confs_filtered = confidences_refined[keep_idx]
    timestamps_filtered = data['timestamps'][keep_idx]
    class_ids_filtered = class_ids[keep_idx]
    class_probs_filtered = class_probs[keep_idx]
    
    n_filtered = len(keep_idx)
    print(f"[Input] Filtered candidates: {n_filtered}")
    
    # Calculate grain centers
    centers = np.array([
        [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        for b in bboxes_filtered
    ])
    
    # Grid search to collect density for all regions
    region_size = args.test_region_size
    step = max(500, region_size // 10)  # Finer search step
    
    x_max = int(centers[:, 0].max())
    y_max = int(centers[:, 1].max())
    
    print(f"[Config] Region size: {region_size} x {region_size} pixels")
    print(f"[Config] Target grains: {args.target_grains}")
    print(f"[Search] Grid step: {step} px")
    
    # Collect all region candidates with their densities
    region_candidates = []
    
    for x0 in range(0, max(1, x_max - region_size + 1), step):
        for y0 in range(0, max(1, y_max - region_size + 1), step):
            mask = (
                (centers[:, 0] >= x0) & (centers[:, 0] < x0 + region_size) &
                (centers[:, 1] >= y0) & (centers[:, 1] < y0 + region_size)
            )
            count = mask.sum()
            if count >= args.min_grains:  # Only consider regions with enough grains
                region_candidates.append((x0, y0, count))
    
    if len(region_candidates) == 0:
        print(f"[WARNING] No regions found with at least {args.min_grains} grains!")
        print("[WARNING] Try a larger --test_region_size or lower --min_grains")
        return
    
    # Sort by density and select median
    region_candidates.sort(key=lambda x: x[2])
    median_idx = len(region_candidates) // 2
    
    # Pick from around the median (slight bias toward higher density for good samples)
    # Select at ~60th percentile to ensure enough grains but not overcrowded
    target_percentile = int(len(region_candidates) * 0.6)
    target_percentile = max(median_idx, target_percentile)
    target_percentile = min(target_percentile, len(region_candidates) - 1)
    
    region_x1, region_y1, n_in_region = region_candidates[target_percentile]
    region_x2 = region_x1 + region_size
    region_y2 = region_y1 + region_size
    
    densities = [r[2] for r in region_candidates]
    print(f"[Search] Found {len(region_candidates)} valid regions")
    print(f"[Search] Density range: {min(densities)} to {max(densities)} grains")
    print(f"[Search] Median density: {densities[median_idx]} grains")
    print(f"[Result] Selected region (60th percentile): ({region_x1}, {region_y1}) to ({region_x2}, {region_y2})")
    print(f"[Result] Grains in region: {n_in_region}")
    
    # Find grains INSIDE the selected region
    inside_mask = (
        (centers[:, 0] >= region_x1) & (centers[:, 0] < region_x2) &
        (centers[:, 1] >= region_y1) & (centers[:, 1] < region_y2)
    )
    inside_indices = np.where(inside_mask)[0]
    n_inside = len(inside_indices)
    
    if n_inside == 0:
        print("[WARNING] No grains found in test region!")
        return
    
    # Calculate target grain count (from grains inside region)
    max_by_fraction = int(n_filtered * args.max_fraction)
    target_count = min(args.target_grains, max_by_fraction, n_inside)
    target_count = max(target_count, min(args.min_grains, n_inside))
    
    print(f"[Config] Max fraction: {args.max_fraction} ({max_by_fraction} grains)")
    print(f"[Result] Actual target: {target_count} grains")
    
    # Among grains inside region, select by confidence + early timestamp
    confs_inside = confs_filtered[inside_indices]
    timestamps_inside = timestamps_filtered[inside_indices]
    
    conf_norm = (confs_inside - confs_inside.min()) / (confs_inside.max() - confs_inside.min() + 1e-6)
    ts_norm = 1.0 - (timestamps_inside - timestamps_inside.min()) / (timestamps_inside.max() - timestamps_inside.min() + 1e-6)
    
    # Combined priority: confidence (60%) + early discovery (40%)
    combined_priority = conf_norm * 0.6 + ts_norm * 0.4
    
    # Sort by priority and select top grains
    priority_order = np.argsort(-combined_priority)
    selected_inside_idx = priority_order[:target_count]
    selected_local_idx = inside_indices[selected_inside_idx]
    
    # Get selected grain data
    selected_bboxes = bboxes_filtered[selected_local_idx]
    selected_centers = centers[selected_local_idx]
    selected_confs = confs_filtered[selected_local_idx]
    selected_class_ids = class_ids_filtered[selected_local_idx]
    selected_class_probs = class_probs_filtered[selected_local_idx]
    selected_original_idx = keep_idx[selected_local_idx]
    
    print(f"[Region] Grains selected: {len(selected_local_idx)}")
    print(f"[Region] Mean confidence: {selected_confs.mean():.4f}")
    
    # Class distribution in test region
    unique_classes, class_counts = np.unique(selected_class_ids, return_counts=True)
    print("\n[Test Region] Class distribution:")
    for cls_id, count in zip(unique_classes, class_counts):
        cls_name = INT_TO_CLASSES.get(cls_id, f"class_{cls_id}")
        print(f"  {cls_name:20s}: {count:3d}")
    
    # Create GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Add test region polygon (large annotation boundary)
    # Using "annotation" objectType so it's editable in QuPath
    geojson["features"].append({
        "type": "Feature",
        "id": "test_region",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [region_x1, region_y1],
                [region_x2, region_y1],
                [region_x2, region_y2],
                [region_x1, region_y2],
                [region_x1, region_y1]
            ]]
        },
        "properties": {
            "objectType": "annotation",
            "name": "test_region",
            "classification": {
                "name": "Test Region",  # QuPath annotation class
                "colorRGB": -16711936  # Green
            },
            "isLocked": False,  # Editable in QuPath
            "n_grains": len(selected_local_idx),
            "description": "Test region for ground truth annotation"
        }
    })
    
    # Add each grain as annotation (same level as test region for QuPath editing)
    # Using "annotation" objectType allows moving/modifying/deleting in QuPath
    for i, (orig_idx, local_idx, bbox, center) in enumerate(
        zip(selected_original_idx, selected_local_idx, selected_bboxes, selected_centers)
    ):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        # Add grain as annotation rectangle (editable in QuPath)
        geojson["features"].append({
            "type": "Feature",
            "id": f"grain_{i}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                    [x1, y1]
                ]]
            },
            "properties": {
                "objectType": "annotation",  # Same as test region - editable in QuPath
                "name": f"pollen_{i}",
                "classification": {
                    "name": "Pollen",  # Capitalized for QuPath class consistency
                    "colorRGB": -16776961  # Blue
                },
                "isLocked": False,  # Ensure annotations are editable
                "original_index": int(orig_idx),
                "confidence": float(selected_confs[i]),
                # Keep detailed predictions in metadata for reference
                "predicted_class": INT_TO_CLASSES.get(int(selected_class_ids[i]), "unknown"),
                "predicted_prob": float(selected_class_probs[i]),
            }
        })
    
    # Extract slide name from output path for slide-specific filenames
    output_basename = os.path.basename(args.output)
    slide_name = output_basename.replace("_filtered.h5", "").replace(".h5", "")
    
    # Save GeoJSON with slide name
    output_dir = os.path.dirname(args.output)
    geojson_path = os.path.join(output_dir, f"{slide_name}_test_region.geojson")
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    print(f"\n[Save] GeoJSON → {geojson_path}")
    
    # Save metadata with slide name
    metadata = {
        "slide_name": slide_name,
        "source_h5": args.h5_path,
        "filtered_h5": args.output,
        "n_filtered_candidates": int(n_filtered),
        "n_grains_inside_region": int(n_inside),
        "n_test_region_grains": int(len(selected_local_idx)),
        "test_region": {
            "x1": region_x1,
            "y1": region_y1,
            "x2": region_x2,
            "y2": region_y2,
            "width": region_size,
            "height": region_size,
        },
        "selection_config": {
            "target_grains": args.target_grains,
            "test_region_size": args.test_region_size,
            "max_fraction": args.max_fraction,
            "min_grains": args.min_grains,
        },
        "statistics": {
            "mean_confidence": float(selected_confs.mean()),
            "std_confidence": float(selected_confs.std()),
            "min_confidence": float(selected_confs.min()),
            "max_confidence": float(selected_confs.max()),
        },
        "class_distribution": {
            INT_TO_CLASSES.get(int(c), f"class_{c}"): int(cnt)
            for c, cnt in zip(unique_classes, class_counts)
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    metadata_path = os.path.join(output_dir, f"{slide_name}_test_region_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Save] Metadata → {metadata_path}")


if __name__ == "__main__":
    main()
