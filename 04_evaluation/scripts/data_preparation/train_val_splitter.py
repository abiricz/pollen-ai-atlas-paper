#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Train/Val Splitter for Pollen AI Atlas

Creates train/validation splits from captioned JSONL files while EXCLUDING
all test regions (TS1 legacy + TS2 expert).

Strategy:
- Source: Captioned JSONL files from 03_captioning/*/production_*_final/
- Test regions: GeoJSON files from 04_evaluation/annotations/{ts1_legacy,ts2_expert}/
- Output: Split files using SAMPLE IDs (not line numbers) for model-agnostic splits

The splitter:
1. Loads all test regions (bounding boxes) from both TS1 and TS2
2. For each captioned sample, checks if its bbox overlaps ANY test region
3. Excludes overlapping samples from train/val pool
4. Randomly splits remaining samples into train (85%) and val (15%)
5. Saves split by SAMPLE ID (works for both Qwen2.5-VL and Qwen3-VL)

IMPORTANT: Since Qwen2.5-VL and Qwen3-VL JSONL files have different line ordering
but identical sample IDs and bboxes, we use sample IDs for splits. This ensures
the same physical samples are in train/val across both caption models.

Usage:
    python train_val_splitter.py --train_ratio 0.85 --seed 42
    python train_val_splitter.py --dry_run  # Preview without saving

Output:
    data/04_evaluation/splits/
    ├── manifest.json           # Master manifest with metadata
    ├── train/
    │   └── {slide}_train.json  # Per-slide train sample IDs
    └── val/
        └── {slide}_val.json    # Per-slide val sample IDs

Author: Pollen AI Atlas Team
Date: January 2026
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

# Directory paths (relative to project root)
CAPTIONING_DIR = PROJECT_ROOT / "data" / "03_captioning"
ANNOTATIONS_DIR = PROJECT_ROOT / "04_evaluation" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "data" / "04_evaluation" / "splits"

# Reference caption model (we use Qwen2.5-VL as reference, but splits work for both)
REFERENCE_CAPTION_MODEL = "production_qwen25vl_final"

# Datasets
DATASETS = ["french", "hungarian", "mediterranean", "swedish"]


# ============================================================================
# Helper Functions
# ============================================================================

def load_test_regions_from_geojson(geojson_path: Path) -> List[Dict]:
    """
    Load test region bounding boxes from a GeoJSON file.
    
    Returns:
        List of dicts with x_min, y_min, x_max, y_max
    """
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    regions = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        name = props.get('name', '')
        classification = props.get('classification', {}).get('name', '')
        
        # Match test regions
        if name == 'test_region' or classification == 'Test Region':
            coords = feature.get('geometry', {}).get('coordinates', [[]])
            if coords and coords[0]:
                xs = [p[0] for p in coords[0]]
                ys = [p[1] for p in coords[0]]
                regions.append({
                    'x_min': min(xs),
                    'y_min': min(ys),
                    'x_max': max(xs),
                    'y_max': max(ys),
                })
    return regions


def extract_slide_name_from_geojson(filename: str) -> str:
    """Extract slide name from GeoJSON filename."""
    return filename.replace('_curated.geojson', '')


def bbox_overlaps_region(bbox: List[int], region: Dict, margin: int = 0) -> bool:
    """
    Check if a sample's bbox overlaps a test region.
    
    Args:
        bbox: [x_min, y_min, x_max, y_max] from JSONL
        region: Dict with x_min, y_min, x_max, y_max
        margin: Extra margin to consider (pixels)
    
    Returns:
        True if overlapping
    """
    # JSONL bbox format: [x_min, y_min, x_max, y_max]
    bx_min, by_min, bx_max, by_max = bbox
    
    # Region bounds
    rx_min = region['x_min'] - margin
    ry_min = region['y_min'] - margin
    rx_max = region['x_max'] + margin
    ry_max = region['y_max'] + margin
    
    # Check for non-overlap
    if bx_max < rx_min or bx_min > rx_max:
        return False
    if by_max < ry_min or by_min > ry_max:
        return False
    
    return True


def load_all_test_regions() -> Dict[str, List[Dict]]:
    """
    Load all test regions from TS1 (legacy) and TS2 (expert).
    
    Returns:
        Dict mapping slide_name -> list of test region bounding boxes
    """
    test_regions = defaultdict(list)
    
    # Load TS1 (legacy)
    ts1_dir = ANNOTATIONS_DIR / "ts1_legacy"
    if ts1_dir.exists():
        for f in ts1_dir.glob("*_curated.geojson"):
            slide_name = extract_slide_name_from_geojson(f.name)
            regions = load_test_regions_from_geojson(f)
            test_regions[slide_name].extend(regions)
    
    # Load TS2 (expert)
    ts2_dir = ANNOTATIONS_DIR / "ts2_expert"
    if ts2_dir.exists():
        for f in ts2_dir.glob("*_curated.geojson"):
            slide_name = extract_slide_name_from_geojson(f.name)
            regions = load_test_regions_from_geojson(f)
            test_regions[slide_name].extend(regions)
    
    return dict(test_regions)


def find_caption_files(caption_model: str = REFERENCE_CAPTION_MODEL) -> Dict[str, Path]:
    """
    Find all caption JSONL files for a given model.
    
    Returns:
        Dict mapping slide_name -> Path to JSONL file
    """
    caption_files = {}
    
    for dataset in DATASETS:
        caption_dir = CAPTIONING_DIR / dataset / caption_model
        if caption_dir.exists():
            for f in caption_dir.glob("*_captions.jsonl"):
                # Skip 'old' subdirectory files if any
                if 'old' in str(f.parent):
                    continue
                slide_name = f.stem.replace('_captions', '')
                caption_files[slide_name] = f
    
    return caption_files


def process_slide_captions(
    jsonl_path: Path,
    test_regions: List[Dict],
    train_ratio: float,
    rng: random.Random
) -> Tuple[List[str], List[str], int, int]:
    """
    Process a single slide's caption file and create train/val splits.
    
    Uses SAMPLE IDs (not line indices) for model-agnostic splits.
    
    Args:
        jsonl_path: Path to the JSONL file
        test_regions: List of test region bboxes for this slide
        train_ratio: Ratio for train split (e.g., 0.85)
        rng: Random number generator for reproducibility
    
    Returns:
        (train_ids, val_ids, total_count, excluded_count)
    """
    # Load all samples
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    
    # Filter out samples that overlap with test regions
    available_ids = []
    excluded_count = 0
    
    for sample in samples:
        sample_id = sample.get('id')
        bbox = sample.get('bbox')
        if bbox is None or sample_id is None:
            continue
        
        # Check against all test regions
        is_in_test = False
        for region in test_regions:
            if bbox_overlaps_region(bbox, region):
                is_in_test = True
                break
        
        if is_in_test:
            excluded_count += 1
        else:
            available_ids.append(sample_id)
    
    # Shuffle and split
    rng.shuffle(available_ids)
    
    split_point = int(len(available_ids) * train_ratio)
    train_ids = sorted(available_ids[:split_point])
    val_ids = sorted(available_ids[split_point:])
    
    return train_ids, val_ids, len(samples), excluded_count


def verify_model_consistency(caption_model_alt: str) -> Dict[str, bool]:
    """
    Verify that the alternative caption model has the same samples.
    
    Returns:
        Dict mapping slide_name -> is_consistent (bool)
    """
    ref_files = find_caption_files(REFERENCE_CAPTION_MODEL)
    alt_files = find_caption_files(caption_model_alt)
    
    consistency = {}
    
    for slide_name in ref_files:
        if slide_name not in alt_files:
            consistency[slide_name] = False
            continue
        
        # Load sample IDs from both
        ref_ids = set()
        with open(ref_files[slide_name], 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ref_ids.add(data.get('id'))
                except:
                    pass
        
        alt_ids = set()
        with open(alt_files[slide_name], 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    alt_ids.add(data.get('id'))
                except:
                    pass
        
        consistency[slide_name] = (ref_ids == alt_ids)
    
    return consistency


# ============================================================================
# Main Splitter
# ============================================================================

def create_splits(
    train_ratio: float = 0.85,
    seed: int = 42,
    dry_run: bool = False,
    verify_both_models: bool = True
) -> Dict:
    """
    Create train/val splits for all slides.
    
    Args:
        train_ratio: Proportion for training set
        seed: Random seed for reproducibility
        dry_run: If True, don't write files
        verify_both_models: If True, verify Qwen2.5 and Qwen3 have same samples
    
    Returns:
        Summary statistics dict
    """
    print("=" * 70)
    print("Train/Val Splitter for Pollen AI Atlas")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Validation ratio: {1 - train_ratio}")
    print(f"  Random seed: {seed}")
    print(f"  Reference model: {REFERENCE_CAPTION_MODEL}")
    print(f"  Dry run: {dry_run}")
    
    # Initialize RNG
    rng = random.Random(seed)
    
    # Verify model consistency (both Qwen models have same samples)
    if verify_both_models:
        print(f"\nVerifying model consistency (Qwen2.5-VL vs Qwen3-VL)...")
        consistency = verify_model_consistency("production_qwen3-fp8_final")
        inconsistent = [s for s, ok in consistency.items() if not ok]
        if inconsistent:
            print(f"  ⚠ WARNING: {len(inconsistent)} slides have inconsistent samples!")
            for s in inconsistent[:5]:
                print(f"    - {s}")
        else:
            print(f"  ✓ All {len(consistency)} slides have identical samples in both models")
    
    # Load test regions
    print(f"\nLoading test regions...")
    test_regions = load_all_test_regions()
    print(f"  Found test regions for {len(test_regions)} slides")
    total_regions = sum(len(r) for r in test_regions.values())
    print(f"  Total test region bboxes: {total_regions}")
    
    # Find caption files (using reference model)
    print(f"\nFinding caption files...")
    caption_files = find_caption_files(REFERENCE_CAPTION_MODEL)
    print(f"  Found {len(caption_files)} slides with captions")
    
    # Process each slide
    print(f"\nProcessing slides...")
    
    results = {
        'train': {},  # slide_name -> list of sample IDs
        'val': {},    # slide_name -> list of sample IDs
        'stats': {
            'total_samples': 0,
            'excluded_test': 0,
            'train_samples': 0,
            'val_samples': 0,
            'slides_processed': 0,
        }
    }
    
    for slide_name, jsonl_path in sorted(caption_files.items()):
        # Get test regions for this slide (if any)
        slide_test_regions = test_regions.get(slide_name, [])
        
        # Process
        train_ids, val_ids, total, excluded = process_slide_captions(
            jsonl_path, slide_test_regions, train_ratio, rng
        )
        
        # Store results
        results['train'][slide_name] = train_ids
        results['val'][slide_name] = val_ids
        
        # Update stats
        results['stats']['total_samples'] += total
        results['stats']['excluded_test'] += excluded
        results['stats']['train_samples'] += len(train_ids)
        results['stats']['val_samples'] += len(val_ids)
        results['stats']['slides_processed'] += 1
        
        # Progress
        test_flag = f" [TEST: {len(slide_test_regions)} regions, {excluded} excluded]" if excluded > 0 else ""
        print(f"  {slide_name}: {len(train_ids)} train, {len(val_ids)} val{test_flag}")
    
    # Summary
    stats = results['stats']
    print(f"\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Slides processed: {stats['slides_processed']}")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Excluded (in test regions): {stats['excluded_test']:,}")
    print(f"Available for split: {stats['total_samples'] - stats['excluded_test']:,}")
    print(f"Train samples: {stats['train_samples']:,} ({stats['train_samples'] / (stats['train_samples'] + stats['val_samples']) * 100:.1f}%)")
    print(f"Val samples: {stats['val_samples']:,} ({stats['val_samples'] / (stats['train_samples'] + stats['val_samples']) * 100:.1f}%)")
    
    # Write output
    if not dry_run:
        print(f"\nWriting split files...")
        
        # Create output directories
        train_dir = OUTPUT_DIR / "train"
        val_dir = OUTPUT_DIR / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Write per-slide split files
        for slide_name, sample_ids in results['train'].items():
            split_file = train_dir / f"{slide_name}_train.json"
            with open(split_file, 'w') as f:
                json.dump({
                    'slide': slide_name,
                    'split': 'train',
                    'sample_ids': sample_ids,
                    'count': len(sample_ids)
                }, f, indent=2)
        
        for slide_name, sample_ids in results['val'].items():
            split_file = val_dir / f"{slide_name}_val.json"
            with open(split_file, 'w') as f:
                json.dump({
                    'slide': slide_name,
                    'split': 'val',
                    'sample_ids': sample_ids,
                    'count': len(sample_ids)
                }, f, indent=2)
        
        # Write master manifest
        manifest = {
            'version': '3.0',
            'created': datetime.now().isoformat(),
            'description': 'Train/val splits by sample ID (works for both Qwen2.5-VL and Qwen3-VL)',
            'config': {
                'train_ratio': train_ratio,
                'val_ratio': 1 - train_ratio,
                'seed': seed,
                'reference_model': REFERENCE_CAPTION_MODEL,
                'compatible_models': [
                    'production_qwen25vl_final',
                    'production_qwen3-fp8_final',
                    'production_qwen35-fp8_final',
                    'production_qwen36-fp8_final',
                    'production_gemma4-bf16_final',
                ]
            },
            'statistics': stats,
            'test_regions_source': {
                'ts1_legacy': str(ANNOTATIONS_DIR / "ts1_legacy"),
                'ts2_expert': str(ANNOTATIONS_DIR / "ts2_expert"),
            },
            'caption_source': str(CAPTIONING_DIR),
            'output_format': 'sample_ids (not line indices)',
            'slides': {
                slide: {
                    'train_count': len(results['train'].get(slide, [])),
                    'val_count': len(results['val'].get(slide, [])),
                }
                for slide in sorted(caption_files.keys())
            }
        }
        
        manifest_path = OUTPUT_DIR / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"  Written {len(results['train'])} train split files to {train_dir}")
        print(f"  Written {len(results['val'])} val split files to {val_dir}")
        print(f"  Manifest: {manifest_path}")
    else:
        print(f"\n[DRY RUN] No files written.")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create train/val splits for Pollen AI Atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run with default settings
    python train_val_splitter.py
    
    # Custom split ratio
    python train_val_splitter.py --train_ratio 0.8
    
    # Different random seed
    python train_val_splitter.py --seed 123
    
    # Preview without writing files
    python train_val_splitter.py --dry_run
    
Notes:
    - Splits use SAMPLE IDs, not line indices
    - This ensures identical splits work for both Qwen2.5-VL and Qwen3-VL captions
    - Both models caption the same physical samples, just in different order
"""
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.85,
        help='Proportion of data for training (default: 0.85)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview splits without writing files'
    )
    
    parser.add_argument(
        '--skip_verify',
        action='store_true',
        help='Skip verification that both VLM models have same samples'
    )
    
    args = parser.parse_args()
    
    # Run splitter
    create_splits(
        train_ratio=args.train_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
        verify_both_models=not args.skip_verify
    )
    
    print(f"\n✓ Done!")


if __name__ == '__main__':
    main()
