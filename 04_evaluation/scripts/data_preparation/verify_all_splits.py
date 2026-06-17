#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Verify All Splits Disjointness

Comprehensive verification that train, validation, and test sets are fully disjoint:
1. No train samples appear in any test region (TS1 or TS2)
2. No validation samples appear in any test region (TS1 or TS2)
3. Train and validation sets do not overlap (by design from splitter)
4. TS1 and TS2 test regions are disjoint (verified separately)

This script should be run AFTER train_val_splitter.py has created the splits.

Usage:
    python verify_all_splits.py
    python verify_all_splits.py --verbose

Output:
    - Console summary with pass/fail for each check
    - data/04_evaluation/results/split_verification_report.json

Author: Pollen AI Atlas Team
Date: January 2026
"""

import argparse
import json
import os
import sys
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

CAPTIONING_DIR = PROJECT_ROOT / "data" / "03_captioning"
ANNOTATIONS_DIR = PROJECT_ROOT / "04_evaluation" / "annotations"
SPLITS_DIR = PROJECT_ROOT / "data" / "04_evaluation" / "splits"
RESULTS_DIR = PROJECT_ROOT / "data" / "04_evaluation" / "results"

DATASETS = ["french", "hungarian", "mediterranean", "swedish"]


# ============================================================================
# Helper Functions
# ============================================================================

def load_test_regions_from_geojson(geojson_path: Path) -> List[Dict]:
    """Load test region bounding boxes from a GeoJSON file."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    regions = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        name = props.get('name', '')
        classification = props.get('classification', {}).get('name', '')
        
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


def bbox_overlaps_region(bbox: List[int], region: Dict) -> bool:
    """Check if a sample's bbox overlaps a test region."""
    bx_min, by_min, bx_max, by_max = bbox
    
    if bx_max < region['x_min'] or bx_min > region['x_max']:
        return False
    if by_max < region['y_min'] or by_min > region['y_max']:
        return False
    
    return True


def load_all_test_regions() -> Dict[str, List[Dict]]:
    """Load all test regions from TS1 and TS2."""
    test_regions = defaultdict(list)
    
    for ts_dir in ["ts1_legacy", "ts2_expert"]:
        ts_path = ANNOTATIONS_DIR / ts_dir
        if ts_path.exists():
            for f in ts_path.glob("*_curated.geojson"):
                slide_name = extract_slide_name_from_geojson(f.name)
                regions = load_test_regions_from_geojson(f)
                test_regions[slide_name].extend(regions)
    
    return dict(test_regions)


def load_split_manifest() -> Optional[Dict]:
    """Load the split manifest file."""
    manifest_path = SPLITS_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_split_indices(split: str) -> Dict[str, List[str]]:
    """Load all split sample IDs for a given split (train/val)."""
    split_dir = SPLITS_DIR / split
    indices = {}
    
    if split_dir.exists():
        for f in split_dir.glob(f"*_{split}.json"):
            with open(f, 'r') as fp:
                data = json.load(fp)
                slide_name = data.get('slide', f.stem.replace(f'_{split}', ''))
                # Handle both old (indices) and new (sample_ids) formats
                indices[slide_name] = data.get('sample_ids', data.get('indices', []))
    
    return indices


def find_jsonl_for_slide(slide_name: str, caption_model: str) -> Optional[Path]:
    """Find the JSONL file for a slide."""
    for dataset in DATASETS:
        jsonl_path = CAPTIONING_DIR / dataset / caption_model / f"{slide_name}_captions.jsonl"
        if jsonl_path.exists():
            return jsonl_path
    return None


def load_samples_by_ids(jsonl_path: Path, sample_ids: List[str]) -> List[Dict]:
    """Load specific samples from a JSONL file by sample IDs."""
    samples = []
    sample_ids_set = set(sample_ids)
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                if sample.get('id') in sample_ids_set:
                    samples.append(sample)
            except json.JSONDecodeError:
                pass
    
    return samples


# ============================================================================
# Verification Checks
# ============================================================================

def verify_train_vs_test(
    train_indices: Dict[str, List[int]],
    test_regions: Dict[str, List[Dict]],
    caption_model: str,
    verbose: bool = False
) -> Tuple[bool, List[Dict]]:
    """
    Verify no training samples overlap with test regions.
    
    Returns:
        (is_valid, list of violations)
    """
    violations = []
    
    for slide_name, sample_ids in train_indices.items():
        if slide_name not in test_regions:
            continue
        
        jsonl_path = find_jsonl_for_slide(slide_name, caption_model)
        if jsonl_path is None:
            continue
        
        samples = load_samples_by_ids(jsonl_path, sample_ids)
        slide_regions = test_regions[slide_name]
        
        for sample in samples:
            bbox = sample.get('bbox')
            if bbox is None:
                continue
            
            for region in slide_regions:
                if bbox_overlaps_region(bbox, region):
                    violations.append({
                        'split': 'train',
                        'slide': slide_name,
                        'sample_id': sample.get('id'),
                        'bbox': bbox,
                        'test_region': region
                    })
                    if verbose:
                        print(f"  ⚠ VIOLATION: Train sample {sample.get('id')} in test region")
                    break
    
    return len(violations) == 0, violations


def verify_val_vs_test(
    val_indices: Dict[str, List[int]],
    test_regions: Dict[str, List[Dict]],
    caption_model: str,
    verbose: bool = False
) -> Tuple[bool, List[Dict]]:
    """
    Verify no validation samples overlap with test regions.
    
    Returns:
        (is_valid, list of violations)
    """
    violations = []
    
    for slide_name, sample_ids in val_indices.items():
        if slide_name not in test_regions:
            continue
        
        jsonl_path = find_jsonl_for_slide(slide_name, caption_model)
        if jsonl_path is None:
            continue
        
        samples = load_samples_by_ids(jsonl_path, sample_ids)
        slide_regions = test_regions[slide_name]
        
        for sample in samples:
            bbox = sample.get('bbox')
            if bbox is None:
                continue
            
            for region in slide_regions:
                if bbox_overlaps_region(bbox, region):
                    violations.append({
                        'split': 'val',
                        'slide': slide_name,
                        'sample_id': sample.get('id'),
                        'bbox': bbox,
                        'test_region': region
                    })
                    if verbose:
                        print(f"  ⚠ VIOLATION: Val sample {sample.get('id')} in test region")
                    break
    
    return len(violations) == 0, violations


def verify_train_val_disjoint(
    train_indices: Dict[str, List[int]],
    val_indices: Dict[str, List[int]]
) -> Tuple[bool, List[Dict]]:
    """
    Verify train and val indices don't overlap (by index within each slide).
    
    Returns:
        (is_valid, list of overlapping slides)
    """
    violations = []
    
    all_slides = set(train_indices.keys()) | set(val_indices.keys())
    
    for slide_name in all_slides:
        train_set = set(train_indices.get(slide_name, []))
        val_set = set(val_indices.get(slide_name, []))
        
        overlap = train_set & val_set
        if overlap:
            violations.append({
                'slide': slide_name,
                'overlapping_indices': list(overlap)[:10],  # First 10
                'overlap_count': len(overlap)
            })
    
    return len(violations) == 0, violations


def verify_ts1_ts2_disjoint(verbose: bool = False) -> Tuple[bool, List[Dict]]:
    """
    Verify TS1 and TS2 test regions don't overlap.
    
    Returns:
        (is_valid, list of overlapping regions)
    """
    violations = []
    
    ts1_regions = {}
    ts1_dir = ANNOTATIONS_DIR / "ts1_legacy"
    if ts1_dir.exists():
        for f in ts1_dir.glob("*_curated.geojson"):
            slide_name = extract_slide_name_from_geojson(f.name)
            ts1_regions[slide_name] = load_test_regions_from_geojson(f)
    
    ts2_regions = {}
    ts2_dir = ANNOTATIONS_DIR / "ts2_expert"
    if ts2_dir.exists():
        for f in ts2_dir.glob("*_curated.geojson"):
            slide_name = extract_slide_name_from_geojson(f.name)
            ts2_regions[slide_name] = load_test_regions_from_geojson(f)
    
    # Check shared slides
    shared_slides = set(ts1_regions.keys()) & set(ts2_regions.keys())
    
    for slide_name in shared_slides:
        for r1 in ts1_regions[slide_name]:
            for r2 in ts2_regions[slide_name]:
                if bbox_overlaps_region(
                    [r1['x_min'], r1['y_min'], r1['x_max'], r1['y_max']],
                    r2
                ):
                    violations.append({
                        'slide': slide_name,
                        'ts1_region': r1,
                        'ts2_region': r2
                    })
                    if verbose:
                        print(f"  ⚠ OVERLAP: {slide_name} TS1/TS2 regions overlap")
    
    return len(violations) == 0, violations


# ============================================================================
# Main Verification
# ============================================================================

def run_verification(verbose: bool = False) -> Dict:
    """Run all verification checks."""
    print("=" * 70)
    print("Verify All Splits Disjointness")
    print("=" * 70)
    
    # Load manifest
    manifest = load_split_manifest()
    if manifest is None:
        print("\n✗ ERROR: No split manifest found!")
        print(f"  Expected: {SPLITS_DIR / 'manifest.json'}")
        print("  Run train_val_splitter.py first.")
        return {'error': 'No manifest'}
    
    # Handle both old (caption_model) and new (reference_model) manifest formats
    caption_model = manifest['config'].get('reference_model') or manifest['config'].get('caption_model', 'production_qwen25vl_final')
    print(f"\nConfiguration from manifest:")
    print(f"  Created: {manifest.get('created', 'unknown')}")
    print(f"  Reference model: {caption_model}")
    print(f"  Train ratio: {manifest['config']['train_ratio']}")
    print(f"  Seed: {manifest['config']['seed']}")
    
    # Load indices
    print(f"\nLoading splits...")
    train_indices = load_split_indices('train')
    val_indices = load_split_indices('val')
    print(f"  Train: {len(train_indices)} slides, {sum(len(v) for v in train_indices.values()):,} samples")
    print(f"  Val: {len(val_indices)} slides, {sum(len(v) for v in val_indices.values()):,} samples")
    
    # Load test regions
    print(f"\nLoading test regions...")
    test_regions = load_all_test_regions()
    print(f"  Slides with test regions: {len(test_regions)}")
    
    # Run checks
    results = {
        'timestamp': datetime.now().isoformat(),
        'manifest_config': manifest['config'],
        'checks': {}
    }
    
    print(f"\n" + "-" * 70)
    print("VERIFICATION CHECKS")
    print("-" * 70)
    
    # Check 1: Train vs Test
    print(f"\n1. Train vs Test Regions...")
    is_valid, violations = verify_train_vs_test(train_indices, test_regions, caption_model, verbose)
    results['checks']['train_vs_test'] = {
        'passed': is_valid,
        'violations': len(violations),
        'details': violations[:20] if violations else []
    }
    if is_valid:
        print(f"   ✓ PASSED: No train samples in test regions")
    else:
        print(f"   ✗ FAILED: {len(violations)} train samples in test regions!")
    
    # Check 2: Val vs Test
    print(f"\n2. Validation vs Test Regions...")
    is_valid, violations = verify_val_vs_test(val_indices, test_regions, caption_model, verbose)
    results['checks']['val_vs_test'] = {
        'passed': is_valid,
        'violations': len(violations),
        'details': violations[:20] if violations else []
    }
    if is_valid:
        print(f"   ✓ PASSED: No val samples in test regions")
    else:
        print(f"   ✗ FAILED: {len(violations)} val samples in test regions!")
    
    # Check 3: Train/Val disjoint
    print(f"\n3. Train/Val Index Disjointness...")
    is_valid, violations = verify_train_val_disjoint(train_indices, val_indices)
    results['checks']['train_val_disjoint'] = {
        'passed': is_valid,
        'violations': len(violations),
        'details': violations[:20] if violations else []
    }
    if is_valid:
        print(f"   ✓ PASSED: Train and val indices are disjoint")
    else:
        print(f"   ✗ FAILED: {len(violations)} slides have overlapping indices!")
    
    # Check 4: TS1/TS2 disjoint (informational, known issue with Alnus)
    print(f"\n4. TS1/TS2 Test Region Disjointness...")
    is_valid, violations = verify_ts1_ts2_disjoint(verbose)
    results['checks']['ts1_ts2_disjoint'] = {
        'passed': is_valid,
        'violations': len(violations),
        'details': violations[:20] if violations else [],
        'note': 'Known issue: Alnus slide has overlapping TS1/TS2 regions'
    }
    if is_valid:
        print(f"   ✓ PASSED: TS1 and TS2 test regions are disjoint")
    else:
        print(f"   ⚠ WARNING: {len(violations)} overlapping TS1/TS2 regions")
        print(f"     (Known issue: Alnus slide documented in paper)")
    
    # Summary
    all_passed = all(
        results['checks'][k]['passed'] 
        for k in ['train_vs_test', 'val_vs_test', 'train_val_disjoint']
    )
    results['all_passed'] = all_passed
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_passed:
        print(f"✓ ALL CRITICAL CHECKS PASSED")
        print(f"  Train/Val splits are properly separated from test regions.")
    else:
        print(f"✗ SOME CHECKS FAILED - Review results carefully!")
    
    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "split_verification_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify all splits (train, val, test) are disjoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script verifies that:
  1. No train samples overlap with test regions (TS1 + TS2)
  2. No validation samples overlap with test regions
  3. Train and validation indices are disjoint
  4. TS1 and TS2 test regions are disjoint (informational)

Run AFTER train_val_splitter.py has created the splits.

Examples:
    python verify_all_splits.py
    python verify_all_splits.py --verbose
"""
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about violations'
    )
    
    args = parser.parse_args()
    
    results = run_verification(verbose=args.verbose)
    
    # Exit code based on critical checks
    if results.get('error'):
        sys.exit(1)
    elif results.get('all_passed'):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
