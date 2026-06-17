#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Verify Disjointness Between Test Sets

This script verifies that the two test sets (TS1: legacy, TS2: new) are disjoint:
1. No overlapping WSI slides between TS1 and TS2
2. For slides that appear in both, no overlapping test regions

This is critical for paper validity - we need to ensure:
- Legacy test regions (TS1) were never used in training
- New test regions (TS2) are independent from legacy regions

Test Sets:
- TS1 (Legacy): 21 slides from previous paper, 6,723 annotations
- TS2 (New): ~85 slides with expert annotations, ~10,000+ annotations

Usage:
    python verify_disjointness.py

Output:
    - Console summary
    - data/04_evaluation/results/disjointness_report.json

Author: Pollen AI Atlas Team
Date: January 2026
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

# Add project root to path
# Script is at 04_evaluation/scripts/data_preparation/verify_disjointness.py
# so parent.parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Directories
TS1_DIR = PROJECT_ROOT / "04_evaluation" / "annotations" / "ts1_legacy"   # Legacy (21 slides)
TS2_DIR = PROJECT_ROOT / "04_evaluation" / "annotations" / "ts2_expert"   # New (~85 slides)


def extract_wsi_name(filename: str) -> str:
    """Extract WSI name from GeoJSON filename."""
    return filename.replace('_curated.geojson', '')


def load_test_regions(geojson_path: Path) -> List[Dict]:
    """Load test region(s) from a GeoJSON file."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    regions = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        if props.get('name') == 'test_region' or props.get('classification', {}).get('name') == 'Test Region':
            coords = feature.get('geometry', {}).get('coordinates', [[]])
            if coords and coords[0]:
                # Extract bounding box from polygon coordinates
                xs = [p[0] for p in coords[0]]
                ys = [p[1] for p in coords[0]]
                regions.append({
                    'x_min': min(xs),
                    'y_min': min(ys),
                    'x_max': max(xs),
                    'y_max': max(ys),
                    'source_file': geojson_path.name
                })
    return regions


def count_pollen_annotations(geojson_path: Path) -> int:
    """Count pollen annotations in a GeoJSON file."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    count = 0
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        name = props.get('name', '')
        classification = props.get('classification', {}).get('name', '')
        if name.startswith('pollen_') or classification == 'Pollen':
            count += 1
    return count


def regions_overlap(r1: Dict, r2: Dict, tolerance: int = 0) -> bool:
    """
    Check if two regions overlap.
    
    Args:
        r1, r2: Regions with x_min, y_min, x_max, y_max
        tolerance: Pixel tolerance for near-overlap detection
    
    Returns:
        True if regions overlap
    """
    # Check for non-overlap conditions
    if r1['x_max'] + tolerance < r2['x_min']:
        return False
    if r2['x_max'] + tolerance < r1['x_min']:
        return False
    if r1['y_max'] + tolerance < r2['y_min']:
        return False
    if r2['y_max'] + tolerance < r1['y_min']:
        return False
    return True


def calculate_overlap_area(r1: Dict, r2: Dict) -> int:
    """Calculate overlapping area between two regions."""
    x_overlap = max(0, min(r1['x_max'], r2['x_max']) - max(r1['x_min'], r2['x_min']))
    y_overlap = max(0, min(r1['y_max'], r2['y_max']) - max(r1['y_min'], r2['y_min']))
    return x_overlap * y_overlap


def main():
    print("=" * 70)
    print("Verify Disjointness Between Test Sets")
    print("=" * 70)
    print(f"\nTS1 (Legacy): {TS1_DIR}")
    print(f"TS2 (New):    {TS2_DIR}")
    print()
    
    # Collect info from TS1 (Legacy)
    ts1_files = sorted(TS1_DIR.glob('*_curated.geojson'))
    ts1_slides = {}  # wsi_name -> {regions: [...], pollen_count: int}
    
    print(f"Loading TS1 (Legacy): {len(ts1_files)} files...")
    ts1_total_pollen = 0
    for f in ts1_files:
        if f.name == 'remap_report.json':
            continue
        wsi_name = extract_wsi_name(f.name)
        regions = load_test_regions(f)
        pollen_count = count_pollen_annotations(f)
        ts1_slides[wsi_name] = {
            'regions': regions,
            'pollen_count': pollen_count,
            'file': f.name
        }
        ts1_total_pollen += pollen_count
    
    # Collect info from TS2 (New)
    ts2_files = sorted(TS2_DIR.glob('*_curated.geojson'))
    ts2_slides = {}
    
    print(f"Loading TS2 (New):    {len(ts2_files)} files...")
    ts2_total_pollen = 0
    for f in ts2_files:
        if f.name.endswith('.log') or f.name.endswith('.json'):
            continue
        wsi_name = extract_wsi_name(f.name)
        regions = load_test_regions(f)
        pollen_count = count_pollen_annotations(f)
        ts2_slides[wsi_name] = {
            'regions': regions,
            'pollen_count': pollen_count,
            'file': f.name
        }
        ts2_total_pollen += pollen_count
    
    # Identify slide overlaps
    ts1_wsi_set = set(ts1_slides.keys())
    ts2_wsi_set = set(ts2_slides.keys())
    
    shared_slides = ts1_wsi_set & ts2_wsi_set
    ts1_only = ts1_wsi_set - ts2_wsi_set
    ts2_only = ts2_wsi_set - ts1_wsi_set
    
    print(f"\n" + "-" * 70)
    print("SLIDE-LEVEL ANALYSIS")
    print("-" * 70)
    print(f"TS1 slides: {len(ts1_wsi_set)}")
    print(f"TS2 slides: {len(ts2_wsi_set)}")
    print(f"Shared slides: {len(shared_slides)}")
    print(f"TS1-only slides: {len(ts1_only)}")
    print(f"TS2-only slides: {len(ts2_only)}")
    
    # Check for region overlaps on shared slides
    print(f"\n" + "-" * 70)
    print("REGION-LEVEL ANALYSIS (Shared Slides)")
    print("-" * 70)
    
    overlaps = []
    for wsi_name in sorted(shared_slides):
        ts1_info = ts1_slides[wsi_name]
        ts2_info = ts2_slides[wsi_name]
        
        ts1_regions = ts1_info['regions']
        ts2_regions = ts2_info['regions']
        
        print(f"\n{wsi_name}:")
        print(f"  TS1: {len(ts1_regions)} region(s), {ts1_info['pollen_count']} pollen")
        print(f"  TS2: {len(ts2_regions)} region(s), {ts2_info['pollen_count']} pollen")
        
        # Check each TS1 region against each TS2 region
        for i, r1 in enumerate(ts1_regions):
            for j, r2 in enumerate(ts2_regions):
                if regions_overlap(r1, r2):
                    overlap_area = calculate_overlap_area(r1, r2)
                    r1_area = (r1['x_max'] - r1['x_min']) * (r1['y_max'] - r1['y_min'])
                    r2_area = (r2['x_max'] - r2['x_min']) * (r2['y_max'] - r2['y_min'])
                    overlap_pct = overlap_area / min(r1_area, r2_area) * 100 if min(r1_area, r2_area) > 0 else 0
                    
                    overlap_info = {
                        'wsi': wsi_name,
                        'ts1_region': r1,
                        'ts2_region': r2,
                        'overlap_area': overlap_area,
                        'overlap_percent': overlap_pct
                    }
                    overlaps.append(overlap_info)
                    
                    print(f"  ⚠ OVERLAP DETECTED!")
                    print(f"      TS1 region {i}: ({r1['x_min']}, {r1['y_min']}) to ({r1['x_max']}, {r1['y_max']})")
                    print(f"      TS2 region {j}: ({r2['x_min']}, {r2['y_min']}) to ({r2['x_max']}, {r2['y_max']})")
                    print(f"      Overlap: {overlap_area:,} px² ({overlap_pct:.1f}%)")
        
        # Check if no overlap
        has_overlap = any(regions_overlap(r1, r2) for r1 in ts1_regions for r2 in ts2_regions)
        if not has_overlap:
            print(f"  ✓ No region overlap")
    
    # Final summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total TS1 slides: {len(ts1_wsi_set)}")
    print(f"Total TS2 slides: {len(ts2_wsi_set)}")
    print(f"Total TS1 pollen: {ts1_total_pollen:,}")
    print(f"Total TS2 pollen: {ts2_total_pollen:,}")
    print(f"Shared slides: {len(shared_slides)}")
    print(f"Region overlaps found: {len(overlaps)}")
    
    if overlaps:
        print(f"\n⚠ WARNING: {len(overlaps)} region overlap(s) detected!")
        print("   These regions may need to be excluded from one test set.")
    else:
        print(f"\n✓ DISJOINTNESS VERIFIED: No overlapping test regions between TS1 and TS2!")
    
    # Prepare report
    report = {
        'ts1': {
            'directory': str(TS1_DIR),
            'slides': len(ts1_wsi_set),
            'pollen': ts1_total_pollen,
            'slide_names': sorted(ts1_wsi_set)
        },
        'ts2': {
            'directory': str(TS2_DIR),
            'slides': len(ts2_wsi_set),
            'pollen': ts2_total_pollen,
            'slide_names': sorted(ts2_wsi_set)
        },
        'shared_slides': sorted(shared_slides),
        'ts1_only_slides': sorted(ts1_only),
        'ts2_only_slides': sorted(ts2_only),
        'overlaps': overlaps,
        'is_disjoint': len(overlaps) == 0
    }
    
    # Write report
    report_path = PROJECT_ROOT / "data" / "04_evaluation" / "results" / "disjointness_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    # Return success/failure
    return len(overlaps) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
