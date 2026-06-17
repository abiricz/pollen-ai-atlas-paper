#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Curate QuPath annotations for evaluation.

This script processes confirmed GeoJSON files from QuPath and produces
curated ground truth files ready for evaluation.

Workflow:
1. Load raw GeoJSON (from filtering step) 
2. Load confirmed GeoJSON (from QuPath after expert review)
3. Validate that original pollen annotations are preserved
4. Add new annotations (drawn by expert) with proper classification
5. Output curated GeoJSON with consistent structure

Usage:
    python curate_qupath_annotations.py --raw path/to/raw.geojson --confirmed path/to/confirmed.geojson --output path/to/curated.geojson
    
Or batch mode:
    python curate_qupath_annotations.py --batch --raw_dir path/to/raw_geojson --confirmed_dir path/to/confirmed_geojson --output_dir path/to/curated_geojson
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


def get_bbox(coords: List) -> Tuple[float, float, float, float]:
    """Extract bounding box from polygon coordinates."""
    if isinstance(coords[0][0], list):
        coords = coords[0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (min(xs), min(ys), max(xs), max(ys))


def validate_geojson(data: Dict) -> Tuple[bool, str]:
    """Validate GeoJSON structure."""
    if data.get('type') != 'FeatureCollection':
        return False, "Not a FeatureCollection"
    
    if 'features' not in data:
        return False, "No features found"
    
    for i, feat in enumerate(data['features']):
        if 'geometry' not in feat:
            return False, f"Feature {i} missing geometry"
        if 'properties' not in feat:
            return False, f"Feature {i} missing properties"
        if feat['geometry'].get('type') not in ['Polygon', 'MultiPolygon']:
            return False, f"Feature {i} has unsupported geometry type: {feat['geometry'].get('type')}"
    
    return True, "Valid"


def analyze_geojson(data: Dict) -> Dict:
    """Analyze GeoJSON contents."""
    analysis = {
        'total_features': len(data['features']),
        'test_regions': 0,
        'pollen_classified': 0,
        'pollen_unclassified': 0,
        'other': 0,
        'pollen_bboxes': set(),
        'test_region_bbox': None
    }
    
    for feat in data['features']:
        props = feat.get('properties', {})
        cls = props.get('classification')
        
        if isinstance(cls, dict):
            cls_name = cls.get('name')
        else:
            cls_name = cls
        
        bbox = get_bbox(feat['geometry']['coordinates'])
        
        if cls_name == 'Test Region':
            analysis['test_regions'] += 1
            analysis['test_region_bbox'] = bbox
        elif cls_name == 'Pollen':
            analysis['pollen_classified'] += 1
            analysis['pollen_bboxes'].add(bbox)
        elif cls_name is None:
            analysis['pollen_unclassified'] += 1
        else:
            analysis['other'] += 1
    
    return analysis


def curate_annotations(raw_data: Dict, confirmed_data: Dict, slide_name: str) -> Tuple[Dict, Dict]:
    """
    Curate annotations by merging raw and confirmed data.
    
    Returns:
        curated_data: Curated GeoJSON
        report: Curation report
    """
    raw_analysis = analyze_geojson(raw_data)
    conf_analysis = analyze_geojson(confirmed_data)
    
    report = {
        'slide_name': slide_name,
        'raw_pollen_count': raw_analysis['pollen_classified'],
        'confirmed_pollen_count': conf_analysis['pollen_classified'],
        'new_annotations_count': conf_analysis['pollen_unclassified'],
        'preserved': 0,
        'modified': 0,
        'issues': []
    }
    
    # Check that original annotations are preserved
    preserved = raw_analysis['pollen_bboxes'] & conf_analysis['pollen_bboxes']
    removed = raw_analysis['pollen_bboxes'] - conf_analysis['pollen_bboxes']
    
    report['preserved'] = len(preserved)
    
    if removed:
        report['issues'].append(f"WARNING: {len(removed)} original pollen annotations were removed")
    
    # Build curated feature list
    curated_features = []
    pollen_idx = 0
    
    # Add test region first
    for feat in confirmed_data['features']:
        props = feat.get('properties', {})
        cls = props.get('classification')
        if isinstance(cls, dict) and cls.get('name') == 'Test Region':
            curated_features.append(feat)
            break
    
    # Add classified pollen
    for feat in confirmed_data['features']:
        props = feat.get('properties', {})
        cls = props.get('classification')
        if isinstance(cls, dict) and cls.get('name') == 'Pollen':
            curated_features.append(feat)
            pollen_idx += 1
    
    # Add unclassified annotations with proper classification
    for feat in confirmed_data['features']:
        props = feat.get('properties', {})
        cls = props.get('classification')
        
        if cls is None:
            # This is a new annotation - add classification
            new_feat = {
                'type': 'Feature',
                'geometry': feat['geometry'],
                'properties': {
                    'objectType': 'annotation',
                    'name': f'pollen_{pollen_idx}',
                    'classification': {
                        'name': 'Pollen',
                        'color': [0, 0, 255]
                    },
                    'source': 'expert_added'  # Mark as expert-added
                }
            }
            curated_features.append(new_feat)
            pollen_idx += 1
    
    curated_data = {
        'type': 'FeatureCollection',
        'features': curated_features
    }
    
    report['final_pollen_count'] = pollen_idx
    report['status'] = 'success' if not report['issues'] else 'warnings'
    
    return curated_data, report


def process_single(raw_path: str, confirmed_path: str, output_path: str) -> Dict:
    """Process a single pair of files."""
    # Load files
    with open(raw_path) as f:
        raw_data = json.load(f)
    with open(confirmed_path) as f:
        confirmed_data = json.load(f)
    
    # Validate
    raw_valid, raw_msg = validate_geojson(raw_data)
    conf_valid, conf_msg = validate_geojson(confirmed_data)
    
    if not raw_valid:
        return {'status': 'error', 'error': f"Raw GeoJSON invalid: {raw_msg}"}
    if not conf_valid:
        return {'status': 'error', 'error': f"Confirmed GeoJSON invalid: {conf_msg}"}
    
    # Extract slide name from path
    slide_name = Path(raw_path).stem.replace('_test_region', '')
    
    # Curate
    curated_data, report = curate_annotations(raw_data, confirmed_data, slide_name)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(curated_data, f, indent=2)
    
    report['output_path'] = output_path
    return report


def process_batch(raw_dir: str, confirmed_dir: str, output_dir: str) -> List[Dict]:
    """Process all matching pairs in directories."""
    reports = []
    
    confirmed_files = {f for f in os.listdir(confirmed_dir) if f.endswith('.geojson')}
    raw_files = {f for f in os.listdir(raw_dir) if f.endswith('.geojson')}
    
    print(f"Found {len(raw_files)} raw files, {len(confirmed_files)} confirmed files")
    
    for conf_file in sorted(confirmed_files):
        # Try to find matching raw file
        # QuPath exports may have different naming (e.g., _pyramid vs _test_region)
        base_name = conf_file.replace('_pyramid.geojson', '').replace('.geojson', '')
        
        raw_candidates = [
            f for f in raw_files 
            if base_name in f or f.replace('_test_region.geojson', '') == base_name
        ]
        
        if not raw_candidates:
            reports.append({
                'confirmed_file': conf_file,
                'status': 'error',
                'error': 'No matching raw file found'
            })
            continue
        
        raw_file = raw_candidates[0]
        output_file = base_name + '_curated.geojson'
        
        print(f"Processing: {conf_file}")
        print(f"  Raw: {raw_file}")
        
        report = process_single(
            os.path.join(raw_dir, raw_file),
            os.path.join(confirmed_dir, conf_file),
            os.path.join(output_dir, output_file)
        )
        report['confirmed_file'] = conf_file
        report['raw_file'] = raw_file
        reports.append(report)
    
    return reports


def main():
    parser = argparse.ArgumentParser(description='Curate QuPath annotations for evaluation')
    parser.add_argument('--raw', type=str, help='Path to raw GeoJSON file')
    parser.add_argument('--confirmed', type=str, help='Path to confirmed GeoJSON file')
    parser.add_argument('--output', type=str, help='Path to output curated GeoJSON file')
    parser.add_argument('--batch', action='store_true', help='Process all files in directories')
    parser.add_argument('--raw_dir', type=str, help='Directory containing raw GeoJSON files')
    parser.add_argument('--confirmed_dir', type=str, help='Directory containing confirmed GeoJSON files')
    parser.add_argument('--output_dir', type=str, help='Directory for curated output files')
    parser.add_argument('--validate_only', action='store_true', help='Only validate, do not curate')
    
    args = parser.parse_args()
    
    if args.batch:
        if not all([args.raw_dir, args.confirmed_dir, args.output_dir]):
            parser.error("Batch mode requires --raw_dir, --confirmed_dir, and --output_dir")
        
        reports = process_batch(args.raw_dir, args.confirmed_dir, args.output_dir)
        
        print("\n=== BATCH PROCESSING SUMMARY ===")
        success = sum(1 for r in reports if r.get('status') == 'success')
        warnings = sum(1 for r in reports if r.get('status') == 'warnings')
        errors = sum(1 for r in reports if r.get('status') == 'error')
        
        print(f"Total: {len(reports)}")
        print(f"Success: {success}")
        print(f"Warnings: {warnings}")
        print(f"Errors: {errors}")
        
        for r in reports:
            if r.get('status') == 'error':
                print(f"\nERROR: {r.get('confirmed_file', 'unknown')}")
                print(f"  {r.get('error')}")
            elif r.get('issues'):
                print(f"\nWARNINGS: {r.get('slide_name', 'unknown')}")
                for issue in r['issues']:
                    print(f"  {issue}")
    
    elif args.validate_only:
        # Just validate the files
        if args.raw:
            with open(args.raw) as f:
                raw_data = json.load(f)
            valid, msg = validate_geojson(raw_data)
            analysis = analyze_geojson(raw_data)
            print(f"Raw file: {args.raw}")
            print(f"  Valid: {valid} ({msg})")
            print(f"  Test regions: {analysis['test_regions']}")
            print(f"  Pollen (classified): {analysis['pollen_classified']}")
            print(f"  Pollen (unclassified): {analysis['pollen_unclassified']}")
        
        if args.confirmed:
            with open(args.confirmed) as f:
                conf_data = json.load(f)
            valid, msg = validate_geojson(conf_data)
            analysis = analyze_geojson(conf_data)
            print(f"\nConfirmed file: {args.confirmed}")
            print(f"  Valid: {valid} ({msg})")
            print(f"  Test regions: {analysis['test_regions']}")
            print(f"  Pollen (classified): {analysis['pollen_classified']}")
            print(f"  Pollen (unclassified): {analysis['pollen_unclassified']}")
    
    else:
        if not all([args.raw, args.confirmed, args.output]):
            parser.error("Single file mode requires --raw, --confirmed, and --output")
        
        report = process_single(args.raw, args.confirmed, args.output)
        
        print("\n=== CURATION REPORT ===")
        print(f"Slide: {report.get('slide_name', 'unknown')}")
        print(f"Status: {report.get('status', 'unknown')}")
        print(f"Raw pollen: {report.get('raw_pollen_count', 0)}")
        print(f"Preserved: {report.get('preserved', 0)}")
        print(f"New annotations: {report.get('new_annotations_count', 0)}")
        print(f"Final pollen count: {report.get('final_pollen_count', 0)}")
        
        if report.get('issues'):
            print("\nIssues:")
            for issue in report['issues']:
                print(f"  - {issue}")
        
        print(f"\nOutput: {report.get('output_path', 'N/A')}")


if __name__ == '__main__':
    main()
