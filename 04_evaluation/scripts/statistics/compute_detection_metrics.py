#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Compute Detection Metrics from Expert-Curated Annotations

This script analyzes the curated GeoJSON files to compute:
- Precision: TP / (TP + FP) - proportion of detected pollen that are true positives
- Recall: TP / (TP + FN) - proportion of actual pollen that were detected
- F1 Score: Harmonic mean of precision and recall

Definitions:
- TP (True Positive): Original detections that were kept (not deleted by expert)
- FP (False Positive): Original detections that were deleted by expert  
- FN (False Negative): Pollen grains added by expert (missed by pipeline)

The analysis is provided at multiple levels:
- Per slide
- Per dataset (French, Hungarian, Mediterranean, Swedish)
- Per taxonomic family
- Per taxonomic species/genus
- Overall

Usage:
    python compute_detection_metrics.py [--output_dir path/to/output]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_taxonomy_from_anchors(anchor_dir: str) -> dict:
    """Load species and family information from anchor files."""
    taxonomy = {}
    
    anchor_files = [f for f in os.listdir(anchor_dir) if f.endswith('_species.txt')]
    
    for species_file in anchor_files:
        slide_base = species_file.replace('_species.txt', '')
        
        species_path = os.path.join(anchor_dir, species_file)
        family_path = os.path.join(anchor_dir, f'{slide_base}_family.txt')
        
        species = None
        family = None
        
        if os.path.exists(species_path):
            with open(species_path) as f:
                species = f.read().strip()
        
        if os.path.exists(family_path):
            with open(family_path) as f:
                family = f.read().strip()
        
        taxonomy[slide_base] = {
            'species': species,
            'family': family
        }
    
    return taxonomy


def get_dataset(fname: str) -> str:
    """Determine dataset from filename."""
    fname_lower = fname.lower()
    if fname_lower.startswith('hun_'):
        return 'Hungarian'
    if 'ambrosia-iva' in fname_lower or 'ambrosia_iva' in fname_lower:
        return 'Hungarian'
    if 'mediterranean' in fname_lower:
        return 'Mediterranean'
    if '_edf_' in fname_lower or fname_lower.endswith('_edf_curated.geojson'):
        return 'French'
    return 'Swedish'


def analyze_curated_file(curated_path: str, raw_path: str) -> dict:
    """Analyze a single curated file against its raw counterpart."""
    with open(curated_path) as f:
        curated_data = json.load(f)
    
    with open(raw_path) as f:
        raw_data = json.load(f)
    
    # Count raw pollen (original detections)
    raw_pollen = sum(1 for f in raw_data['features']
                     if f.get('properties', {}).get('classification', {}).get('name') == 'Pollen')
    
    # Count curated pollen
    curated_original = 0
    curated_expert_added = 0
    test_region = None
    
    for feat in curated_data['features']:
        props = feat.get('properties', {})
        cls = props.get('classification', {})
        
        if isinstance(cls, dict) and cls.get('name') == 'Pollen':
            if props.get('source') == 'expert_added':
                curated_expert_added += 1
            else:
                curated_original += 1
        elif isinstance(cls, dict) and cls.get('name') == 'Test Region':
            coords = feat['geometry']['coordinates']
            if isinstance(coords[0][0], list):
                coords = coords[0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            test_region = {
                'x': int(min(xs)),
                'y': int(min(ys)),
                'width': int(max(xs) - min(xs)),
                'height': int(max(ys) - min(ys))
            }
    
    # Compute metrics
    # TP = curated_original (detections that were kept)
    # FP = raw_pollen - curated_original (detections that were deleted)
    # FN = curated_expert_added (missed detections added by expert)
    
    tp = curated_original
    fp = raw_pollen - curated_original
    fn = curated_expert_added
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'raw_detections': raw_pollen,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'total_ground_truth': tp + fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'test_region': test_region
    }


def aggregate_metrics(metrics_list: list) -> dict:
    """Aggregate metrics from multiple slides."""
    total_tp = sum(m['true_positives'] for m in metrics_list)
    total_fp = sum(m['false_positives'] for m in metrics_list)
    total_fn = sum(m['false_negatives'] for m in metrics_list)
    total_gt = sum(m['total_ground_truth'] for m in metrics_list)
    total_raw = sum(m['raw_detections'] for m in metrics_list)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'num_slides': len(metrics_list),
        'total_raw_detections': total_raw,
        'total_true_positives': total_tp,
        'total_false_positives': total_fp,
        'total_false_negatives': total_fn,
        'total_ground_truth': total_gt,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def main():
    parser = argparse.ArgumentParser(description='Compute detection metrics from curated annotations')
    parser.add_argument('--output_dir', type=str, 
                        default='data/04_evaluation/qupath_annotator_process',
                        help='Directory to save output files')
    parser.add_argument('--curated_dir', type=str,
                        default='data/04_evaluation/qupath_annotator_process/curated_geojson',
                        help='Directory containing curated GeoJSON files')
    parser.add_argument('--raw_dir', type=str,
                        default='data/04_evaluation/qupath_annotator_process/raw_geojson',
                        help='Directory containing raw GeoJSON files')
    parser.add_argument('--anchor_dir', type=str,
                        default='03_captioning/caption_anchors',
                        help='Directory containing taxonomy anchor files')
    args = parser.parse_args()
    
    # Get project root (statistics → scripts → 04_evaluation → project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    
    curated_dir = project_root / args.curated_dir
    raw_dir = project_root / args.raw_dir
    anchor_dir = project_root / args.anchor_dir
    output_dir = project_root / args.output_dir
    
    # Load taxonomy
    print("Loading taxonomy from anchor files...")
    taxonomy = load_taxonomy_from_anchors(str(anchor_dir))
    print(f"  Loaded taxonomy for {len(taxonomy)} slides")
    
    # Find all curated files
    curated_files = sorted([f for f in os.listdir(curated_dir) if f.endswith('.geojson')])
    print(f"\nFound {len(curated_files)} curated files")
    
    # Analyze each file
    results = {
        'generated': datetime.now().isoformat(),
        'slides': [],
        'by_dataset': {},
        'by_family': {},
        'by_species': {},
        'overall': {}
    }
    
    by_dataset = defaultdict(list)
    by_family = defaultdict(list)
    by_species = defaultdict(list)
    all_metrics = []
    
    print("\nAnalyzing files...")
    for curated_file in curated_files:
        slide_base = curated_file.replace('_curated.geojson', '')
        
        # Find matching raw file - EXACT match required
        raw_file = None
        for pattern in [f'{slide_base}_test_region.geojson', f'{slide_base}.geojson']:
            if pattern in os.listdir(raw_dir):
                raw_file = pattern
                break
        
        if not raw_file:
            print(f"  WARNING: No raw file found for {curated_file}")
            continue
        
        # Analyze
        metrics = analyze_curated_file(
            str(curated_dir / curated_file),
            str(raw_dir / raw_file)
        )
        
        # Get taxonomy
        tax = taxonomy.get(slide_base, {})
        species = tax.get('species', 'Unknown')
        family = tax.get('family', 'Unknown')
        dataset = get_dataset(curated_file)
        
        slide_result = {
            'file': curated_file,
            'slide_base': slide_base,
            'dataset': dataset,
            'species': species,
            'family': family,
            **metrics
        }
        
        results['slides'].append(slide_result)
        all_metrics.append(metrics)
        by_dataset[dataset].append(metrics)
        by_family[family].append(metrics)
        by_species[species].append(metrics)
    
    # Aggregate by dataset
    print("\n=== METRICS BY DATASET ===")
    print(f"{'Dataset':<15} {'Slides':>6} {'GT':>6} {'TP':>6} {'FP':>5} {'FN':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 75)
    
    for dataset in ['French', 'Hungarian', 'Mediterranean', 'Swedish']:
        if dataset in by_dataset:
            agg = aggregate_metrics(by_dataset[dataset])
            results['by_dataset'][dataset] = agg
            print(f"{dataset:<15} {agg['num_slides']:>6} {agg['total_ground_truth']:>6} "
                  f"{agg['total_true_positives']:>6} {agg['total_false_positives']:>5} "
                  f"{agg['total_false_negatives']:>6} {agg['precision']:>6.1%} "
                  f"{agg['recall']:>6.1%} {agg['f1_score']:>6.1%}")
    
    # Overall
    overall = aggregate_metrics(all_metrics)
    results['overall'] = overall
    print("-" * 75)
    print(f"{'OVERALL':<15} {overall['num_slides']:>6} {overall['total_ground_truth']:>6} "
          f"{overall['total_true_positives']:>6} {overall['total_false_positives']:>5} "
          f"{overall['total_false_negatives']:>6} {overall['precision']:>6.1%} "
          f"{overall['recall']:>6.1%} {overall['f1_score']:>6.1%}")
    
    # By family
    print("\n=== METRICS BY FAMILY ===")
    print(f"{'Family':<25} {'Slides':>6} {'GT':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 60)
    
    for family in sorted(by_family.keys()):
        agg = aggregate_metrics(by_family[family])
        results['by_family'][family] = agg
        if agg['num_slides'] >= 1:
            print(f"{family[:25]:<25} {agg['num_slides']:>6} {agg['total_ground_truth']:>6} "
                  f"{agg['precision']:>6.1%} {agg['recall']:>6.1%} {agg['f1_score']:>6.1%}")
    
    # By species (top contributors)
    print("\n=== METRICS BY SPECIES (sorted by pollen count) ===")
    print(f"{'Species':<35} {'Slides':>6} {'GT':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 75)
    
    species_sorted = sorted(by_species.items(), 
                           key=lambda x: sum(m['total_ground_truth'] for m in x[1]), 
                           reverse=True)
    
    for species, metrics_list in species_sorted[:30]:
        agg = aggregate_metrics(metrics_list)
        results['by_species'][species] = agg
        print(f"{species[:35]:<35} {agg['num_slides']:>6} {agg['total_ground_truth']:>6} "
              f"{agg['precision']:>6.1%} {agg['recall']:>6.1%} {agg['f1_score']:>6.1%}")
    
    # Per-slide details (sorted by F1)
    print("\n=== PER-SLIDE METRICS (sorted by recall) ===")
    print(f"{'Slide':<50} {'GT':>5} {'TP':>5} {'FP':>4} {'FN':>5} {'Prec':>6} {'Rec':>6}")
    print("-" * 90)
    
    slides_sorted = sorted(results['slides'], key=lambda x: x['recall'])
    for s in slides_sorted[:20]:
        name = s['slide_base'][:50]
        print(f"{name:<50} {s['total_ground_truth']:>5} {s['true_positives']:>5} "
              f"{s['false_positives']:>4} {s['false_negatives']:>5} "
              f"{s['precision']:>6.1%} {s['recall']:>6.1%}")
    
    if len(slides_sorted) > 20:
        print(f"... and {len(slides_sorted) - 20} more slides")
    
    # Save results
    output_path = output_dir / 'detection_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved detailed metrics to: {output_path}")
    
    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"Total slides analyzed: {overall['num_slides']}")
    print(f"Total ground truth pollen: {overall['total_ground_truth']}")
    print(f"  - True Positives (correctly detected): {overall['total_true_positives']}")
    print(f"  - False Positives (incorrectly detected): {overall['total_false_positives']}")
    print(f"  - False Negatives (missed): {overall['total_false_negatives']}")
    print(f"\nOverall Precision: {overall['precision']:.1%}")
    print(f"Overall Recall: {overall['recall']:.1%}")
    print(f"Overall F1 Score: {overall['f1_score']:.1%}")


if __name__ == '__main__':
    main()
