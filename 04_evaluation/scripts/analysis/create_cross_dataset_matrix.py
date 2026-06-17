#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Cross-Dataset Species Overlap Matrix
=====================================

Analyzes species overlap between datasets (French, Hungarian, Mediterranean, Swedish)
to determine which cross-region train→test combinations are valid.

This is CRITICAL for designing inter-region experiments:
- Training on dataset A, testing on dataset B requires overlapping taxa
- Only shared species can be evaluated in cross-region experiments

Usage:
    python 04_evaluation/scripts/analysis/create_cross_dataset_matrix.py
    
Output:
    data/04_evaluation/results/cross_dataset_matrix.json
    data/04_evaluation/results/cross_dataset_matrix.txt (human-readable)

"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.species_mapping import load_caption_anchors


DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))


def resolve_splits_manifest() -> Path:
    """Resolve the split manifest from the public data root, with legacy fallback."""
    candidates = [
        DATA_ROOT / "04_evaluation" / "splits" / "manifest.json",
        PROJECT_ROOT / "04_evaluation" / "splits" / "manifest.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find split manifest. Expected data/04_evaluation/splits/manifest.json "
        "or legacy 04_evaluation/splits/manifest.json."
    )


def get_dataset_from_slide(slide_name: str) -> str:
    """
    Determine dataset from slide name.
    
    Naming conventions:
    - mediterranean_*: Mediterranean
    - hun_* or Ambrosia-Iva_*: Hungarian
    - Contains 40x_ZS, 20x_ZS, layers_40x, merged_reference (uppercase): Swedish
    - Everything else with _edf suffix: French
    """
    # Check explicit prefixes first
    if slide_name.startswith("mediterranean_"):
        return "mediterranean"
    elif slide_name.startswith("hun_") or slide_name.startswith("Ambrosia-Iva"):
        return "hungarian"
    
    # Swedish slides have specific patterns (case-sensitive naming)
    swedish_patterns = [
        "_40x_ZS", "_20x_ZS", "layers_40x", 
        "merged_reference", "mm_circle"
    ]
    if any(pattern in slide_name for pattern in swedish_patterns):
        return "swedish"
    
    # Default: French (most slides with _edf pattern)
    return "french"


def build_dataset_species_mapping(
    caption_anchors: Dict[str, str]
) -> Tuple[Dict[str, Set[str]], Dict[str, List[str]], Dict[str, Dict[str, int]]]:
    """
    Build mappings from dataset to species and slides.
    
    Returns:
        dataset_species: {dataset: {species1, species2, ...}}
        dataset_slides: {dataset: [slide1, slide2, ...]}
        dataset_counts: {dataset: {species: sample_count}}
    """
    # Load splits manifest for sample counts
    manifest_path = resolve_splits_manifest()
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    slides_info = manifest.get("slides", {})
    
    dataset_species: Dict[str, Set[str]] = defaultdict(set)
    dataset_slides: Dict[str, List[str]] = defaultdict(list)
    dataset_species_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for slide_name, species in caption_anchors.items():
        # Skip Unknown species
        if species.lower() == "unknown":
            continue
            
        dataset = get_dataset_from_slide(slide_name)
        dataset_species[dataset].add(species)
        dataset_slides[dataset].append(slide_name)
        
        # Add sample counts
        if slide_name in slides_info:
            train_count = slides_info[slide_name].get("train_count", 0)
            val_count = slides_info[slide_name].get("val_count", 0)
            dataset_species_counts[dataset][species] += train_count + val_count
    
    return (
        {k: v for k, v in dataset_species.items()},
        {k: v for k, v in dataset_slides.items()},
        {k: dict(v) for k, v in dataset_species_counts.items()}
    )


def compute_overlap_matrix(
    dataset_species: Dict[str, Set[str]]
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Compute species overlap for all dataset pairs.
    
    Returns:
        {train_dataset: {test_dataset: {overlapping_species}}}
    """
    datasets = ["french", "hungarian", "mediterranean", "swedish"]
    overlap_matrix = {}
    
    for train_ds in datasets:
        overlap_matrix[train_ds] = {}
        train_species = dataset_species.get(train_ds, set())
        
        for test_ds in datasets:
            test_species = dataset_species.get(test_ds, set())
            overlap = train_species & test_species
            overlap_matrix[train_ds][test_ds] = overlap
    
    return overlap_matrix


def generate_report(
    dataset_species: Dict[str, Set[str]],
    dataset_slides: Dict[str, List[str]],
    dataset_counts: Dict[str, Dict[str, int]],
    overlap_matrix: Dict[str, Dict[str, Set[str]]],
) -> str:
    """Generate human-readable report."""
    datasets = ["french", "hungarian", "mediterranean", "swedish"]
    
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-DATASET SPECIES OVERLAP MATRIX")
    lines.append("Pollen AI Atlas - Inter-Region Evaluation Analysis")
    lines.append("=" * 80)
    
    # Dataset summary
    lines.append("\n" + "=" * 80)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 80)
    
    for ds in datasets:
        species = dataset_species.get(ds, set())
        slides = dataset_slides.get(ds, [])
        total_samples = sum(dataset_counts.get(ds, {}).values())
        
        lines.append(f"\n{ds.upper()}")
        lines.append(f"  Slides: {len(slides)}")
        lines.append(f"  Species: {len(species)}")
        lines.append(f"  Samples: {total_samples:,}")
        lines.append(f"  Taxa: {', '.join(sorted(species))}")
    
    # Overlap matrix (counts only)
    lines.append("\n" + "=" * 80)
    lines.append("OVERLAP MATRIX (Number of Shared Species)")
    lines.append("=" * 80)
    
    # Header
    header = f"{'Train→Test':<15}"
    for ds in datasets:
        header += f"{ds[:10]:>12}"
    lines.append(header)
    lines.append("-" * 65)
    
    # Matrix rows
    for train_ds in datasets:
        row = f"{train_ds:<15}"
        for test_ds in datasets:
            overlap = overlap_matrix[train_ds][test_ds]
            count = len(overlap)
            if train_ds == test_ds:
                row += f"{'(self)':>12}"
            else:
                row += f"{count:>12}"
        lines.append(row)
    
    # Detailed overlap for each pair
    lines.append("\n" + "=" * 80)
    lines.append("DETAILED OVERLAP (for cross-region evaluation)")
    lines.append("=" * 80)
    
    for train_ds in datasets:
        for test_ds in datasets:
            if train_ds == test_ds:
                continue
            
            overlap = overlap_matrix[train_ds][test_ds]
            if not overlap:
                lines.append(f"\n{train_ds.upper()} → {test_ds.upper()}: NO OVERLAP (cannot evaluate)")
            else:
                lines.append(f"\n{train_ds.upper()} → {test_ds.upper()} ({len(overlap)} shared taxa)")
                lines.append(f"  Shared: {', '.join(sorted(overlap))}")
                
                # Species unique to each
                train_only = dataset_species.get(train_ds, set()) - overlap
                test_only = dataset_species.get(test_ds, set()) - overlap
                
                if train_only:
                    lines.append(f"  Train-only ({len(train_only)}): {', '.join(sorted(train_only))}")
                if test_only:
                    lines.append(f"  Test-only ({len(test_only)}): {', '.join(sorted(test_only))}")
    
    # Recommendations
    lines.append("\n" + "=" * 80)
    lines.append("RECOMMENDED EXPERIMENTS")
    lines.append("=" * 80)
    
    # Sort by overlap count
    pairs = []
    for train_ds in datasets:
        for test_ds in datasets:
            if train_ds != test_ds:
                overlap = overlap_matrix[train_ds][test_ds]
                pairs.append((train_ds, test_ds, len(overlap), overlap))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    lines.append("\nRanked by overlap size (higher = more comparable):")
    for i, (train, test, count, species) in enumerate(pairs, 1):
        if count > 0:
            lines.append(f"  {i}. {train:>13} → {test:<13} : {count} shared taxa")
    
    # Hungarian as source (user's preference for debugging)
    lines.append("\n" + "=" * 80)
    lines.append("HUNGARIAN AS TRAINING SET (recommended for debugging)")
    lines.append("=" * 80)
    
    hun_species = dataset_species.get("hungarian", set())
    hun_samples = sum(dataset_counts.get("hungarian", {}).values())
    lines.append(f"\nHungarian has {len(hun_species)} species, {hun_samples:,} samples")
    lines.append("\nCan evaluate on:")
    
    for test_ds in ["french", "mediterranean", "swedish"]:
        overlap = overlap_matrix["hungarian"][test_ds]
        if overlap:
            test_samples = sum(
                dataset_counts.get(test_ds, {}).get(sp, 0) 
                for sp in overlap
            )
            lines.append(f"  - {test_ds}: {len(overlap)} taxa, ~{test_samples:,} test samples (filtered by overlap)")
    
    return "\n".join(lines)


def main():
    """Generate cross-dataset matrix and save outputs."""
    print("Loading caption anchors...")
    caption_anchors = load_caption_anchors()
    print(f"  Loaded {len(caption_anchors)} slide→species mappings")
    
    print("\nBuilding dataset mappings...")
    dataset_species, dataset_slides, dataset_counts = build_dataset_species_mapping(caption_anchors)
    
    for ds in ["french", "hungarian", "mediterranean", "swedish"]:
        print(f"  {ds}: {len(dataset_slides.get(ds, []))} slides, {len(dataset_species.get(ds, set()))} species")
    
    print("\nComputing overlap matrix...")
    overlap_matrix = compute_overlap_matrix(dataset_species)
    
    # Generate report
    report = generate_report(dataset_species, dataset_slides, dataset_counts, overlap_matrix)
    print(report)
    
    # Save outputs
    output_dir = DATA_ROOT / "04_evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save human-readable report
    report_path = output_dir / "cross_dataset_matrix.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n[Saved] {report_path}")
    
    # Save structured JSON
    json_data = {
        "description": "Cross-dataset species overlap for inter-region evaluation",
        "datasets": {
            ds: {
                "slides": dataset_slides.get(ds, []),
                "species": sorted(dataset_species.get(ds, set())),
                "species_counts": dataset_counts.get(ds, {}),
                "total_samples": sum(dataset_counts.get(ds, {}).values()),
            }
            for ds in ["french", "hungarian", "mediterranean", "swedish"]
        },
        "overlap_matrix": {
            train_ds: {
                test_ds: sorted(overlap_matrix[train_ds][test_ds])
                for test_ds in ["french", "hungarian", "mediterranean", "swedish"]
            }
            for train_ds in ["french", "hungarian", "mediterranean", "swedish"]
        },
        "recommendations": {
            "best_pairs": [
                {"train": train, "test": test, "shared_taxa": count}
                for train, test, count, _ in sorted(
                    [
                        (t, e, len(overlap_matrix[t][e]), overlap_matrix[t][e])
                        for t in ["french", "hungarian", "mediterranean", "swedish"]
                        for e in ["french", "hungarian", "mediterranean", "swedish"]
                        if t != e and len(overlap_matrix[t][e]) > 0
                    ],
                    key=lambda x: x[2], 
                    reverse=True
                )
            ]
        }
    }
    
    json_path = output_dir / "cross_dataset_matrix.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
