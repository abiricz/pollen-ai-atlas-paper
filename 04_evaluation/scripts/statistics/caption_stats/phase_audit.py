# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Expert audit sample export.

  The stratification ensures representation across:
    • Datasets (geographic sites)
    • Species (taxonomic diversity)
    • Slides (imaging variability)

  Typical audit axes (Likert 1-5):
    • Morphology accuracy
    • Completeness
    • Clarity / fluency
    • Non-hallucination
    • Correct use of uncertainty

  Inter-rater agreement (Cohen's κ) should be computed on the returned
  sample and reported in the paper. If this phase is to be done.

Outputs:
  • JSONL + CSV with all-model captions for human scoring
  • Summary JSON with allocation breakdown
"""

import csv
import json
from collections import Counter, defaultdict
from random import Random

from .constants import MODELS
from .helpers import (
    allocate_budget_with_min,
    pair_caption_entries,
    resolve_dataset,
    safe_model_name,
)


# ── Phase entry point ────────────────────────────────────────────────

def phase_export_expert_audit_sample(
    model_data,
    model_dataset,
    slide_species,
    slide_dataset_truth,
    output_dir,
    sample_size=240,
    seed=42,
):
    """Export a hierarchical stratified caption sample for expert audit.

    Uses all available models. Matching is based on grain pair keys
    across the first two models, and captions from remaining models
    are included where available.

    Returns a summary dict.
    """
    print("\n" + "=" * 60)
    print("Phase 7: Expert audit sample export")
    print("=" * 60)

    sample_size = int(sample_size)
    if sample_size <= 0:
        print("  Skipping (sample_size <= 0)")
        return {"enabled": False, "reason": "sample_size <= 0"}

    # Determine available models
    available_models = sorted(mk for mk in MODELS if mk in model_data and model_data[mk])
    if len(available_models) < 2:
        print("  Skipping (fewer than 2 models available)")
        return {"enabled": False, "reason": "fewer than 2 models"}

    # Use first two models as the anchor pair for matching
    mk_a = available_models[0]
    mk_b = available_models[1]
    other_models = available_models[2:]

    print(f"  Anchor pair: {safe_model_name(mk_a)} vs {safe_model_name(mk_b)}")
    print(f"  Additional models: {', '.join(safe_model_name(m) for m in other_models)}")

    # Count available matched pairs per slide and build structure
    slide_counts = {}
    structure = defaultdict(lambda: defaultdict(dict))
    common_slides = sorted(set(model_data[mk_a].keys()) & set(model_data[mk_b].keys()))
    for slide in common_slides:
        pairs, _, _, _, _ = pair_caption_entries(model_data[mk_a][slide], model_data[mk_b][slide])
        n = len(pairs)
        if n <= 0:
            continue
        dataset = resolve_dataset(slide, model_dataset[mk_a].get(slide), slide_dataset_truth)
        species = slide_species.get(slide, "Unknown")
        slide_counts[slide] = n
        structure[dataset][species][slide] = n

    if not slide_counts:
        print("  Skipping (no matched pairs)")
        return {"enabled": False, "reason": "no matched pairs"}

    # Hierarchical budget allocation: dataset → species → slide
    dataset_counts = {
        dataset: sum(sum(slide_map.values()) for slide_map in species_map.values())
        for dataset, species_map in structure.items()
    }
    dataset_budget = allocate_budget_with_min(dataset_counts, sample_size, min_each=1)

    species_budget = {}
    slide_budget = {}
    for dataset, species_map in structure.items():
        sp_counts = {sp: sum(slides.values()) for sp, slides in species_map.items()}
        sp_alloc = allocate_budget_with_min(sp_counts, dataset_budget.get(dataset, 0), min_each=1)
        species_budget[dataset] = sp_alloc
        for sp, slides in species_map.items():
            alloc = allocate_budget_with_min(slides, sp_alloc.get(sp, 0), min_each=1)
            for slide, n_alloc in alloc.items():
                if n_alloc > 0:
                    slide_budget[slide] = n_alloc

    # Sample from each slide
    rng = Random(seed)
    rows = []
    for slide, k in sorted(slide_budget.items()):
        pairs, _, _, _, _ = pair_caption_entries(model_data[mk_a][slide], model_data[mk_b][slide])
        if not pairs:
            continue
        take = min(k, len(pairs))
        chosen = rng.sample(pairs, take) if take < len(pairs) else pairs
        dataset = resolve_dataset(slide, model_dataset[mk_a].get(slide), slide_dataset_truth)
        species = slide_species.get(slide, "Unknown")

        # Build lookup dicts for other models on this slide
        other_lookups = {}
        for mk_other in other_models:
            if slide in model_data.get(mk_other, {}):
                lookup = {}
                for caption, pair_key in model_data[mk_other][slide]:
                    lookup[pair_key] = caption
                other_lookups[mk_other] = lookup

        for pair_key, cap_a, cap_b in chosen:
            row = {
                "dataset": dataset,
                "species": species,
                "slide": slide,
                "pair_key": pair_key,
                f"caption_{safe_model_name(mk_a)}": cap_a,
                f"caption_{safe_model_name(mk_b)}": cap_b,
            }
            for mk_other in other_models:
                lookup = other_lookups.get(mk_other, {})
                row[f"caption_{safe_model_name(mk_other)}"] = lookup.get(pair_key, "")
            rows.append(row)

    if len(rows) > sample_size:
        rows = rng.sample(rows, sample_size)

    for i, row in enumerate(rows):
        row["audit_id"] = f"audit_{i:05d}"

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "caption_expert_audit_sample.jsonl"
    csv_path = output_dir / "caption_expert_audit_sample.csv"
    summary_path = output_dir / "caption_expert_audit_sample_summary.json"

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Build fieldnames dynamically from all models
    caption_fields = [f"caption_{safe_model_name(mk)}" for mk in available_models]
    fieldnames = ["audit_id", "dataset", "species", "slide", "pair_key"] + caption_fields

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    per_dataset = Counter(row["dataset"] for row in rows)
    per_species = Counter(row["species"] for row in rows)
    per_slide = Counter(row["slide"] for row in rows)

    summary = {
        "enabled": True,
        "seed": seed,
        "n_models": len(available_models),
        "model_keys": available_models,
        "model_labels": [safe_model_name(mk) for mk in available_models],
        "requested_sample_size": sample_size,
        "actual_sample_size": len(rows),
        "n_datasets": len(per_dataset),
        "n_species": len(per_species),
        "n_slides": len(per_slide),
        "allocation": {
            "dataset_budget": dataset_budget,
            "species_budget": species_budget,
            "slide_budget": slide_budget,
        },
        "distribution": {
            "per_dataset": dict(sorted(per_dataset.items())),
            "per_species_top20": dict(per_species.most_common(20)),
            "per_slide_top20": dict(per_slide.most_common(20)),
        },
        "paths": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  Exported {len(rows)} rows across {len(per_dataset)} datasets, "
        f"{len(per_species)} species, {len(per_slide)} slides"
    )
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")

    summary["paths"]["summary_json"] = str(summary_path)
    return summary
