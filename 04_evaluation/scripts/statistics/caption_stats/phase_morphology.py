# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Morphological vocabulary coverage.

Metrics computed per model / dataset / species / slide:
  • Per-caption precision, recall, F1 against slide-specific term set
  • Global vocabulary coverage %
  • Top-30 term frequency list
  • Unused terms
"""

from collections import Counter, defaultdict

from tqdm import tqdm

from .constants import MODELS
from .helpers import (
    new_stat,
    resolve_dataset,
    safe_model_name,
    stat_add,
    stat_finalize,
    tokenize,
)


# ── Group helpers ────────────────────────────────────────────────────

def _new_overlap_group():
    return {
        "n": 0,
        "precision": new_stat(),
        "recall": new_stat(),
        "f1": new_stat(),
        "caption_anchor_term_count": new_stat(),
        "caption_vocab_term_count": new_stat(),
    }


def _update_overlap_group(group, precision, recall, f1, anchor_term_count, vocab_term_count):
    group["n"] += 1
    stat_add(group["precision"], precision)
    stat_add(group["recall"], recall)
    stat_add(group["f1"], f1)
    stat_add(group["caption_anchor_term_count"], anchor_term_count)
    stat_add(group["caption_vocab_term_count"], vocab_term_count)


def _finalize_overlap_group(group, labels):
    return {
        **labels,
        "n": group["n"],
        "precision": stat_finalize(group["precision"]),
        "recall": stat_finalize(group["recall"]),
        "f1": stat_finalize(group["f1"]),
        "caption_anchor_term_count": stat_finalize(group["caption_anchor_term_count"]),
        "caption_vocab_term_count": stat_finalize(group["caption_vocab_term_count"]),
    }


# ── Phase entry point ────────────────────────────────────────────────

def phase_morphological_coverage(
    model_data,
    model_dataset,
    slide_species,
    slide_dataset_truth,
    vocab_terms,
    slide_term_sets,
):
    """Compute anchor-term coverage with per-slide/species/dataset breakdown.

    Returns a dict keyed by model_key.
    """
    print("\n" + "=" * 60)
    print("Phase 4: Morphological vocabulary coverage")
    print("=" * 60)

    results = {}

    for mk in MODELS:
        term_usage = Counter()       # terms correctly used (overlap with slide anchor)
        all_term_usage = Counter()   # ALL vocab terms mentioned in captions (any slide)
        total_captions = 0

        global_group = _new_overlap_group()
        ds_groups = defaultdict(_new_overlap_group)
        sp_groups = defaultdict(_new_overlap_group)
        slide_groups = defaultdict(_new_overlap_group)

        for slide, entries in tqdm(
            sorted(model_data[mk].items()),
            desc=f"  {safe_model_name(mk)}",
            leave=False,
        ):
            dataset = resolve_dataset(slide, model_dataset[mk].get(slide), slide_dataset_truth)
            species = slide_species.get(slide, "Unknown")
            slide_terms = set(slide_term_sets.get(slide, set()))

            for caption, _ in entries:
                total_captions += 1
                cap_terms = set(tokenize(caption))
                cap_vocab_terms = cap_terms & vocab_terms
                overlap = cap_vocab_terms & slide_terms

                precision = len(overlap) / max(len(cap_vocab_terms), 1)
                recall = len(overlap) / max(len(slide_terms), 1) if slide_terms else 0.0
                f1 = (
                    0.0
                    if (precision + recall) == 0
                    else 2.0 * precision * recall / (precision + recall)
                )

                term_usage.update(overlap)
                all_term_usage.update(cap_vocab_terms)

                _update_overlap_group(global_group, precision, recall, f1, len(overlap), len(cap_vocab_terms))
                _update_overlap_group(ds_groups[dataset], precision, recall, f1, len(overlap), len(cap_vocab_terms))
                _update_overlap_group(sp_groups[species], precision, recall, f1, len(overlap), len(cap_vocab_terms))
                _update_overlap_group(slide_groups[slide], precision, recall, f1, len(overlap), len(cap_vocab_terms))

        used_terms = {term for term, count in all_term_usage.items() if count > 0}
        unused_terms = sorted(vocab_terms - used_terms)
        correctly_used_terms = {term for term, count in term_usage.items() if count > 0}

        out = {
            "model_name": safe_model_name(mk),
            "total_anchor_terms": len(vocab_terms),
            "terms_used_in_captions": len(used_terms),
            "terms_correctly_used": len(correctly_used_terms),
            "terms_not_used": len(unused_terms),
            "global_vocab_coverage_pct": round(100.0 * len(used_terms) / max(len(vocab_terms), 1), 2),
            "total_captions": total_captions,
            "global_anchor_overlap": _finalize_overlap_group(global_group, {}),
            "top_30_terms": [
                {
                    "term": term,
                    "count": count,
                    "pct_captions": round(100.0 * count / max(total_captions, 1), 3),
                }
                for term, count in all_term_usage.most_common(30)
            ],
            "unused_terms": unused_terms[:80],
            "per_dataset": {},
            "per_species": {},
            "per_slide": {},
        }

        for dataset, group in ds_groups.items():
            key = f"{mk}__{dataset}"
            out["per_dataset"][key] = _finalize_overlap_group(
                group, {"model": safe_model_name(mk), "dataset": dataset}
            )

        for species, group in sp_groups.items():
            key = f"{mk}__{species}"
            out["per_species"][key] = _finalize_overlap_group(
                group, {"model": safe_model_name(mk), "species": species}
            )

        for slide, group in slide_groups.items():
            dataset = resolve_dataset(slide, model_dataset[mk].get(slide), slide_dataset_truth)
            species = slide_species.get(slide, "Unknown")
            key = f"{mk}__{slide}"
            out["per_slide"][key] = _finalize_overlap_group(
                group,
                {
                    "model": safe_model_name(mk),
                    "slide": slide,
                    "dataset": dataset,
                    "species": species,
                    "slide_anchor_term_count": len(slide_term_sets.get(slide, set())),
                },
            )

        results[mk] = out

        print(
            f"  {safe_model_name(mk)}: {len(used_terms)}/{len(vocab_terms)} terms used "
            f"({out['global_vocab_coverage_pct']:.1f}% global coverage)"
        )

    return results
