# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Basic statistics and prompt compliance.

Metrics computed (whole dataset, per-model/dataset/species/slide):
  • Word-count distribution + percentiles (P5/P25/P50/P75/P95)
  • Word-range compliance (60-80 words)
  • Digit incidence (any digit in caption)
  • Size-numeric leakage (measurement units like µm, mm)
  • Taxon-name leakage (species/family names in caption text)
  • Debris-opener rate
  • Hint-exemplar-inconsistent rate
  • Prompt-qualifier phrase frequencies
  • Prompt-marker frequencies
  • Vocabulary size and type-token ratio

These map directly to a "constraint compliance dashboard" recommended
for the Technical Validation section of the paper.
"""

from collections import Counter, defaultdict

from tqdm import tqdm

from .constants import (
    DEBRIS_OPENER,
    MODELS,
    NUMBER_RE,
    SIZE_NUMBER_RE,
)
from .helpers import (
    compute_hist_percentiles,
    new_stat,
    pct,
    resolve_dataset,
    safe_model_name,
    stat_add,
    stat_finalize,
    tokenize,
)


# ── Group helpers ────────────────────────────────────────────────────

def _new_basic_group():
    return {
        "n": 0,
        "word_count": new_stat(),
        "word_hist": Counter(),
        "in_word_range": 0,
        "digit_incidence": 0,
        "size_numeric_leakage": 0,
        "taxon_leakage": 0,
        "debris_opener": 0,
        "hint_exemplar_inconsistent": 0,
    }


def _update_basic_group(
    group,
    word_count,
    in_range,
    has_digit,
    has_size_numeric,
    has_taxon,
    is_debris,
    has_hint_inconsistent,
):
    group["n"] += 1
    stat_add(group["word_count"], word_count)
    group["word_hist"][int(word_count)] += 1
    group["in_word_range"] += int(in_range)
    group["digit_incidence"] += int(has_digit)
    group["size_numeric_leakage"] += int(has_size_numeric)
    group["taxon_leakage"] += int(has_taxon)
    group["debris_opener"] += int(is_debris)
    group["hint_exemplar_inconsistent"] += int(has_hint_inconsistent)


def _finalize_basic_group(group, labels):
    n = group["n"]
    return {
        **labels,
        "n": n,
        "word_count": stat_finalize(group["word_count"]),
        "word_count_percentiles": compute_hist_percentiles(group["word_hist"]),
        "word_range_60_80_pct": pct(group["in_word_range"], n),
        "digit_incidence_pct": pct(group["digit_incidence"], n),
        "size_numeric_leakage_pct": pct(group["size_numeric_leakage"], n),
        # Backward-compatible alias
        "numeric_leakage_pct": pct(group["digit_incidence"], n),
        "taxon_leakage_pct": pct(group["taxon_leakage"], n),
        "debris_opener_pct": pct(group["debris_opener"], n),
        "hint_exemplar_inconsistent_pct": pct(group["hint_exemplar_inconsistent"], n),
    }


# ── Phase entry point ────────────────────────────────────────────────

def phase_basic_stats(
    model_data,
    model_dataset,
    slide_species,
    slide_family,
    slide_dataset_truth,
    prompt_qualifiers,
    prompt_markers,
):
    """Compute basic statistics and prompt compliance metrics.

    Returns a dict with keys: ``per_model``, ``per_dataset``,
    ``per_species``, ``per_slide``.
    """
    print("\n" + "=" * 60)
    print("Phase 2: Basic statistics and prompt compliance")
    print("=" * 60)

    results = {"per_model": {}, "per_dataset": {}, "per_species": {}, "per_slide": {}}

    # Build taxon-name token set for leakage detection
    species_tokens = set()
    family_tokens = set()
    for species in slide_species.values():
        species_tokens.update(tokenize(species))
    for family in slide_family.values():
        family_tokens.update(tokenize(family))
    taxon_tokens = (species_tokens | family_tokens) - {"unknown"}

    # Normalize prompt qualifiers / markers for case-insensitive matching (codex fix)
    normalized_prompt_qualifiers = [
        (phrase, str(phrase).strip().lower())
        for phrase in prompt_qualifiers
        if str(phrase).strip()
    ]
    normalized_prompt_markers = [
        (marker, str(marker).strip().lower().rstrip(".,;:!?"))
        for marker in prompt_markers
        if str(marker).strip()
    ]

    for mk in MODELS:
        word_stat = new_stat()
        char_stat = new_stat()
        sentence_stat = new_stat()
        word_hist = Counter()

        token_total = 0
        unique_tokens = set()

        qualifier_counts = Counter()
        marker_counts = Counter()

        compliance_counts = Counter()
        total_captions = 0

        ds_groups = defaultdict(_new_basic_group)
        sp_groups = defaultdict(_new_basic_group)
        slide_groups = defaultdict(_new_basic_group)

        for slide, entries in tqdm(
            sorted(model_data[mk].items()),
            desc=f"  {safe_model_name(mk)}",
            leave=False,
        ):
            dataset = resolve_dataset(slide, model_dataset[mk].get(slide), slide_dataset_truth)
            species = slide_species.get(slide, "Unknown")

            for caption, _ in entries:
                total_captions += 1
                cap_lower = caption.lower()
                tokens = tokenize(cap_lower)

                word_count = len(tokens)
                char_count = len(caption)
                sentence_count = max(
                    caption.count(".") + caption.count("!") + caption.count("?"), 1
                )

                stat_add(word_stat, word_count)
                stat_add(char_stat, char_count)
                stat_add(sentence_stat, sentence_count)
                word_hist[int(word_count)] += 1

                token_total += len(tokens)
                unique_tokens.update(tokens)

                in_range = 60 <= word_count <= 80
                has_digit = bool(NUMBER_RE.search(caption))
                has_size_numeric = bool(SIZE_NUMBER_RE.search(caption))
                has_taxon = bool(set(tokens) & taxon_tokens)
                is_debris = cap_lower.startswith(DEBRIS_OPENER)
                has_hint_inconsistent = "hint or exemplar inconsistent" in cap_lower

                compliance_counts["in_word_range"] += int(in_range)
                compliance_counts["digit_incidence"] += int(has_digit)
                compliance_counts["size_numeric_leakage"] += int(has_size_numeric)
                compliance_counts["taxon_leakage"] += int(has_taxon)
                compliance_counts["debris_opener"] += int(is_debris)
                compliance_counts["hint_exemplar_inconsistent"] += int(has_hint_inconsistent)

                for phrase, phrase_norm in normalized_prompt_qualifiers:
                    if phrase_norm in cap_lower:
                        qualifier_counts[phrase] += 1

                for marker, marker_norm in normalized_prompt_markers:
                    if marker_norm in cap_lower:
                        marker_counts[marker] += 1

                _update_basic_group(
                    ds_groups[dataset], word_count, in_range,
                    has_digit, has_size_numeric, has_taxon, is_debris, has_hint_inconsistent,
                )
                _update_basic_group(
                    sp_groups[species], word_count, in_range,
                    has_digit, has_size_numeric, has_taxon, is_debris, has_hint_inconsistent,
                )
                _update_basic_group(
                    slide_groups[slide], word_count, in_range,
                    has_digit, has_size_numeric, has_taxon, is_debris, has_hint_inconsistent,
                )

        results["per_model"][mk] = {
            "model_name": safe_model_name(mk),
            "total_captions": total_captions,
            "word_count": stat_finalize(word_stat),
            "word_count_percentiles": compute_hist_percentiles(word_hist),
            "char_count": stat_finalize(char_stat),
            "sentence_count": stat_finalize(sentence_stat),
            "vocabulary_size": len(unique_tokens),
            "total_tokens": int(token_total),
            "type_token_ratio": round(len(unique_tokens) / max(token_total, 1), 6),
            "prompt_compliance": {
                "word_range_60_80_pct": pct(compliance_counts["in_word_range"], total_captions),
                "digit_incidence_pct": pct(compliance_counts["digit_incidence"], total_captions),
                "size_numeric_leakage_pct": pct(
                    compliance_counts["size_numeric_leakage"], total_captions
                ),
                # Backward-compatible alias
                "numeric_leakage_pct": pct(compliance_counts["digit_incidence"], total_captions),
                "taxon_leakage_pct": pct(compliance_counts["taxon_leakage"], total_captions),
                "debris_opener_pct": pct(compliance_counts["debris_opener"], total_captions),
                "hint_exemplar_inconsistent_pct": pct(
                    compliance_counts["hint_exemplar_inconsistent"], total_captions
                ),
            },
            "qualifier_phrases": {
                phrase: {
                    "count": int(qualifier_counts[phrase]),
                    "pct": pct(qualifier_counts[phrase], total_captions),
                }
                for phrase in prompt_qualifiers
            },
            "prompt_markers": {
                marker: {
                    "count": int(marker_counts[marker]),
                    "pct": pct(marker_counts[marker], total_captions),
                }
                for marker in prompt_markers
            },
        }

        for dataset, group in ds_groups.items():
            key = f"{mk}__{dataset}"
            results["per_dataset"][key] = _finalize_basic_group(
                group, {"model": safe_model_name(mk), "dataset": dataset}
            )

        for species, group in sp_groups.items():
            key = f"{mk}__{species}"
            results["per_species"][key] = _finalize_basic_group(
                group, {"model": safe_model_name(mk), "species": species}
            )

        for slide, group in slide_groups.items():
            dataset = resolve_dataset(slide, model_dataset[mk].get(slide), slide_dataset_truth)
            species = slide_species.get(slide, "Unknown")
            key = f"{mk}__{slide}"
            results["per_slide"][key] = _finalize_basic_group(
                group,
                {
                    "model": safe_model_name(mk),
                    "slide": slide,
                    "dataset": dataset,
                    "species": species,
                },
            )

        print(
            f"  {safe_model_name(mk)}: {total_captions:,} captions, "
            f"mean {results['per_model'][mk]['word_count']['mean']:.1f} words/caption"
        )

    return results
