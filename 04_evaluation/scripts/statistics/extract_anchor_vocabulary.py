#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Pollen AI Atlas — Extract Anchor Vocabulary
=============================================

Reads ALL caption anchor and hint files from 03_captioning/caption_anchors/
and extracts the full vocabulary with zero hardcoded terms.

Every unique content-word that appears in the expert-curated PalDat-derived
anchor descriptions and one-liner hints is extracted. These are inherently
palynological/morphological terms because the anchor texts are purely
scientific pollen grain descriptions.

Outputs:
  data/04_evaluation/results/caption_statistics/anchor_vocabulary.json

Fields:
  morphological_terms     — sorted list of all content-words from anchors
  morphological_term_freq — {term: count} frequency in the corpus
  multi_word_phrases      — hyphenated and recognized multi-word terms
  prompt_qualifiers       — qualifier phrases injected by the VLM prompt
  all_raw_tokens          — every unique token (including stop-words)
  per_slide_metadata      — {slide: {species, family, anchor/hint lengths}}
  corpus_stats            — aggregate statistics

Usage:
    python extract_anchor_vocabulary.py

"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
ANCHOR_DIR = REPO_ROOT / "03_captioning" / "caption_anchors"
OUTPUT_DIR = REPO_ROOT / "data" / "04_evaluation" / "results" / "caption_statistics"

# Minimal English stop-words to separate content from function words.
# Intentionally conservative — anything scientific stays.
STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "each", "either", "else", "even",
    "for", "from", "get", "got", "had", "has", "have", "how", "if", "in",
    "into", "is", "it", "its", "just", "may", "might", "more", "most",
    "much", "my", "no", "nor", "not", "of", "on", "one", "only", "or",
    "other", "our", "out", "per", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "though", "to", "too", "under", "until", "up",
    "upon", "very", "was", "we", "were", "what", "when", "where", "whether",
    "which", "while", "who", "whom", "whose", "why", "will", "with",
    "without", "would", "yet", "you", "your", "about", "also", "any",
})

# Qualifier phrases injected by the VLM prompt (build_prompt_morphology).
# These appear in generated captions but are NOT in the anchor texts.
PROMPT_QUALIFIERS = [
    "not resolved",
    "partially resolved",
    "consistent with",
    "suggestive of",
]

# Additional prompt-governed markers used for compliance and behavior checks.
PROMPT_MARKERS = [
    "debris, dust or artifact detected; no matching pollen grain present",
    "hint or exemplar inconsistent",
]

TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract vocabulary from anchor/hint files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for anchor_vocabulary.json (default: data/04_evaluation/results/caption_statistics)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all anchors and hints ───────────────────────────────────────
    anchor_files = sorted(ANCHOR_DIR.glob("*_anchor.txt"))
    if not anchor_files:
        print(f"ERROR: No anchor files found in {ANCHOR_DIR}", file=sys.stderr)
        return 1

    per_slide = {}
    per_slide_terms = {}
    species_to_terms = defaultdict(set)
    family_to_terms = defaultdict(set)
    all_anchor_text = []
    all_hint_text = []

    for af in anchor_files:
        slide = af.stem.replace("_anchor", "")
        anchor = af.read_text().strip()

        species_f = ANCHOR_DIR / f"{slide}_species.txt"
        family_f = ANCHOR_DIR / f"{slide}_family.txt"
        hint_f = ANCHOR_DIR / f"{slide}_hint.txt"

        species = species_f.read_text().strip() if species_f.exists() else "Unknown"
        family = family_f.read_text().strip() if family_f.exists() else "Unknown"
        hint = hint_f.read_text().strip() if hint_f.exists() else ""

        slide_text = f"{anchor.lower()} {hint.lower()}".strip()
        slide_terms = sorted(set(
            t for t in TOKEN_RE.findall(slide_text)
            if t not in STOP and len(t) > 2
        ))
        per_slide_terms[slide] = slide_terms
        species_to_terms[species].update(slide_terms)
        family_to_terms[family].update(slide_terms)

        per_slide[slide] = {
            "species": species,
            "family": family,
            "anchor_length_words": len(anchor.split()),
            "hint_length_words": len(hint.split()),
            "n_unique_terms": len(slide_terms),
        }
        all_anchor_text.append(anchor.lower())
        all_hint_text.append(hint.lower())

    combined_text = " ".join(all_anchor_text + all_hint_text)

    # ── Extract ALL tokens ───────────────────────────────────────────────
    # Captures hyphenated compounds like oblate-spheroidal, verrucate-scabrate
    all_tokens = TOKEN_RE.findall(combined_text)
    raw_counts = Counter(all_tokens)

    # ── Separate content-words from stop-words ───────────────────────────
    # Content-words = everything NOT in the stop-word list and length > 2
    content_tokens = sorted(set(
        t for t in all_tokens if t not in STOP and len(t) > 2
    ))
    content_counts = {t: raw_counts[t] for t in content_tokens}

    # ── Hyphenated compound terms (data-driven) ──────────────────────────
    hyphenated = sorted(set(re.findall(r"[a-z]+-[a-z]+(?:-[a-z]+)*", combined_text)))

    # ── Multi-word phrases (data-driven: scan for adjacent pairs) ────────
    # Instead of pre-defining, find frequently co-occurring bigrams
    words_list = combined_text.split()
    bigrams = Counter()
    for i in range(len(words_list) - 1):
        w1 = re.sub(r"[^a-z-]", "", words_list[i])
        w2 = re.sub(r"[^a-z-]", "", words_list[i + 1])
        if w1 and w2 and w1 not in STOP and w2 not in STOP:
            bigrams[f"{w1} {w2}"] += 1

    # Keep bigrams occurring 3+ times (truly characteristic phrases)
    frequent_bigrams = sorted(
        [bg for bg, cnt in bigrams.items() if cnt >= 3],
    )

    # Count in how many slides each term appears.
    term_document_frequency = Counter()
    for terms in per_slide_terms.values():
        term_document_frequency.update(set(terms))

    # ── Build output ─────────────────────────────────────────────────────
    output = {
        "_description": (
            "Vocabulary extracted from caption anchor and hint texts. "
            "All terms are data-driven — no hardcoded morphological vocabulary. "
            "Used by compute_caption_stats.py for morphological coverage analysis."
        ),
        "source_dir": str(ANCHOR_DIR),
        "tokenization_regex": TOKEN_RE.pattern,
        "n_slides": len(per_slide),
        "n_anchor_files": len(anchor_files),
        "corpus_stats": {
            "total_unique_content_words": len(content_tokens),
            "total_unique_raw_tokens": len(raw_counts),
            "total_token_occurrences": sum(raw_counts.values()),
            "mean_unique_terms_per_slide": round(
                sum(len(v) for v in per_slide_terms.values()) / max(len(per_slide_terms), 1), 1
            ),
            "anchor_mean_length_words": round(
                sum(s["anchor_length_words"] for s in per_slide.values()) / len(per_slide), 1
            ),
            "hint_mean_length_words": round(
                sum(s["hint_length_words"] for s in per_slide.values()) / len(per_slide), 1
            ),
        },
        "morphological_terms": content_tokens,
        "morphological_terms_count": len(content_tokens),
        "morphological_term_freq": dict(
            sorted(content_counts.items(), key=lambda x: -x[1])
        ),
        "hyphenated_compounds": hyphenated,
        "frequent_bigrams": frequent_bigrams,
        "prompt_qualifiers": PROMPT_QUALIFIERS,
        "prompt_markers": PROMPT_MARKERS,
        "species_names": sorted({v["species"] for v in per_slide.values()}),
        "family_names": sorted({v["family"] for v in per_slide.values()}),
        "per_slide_terms": per_slide_terms,
        "term_document_frequency": dict(
            sorted(term_document_frequency.items(), key=lambda x: (-x[1], x[0]))
        ),
        "species_to_terms": {
            sp: sorted(terms) for sp, terms in sorted(species_to_terms.items())
        },
        "family_to_terms": {
            fam: sorted(terms) for fam, terms in sorted(family_to_terms.items())
        },
        "all_raw_tokens": sorted(raw_counts.keys()),
        "per_slide_metadata": per_slide,
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = output_dir / "anchor_vocabulary.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"Anchor vocabulary extracted from {len(per_slide)} slides")
    print(f"  Unique content-words:     {len(content_tokens)}")
    print(f"  Hyphenated compounds:     {len(hyphenated)}")
    print(f"  Frequent bigrams (3+):    {len(frequent_bigrams)}")
    print(f"  Prompt qualifiers:        {len(PROMPT_QUALIFIERS)}")
    print(f"  Mean anchor length:       {output['corpus_stats']['anchor_mean_length_words']} words")
    print(f"  Mean hint length:         {output['corpus_stats']['hint_mean_length_words']} words")
    print(f"  Output: {out_path}")

    # Top 20 terms
    print(f"\n  Top 20 morphological terms:")
    for term, count in list(output["morphological_term_freq"].items())[:20]:
        print(f"    {count:4d}  {term}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
