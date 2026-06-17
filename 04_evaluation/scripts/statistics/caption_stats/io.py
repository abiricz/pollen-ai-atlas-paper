# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Data I/O: JSONL discovery, caption reading, anchor/vocabulary loading.

This module handles all file-system interactions for the caption
statistics pipeline.  Phase modules import from here to get data;
they never touch the filesystem directly.
"""

import json
from pathlib import Path

from .constants import ANCHOR_DIR, DATA_ROOT, DATASETS, EXCLUSION_YAML, MODELS
from .helpers import extract_pair_key, run_parallel_tasks, tokenize


# ── Slide exclusion loading ──────────────────────────────────────────

def load_excluded_slides():
    """Load excluded slide names from the slide_exclusions.yaml.

    Returns a set of slide names that should be excluded.
    """
    excluded = set()
    if not EXCLUSION_YAML.exists():
        return excluded
    try:
        import yaml
        with open(EXCLUSION_YAML) as f:
            data = yaml.safe_load(f) or {}
        for dataset in DATASETS:
            ds_info = data.get(dataset, {})
            for entry in ds_info.get("excluded", []) or []:
                if isinstance(entry, dict) and "slide" in entry:
                    excluded.add(entry["slide"])
                elif isinstance(entry, str):
                    excluded.add(entry)
    except ImportError:
        # Fallback: hardcoded exclusions if PyYAML not available
        excluded = {
            "ericaceae_edf",
            "juncaceae_edf",
            "mediterranean_pollen_palmaceae_reference",
            "mediterranean_pollen_poaceae_reference",
            "betula_sp_10_betulaceae_14_layers_40x_blue_colou_zs017_5_mm_circle",
        }
    except Exception as exc:
        print(f"  WARNING: failed to load slide exclusions: {exc}")
    return excluded


# ── Metadata loading ─────────────────────────────────────────────────

def load_anchor_texts():
    """Load per-slide anchor text, species, family, and hint.

    Returns four dicts keyed by slide name:
      slide_anchor, slide_species, slide_family, slide_hint
    """
    slide_anchor = {}
    slide_species = {}
    slide_family = {}
    slide_hint = {}

    for anchor_file in sorted(ANCHOR_DIR.glob("*_anchor.txt")):
        slide = anchor_file.stem.replace("_anchor", "")
        slide_anchor[slide] = anchor_file.read_text().strip()

        species_file = ANCHOR_DIR / f"{slide}_species.txt"
        family_file = ANCHOR_DIR / f"{slide}_family.txt"
        hint_file = ANCHOR_DIR / f"{slide}_hint.txt"

        slide_species[slide] = (
            species_file.read_text().strip() if species_file.exists() else "Unknown"
        )
        slide_family[slide] = (
            family_file.read_text().strip() if family_file.exists() else "Unknown"
        )
        slide_hint[slide] = (
            hint_file.read_text().strip() if hint_file.exists() else ""
        )

    return slide_anchor, slide_species, slide_family, slide_hint


def load_cross_dataset_slide_map(path):
    """Load slide → dataset truth map from ``cross_dataset_matrix.json``.

    Returns an empty dict if the file does not exist or cannot be parsed.
    """
    mapping = {}
    p = Path(path)
    if not p.exists():
        return mapping

    try:
        with open(p) as f:
            payload = json.load(f)
        for dataset, ds_info in payload.get("datasets", {}).items():
            for slide in ds_info.get("slides", []):
                mapping[slide] = dataset
    except Exception as exc:
        print(f"  WARNING: failed to parse cross-dataset matrix {p}: {exc}")

    return mapping


def build_slide_term_sets(vocab, slide_anchor, slide_hint):
    """Build per-slide morphological term sets from vocabulary JSON.

    If the vocabulary JSON contains ``per_slide_terms``, those are used directly.
    Otherwise, terms are extracted from anchor + hint texts.
    """
    vocab_terms = set(vocab.get("morphological_terms", []))
    per_slide_terms_raw = vocab.get("per_slide_terms", {})

    if per_slide_terms_raw:
        out = {
            slide: set(terms) & vocab_terms
            for slide, terms in per_slide_terms_raw.items()
        }
        for slide in slide_anchor:
            out.setdefault(slide, set())
        return out

    out = {}
    for slide, anchor in slide_anchor.items():
        hint = slide_hint.get(slide, "")
        toks = set(tokenize(f"{anchor} {hint}"))
        out[slide] = toks & vocab_terms
    return out


# ── JSONL discovery ──────────────────────────────────────────────────

def discover_jsonl_files():
    """Discover caption JSONL files grouped by model and dataset.

    Excludes slides listed in ``slide_exclusions.yaml``.
    Returns ``{model_key: [(path_str, dataset, slide), ...]}``.
    """
    excluded_slides = load_excluded_slides()
    if excluded_slides:
        print(f"  Excluding {len(excluded_slides)} slides: {', '.join(sorted(excluded_slides))}")

    result = {mk: [] for mk in MODELS}
    for dataset in DATASETS:
        dataset_dir = DATA_ROOT / dataset
        if not dataset_dir.is_dir():
            continue
        for model_key in MODELS:
            model_dir = dataset_dir / model_key
            if not model_dir.is_dir():
                continue
            for jsonl_file in sorted(model_dir.glob("*_captions.jsonl")):
                slide = jsonl_file.stem.replace("_captions", "")
                if slide in excluded_slides:
                    continue
                result[model_key].append((str(jsonl_file), dataset, slide))
    return result


# ── JSONL reading ────────────────────────────────────────────────────

def _read_jsonl_worker(args):
    """Read one JSONL file (process-pool worker).

    Returns ``(model_key, dataset, slide, [(caption, pair_key), ...])``.
    """
    path_str, model_key, dataset, slide, max_captions = args
    entries = []

    with open(path_str) as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            caption = str(obj.get("caption", "")).strip()
            if not caption:
                continue

            pair_key = extract_pair_key(obj, row_idx)
            entries.append((caption, pair_key))

            if max_captions > 0 and len(entries) >= max_captions:
                break

    return model_key, dataset, slide, entries


def phase_read_captions(model_files, workers, max_captions_per_slide=0):
    """Phase 1: Read captions from all discovered JSONL files.

    Returns ``(model_data, model_dataset)`` where:
      - ``model_data[mk][slide] = [(caption, pair_key), ...]``
      - ``model_dataset[mk][slide] = dataset``
    """
    model_data = {mk: {} for mk in MODELS}
    model_dataset = {mk: {} for mk in MODELS}

    tasks = []
    for mk, files in model_files.items():
        for path, ds, slide in files:
            tasks.append((path, mk, ds, slide, max_captions_per_slide))

    print("\n" + "=" * 60)
    print(f"Phase 1: Reading {len(tasks)} JSONL files with {workers} workers")
    if max_captions_per_slide > 0:
        print(f"  NOTE: limiting to {max_captions_per_slide} captions per slide")
    print("=" * 60)

    for mk, ds, slide, entries in run_parallel_tasks(
        tasks, _read_jsonl_worker, workers, "Reading"
    ):
        model_data[mk][slide] = entries
        model_dataset[mk][slide] = ds

    from .helpers import safe_model_name

    for mk in MODELS:
        total = sum(len(entries) for entries in model_data[mk].values())
        print(f"  {safe_model_name(mk)}: {total:,} captions from {len(model_data[mk])} slides")

    return model_data, model_dataset
