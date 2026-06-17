#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Cross-Regional Multimodal Retrieval Experiment
=================================================

Purpose
-------
Demonstrate the dataset as a reusable retrieval resource beyond classification.
Given a single exemplar image (one-shot crop) and/or an expert morphological
descriptor, retrieve the most morphologically consistent pollen grains from the
full ~1.48 M corpus — including across geographic origins and scanner setups.

This evaluation has NO learned parameters, making it complementary to the
classification experiments.  It directly tests whether (a) the ViT image
embeddings and (b) the VLM-generated caption embeddings faithfully encode
species-level morphology.

Scientific Design
-----------------
Queries — two independent, non-circular signals per slide:

  IMAGE   The one-shot exemplar crop from 01_initialization/query_images/.
          This is an independently selected reference grain — it was hand-picked
          by the expert for OWL-ViT initialization, not explicitly part of the mined
          corpus, but can be. Using it as a query mimics the real use case: "I have a
          reference grain from a palynological atlas; find the same taxon in
          my unsorted digital collection."
          Note: these images were shown to the VLM during captioning and used
          to seed the mining pipeline. This creates a soft pipeline bias
          (disclosed), not direct data leakage.

  TEXT    Expert-derived morphological descriptors defined in
          retrieval_config.yaml. IMPORTANT: these are NOT the raw caption
          anchors from 03_captioning/caption_anchors/. The anchors were fed
          into the VLM captioning prompt, so using them directly as queries
          creates prompt-coupled circularity (inflated text metrics).
          Instead, we derive independent descriptions from the same
          palynological knowledge base: same diagnostic features, but
          different phrasing and no numeric measurements.
          One text query per species (N_species distinct embeddings).

  COMBINED  alpha * sim_image + (1 - alpha) * sim_text  (late fusion).
          ViT and SBERT occupy separate, independently trained embedding
          spaces; score-level fusion is more principled than feature
          concatenation for heterogeneous embeddings.

Corpus:
  ~1.48 M pre-computed embeddings (per-slide H5 files):
    - Image:   data/04_evaluation/vit_embeddings/{slide}_embeddings.h5
    - Caption: data/04_evaluation/caption_embeddings/{vlm}/{slide}_embeddings.h5

Focus:
  Only cross-regional species (present in >=2 geographic origins) are tested.
  This is the scientifically interesting case — can retrieval bridge domain
  shift across different scanners, staining protocols, and regions?

Exclusion modes (three tiers of exclusion stringency):
  ALL             No exclusion — the full ~1.5M corpus queried as-is.
                  Text: reduced circularity (independent query, not
                  raw anchor); pipeline-coupled → ceiling condition.
                  Image: mildly inflated (same-slide grains are similar
                  by construction), reported with this caveat disclosed.
                  -> ceiling estimate for perfect knowledge scenario
  FULL            Exclude query slide AND its scanner-step siblings
                  (betula_edf <-> betula_2_edf: same prep, different passes).
                  -> "find same-species grains on other slides"
  CROSS_REGIONAL  Exclude ALL slides from the query's geographic origin.
                  Each query slide produces an independent leave-origin-out
                  evaluation, so 3-origin species yield 3 queries each.
                  -> "bridge French <-> Hungarian <-> Mediterranean <-> Swedish"

Relevance:
  Binary species label from *_species.txt (authoritative HITL annotation).

Aggregation:
  Per-slide -> per-species (average) -> global (macro-average across species).
  Every species contributes equally regardless of sample count.

Metrics:
  P@K, R@K, nDCG@K, MRR, mAP@max(topk)  (K in {1, 5, 10, 20}).

Usage:
  python retrieval_experiments.py                              # full run (all modes)
  python retrieval_experiments.py --vlm qwen25vl               # one VLM
  python retrieval_experiments.py --modality image              # image only
  python retrieval_experiments.py --mode all                    # ceiling only
  python retrieval_experiments.py --mode cross_regional         # cross only
  python retrieval_experiments.py --alpha 0.5                   # one alpha
  python retrieval_experiments.py --dry_run                     # list queries
  python retrieval_experiments.py --negative_control             # label-shuffle control
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml


# =====================================================================
# PATHS
# =====================================================================

ROOT           = Path(__file__).resolve().parents[3]          # repo root
DATA           = ROOT / "data"
VIT_DIR        = DATA / "04_evaluation" / "vit_embeddings"
CAPTION_DIR    = DATA / "04_evaluation" / "caption_embeddings"
QUERY_IMG_DIR  = ROOT / "01_initialization" / "query_images"
ANCHOR_DIR     = ROOT / "03_captioning" / "caption_anchors"
DEFAULT_OUTPUT_DIR = DATA / "04_evaluation" / "results" / "retrieval"
OUTPUT_DIR     = Path(
    os.environ.get("RETRIEVAL_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
)
CONFIG_PATH    = Path(__file__).parent / "retrieval_config.yaml"

# ViT model name — must match extract_vit_embeddings.py exactly.
# pretrained=True (plain LVD-142M from timm) — the finetuned
# 01_initialization checkpoint is NOT used here (matches Option A,
# backbone_checkpoint: null).  Changing this makes query and corpus
# embeddings incommensurable.
VIT_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"

# SBERT model (same model used for corpus caption embedding extraction)
SBERT_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"


# =====================================================================
# METADATA LOOKUP  (authoritative, from cross_dataset_matrix.json)
# =====================================================================

def load_metadata():
    """Build slide -> origin and slide -> species mappings.

    Origin: from cross_dataset_matrix.json (authoritative).
    Species: from *_species.txt in caption_anchors/ (authoritative HITL).

    Returns (slide_to_origin, slide_to_species) dicts.
    """
    # --- Origin from cross_dataset_matrix.json ---
    cdm_path = ROOT / "04_evaluation" / "results" / "cross_dataset_matrix.json"
    slide_to_origin = {}
    if cdm_path.exists():
        with open(cdm_path) as f:
            cdm = json.load(f)
        for dataset, info in cdm["datasets"].items():
            origin = dataset.capitalize()
            for slide in info["slides"]:
                slide_to_origin[slide] = origin
    else:
        print(f"  [WARNING] {cdm_path} not found, falling back to name heuristics")

    # --- Species from *_species.txt ---
    slide_to_species = {}
    for sp_file in sorted(ANCHOR_DIR.glob("*_species.txt")):
        slide = sp_file.stem.replace("_species", "")
        species = sp_file.read_text().strip()
        if species:
            slide_to_species[slide] = species

    return slide_to_origin, slide_to_species


def get_origin_fallback(slide):
    """Fallback origin heuristic when cross_dataset_matrix.json is absent."""
    if slide.startswith("mediterranean"):
        return "Mediterranean"
    if slide.startswith("hun_") or slide.startswith("Ambrosia-Iva_reference"):
        return "Hungarian"
    if slide.endswith("_edf"):
        return "French"
    return "Swedish"


# =====================================================================
# CORPUS LOADING  (pre-computed per-slide H5 files)
# =====================================================================

def load_corpus(emb_dir, slide_to_origin, slide_to_species):
    """Load all per-slide H5 embeddings into one aligned corpus.

    Returns dict with numpy arrays:
        embeddings  (N, 384) float32, L2-normed
        sample_ids  (N,)     str
        slides      (N,)     str
        species     (N,)     str
        origins     (N,)     str
    """
    all_embs, all_ids = [], []
    all_slides, all_species, all_origins = [], [], []
    n_zero_norm_total = 0

    for h5 in sorted(Path(emb_dir).glob("*.h5")):
        slide = h5.stem.replace("_embeddings", "")
        species = slide_to_species.get(slide, "Unknown")
        if species == "Unknown":
            continue
        origin = slide_to_origin.get(slide, get_origin_fallback(slide))

        with h5py.File(h5, "r") as f:
            embs = f["embeddings"][:]
            ids = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in f["sample_ids"][:]
            ]

        # Filter out zero-norm vectors (failed extraction placeholders).
        # These are produced when crop extraction fails during ViT embedding;
        # keeping them would insert meaningless data points into the corpus
        # (Finding 2).  Currently total_failed=0 in extraction_metadata.json,
        # but this guard makes the pipeline resilient to any future re-run.
        norms = np.linalg.norm(embs, axis=1)
        nonzero_mask = norms > 1e-6
        n_zero = int((~nonzero_mask).sum())
        if n_zero > 0:
            n_zero_norm_total += n_zero
            print(f"  [WARNING] {slide}: dropping {n_zero} zero-norm embeddings")
            embs = embs[nonzero_mask]
            ids = [sid for sid, keep in zip(ids, nonzero_mask) if keep]

        n = embs.shape[0]
        all_embs.append(embs)
        all_ids.extend(ids)
        all_slides.extend([slide] * n)
        all_species.extend([species] * n)
        all_origins.extend([origin] * n)

    if n_zero_norm_total > 0:
        print(f"  [WARNING] Dropped {n_zero_norm_total:,} zero-norm embeddings "
              f"across all slides")

    return {
        "embeddings": np.vstack(all_embs).astype(np.float32),
        "sample_ids": np.array(all_ids),
        "slides":     np.array(all_slides),
        "species":    np.array(all_species),
        "origins":    np.array(all_origins),
    }


def align_corpora(img_corpus, txt_corpus):
    """Reorder text corpus to match image corpus sample order.

    Also performs data integrity checks:
      - Sample counts must match
      - Sample IDs must be globally unique (no silent overwrites)
      - ID sets must be identical
    """
    n_img = img_corpus["embeddings"].shape[0]
    n_txt = txt_corpus["embeddings"].shape[0]

    # --- Integrity: ID uniqueness ---
    img_unique = len(set(img_corpus["sample_ids"]))
    txt_unique = len(set(txt_corpus["sample_ids"]))
    assert img_unique == n_img, (
        f"Duplicate sample_ids in image corpus: {n_img} rows, "
        f"{img_unique} unique"
    )
    assert txt_unique == n_txt, (
        f"Duplicate sample_ids in text corpus: {n_txt} rows, "
        f"{txt_unique} unique"
    )

    # --- Integrity: set equality ---
    assert n_img == n_txt, (
        f"Cannot align: image {n_img:,} != text {n_txt:,}"
    )
    img_set = set(img_corpus["sample_ids"])
    txt_set = set(txt_corpus["sample_ids"])
    assert img_set == txt_set, (
        f"ID sets differ: {len(img_set - txt_set)} in image only, "
        f"{len(txt_set - img_set)} in text only"
    )

    # Fast path: already aligned
    if np.array_equal(
        img_corpus["sample_ids"], txt_corpus["sample_ids"]
    ):
        print(f"  Already aligned ({n_img:,} samples)")
        return txt_corpus

    txt_id_to_idx = {
        sid: i for i, sid in enumerate(txt_corpus["sample_ids"])
    }
    reorder = np.array(
        [txt_id_to_idx[sid] for sid in img_corpus["sample_ids"]]
    )

    aligned = {k: txt_corpus[k][reorder] for k in txt_corpus}
    assert np.array_equal(img_corpus["sample_ids"], aligned["sample_ids"])
    print(f"  Aligned ({n_img:,} samples)")
    return aligned


# =====================================================================
# QUERY EMBEDDING -- ONE-SHOT IMAGE  (ViT-Small-LVD)
# =====================================================================

def embed_query_images(slides, device="cuda:0", model_name=VIT_MODEL_NAME,
                       checkpoint_path=None):
    """Embed one-shot query crops with the same ViT backbone.

    Each slide is expected to have a query image at:
        01_initialization/query_images/{slide}.png

    IMPORTANT: Model construction and checkpoint loading must be
    byte-for-byte identical to extract_vit_embeddings.py to ensure
    query and corpus embeddings live in the same feature space.

    Returns: {slide: np.array(384,)} L2-normalised.
    """
    import timm
    from torchvision import transforms
    from PIL import Image

    # Build model — MUST match extract_vit_embeddings.py exactly:
    #   model_name (from config) + pretrained=True for plain LVD-142M weights
    #   init_values=1e-5 for LayerScale, num_classes=0 for feature mode
    model = timm.create_model(
        model_name,
        pretrained=(checkpoint_path is None),
        img_size=518,
        init_values=1e-5,
        num_classes=0,
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  [ViT query] Loading checkpoint: {os.path.basename(checkpoint_path)}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state = {
            k: v for k, v in state.items()
            if not k.startswith("head.") and not k.startswith("classifier.")
        }
        model.load_state_dict(state, strict=False)

    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    query_embs = {}
    for slide in slides:
        img_path = QUERY_IMG_DIR / f"{slide}.png"
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        ctx = torch.amp.autocast("cuda") if device.startswith("cuda") else nullcontext()
        with torch.no_grad(), ctx:
            feat = model(tensor)              # (1, 384)
        feat = feat.float().cpu().numpy().squeeze()
        feat /= np.linalg.norm(feat) + 1e-8
        query_embs[slide] = feat

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return query_embs


# =====================================================================
# QUERY EMBEDDING -- EXPERT-DERIVED TEXT  (SBERT)
# =====================================================================

def embed_text_queries(species_queries, slide_to_species, device="cuda:0", model_name=SBERT_MODEL):
    """Embed expert-derived text queries with the same SBERT model.

    Text queries are defined per-species in the config (NOT the raw
    caption anchors, which are prompt-coupled to VLM output).

    Each species has one text query. Each slide of that species gets
    the same embedding.  This is transparent: text retrieval operates
    at species-level granularity (N_species distinct query vectors),
    not slide-level. The per-slide metric aggregation averages over
    different exclusion masks, not different query signals.

    Returns: {slide: np.array(384,)} L2-normalised.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    # Collect unique species -> text_query
    species_list, texts = [], []
    for species, qdef in sorted(species_queries.items()):
        text = qdef.get("text_query", "").strip()
        if text:
            species_list.append(species)
            texts.append(text)

    if not texts:
        return {}

    embs = model.encode(texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True)

    # Build species -> embedding, then expand to slide -> embedding
    species_embs = {
        sp: embs[i].astype(np.float32)
        for i, sp in enumerate(species_list)
    }

    # Map every slide of each species to the species embedding
    slide_embs = {}
    for species, qdef in species_queries.items():
        if species not in species_embs:
            continue
        for entry in qdef.get("exemplar_slides", []):
            slide = entry["slide"]
            slide_embs[slide] = species_embs[species]

    # Also cover slides that are in the corpus but not listed as exemplars
    # (they still need text queries for "full" mode)
    for slide, sp in slide_to_species.items():
        if sp in species_embs and slide not in slide_embs:
            slide_embs[slide] = species_embs[sp]

    print(f"  {len(species_embs)} species text queries -> "
          f"{len(slide_embs)} slide mappings")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return slide_embs


# =====================================================================
# QUERY CONSTRUCTION  (cross-regional species from config)
# =====================================================================

def build_queries(corpus, query_img_embs, text_embs, species_queries,
                  excluded_slides=None):
    """Build one query per slide that has at least one query signal.

    Only considers slides explicitly listed in species_queries.exemplar_slides
    (cross-regional species).  Image queries are restricted to slides declared
    in the config — a slide with a query_images/ file but NOT listed in
    exemplar_slides is silently ignored.  This ensures the config is the single
    authoritative source of query definitions (Finding 3).

    Each query dict contains:
        slide, species, origin, n_grains,
        img_emb (384-d or None), txt_emb (384-d or None)
    """
    excluded = set(excluded_slides or [])

    # Collect per-slide info from corpus
    slide_info = {}
    for i in range(len(corpus["slides"])):
        slide = corpus["slides"][i]
        if slide not in slide_info:
            slide_info[slide] = {
                "species": corpus["species"][i],
                "origin":  corpus["origins"][i],
                "count":   0,
            }
        slide_info[slide]["count"] += 1

    # Build the set of slides explicitly declared in config exemplar_slides.
    # Only these are eligible to generate queries — the config is the single
    # source of truth, not the filesystem (fixes Finding 3).
    config_exemplar_slides = set()
    for species, qdef in species_queries.items():
        for entry in qdef.get("exemplar_slides", []):
            config_exemplar_slides.add(entry["slide"])

    queries = []
    for slide in sorted(config_exemplar_slides):
        if slide in excluded:
            continue
        info = slide_info.get(slide)
        if info is None:
            # Slide listed in config but not present in corpus (e.g. missing H5)
            print(f"  [WARNING] Config slide '{slide}' not found in corpus "
                  f"— skipping (missing H5 file?)")
            continue
        img_emb = query_img_embs.get(slide)
        txt_emb = text_embs.get(slide)
        if img_emb is None and txt_emb is None:
            continue
        queries.append({
            "slide":    slide,
            "species":  info["species"],
            "origin":   info["origin"],
            "n_grains": info["count"],
            "img_emb":  img_emb,
            "txt_emb":  txt_emb,
        })
    return queries


# =====================================================================
# METRICS  (corrected nDCG with proper IDCG)
# =====================================================================

def ndcg_at_k(relevant_sorted, total_relevant, k):
    """Normalized discounted cumulative gain at k.

    IDCG is computed from total_relevant (not from the observed top-K),
    which is the standard definition and prevents inflation/deflation
    artifacts that occur when IDCG is derived only from observed hits.
    """
    rel = relevant_sorted[:k].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((rel * discounts[:len(rel)]).sum())

    ideal_hits = min(total_relevant, k)
    if ideal_hits == 0:
        return 0.0
    idcg = float(discounts[:ideal_hits].sum())
    return dcg / idcg


def compute_metrics(relevant, total_relevant, topk):
    """Compute P@K, R@K, nDCG@K, MRR, mAP from a binary relevance vector.

    Args:
        relevant:       boolean array, True if item at that rank is relevant.
        total_relevant: total number of relevant items in the (masked) corpus.
        topk:           list of K values [1, 5, 10, 20, ...].
    """
    total_relevant = max(total_relevant, 1)
    max_k = max(topk)

    out = {"total_relevant": total_relevant}
    for k in topk:
        ka = min(k, len(relevant))
        hits = float(relevant[:ka].sum())
        out[f"P@{k}"]    = hits / ka if ka > 0 else 0.0
        out[f"R@{k}"]    = hits / total_relevant
        out[f"nDCG@{k}"] = ndcg_at_k(relevant, total_relevant, k)

    # MRR: reciprocal rank of first relevant item
    first_hits = np.where(relevant)[0]
    out["MRR"] = 1.0 / (first_hits[0] + 1) if len(first_hits) > 0 else 0.0

    # mAP@K: mean average precision truncated at max(topk)
    map_key = f"mAP@{max_k}"
    n = min(max_k, len(relevant))
    cumsum    = np.cumsum(relevant[:n])
    positions = np.arange(1, n + 1)
    prec_at_j = cumsum / positions
    out[map_key] = float(
        (prec_at_j * relevant[:n]).sum()
    ) / min(total_relevant, n)

    return out


# =====================================================================
# RETRIEVAL
# =====================================================================

def _score_and_rank(scores, q_species, corpus_species, exclude, topk):
    """Score -> rank -> metrics.  Shared by all modalities."""
    scores = scores.copy()
    scores[exclude] = -np.inf

    sorted_idx     = np.argsort(-scores)
    sorted_species = corpus_species[sorted_idx]
    sorted_scores  = scores[sorted_idx]
    relevant       = (sorted_species == q_species)

    total_relevant = int(((corpus_species == q_species) & ~exclude).sum())

    result = compute_metrics(relevant, total_relevant, topk)
    result["top10_species"] = [str(s) for s in sorted_species[:10]]
    result["top10_scores"]  = [round(float(s), 5) for s in sorted_scores[:10]]
    return result


def retrieve_image(q_img, q_species, corpus_img, corpus_species,
                   exclude, topk):
    """Image-only retrieval: one-shot crop vs corpus ViT embeddings."""
    scores = (q_img @ corpus_img.T).squeeze()
    return _score_and_rank(scores, q_species, corpus_species, exclude, topk)


def retrieve_text(q_txt, q_species, corpus_txt, corpus_species,
                  exclude, topk):
    """Text-only retrieval: expert text query vs corpus caption embeddings."""
    scores = (q_txt @ corpus_txt.T).squeeze()
    return _score_and_rank(scores, q_species, corpus_species, exclude, topk)


def retrieve_combined(q_img, q_txt, q_species, corpus_img, corpus_txt,
                      corpus_species, alpha, exclude, topk):
    """Late-fusion: alpha * sim_image + (1 - alpha) * sim_text."""
    sim_img = (q_img @ corpus_img.T).squeeze()
    sim_txt = (q_txt @ corpus_txt.T).squeeze()
    scores  = alpha * sim_img + (1.0 - alpha) * sim_txt
    return _score_and_rank(scores, q_species, corpus_species, exclude, topk)


# =====================================================================
# SLIDE SIBLING DETECTION
# =====================================================================

def build_slide_groups(all_slides):
    """Group slides that are multi-step scanner variants of the same prep.

    Detects slides that share a base name and differ only by a numeric
    suffix inserted before the final '_edf' token, e.g.:
        betula_edf   and  betula_2_edf
        corylus_edf  and  corylus_2_edf
        pinus_edf    and  pinus_2_edf

    Returns: dict {slide: [sibling_slide, ...]} — empty list if no siblings.
    """
    import re

    # Compute canonical key for each slide: strip trailing _N before _edf
    # betula_2_edf -> betula_edf  |  betula_edf -> betula_edf (unchanged)
    # Non-_edf slides: betula_sp_01_40x_... -> no stripping
    def canonical(s):
        m = re.match(r'^(.*?)_(\d+)(_edf)$', s)
        return m.group(1) + m.group(3) if m else s

    groups = {}     # canonical_key -> set of slide names
    for slide in all_slides:
        key = canonical(slide)
        groups.setdefault(key, set()).add(slide)

    # Build slide -> siblings (all group members except itself)
    slide_to_siblings = {}
    for group in groups.values():
        for s in group:
            siblings = sorted(group - {s})
            slide_to_siblings[s] = siblings  # [] if lone member

    return slide_to_siblings


def merge_extra_siblings(slide_siblings, extra_groups):
    """Merge manually declared sibling groups into auto-detected ones."""
    for group_list in extra_groups:
        group = set(group_list)
        for s in group:
            existing = set(slide_siblings.get(s, []))
            slide_siblings[s] = sorted(existing | (group - {s}))
    return slide_siblings


# =====================================================================
# EXCLUSION MASKS
# =====================================================================

def build_exclude(corpus, slide, origin, mode, slide_siblings=None):
    """Boolean mask: True => exclude from retrieval.

    all:            No exclusion — the full corpus is queried as-is.
                    Establishes the performance ceiling.
                    Text: valid (no leakage).
                    Image: mildly inflated (same-slide grains are
                    morphologically similar by mining construction).

    full:           Exclude the query slide AND its scanner-step siblings
                    (e.g. betula_edf and betula_2_edf are different z-stack
                    passes of the same physical preparation).
                    Tests cross-slide retrieval within any origin.

    cross_regional: Exclude ALL slides from the query's geographic origin.
                    Tests whether retrieval bridges domain shift.
                    Each query slide acts as its own leave-origin-out query,
                    so a 3-origin species yields 3 independent queries each
                    retrieving from the remaining 2 origins.
    """
    if mode == "all":
        # No exclusion — include the full corpus, even the query slide.
        # Text: reduced circularity (independent query, not raw anchor),
        #        but same-slide captions are still pipeline-coupled —
        #        treat as a ceiling condition.
        # Image: mildly inflated (same-slide grains are structurally similar
        #        by mining construction), reported with this caveat disclosed.
        # Establishes the performance ceiling.
        mask = np.zeros(len(corpus["slides"]), dtype=bool)

    elif mode == "full":
        # Exclude query slide and its _N_edf scanner siblings
        mask = (corpus["slides"] == slide)
        siblings = (slide_siblings or {}).get(slide, [])
        for sib in siblings:
            mask = mask | (corpus["slides"] == sib)

    else:  # cross_regional
        mask = (corpus["origins"] == origin)

    return mask


# =====================================================================
# AGGREGATION  (per-slide -> per-species -> global macro-average)
# =====================================================================

# =====================================================================
# HIERARCHICAL BOOTSTRAP CIs
# =====================================================================

def bootstrap_ci(per_query, topk, n_boot=10000, ci=0.95, seed=42):
    """Origin-balanced hierarchical bootstrap 95% CIs.

    Three-level resampling mirrors the origin-balanced aggregation:
      1. Resample species WITH replacement (N = n_species)
      2. For each sampled species, resample origins WITH replacement
      3. For each sampled origin, resample queries WITH replacement
      4. Compute origin means -> species mean -> global macro-average
    Take percentile [2.5, 97.5] for 95% CI.

    This respects the origin-balanced macro-average aggregation,
    properly handles the hierarchical variance structure (species >
    origin > queries), and avoids overweighting origins with many
    slides. With ~15 species CIs are appropriately wide.

    For text queries (1 query per origin), levels 2-3 collapse to
    the equivalent of the simpler 2-level bootstrap.

    Returns: dict {metric_name: [ci_low, ci_high]}
    """
    map_key = f"mAP@{max(topk)}"
    metric_keys = (
        [f"P@{k}" for k in topk] +
        [f"R@{k}" for k in topk] +
        [f"nDCG@{k}" for k in topk] +
        ["MRR", map_key]
    )

    # Group by species -> origin -> queries
    by_sp_origin = {}
    for r in per_query:
        sp = r["query_species"]
        origin = r["query_origin"]
        by_sp_origin.setdefault(sp, {}).setdefault(origin, []).append(r)

    species_list = sorted(by_sp_origin.keys())
    n_species = len(species_list)
    if n_species == 0:
        return {}

    rng = np.random.RandomState(seed)
    boot_globals = {mk: [] for mk in metric_keys}

    for _ in range(n_boot):
        # Level 1: resample species with replacement
        sampled_species = rng.choice(species_list, size=n_species, replace=True)

        species_means = {mk: [] for mk in metric_keys}
        for sp in sampled_species:
            sp_origins = by_sp_origin[sp]
            origin_list = list(sp_origins.keys())
            n_origins = len(origin_list)

            # Level 2: resample origins within species
            sampled_origins = rng.choice(origin_list, size=n_origins,
                                         replace=True)

            origin_means = {mk: [] for mk in metric_keys}
            for origin in sampled_origins:
                oq = sp_origins[origin]
                # Level 3: resample queries within origin
                idx = rng.choice(len(oq), size=len(oq), replace=True)
                for mk in metric_keys:
                    vals = [oq[i][mk] for i in idx if mk in oq[i]]
                    if vals:
                        origin_means[mk].append(float(np.mean(vals)))

            for mk in metric_keys:
                if origin_means[mk]:
                    species_means[mk].append(float(np.mean(origin_means[mk])))

        # Global macro-average across sampled species
        for mk in metric_keys:
            if species_means[mk]:
                boot_globals[mk].append(float(np.mean(species_means[mk])))

    # Compute percentile CIs
    alpha_ci = (1 - ci) / 2
    cis = {}
    for mk in metric_keys:
        if boot_globals[mk]:
            lo = float(np.percentile(boot_globals[mk], 100 * alpha_ci))
            hi = float(np.percentile(boot_globals[mk], 100 * (1 - alpha_ci)))
            cis[mk] = [round(lo, 4), round(hi, 4)]
    return cis


def aggregate(per_query, topk, species_all_origins=None, n_boot=10000, no_ci=False):
    """Origin-balanced macro-average: origin -> species -> global.

    For each species, first average within each origin, then average
    across origins. This ensures each geographic origin contributes
    equally regardless of slide count. For text (one query per origin),
    this is equivalent to simple averaging.

    Args:
        per_query: list of per-query result dicts.
        topk: list of K values.
        species_all_origins: optional dict {species: [origin, ...]} for
            complete origin metadata (used when text dedup collapses
            queries to fewer than the full set of origins).
    """
    metric_keys = (
        [f"P@{k}"    for k in topk] +
        [f"R@{k}"    for k in topk] +
        [f"nDCG@{k}" for k in topk] +
        ["MRR", "mAP@20"]
    )

    # Group by species -> origin
    by_species = {}
    for r in per_query:
        sp = r["query_species"]
        origin = r["query_origin"]
        by_species.setdefault(sp, {}).setdefault(origin, []).append(r)

    per_species = {}
    for sp, by_origin in sorted(by_species.items()):
        m = {}
        for mk in metric_keys:
            # Average within each origin first
            origin_means = []
            for origin, origin_results in by_origin.items():
                vals = [r[mk] for r in origin_results if mk in r]
                if vals:
                    origin_means.append(float(np.mean(vals)))
            # Then average across origins
            m[mk] = float(np.mean(origin_means)) if origin_means else 0.0
        m["n_queries"] = sum(len(oq) for oq in by_origin.values())
        # Use species_all_origins for complete metadata when available
        if species_all_origins and sp in species_all_origins:
            m["origins"] = species_all_origins[sp]
        else:
            m["origins"] = sorted(by_origin.keys())
        m["n_origins"] = len(by_origin)
        m["queries_per_origin"] = {
            origin: len(oq) for origin, oq in sorted(by_origin.items())
        }
        per_species[sp] = m

    # Global macro-average across species
    global_metrics = {}
    for mk in metric_keys:
        global_metrics[mk] = float(
            np.mean([per_species[sp][mk] for sp in per_species])
        )
    global_metrics["n_species"]  = len(per_species)
    global_metrics["n_queries"]  = len(per_query)

    # Origin-balanced hierarchical bootstrap CIs
    if no_ci:
        global_ci_95 = {}
    else:
        global_ci_95 = bootstrap_ci(per_query, topk, n_boot=n_boot)

    return {
        "global": global_metrics,
        "global_ci_95": global_ci_95,
        "per_species": per_species,
    }


# =====================================================================
# DISPLAY
# =====================================================================

def print_header(mode, modality, n_queries, alpha=None, n_dedup=None):
    labels = {
        "all":           "ALL        (no exclusion — full corpus)",
        "full":          "FULL       (excl. query slide + siblings)",
        "cross_regional":"CROSS-REG  (excl. query origin)",
    }
    label = labels.get(mode, mode.upper())
    mod = modality.upper()
    if alpha is not None:
        mod += f" a={alpha:.2f}"
    q_str = f"{n_queries} queries"
    if n_dedup is not None and n_dedup != n_queries:
        q_str += f" ({n_dedup} deduplicated — text uses 1 per unique exclusion set)"
    print(f"\n{'_'*70}")
    print(f"  {label}  |  {mod}  |  {q_str}")
    print(f"{'_'*70}")


def print_results(agg, topk):
    g = agg["global"]
    ci = agg.get("global_ci_95", {})
    map_key = f"mAP@{max(topk)}"
    cols = [f"P@{k}" for k in topk] + ["MRR", map_key]
    hdr  = "  ".join(f"{c:>7s}" for c in cols)
    vals = "  ".join(f"{g[c]:>7.3f}" for c in cols)
    print(f"  Global ({g['n_species']} species, {g['n_queries']} queries):")
    print(f"    {hdr}")
    print(f"    {vals}")
    # Print CIs for primary metrics (P@20, MRR, mAP@20)
    if ci:
        ci_cols = ["P@20", "MRR", map_key]
        ci_strs = []
        for c in ci_cols:
            if c in ci:
                ci_strs.append(f"{c}: [{ci[c][0]:.3f}–{ci[c][1]:.3f}]")
        if ci_strs:
            print(f"    95% CI: {', '.join(ci_strs)}")
    print()
    print(f"  Per-species:")
    short_cols = [f"P@{topk[0]}", f"P@{topk[-1]}", "MRR", map_key]
    hdr2 = "  ".join(f"{c:>7s}" for c in short_cols + ["#Q", "origins"])
    print(f"    {'species':<22s} {hdr2}")
    for sp, m in sorted(agg["per_species"].items()):
        row = "  ".join(f"{m[c]:>7.3f}" for c in short_cols)
        ostr = ",".join(o[:3] for o in m["origins"])
        print(f"    {sp:<22s} {row}  {m['n_queries']:>3d}  {ostr}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Regional Multimodal Retrieval Experiment",
    )
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--vlm", type=str, nargs="+", default=None)
    parser.add_argument("--modality", type=str, nargs="+", default=None,
                        choices=["image", "text", "combined"])
    parser.add_argument("--mode", type=str, nargs="+", default=None,
                        choices=["all", "full", "cross_regional"])
    parser.add_argument("--alpha", type=float, nargs="+", default=None)
    parser.add_argument("--topk", type=int, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vit_dir", type=str, default=None,
                        help="Override ViT embedding directory (default: "
                             "data/04_evaluation/vit_embeddings)")
    parser.add_argument("--vit_checkpoint", type=str, default=None,
                        help="ViT checkpoint for query embedding (default: "
                             "pretrained from timm). Must match the corpus "
                             "embeddings in --vit_dir.")
    parser.add_argument("--dry_run", action="store_true",
                        help="List queries only, don't run retrieval")
    parser.add_argument("--negative_control", action="store_true",
                        help="Run negative control: shuffle corpus species "
                             "labels and verify metrics collapse. Proves "
                             "retrieval measures genuine species signal.")
    parser.add_argument("--negative_control_only", action="store_true",
                        help="Run ONLY the negative control (skip main "
                             "experiment). Implies --negative_control.")
    parser.add_argument("--negative_control_seeds", type=int, nargs="+",
                        default=[42, 123, 456],
                        help="RNG seeds for label-shuffle negative control "
                             "(default: 3 seeds for stability)")
    parser.add_argument("--n_boot", type=int, default=10000,
                        help="Number of bootstrap iterations for CIs "
                             "(default: 10000; use 100 for fast iteration)")
    parser.add_argument("--no_ci", action="store_true",
                        help="Skip bootstrap CI computation entirely "
                             "(fastest mode for checking point estimates)")
    args = parser.parse_args()

    # --negative_control_only implies --negative_control
    if args.negative_control_only:
        args.negative_control = True
    skip_main = args.negative_control_only

    # -- Config --------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vlm_models      = args.vlm      or cfg["vlm_models"]
    modalities      = args.modality  or cfg["retrieval"]["modalities"]
    modes           = args.mode      or cfg["retrieval"]["modes"]
    topk            = args.topk      or cfg["retrieval"]["topk"]
    alphas          = args.alpha     or cfg["retrieval"]["combined_alpha"]
    species_queries = cfg.get("species_queries", {})
    excluded_slides = set(cfg.get("excluded_slides", []))

    # -- Model names from config (override module-level defaults) ------
    # This ensures query-embedding models match what the config declares.
    # vit_checkpoint: null means plain pretrained weights (matching Option A).
    cfg_models   = cfg.get("models", {})
    vit_model    = cfg_models.get("vit_backbone", VIT_MODEL_NAME)
    sbert_model  = cfg_models.get("sbert_model", SBERT_MODEL)

    # CLI --vit_checkpoint overrides config; None means pretrained
    vit_ckpt = args.vit_checkpoint
    if vit_ckpt is None:
        cfg_ckpt = cfg_models.get("vit_checkpoint", None)
        if cfg_ckpt is not None:
            print(f"  [WARNING] Config vit_checkpoint={cfg_ckpt!r} ignored; "
                  f"use --vit_checkpoint to load finetuned weights.")

    # CLI --vit_dir overrides default VIT_DIR
    vit_dir = Path(args.vit_dir) if args.vit_dir else VIT_DIR

    ckpt_label = os.path.basename(vit_ckpt) if vit_ckpt else "pretrained (LVD-142M)"
    print(f"  ViT model  : {vit_model} (checkpoint: {ckpt_label})")
    print(f"  ViT embeds : {vit_dir}")
    print(f"  SBERT model: {sbert_model}")
    print(f"  Output dir : {OUTPUT_DIR}")

    if not species_queries:
        print("ERROR: No species_queries defined in config. Exiting.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- Metadata ------------------------------------------------------
    print("Loading metadata...")
    slide_to_origin, slide_to_species = load_metadata()

    # Apply exclusions to metadata
    for slide in excluded_slides:
        if slide in slide_to_species:
            print(f"  [EXCLUDED] {slide} (species: {slide_to_species[slide]})")
            del slide_to_species[slide]

    print(f"  {len(slide_to_origin)} slides with origin, "
          f"{len(slide_to_species)} with species")
    print(f"  {len(species_queries)} species defined in config")
    print(f"  {len(excluded_slides)} slides excluded")

    # -- Corpus: image embeddings --------------------------------------
    t0 = time.time()
    print("\nLoading ViT image embeddings...")
    img_corpus = load_corpus(vit_dir, slide_to_origin, slide_to_species)
    n_total    = img_corpus["embeddings"].shape[0]
    n_slides   = len(set(img_corpus["slides"]))
    n_species  = len(set(img_corpus["species"]))
    print(f"  {n_total:,} samples, {n_slides} slides, "
          f"{n_species} species  ({time.time()-t0:.1f}s)")

    # -- Embedding provenance assertion --------------------------------
    # Verify corpus embeddings were produced with the same model/config
    # as the query embedder.  Catches the most dangerous silent bug:
    # query and corpus in different feature spaces.
    meta_path = vit_dir / "extraction_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            emb_meta = json.load(f)
        assert emb_meta["backbone"] == vit_model, (
            f"Embedding-space mismatch: corpus backbone={emb_meta['backbone']!r} "
            f"vs query backbone={vit_model!r}"
        )
        # Verify checkpoint consistency between corpus and query embedder
        corpus_ckpt = emb_meta.get("checkpoint")
        if vit_ckpt is None:
            # Query uses pretrained — corpus must also be pretrained
            assert corpus_ckpt in (None, "null",
                   "pretrained_from_timm (LVD-142M)"), (
                f"Corpus was built with a custom checkpoint: "
                f"{corpus_ckpt!r}.  Query embedder uses "
                f"pretrained=True.  These are different feature spaces."
            )
        else:
            # Query uses finetuned — corpus must also use a checkpoint
            assert corpus_ckpt not in (None, "null",
                   "pretrained_from_timm (LVD-142M)"), (
                f"Corpus was built with pretrained weights but query uses "
                f"checkpoint {vit_ckpt!r}.  These are different feature spaces."
            )
        assert emb_meta["img_size"] == 518, (
            f"Image size mismatch: corpus={emb_meta['img_size']} vs query=518"
        )
        ckpt_raw = emb_meta.get("checkpoint")
        ckpt_info = (os.path.basename(ckpt_raw) if ckpt_raw else "pretrained")
        print(f"  Provenance OK: backbone={emb_meta['backbone']}, "
              f"checkpoint={ckpt_info}, img_size={emb_meta['img_size']}")
    else:
        print(f"  [WARNING] No extraction_metadata.json — cannot verify "
              f"embedding provenance.  Proceed with caution.")

    # -- Integrity: verify L2 norms ------------------------------------
    norms = np.linalg.norm(img_corpus["embeddings"], axis=1)
    bad_norms = np.sum(np.abs(norms - 1.0) > 0.01)
    if bad_norms > 0:
        print(f"  [WARNING] {bad_norms:,} image embeddings have "
              f"||v|| != 1.0 (range: {norms.min():.4f}–{norms.max():.4f})")
    else:
        print(f"  L2 norms OK (all within 1.0 ± 0.01)")

    # -- All corpus slides and sibling groups --------------------------
    corpus_slides = sorted(set(img_corpus["slides"]))

    # Detect scanner-step sibling groups (betula_edf <-> betula_2_edf, etc.)
    # Used by FULL mode to exclude both variants of the same physical slide.
    slide_siblings = build_slide_groups(corpus_slides)
    extra_sibs = cfg.get("extra_siblings", [])
    if extra_sibs:
        slide_siblings = merge_extra_siblings(slide_siblings, extra_sibs)
    n_with_siblings = sum(1 for v in slide_siblings.values() if v)
    if n_with_siblings:
        print(f"  Sibling groups: {n_with_siblings} slides have sibling slides")
        for s, sibs in sorted(slide_siblings.items()):
            if sibs:
                print(f"    {s}  <->  {', '.join(sibs)}")

    # -- Embed one-shot query images (ViT) -----------------------------
    # Only embed slides explicitly listed in config exemplar_slides
    # (the config is the single source of truth for query definitions).
    config_exemplar_slides = set()
    for species, qdef in species_queries.items():
        for entry in qdef.get("exemplar_slides", []):
            config_exemplar_slides.add(entry["slide"])

    need_image = ("image" in modalities) or ("combined" in modalities)
    query_img_embs = {}
    if need_image:
        print("\nEmbedding one-shot query crops (ViT)...")
        query_img_embs = embed_query_images(
            sorted(config_exemplar_slides), device=args.device,
            model_name=vit_model, checkpoint_path=vit_ckpt,
        )
        print(f"  {len(query_img_embs)}/{len(config_exemplar_slides)} "
              f"config exemplar slides have query images")

    # -- Embed text queries from config (SBERT) ------------------------
    need_text = ("text" in modalities) or ("combined" in modalities)
    text_embs = {}
    if need_text:
        print("\nEmbedding expert-derived text queries (SBERT)...")
        text_embs = embed_text_queries(
            species_queries, slide_to_species, device=args.device,
            model_name=sbert_model,
        )

    # -- Build queries (per slide, cross-regional species only) --------
    queries = build_queries(
        img_corpus, query_img_embs, text_embs,
        species_queries, excluded_slides,
    )
    print(f"\nQueries: {len(queries)} slides")

    if args.dry_run:
        print("\n--- DRY RUN: query list ---")
        for q in queries:
            has_img = "Y" if q["img_emb"] is not None else "N"
            has_txt = "Y" if q["txt_emb"] is not None else "N"
            print(f"  {q['slide']:<55s}  {q['species']:<15s}  "
                  f"{q['origin']:<14s}  img={has_img}  txt={has_txt}  "
                  f"n={q['n_grains']:>6,}")
        return

    # -- Species overview ----------------------------------------------
    sp_origins = {}
    for q in queries:
        sp_origins.setdefault(q["species"], set()).add(q["origin"])
    cross_regional = {sp for sp, o in sp_origins.items() if len(o) >= 2}
    print(f"\nSpecies: {len(sp_origins)} total, "
          f"{len(cross_regional)} cross-regional")

    # Build species -> all origins mapping (for metadata in
    # species-collapsed text modes where dedup drops some origins)
    species_all_origins = {
        sp: sorted(origins) for sp, origins in sp_origins.items()
    }

    # Transparency: how many unique text queries vs slide-level queries?
    n_unique_txt = len(set(
        id(q["txt_emb"]) for q in queries if q["txt_emb"] is not None
    ))
    n_txt_queries = sum(1 for q in queries if q["txt_emb"] is not None)
    if n_txt_queries > 0:
        print(f"  Text queries: {n_txt_queries} slide-level, "
              f"{n_unique_txt} unique species-level embeddings")

    # -- Run per VLM ---------------------------------------------------
    if skip_main:
        print("\n  [SKIP] Main experiment (--negative_control_only)")
    for vlm in vlm_models:
        if skip_main:
            break
        print(f"\n{'='*70}")
        print(f"  VLM: {vlm}")
        print(f"{'='*70}")

        # Load text corpus if needed
        txt_corpus = None
        if need_text:
            t1 = time.time()
            print(f"\nLoading SBERT caption embeddings ({vlm})...")
            txt_corpus = load_corpus(
                CAPTION_DIR / vlm, slide_to_origin, slide_to_species
            )
            txt_corpus = align_corpora(img_corpus, txt_corpus)
            print(f"  {txt_corpus['embeddings'].shape[0]:,} samples  "
                  f"({time.time()-t1:.1f}s)")

            # Verify text corpus L2 norms
            tnorms = np.linalg.norm(txt_corpus["embeddings"], axis=1)
            bad_tnorms = np.sum(np.abs(tnorms - 1.0) > 0.01)
            if bad_tnorms > 0:
                print(f"  [WARNING] {bad_tnorms:,} text embeddings have "
                      f"||v|| != 1.0")
            else:
                print(f"  Text L2 norms OK")

        all_results = {}

        for mode in modes:
            for mod in modalities:
                alpha_list = alphas if mod == "combined" else [None]
                for alpha in alpha_list:
                    # Filter queries to those with the required signals
                    if mod == "image":
                        active = [q for q in queries if q["img_emb"] is not None]
                    elif mod == "text":
                        active = [q for q in queries if q["txt_emb"] is not None]
                    else:  # combined
                        active = [q for q in queries
                                  if q["img_emb"] is not None
                                  and q["txt_emb"] is not None]

                    if not active:
                        print(f"\n  [SKIP] {mode}/{mod}: no queries available")
                        continue

                    # Sibling dedup for image/combined: same physical
                    # preparation produces near-identical visual queries.
                    # Keep one representative per sibling group to avoid
                    # pseudo-replication.
                    # ALL mode: no dedup — ceiling uses all 44 query slides.
                    if mod in ("image", "combined") and mode != "all":
                        seen_sib = set()
                        deduped = []
                        for q in active:
                            sib_group = frozenset(
                                [q["slide"]]
                                + slide_siblings.get(q["slide"], [])
                            )
                            if sib_group not in seen_sib:
                                seen_sib.add(sib_group)
                                deduped.append(q)
                        if len(deduped) < len(active):
                            print(f"  Sibling dedup: {len(active)} -> "
                                  f"{len(deduped)} queries "
                                  f"({len(active) - len(deduped)} "
                                  f"sibling duplicates removed)")
                            active = deduped

                    # Pre-compute effective unique query count for text mode
                    # Text dedup keys:
                    #   all/full: (species,) — species-level only
                    #   cross_reg: (species, origin) — one per direction
                    n_dedup = None
                    if mod == "text":
                        dedup_keys = set()
                        for q in active:
                            if mode == "cross_regional":
                                k = (q["species"], q["origin"])
                            else:  # all, full
                                k = (q["species"],)
                            dedup_keys.add(k)
                        n_dedup = len(dedup_keys)

                    print_header(mode, mod, len(active), alpha,
                                 n_dedup=n_dedup)
                    per_query = []

                    # Text-only deduplication:
                    # All slides of the same species share the same text query
                    # embedding. Text retrieval is species-level by nature:
                    #   all/full: (species,) — one query per species
                    #   cross_reg: (species, origin) — one per leave-origin-out
                    #              direction (different exclusion = different eval)
                    #
                    # Image and combined: NOT deduplicated here (each slide
                    # has a distinct image query); sibling dedup is applied
                    # above instead.
                    seen_txt_keys: set = set()

                    for q in active:
                        excl = build_exclude(
                            img_corpus, q["slide"], q["origin"], mode,
                            slide_siblings=slide_siblings,
                        )

                        # Check there are relevant items after exclusion
                        n_rel = int(
                            ((img_corpus["species"] == q["species"])
                             & ~excl).sum()
                        )
                        if n_rel == 0:
                            continue

                        # Text deduplication: species-level for all/full,
                        # (species, origin) for cross-regional.
                        if mod == "text":
                            if mode == "cross_regional":
                                txt_key = (q["species"], q["origin"])
                            else:  # all, full
                                txt_key = (q["species"],)
                            if txt_key in seen_txt_keys:
                                continue
                            seen_txt_keys.add(txt_key)

                        if mod == "image":
                            q_emb = q["img_emb"][np.newaxis, :]
                            r = retrieve_image(
                                q_emb, q["species"],
                                img_corpus["embeddings"],
                                img_corpus["species"],
                                excl, topk,
                            )
                        elif mod == "text":
                            q_emb = q["txt_emb"][np.newaxis, :]
                            r = retrieve_text(
                                q_emb, q["species"],
                                txt_corpus["embeddings"],
                                txt_corpus["species"],
                                excl, topk,
                            )
                        else:  # combined
                            r = retrieve_combined(
                                q["img_emb"][np.newaxis, :],
                                q["txt_emb"][np.newaxis, :],
                                q["species"],
                                img_corpus["embeddings"],
                                txt_corpus["embeddings"],
                                img_corpus["species"],
                                alpha, excl, topk,
                            )

                        r["query_slide"]   = q["slide"]
                        r["query_species"] = q["species"]
                        r["query_origin"]  = q["origin"]
                        r["n_grains"]      = q["n_grains"]

                        # ── Per-query provenance trace ──────────────
                        searched_mask = ~excl
                        n_searched = int(searched_mask.sum())
                        excl_slides = sorted(set(img_corpus["slides"][excl]))
                        target_slides = sorted(set(
                            img_corpus["slides"][
                                (img_corpus["species"] == q["species"]) & searched_mask
                            ]
                        ))

                        # Build full pool composition: per-slide grain counts
                        pool_slides = img_corpus["slides"][searched_mask]
                        pool_species = img_corpus["species"][searched_mask]
                        unique_pool_slides = sorted(set(pool_slides))
                        pool_composition = []
                        for ps in unique_pool_slides:
                            ps_mask = pool_slides == ps
                            ps_sp = pool_species[ps_mask][0]
                            ps_n = int(ps_mask.sum())
                            is_target = (str(ps_sp) == q["species"])
                            pool_composition.append({
                                "slide": str(ps), "species": str(ps_sp),
                                "n_grains": ps_n, "is_target": is_target,
                            })

                        r["n_corpus_searched"] = n_searched
                        r["n_excluded"]        = int(excl.sum())
                        r["excluded_slides"]   = [str(s) for s in excl_slides]
                        r["target_slides"]     = [str(s) for s in target_slides]
                        r["pool_composition"]  = pool_composition
                        r["probed_against_slides"] = [
                            str(pc["slide"]) for pc in pool_composition
                        ]
                        r["n_probed_slides"] = len(pool_composition)

                        # Query source description
                        if mod in ("image", "combined"):
                            r["query_image_path"] = str(
                                QUERY_IMG_DIR / f"{q['slide']}.png"
                            )
                        if mod in ("text", "combined"):
                            sp_qdef = species_queries.get(q["species"], {})
                            full_query_text = sp_qdef.get("text_query", "").strip()
                            r["query_text"] = full_query_text
                            r["query_text_excerpt"] = (
                                full_query_text[:120]
                            )
                            r["query_text_source"] = str(CONFIG_PATH)

                        # Console trace — header
                        print(
                            f"    ┌─ Q: {q['slide']}  [{q['species']}]  "
                            f"origin={q['origin']}  mode={mode}"
                        )
                        if mod in ("image", "combined"):
                            print(
                                f"    │  img query: "
                                f"01_initialization/query_images/{q['slide']}.png"
                            )
                        if mod in ("text", "combined"):
                            sp_qdef = species_queries.get(q["species"], {})
                            txt_exc = sp_qdef.get("text_query", "").strip()[:100]
                            print(f'    │  txt query: "{txt_exc}..."')
                        # Pool summary
                        n_pool_slides = len(unique_pool_slides)
                        n_target_slides = len(target_slides)
                        n_distractor_slides = n_pool_slides - n_target_slides
                        print(
                            f"    │  pool: {n_searched:,} grains across "
                            f"{n_pool_slides} slides  "
                            f"({n_target_slides} target, "
                            f"{n_distractor_slides} distractor)  "
                            f"| excluded {len(excl_slides)} slides"
                        )
                        # Target slides detail
                        target_details = [
                            pc for pc in pool_composition if pc["is_target"]
                        ]
                        for td in target_details:
                            print(
                                f"    │    ✓ {td['slide']:<50s} "
                                f"{td['n_grains']:>7,} grains"
                            )
                        # Results
                        top10 = r['top10_species'][:10]
                        print(
                            f"    │  → P@1={r['P@1']:.3f}  "
                            f"mAP@20={r['mAP@20']:.3f}  "
                            f"MRR={r['MRR']:.3f}  "
                            f"total_rel={r['total_relevant']:,}  "
                            f"top10={top10}"
                        )
                        print(f"    └{'─'*70}")

                        per_query.append(r)

                    if not per_query:
                        print(f"  No valid queries after exclusion -- skipped")
                        continue

                    agg = aggregate(per_query, topk,
                                    species_all_origins=species_all_origins,
                                    n_boot=args.n_boot, no_ci=args.no_ci)
                    print_results(agg, topk)

                    key = f"{mode}_{mod}"
                    if alpha is not None:
                        key += f"_a{alpha:.2f}"
                    all_results[key] = {
                        "config": {
                            "mode": mode,
                            "modality": mod,
                            "vlm": vlm,
                            "alpha": alpha,
                            "topk": topk,
                            "n_queries": len(per_query),
                            "corpus_size": n_total,
                            "query_source_image": (
                                "one-shot crop "
                                "(01_initialization/query_images/)"
                            ),
                            "query_source_text": (
                                "expert-derived morphological descriptor "
                                "(defined in retrieval_config.yaml, "
                                "independent of VLM caption anchors)"
                            ),
                        },
                        "global": agg["global"],
                        "global_ci_95": agg.get("global_ci_95", {}),
                        "per_species": agg["per_species"],
                        "per_query": per_query,
                    }

        # -- Save ------------------------------------------------------
        out = {
            "experiment": {
                "name": cfg["experiment"]["name"],
                "description": cfg["experiment"]["description"],
                "version": cfg["experiment"].get("version", "2.0"),
                "timestamp": datetime.now().isoformat(),
                "vlm": vlm,
                "query_design": {
                    "image": (
                        "One-shot exemplar crop from "
                        "01_initialization/query_images/{slide}.png — "
                        "independently selected, never in mined corpus."
                    ),
                    "text": (
                        "Expert-derived morphological descriptor from "
                        "retrieval_config.yaml — written independently "
                        "from VLM caption anchors. Same palynological "
                        "vocabulary but different phrasing, no numeric "
                        "measurements. One text query per species."
                    ),
                    "combined": (
                        "Late fusion: alpha * sim_image + (1-alpha) * "
                        "sim_text. Score-level fusion of heterogeneous "
                        "embedding spaces."
                    ),
                },
                "exclusion": {
                    "all": (
                        "No exclusion — full corpus queried as-is. "
                        "Text: reduced circularity (independent query, "
                        "not raw anchor), but same-slide captions are "
                        "still pipeline-coupled — ceiling condition. "
                        "Image: mildly inflated (same-slide grains are "
                        "morphologically similar by mining construction); "
                        "reported with this caveat disclosed."
                    ),
                    "full": (
                        "Exclude query slide AND its scanner-step "
                        "siblings (e.g. betula_edf and betula_2_edf "
                        "are different z-stack passes of the same "
                        "physical preparation)."
                    ),
                    "cross_regional": (
                        "Exclude all slides from query's geographic "
                        "origin. Each query slide acts as independent "
                        "leave-origin-out query."
                    ),
                },
                "excluded_slides": sorted(excluded_slides),
                "circularity_notes": {
                    "text": (
                        "Text queries are NOT the raw caption anchors. "
                        "Anchors were used in VLM prompts, creating "
                        "prompt-coupled circularity. These queries are "
                        "independently written descriptions using "
                        "standard palynological terminology."
                    ),
                    "image": (
                        "One-shot query images were shown to the VLM "
                        "during captioning and used to seed the mining "
                        "pipeline. This creates a soft pipeline bias "
                        "(disclosed). Not direct data leakage."
                    ),
                },
            },
            "slides_summary": {
                "corpus_slides": len(corpus_slides),
                "with_query_image": len(query_img_embs),
                "with_text_query": len(text_embs),
                "target_species": len(species_queries),
                "cross_regional_species": len(cross_regional),
                "species_list": sorted(species_queries.keys()),
            },
            "results": all_results,
        }
        out_path = OUTPUT_DIR / f"retrieval_{vlm}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")

    # ==================================================================
    # NEGATIVE CONTROL: label-shuffle validation
    # ==================================================================
    if args.negative_control:
        print(f"\n{'='*70}")
        print(f"  NEGATIVE CONTROL: Species Label Shuffle")
        print(f"{'='*70}")
        print(f"  Purpose: Shuffle corpus species labels and verify retrieval")
        print(f"  metrics collapse to chance level.  This validates that the")
        print(f"  main experiment measures genuine species-level signal.")
        print(f"  (Permutation sanity check — not an error bar.)")
        print(f"  Seeds: {args.negative_control_seeds}")
        print(f"  VLMs:  {vlm_models}")
        print()

        neg_all_vlms = {}

        for vlm_neg in vlm_models:
            print(f"\n  --- Negative control: {vlm_neg} ---")

            if need_text:
                print(f"  Loading text corpus ({vlm_neg}) for negative control...")
                txt_corpus_neg = load_corpus(
                    CAPTION_DIR / vlm_neg, slide_to_origin, slide_to_species
                )
                txt_corpus_neg = align_corpora(img_corpus, txt_corpus_neg)
            else:
                txt_corpus_neg = None

            neg_results = {}

            for seed in args.negative_control_seeds:
                rng = np.random.RandomState(seed)
                shuffled_species = img_corpus["species"].copy()
                rng.shuffle(shuffled_species)

                # Verify shuffle actually changed labels
                n_changed = int((shuffled_species != img_corpus["species"]).sum())
                pct_changed = 100.0 * n_changed / len(shuffled_species)
                print(f"  Seed {seed}: {n_changed:,}/{len(shuffled_species):,} "
                      f"labels changed ({pct_changed:.1f}%)")

                seed_results = {}

                for mode in modes:
                    for mod in modalities:
                        alpha_list = alphas if mod == "combined" else [None]
                        for alpha in alpha_list:
                            if mod == "image":
                                active = [q for q in queries if q["img_emb"] is not None]
                            elif mod == "text":
                                active = [q for q in queries if q["txt_emb"] is not None]
                            else:
                                active = [q for q in queries
                                          if q["img_emb"] is not None
                                          and q["txt_emb"] is not None]

                            if not active:
                                continue

                            # Sibling dedup for image/combined
                            # (mirrors main loop; skip for ALL mode)
                            if mod in ("image", "combined") and mode != "all":
                                seen_sib = set()
                                deduped = []
                                for q in active:
                                    sib_group = frozenset(
                                        [q["slide"]]
                                        + slide_siblings.get(q["slide"], [])
                                    )
                                    if sib_group not in seen_sib:
                                        seen_sib.add(sib_group)
                                        deduped.append(q)
                                active = deduped

                            seen_txt_keys: set = set()
                            per_query_neg = []

                            for q in active:
                                excl = build_exclude(
                                    img_corpus, q["slide"], q["origin"], mode,
                                    slide_siblings=slide_siblings,
                                )

                                # Relevance against SHUFFLED labels
                                n_rel = int(
                                    ((shuffled_species == q["species"])
                                     & ~excl).sum()
                                )
                                if n_rel == 0:
                                    continue

                                if mod == "text":
                                    if mode == "cross_regional":
                                        txt_key = (q["species"], q["origin"])
                                    else:  # all, full
                                        txt_key = (q["species"],)
                                    if txt_key in seen_txt_keys:
                                        continue
                                    seen_txt_keys.add(txt_key)

                                # Score using REAL embeddings but SHUFFLED labels
                                if mod == "image":
                                    q_emb = q["img_emb"][np.newaxis, :]
                                    scores = (q_emb @ img_corpus["embeddings"].T).squeeze()
                                elif mod == "text":
                                    q_emb = q["txt_emb"][np.newaxis, :]
                                    scores = (q_emb @ txt_corpus_neg["embeddings"].T).squeeze()
                                else:
                                    sim_img = (q["img_emb"][np.newaxis, :] @ img_corpus["embeddings"].T).squeeze()
                                    sim_txt = (q["txt_emb"][np.newaxis, :] @ txt_corpus_neg["embeddings"].T).squeeze()
                                    scores = alpha * sim_img + (1.0 - alpha) * sim_txt

                                scores_copy = scores.copy()
                                scores_copy[excl] = -np.inf
                                sorted_idx = np.argsort(-scores_copy)
                                sorted_species_neg = shuffled_species[sorted_idx]
                                relevant = (sorted_species_neg == q["species"])
                                total_relevant = n_rel

                                r = compute_metrics(relevant, total_relevant, topk)
                                r["query_species"] = q["species"]
                                r["query_origin"] = q["origin"]
                                per_query_neg.append(r)

                            if not per_query_neg:
                                continue

                            agg_neg = aggregate(per_query_neg, topk,
                                                species_all_origins=species_all_origins,
                                                n_boot=args.n_boot, no_ci=args.no_ci)
                            if args.no_ci:
                                ci_neg = {}
                            else:
                                ci_neg = bootstrap_ci(per_query_neg, topk,
                                                      n_boot=args.n_boot, seed=seed)
                            key = f"{mode}_{mod}"
                            if alpha is not None:
                                key += f"_a{alpha:.2f}"
                            seed_results[key] = {
                                "global": agg_neg["global"],
                                "global_ci_95": ci_neg,
                                "per_species": agg_neg["per_species"],
                            }

                neg_results[f"seed_{seed}"] = seed_results

            # Average across seeds (global level)
            all_keys = set()
            for sr in neg_results.values():
                all_keys.update(sr.keys())

            neg_averaged = {}
            for key in sorted(all_keys):
                vals_per_metric = {}
                for sr in neg_results.values():
                    if key in sr:
                        for mk, mv in sr[key]["global"].items():
                            if isinstance(mv, (int, float)):
                                vals_per_metric.setdefault(mk, []).append(mv)
                neg_averaged[key] = {
                    mk: float(np.mean(vs)) for mk, vs in vals_per_metric.items()
                }

            # Average bootstrap CIs across seeds
            neg_averaged_ci = {}
            for key in sorted(all_keys):
                ci_per_metric_lo = {}
                ci_per_metric_hi = {}
                for sr in neg_results.values():
                    if key in sr:
                        ci_seed = sr[key].get("global_ci_95", {})
                        for mk, bounds in ci_seed.items():
                            if bounds and len(bounds) == 2:
                                ci_per_metric_lo.setdefault(mk, []).append(bounds[0])
                                ci_per_metric_hi.setdefault(mk, []).append(bounds[1])
                neg_averaged_ci[key] = {
                    mk: [round(float(np.mean(ci_per_metric_lo[mk])), 4),
                         round(float(np.mean(ci_per_metric_hi[mk])), 4)]
                    for mk in ci_per_metric_lo
                }

            # Also average per-species across seeds
            neg_averaged_per_species = {}
            for key in sorted(all_keys):
                all_sp = set()
                for sr in neg_results.values():
                    if key in sr:
                        all_sp.update(sr[key].get("per_species", {}).keys())
                sp_avg = {}
                for sp in sorted(all_sp):
                    sp_vals = {}
                    for sr in neg_results.values():
                        if key in sr:
                            sp_data = sr[key].get("per_species", {}).get(sp, {})
                            for mk, mv in sp_data.items():
                                if isinstance(mv, (int, float)):
                                    sp_vals.setdefault(mk, []).append(mv)
                    sp_avg[sp] = {
                        mk: float(np.mean(vs)) for mk, vs in sp_vals.items()
                    }
                neg_averaged_per_species[key] = sp_avg

            # Print comparison table
            map_key = f"mAP@{max(topk)}"
            print(f"\n  Negative Control Results for {vlm_neg} (averaged over "
                  f"{len(args.negative_control_seeds)} seeds):")
            print(f"  {'Condition':<30s} {'P@1':>7s} {'P@20':>7s} "
                  f"{'MRR':>7s} {map_key:>7s}")
            print(f"  {'-'*65}")
            for key in sorted(neg_averaged.keys()):
                nm = neg_averaged[key]
                print(f"  SHUFFLED  {key:<19s} "
                      f"{nm.get('P@1',0):>7.3f} {nm.get('P@20',0):>7.3f} "
                      f"{nm.get('MRR',0):>7.3f} "
                      f"{nm.get(map_key,0):>7.3f}")

            neg_all_vlms[vlm_neg] = {
                "per_seed": neg_results,
                "averaged": neg_averaged,
                "averaged_ci_95": neg_averaged_ci,
                "averaged_per_species": neg_averaged_per_species,
            }

        # Save negative control output (all VLMs)
        neg_out = {
            "experiment": "negative_control_label_shuffle",
            "description": (
                "Permutation sanity check: species labels in the corpus "
                "are randomly shuffled (embeddings unchanged). If "
                "retrieval metrics collapse to near-chance, it validates "
                "that the main experiment measures genuine species-level "
                "signal rather than structural artifacts."
            ),
            "seeds": args.negative_control_seeds,
            "vlms": vlm_models,
            "corpus_size": n_total,
            "per_vlm": neg_all_vlms,
            "timestamp": datetime.now().isoformat(),
        }
        neg_path = OUTPUT_DIR / "retrieval_negative_control.json"
        with open(neg_path, "w") as f:
            json.dump(neg_out, f, indent=2, default=str)
        print(f"\n  Saved: {neg_path}")

    print(f"\nDone. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
