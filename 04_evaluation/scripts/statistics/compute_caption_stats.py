#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Pollen AI Atlas - Caption Statistics Driver
===========================================

Public-release driver for the caption analyses used in the paper codebase.
The retained phases are intentionally limited to manuscript-facing metrics:

  Phase 1 (Read)         -> caption_stats.io.phase_read_captions
  Phase 2 (Basic stats)  -> caption_stats.phase_basic.phase_basic_stats
  Phase 3 (Morphology)   -> caption_stats.phase_morphology.phase_morphological_coverage
  Phase 4 (Cross-model)  -> caption_stats.phase_cross_model.phase_cross_model
  Phase 5 (Audit export) -> caption_stats.phase_audit.phase_export_expert_audit_sample

BLEU/ROUGE and TF-IDF consistency analyses were exploratory and are not
included in this release.

Usage:
  python compute_caption_stats.py
  python compute_caption_stats.py --skip-cross-model
  python compute_caption_stats.py --max-captions-per-slide 100
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from caption_stats.constants import (
    CROSS_DATASET_MATRIX_PATH,
    DEFAULT_PROMPT_MARKERS,
    HAS_SBERT,
    HAS_TORCH,
    MODELS,
    OUTPUT_DIR,
    VOCAB_PATH,
)
from caption_stats import (
    build_slide_term_sets,
    discover_jsonl_files,
    generate_markdown,
    load_anchor_texts,
    load_cross_dataset_slide_map,
    phase_basic_stats,
    phase_cross_model,
    phase_export_expert_audit_sample,
    phase_morphological_coverage,
    phase_read_captions,
)
from caption_stats.helpers import safe_model_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute final caption statistics for Pollen AI Atlas"
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for SBERT cross-model agreement when available",
    )
    parser.add_argument(
        "--require-gpu", action="store_true",
        help="Fail fast if CUDA is not available",
    )
    parser.add_argument("--vocab-path", type=str, default=None)
    parser.add_argument(
        "--cross-dataset-path", type=str,
        default=str(CROSS_DATASET_MATRIX_PATH),
        help="Path to cross_dataset_matrix.json",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--max-captions-per-slide", type=int, default=0,
        help="Limit captions read per slide for quick validation (0=all)",
    )
    parser.add_argument(
        "--skip-cross-model", action="store_true",
        help="Skip pairwise cross-model Jaccard/SBERT agreement",
    )
    parser.add_argument(
        "--skip-sbert", action="store_true",
        help="Skip SBERT inside cross-model agreement",
    )
    parser.add_argument(
        "--sbert-model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic similarity",
    )
    parser.add_argument(
        "--sbert-max-pairs", type=int, default=0,
        help="Max matched pairs per model pair for SBERT (0=all)",
    )
    parser.add_argument(
        "--sbert-batch-size", type=int, default=256,
        help="Batch size for SBERT encoding",
    )
    parser.add_argument(
        "--sbert-allow-download", action="store_true",
        help="Allow SBERT model download if it is not cached locally",
    )
    parser.add_argument(
        "--audit-sample-size", type=int, default=240,
        help="Expert audit sample size (0 disables export)",
    )
    parser.add_argument(
        "--audit-seed", type=int, default=42,
        help="Random seed for expert audit sample export",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = perf_counter()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = Path(args.vocab_path) if args.vocab_path else VOCAB_PATH

    print("Caption Statistics - Pollen AI Atlas")
    print(f"  torch:    {'yes' if HAS_TORCH else 'no'}")
    print(f"  SBERT:    {'yes' if HAS_SBERT else 'no'}")
    print(f"  workers:  {args.workers}")
    print(f"  device:   {args.device}")

    cuda_available = False
    cuda_error = None
    if HAS_TORCH:
        import torch
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:
            cuda_error = str(exc)

    print(f"  cuda:     {'yes' if cuda_available else 'no'}")
    if args.device.startswith("cuda") and not cuda_available:
        msg = f"  WARNING: CUDA requested via --device={args.device}, but unavailable"
        print(f"{msg}: {cuda_error}" if cuda_error else msg)

    if args.require_gpu and not cuda_available:
        print("ERROR: --require-gpu was set, but CUDA is not available.")
        return 1

    if not vocab_path.exists():
        print(f"\nERROR: missing vocabulary file: {vocab_path}")
        print("Run extract_anchor_vocabulary.py first.")
        return 1

    with open(vocab_path) as f:
        vocab = json.load(f)

    vocab_terms = set(vocab.get("morphological_terms", []))
    prompt_qualifiers = vocab.get("prompt_qualifiers", [])
    prompt_markers = vocab.get("prompt_markers", DEFAULT_PROMPT_MARKERS)
    print(f"\n  Loaded {len(vocab_terms)} morphological terms from {vocab_path}")

    slide_anchor, slide_species, slide_family, slide_hint = load_anchor_texts()
    slide_term_sets = build_slide_term_sets(vocab, slide_anchor, slide_hint)
    print(f"  Loaded anchors for {len(slide_anchor)} slides")

    slide_dataset_truth = load_cross_dataset_slide_map(args.cross_dataset_path)
    if slide_dataset_truth:
        print(f"  Loaded dataset truth for {len(slide_dataset_truth)} slides from {args.cross_dataset_path}")
    else:
        print("  WARNING: cross-dataset mapping missing or empty; falling back to discovered/inferred datasets")

    model_files = discover_jsonl_files()
    for mk in MODELS:
        print(f"  {safe_model_name(mk)}: {len(model_files[mk])} JSONL files")

    model_data, model_dataset = phase_read_captions(
        model_files,
        workers=args.workers,
        max_captions_per_slide=args.max_captions_per_slide,
    )

    basic_stats = phase_basic_stats(
        model_data, model_dataset,
        slide_species, slide_family,
        slide_dataset_truth,
        prompt_qualifiers, prompt_markers,
    )

    morph_coverage = phase_morphological_coverage(
        model_data, model_dataset,
        slide_species, slide_dataset_truth,
        vocab_terms, slide_term_sets,
    )

    if args.skip_cross_model:
        cross_model = {}
    else:
        cross_model = phase_cross_model(
            model_data, model_dataset,
            slide_species, slide_dataset_truth,
            workers=args.workers,
            skip_sbert=args.skip_sbert,
            sbert_model_name=args.sbert_model,
            sbert_max_pairs=args.sbert_max_pairs,
            sbert_batch_size=args.sbert_batch_size,
            sbert_device=args.device,
            sbert_allow_download=args.sbert_allow_download,
        )

    expert_audit_sample = phase_export_expert_audit_sample(
        model_data, model_dataset,
        slide_species, slide_dataset_truth,
        output_dir=output_dir,
        sample_size=args.audit_sample_size,
        seed=args.audit_seed,
    )

    elapsed = perf_counter() - t0
    stats = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "runtime_seconds": round(elapsed, 1),
            "n_workers": args.workers,
            "device": args.device,
            "vocab_path": str(vocab_path),
            "cross_dataset_path": str(args.cross_dataset_path),
            "n_vocab_terms": len(vocab_terms),
            "max_captions_per_slide": args.max_captions_per_slide,
            "cross_model_executed": not args.skip_cross_model,
            "sbert_enabled": not args.skip_sbert,
            "sbert_allow_download": bool(args.sbert_allow_download),
            "audit_sample_size": args.audit_sample_size,
            "audit_seed": args.audit_seed,
        },
        "basic_stats": basic_stats,
        "morphological_coverage": morph_coverage,
        "cross_model": cross_model,
        "expert_audit_sample": expert_audit_sample,
    }

    json_path = output_dir / "caption_statistics.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    md_path = output_dir / "caption_statistics.md"
    generate_markdown(stats, md_path)
    print(f"  MD:   {md_path}")

    print("\n" + "=" * 60)
    print(f"Total runtime: {elapsed:.1f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
