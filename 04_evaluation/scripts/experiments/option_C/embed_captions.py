#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Pre-compute SBERT Caption Embeddings for LUPI Training
=======================================================

Encodes all VLM-generated captions into 384-dim sentence embeddings using
sentence-transformers/all-MiniLM-L6-v2 and saves per-slide H5 files.

These pre-computed embeddings are loaded during LUPI training to avoid
SBERT forward passes on every epoch.

Usage:
    python embed_captions.py                     # All datasets, qwen25vl
    python embed_captions.py --caption_model production_qwen3-fp8_final
    python embed_captions.py --datasets french   # Single dataset

Output:
    data/04_evaluation/caption_embeddings/qwen25vl/{slide_name}_embeddings.h5
    Each H5 has:
      - sample_ids: string array of sample IDs
      - embeddings: float32 array (N, 384)
"""

import os
import sys
import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))


def embed_captions(
    data_root: Path,
    output_dir: Path,
    caption_model: str = "production_qwen25vl_final",
    datasets: list = None,
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
    device: str = "cuda:0",
):
    """Pre-compute and cache SBERT embeddings for all captions.
    
    Args:
        data_root: Path to data/ directory
        output_dir: Where to save H5 files
        caption_model: Caption model subfolder name
        datasets: List of dataset names to process
        sbert_model: SBERT model identifier
        batch_size: Encoding batch size
        device: Device for SBERT inference
    """
    from sentence_transformers import SentenceTransformer
    
    if datasets is None:
        datasets = ["french", "hungarian", "mediterranean", "swedish"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SBERT model
    print(f"[SBERT] Loading {sbert_model}...")
    model = SentenceTransformer(sbert_model, device=device)
    embed_dim = model.get_sentence_embedding_dimension()
    print(f"[SBERT] Embedding dimension: {embed_dim}")
    
    total_embedded = 0
    total_slides = 0
    
    for dataset in datasets:
        caption_dir = data_root / "03_captioning" / dataset / caption_model
        if not caption_dir.exists():
            print(f"[WARNING] Caption dir not found: {caption_dir}")
            continue
        
        jsonl_files = sorted(caption_dir.glob("*_captions.jsonl"))
        print(f"\n[{dataset}] Found {len(jsonl_files)} caption files")
        
        for jsonl_path in tqdm(jsonl_files, desc=f"Embedding {dataset}"):
            slide_name = jsonl_path.stem.replace("_captions", "")
            h5_path = output_dir / f"{slide_name}_embeddings.h5"

            # --- Slide-level species check via anchor _species.txt ---
            # Use first word of _species.txt (genus) to determine if slide has valid taxonomy.
            # This correctly includes slides like brassicaceae_edf (species.txt='Brassica')
            # even when individual JSONL records carry species='Unknown'.
            anchor_dir = PROJECT_ROOT / "03_captioning" / "caption_anchors"
            species_file = anchor_dir / f"{slide_name}_species.txt"
            if species_file.exists():
                genus = species_file.read_text().strip().split()[0].lower() if species_file.read_text().strip() else "unknown"
            else:
                genus = "unknown"
            if genus == "unknown" or len(genus) < 2:
                continue  # Skip slides with no valid taxonomy

            # Skip if already computed with same SBERT model
            if h5_path.exists():
                try:
                    with h5py.File(h5_path, "r") as hf:
                        stored_model = hf.attrs.get("sbert_model", "")
                        stored_caption = hf.attrs.get("caption_model", "")
                        if stored_model == sbert_model and stored_caption == caption_model:
                            continue
                        else:
                            print(f"  [STALE] {h5_path.name}: model mismatch "
                                  f"({stored_model}/{stored_caption} vs {sbert_model}/{caption_model}), re-embedding")
                except Exception:
                    print(f"  [CORRUPT] {h5_path.name}: cannot read, re-embedding")

            # Load captions (all records with non-empty caption)
            sample_ids = []
            captions = []

            with open(jsonl_path) as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        caption = record.get("caption", "")
                        if caption:
                            sample_ids.append(record["id"])
                            captions.append(caption)
                    except json.JSONDecodeError:
                        continue
            
            if not captions:
                continue
            
            # Encode in batches
            embeddings = model.encode(
                captions,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for consistency
            )
            
            # Save to H5
            with h5py.File(h5_path, "w") as hf:
                # Store as variable-length strings
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset("sample_ids", data=np.array(sample_ids, dtype=object), dtype=dt)
                hf.create_dataset("embeddings", data=embeddings.astype(np.float32),
                                  compression="gzip", compression_opts=4)
                hf.attrs["sbert_model"] = sbert_model
                hf.attrs["embed_dim"] = embed_dim
                hf.attrs["num_samples"] = len(sample_ids)
                hf.attrs["caption_model"] = caption_model
                hf.attrs["slide_name"] = slide_name
            
            total_embedded += len(captions)
            total_slides += 1
    
    print(f"\n{'='*60}")
    print(f"Embedding complete!")
    print(f"Slides: {total_slides}")
    print(f"Samples: {total_embedded:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SBERT caption embeddings")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data/ directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for H5 files")
    parser.add_argument("--caption_model", type=str, default="production_qwen25vl_final",
                        choices=[
                            "production_qwen25vl_final",
                            "production_qwen3-fp8_final",
                            "production_qwen35-fp8_final",
                            "production_qwen36-fp8_final",
                            "production_gemma4-bf16_final",
                        ],
                        help="Caption model to embed")
    parser.add_argument("--datasets", nargs="+",
                        default=["french", "hungarian", "mediterranean", "swedish"],
                        help="Datasets to process")
    parser.add_argument("--sbert_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SBERT model to use")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Encoding batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for SBERT")
    
    args = parser.parse_args()
    
    # Defaults
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    
    # Output naming: strip production_ prefix and _final suffix
    # e.g. "production_gemma4-bf16_final" -> "gemma4-bf16"
    model_short = args.caption_model
    if model_short.startswith("production_") and model_short.endswith("_final"):
        model_short = model_short[len("production_"):-len("_final")]
    output_dir = (Path(args.output_dir) if args.output_dir 
                  else data_root / "04_evaluation" / "caption_embeddings" / model_short)
    
    embed_captions(
        data_root=data_root,
        output_dir=output_dir,
        caption_model=args.caption_model,
        datasets=args.datasets,
        sbert_model=args.sbert_model,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
