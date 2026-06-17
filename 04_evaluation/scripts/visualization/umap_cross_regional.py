#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Cross-Regional UMAP Visualization — 15 Retrieval Species
==========================================================

Publication-ready UMAP plots for the 15 cross-regional species used
in retrieval experiments. Produces a 3-panel figure:

  (a) Text embeddings (SBERT caption space)
  (b) Image embeddings (ViT-Small-LVD — pretrained or finetuned)
  (c) Combined (α=0.5 late-fusion distance, matching retrieval)

Points are colored by species (15 colors) and shaped by geographic
origin (French, Hungarian, Mediterranean, Swedish).

Sampling is balanced per (species, origin) to avoid visual bias.
Uses pre-extracted embeddings from H5 files — no on-the-fly WSI reading.

No figure title (intended for LaTeX \\includegraphics).

Usage:
    python umap_cross_regional.py
    python umap_cross_regional.py --max_per_origin 80 --dpi 300
    python umap_cross_regional.py --vlm qwen3vl
    python umap_cross_regional.py --vit_mode finetuned
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# ── The 15 cross-regional species (from retrieval_config.yaml) ──────
CROSS_REGIONAL_SPECIES = [
    "Alnus", "Ambrosia", "Betula", "Chenopodium", "Corylus",
    "Cupressus", "Picea", "Pinus", "Plantago", "Platanus",
    "Quercus", "Rumex", "Salix", "Ulmus", "Urtica",
]

# ── Dataset identification from slide name ──────────────────────────
DATASET_PREFIXES = {
    "mediterranean_pollen": "Mediterranean",
    "hun_": "Hungarian",
    "Ambrosia-Iva_reference": "Hungarian",
    "Ambrosia_artemisifolia_Asteraceae": "Swedish",
    "Alnus_cf_incana": "Swedish", "Alnus_glutinosa": "Swedish",
    "Betula_cf_pendula": "Swedish", "Betula_sp_01": "Swedish",
    "Corylus_avellana": "Swedish", "Corylus_sp_01": "Swedish",
    "Dactylus_glomerata": "Swedish", "Picea_abies": "Swedish",
    "Pinus_sp_10": "Swedish", "Quercus_robur": "Swedish",
    "Salix_sp_10": "Swedish", "Ulmus_glabra": "Swedish",
    "Urtica_dioica": "Swedish", "calluna_vulgaris_ericaceae": "Swedish",
    "plantago_major_plantaginaceae": "Swedish",
    "acer_platanoides": "Swedish", "aesculus_hippocustanum": "Swedish",
}

# Origin abbreviations and marker styles
ORIGIN_ABBREV = {
    "French": "F", "Hungarian": "H",
    "Swedish": "S", "Mediterranean": "M",
}
ORIGIN_MARKERS = {
    "French": "o", "Hungarian": "^",
    "Swedish": "s", "Mediterranean": "D",
}
ORIGIN_ORDER = ["French", "Hungarian", "Mediterranean", "Swedish"]


def get_dataset(slide_name: str) -> str:
    for prefix, dataset in DATASET_PREFIXES.items():
        if slide_name.startswith(prefix):
            return dataset
    return "French" if slide_name.endswith("_edf") else "Unknown"


def load_cross_regional_data(
    splits_dir: Path,
    sbert_dir: Path,
    vit_dir: Path,
    anchor_dir: Path,
    max_per_origin: int = 60,
    seed: int = 42,
) -> dict:
    """Load val-split embeddings for the 15 cross-regional species.

    Balances sampling per (species, origin).

    Returns dict with:
        sbert_embeddings, vit_embeddings, species, origins, sample_ids, slides
        Also: counts dict for metadata export.
    """
    rng = np.random.RandomState(seed)
    cross_set = set(CROSS_REGIONAL_SPECIES)

    # Collect per (species, origin)
    buckets = defaultdict(lambda: {
        "sbert": [], "vit": [], "ids": [], "slides": [],
    })

    val_dir = splits_dir / "val"
    for f in sorted(val_dir.glob("*_val.json")):
        with open(f) as fh:
            data = json.load(fh)
        slide = data["slide"]

        species_file = anchor_dir / f"{slide}_species.txt"
        if not species_file.exists():
            continue
        species = species_file.read_text().strip()
        if species not in cross_set:
            continue

        origin = get_dataset(slide)
        if origin == "Unknown":
            continue

        val_ids = set(data["sample_ids"])

        # Load SBERT
        sbert_path = sbert_dir / f"{slide}_embeddings.h5"
        vit_path = vit_dir / f"{slide}_embeddings.h5"
        if not sbert_path.exists() or not vit_path.exists():
            continue

        # Read SBERT
        with h5py.File(sbert_path, "r") as hf:
            s_ids = [x.decode() if isinstance(x, bytes) else x for x in hf["sample_ids"][:]]
            s_embs = hf["embeddings"][:]
        s_id_to_idx = {sid: i for i, sid in enumerate(s_ids)}

        # Read ViT
        with h5py.File(vit_path, "r") as hf:
            v_ids = [x.decode() if isinstance(x, bytes) else x for x in hf["sample_ids"][:]]
            v_embs = hf["embeddings"][:]
        v_id_to_idx = {sid: i for i, sid in enumerate(v_ids)}

        # Only include samples that exist in both embedding spaces AND val split
        common_ids = val_ids & set(s_id_to_idx.keys()) & set(v_id_to_idx.keys())

        key = (species, origin)
        for sid in common_ids:
            buckets[key]["sbert"].append(s_embs[s_id_to_idx[sid]])
            buckets[key]["vit"].append(v_embs[v_id_to_idx[sid]])
            buckets[key]["ids"].append(sid)
            buckets[key]["slides"].append(slide)

    # Balanced subsample per (species, origin)
    all_sbert, all_vit = [], []
    all_species, all_origins, all_ids, all_slides = [], [], [], []
    counts = {}

    for species in sorted(cross_set):
        for origin in ORIGIN_ORDER:
            key = (species, origin)
            if key not in buckets:
                continue
            b = buckets[key]
            n = len(b["ids"])
            counts[f"{ORIGIN_ABBREV[origin]}-{species}"] = {
                "available": n, "sampled": min(n, max_per_origin),
            }

            if n > max_per_origin:
                chosen = rng.choice(n, max_per_origin, replace=False)
            else:
                chosen = np.arange(n)

            for i in chosen:
                all_sbert.append(b["sbert"][i])
                all_vit.append(b["vit"][i])
                all_species.append(species)
                all_origins.append(origin)
                all_ids.append(b["ids"][i])
                all_slides.append(b["slides"][i])

    return {
        "sbert_embeddings": np.array(all_sbert, dtype=np.float32),
        "vit_embeddings": np.array(all_vit, dtype=np.float32),
        "species": all_species,
        "origins": all_origins,
        "sample_ids": all_ids,
        "slides": all_slides,
        "counts": counts,
    }


def make_combined_distance(sbert: np.ndarray, vit: np.ndarray,
                           alpha: float = 0.5) -> np.ndarray:
    """Compute combined distance matrix matching retrieval late fusion.

    dist = 1 - (alpha * cos_sim_image + (1-alpha) * cos_sim_text)
    Returns a precomputed distance matrix for UMAP metric='precomputed'.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    n = sbert.shape[0]
    print(f"  Computing combined distance matrix ({n}x{n})...")
    sim_text = cosine_similarity(sbert)
    sim_image = cosine_similarity(vit)
    combined_sim = alpha * sim_image + (1 - alpha) * sim_text
    dist = 1.0 - combined_sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)  # ensure non-negative
    return dist.astype(np.float32)


def run_umap(embeddings: np.ndarray, seed: int = 42,
             n_neighbors: int = 15, min_dist: float = 0.1,
             metric: str = "cosine") -> np.ndarray:
    """UMAP projection to 2D."""
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        random_state=seed, n_jobs=4, metric=metric,
    )
    return reducer.fit_transform(embeddings)


def create_three_panel_figure(
    projections: dict,
    species: list,
    origins: list,
    output_dir: Path,
    vlm_short: str,
    dpi: int = 300,
    point_size: int = 12,
    alpha: float = 0.55,
):
    """Create a 3-panel (a,b,c) UMAP figure — no title, LaTeX-ready.

    projections: dict with keys 'text', 'image', 'combined' -> (N,2) arrays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans']

    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False
        print("[WARNING] adjustText not installed — centroid labels may overlap")

    species_arr = np.array(species)
    origins_arr = np.array(origins)
    unique_species = sorted(set(species))
    unique_origins = [o for o in ORIGIN_ORDER if o in set(origins)]

    # ── Species colors (15 species → tab20 subset) ──────────────────
    cmap = plt.colormaps.get_cmap("tab20")
    species_colors = {sp: cmap(i / max(len(unique_species) - 1, 1))
                      for i, sp in enumerate(unique_species)}

    # ── Figure layout ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), dpi=dpi)
    panel_labels = ["(a)", "(b)", "(c)"]
    panel_keys = ["text", "image", "combined"]

    for ax, pkey, plabel in zip(axes, panel_keys, panel_labels):
        proj = projections[pkey]

        # Plot per (species, origin) for marker differentiation
        for sp in unique_species:
            for orig in unique_origins:
                mask = (species_arr == sp) & (origins_arr == orig)
                if not mask.any():
                    continue
                ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    c=[species_colors[sp]],
                    marker=ORIGIN_MARKERS[orig],
                    s=point_size,
                    alpha=alpha,
                    edgecolors="none",
                    rasterized=True,
                )

        # Centroid labels per (species, origin) — e.g. F-Alnus, H-Alnus
        texts = []
        for sp in unique_species:
            for orig in unique_origins:
                mask = (species_arr == sp) & (origins_arr == orig)
                if not mask.any():
                    continue
                cx, cy = proj[mask].mean(axis=0)
                label = f"{ORIGIN_ABBREV[orig]}-{sp}"
                t = ax.text(
                    cx, cy, label, fontsize=10.0, fontstyle="italic",
                    weight="bold", ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=0.6, boxstyle="round,pad=0.15"),
                )
                texts.append(t)

        if has_adjust and texts:
            adjust_text(
                texts, ax=ax,
                expand_text=(1.15, 1.35),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.5),
            )

        ax.set_xlabel("UMAP-1", fontsize=22)
        # Only show UMAP-2 label on the first (leftmost) panel
        if pkey == panel_keys[0]:
            ax.set_ylabel("UMAP-2", fontsize=22)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=18)

        # Remove top/right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel label (a), (b), (c) — top-left, no subtitle
        ax.text(
            0.02, 0.98, plabel, transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top", ha="left",
        )

    # ── Shared legend: species colors + origin markers ──────────────
    species_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=species_colors[sp], markersize=9,
               label=sp, markeredgecolor="none")
        for sp in unique_species
    ]
    origin_handles = [
        Line2D([0], [0], marker=ORIGIN_MARKERS[o], color="w",
               markerfacecolor="0.4", markersize=9,
               label=f"{ORIGIN_ABBREV[o]}: {o}", markeredgecolor="none")
        for o in unique_origins
    ]
    all_handles = species_handles + origin_handles

    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(len(all_handles), 10),
        fontsize=20,
        frameon=True,
        framealpha=0.95,
        edgecolor="0.8",
        columnspacing=0.8,
        handletextpad=0.3,
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.20, wspace=0.23)

    # ── Save ────────────────────────────────────────────────────────
    stem = f"umap_cross_regional_{vlm_short}"
    for ext in ["pdf", "png", "svg"]:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def create_individual_panels(
    projections: dict,
    species: list,
    origins: list,
    output_dir: Path,
    vlm_short: str,
    dpi: int = 300,
    point_size: int = 18,
    alpha: float = 0.55,
):
    """Create individual per-modality panels for flexible LaTeX layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans']

    try:
        from adjustText import adjust_text
        has_adjust = True
    except ImportError:
        has_adjust = False

    species_arr = np.array(species)
    origins_arr = np.array(origins)
    unique_species = sorted(set(species))
    unique_origins = [o for o in ORIGIN_ORDER if o in set(origins)]

    cmap = plt.colormaps.get_cmap("tab20")
    species_colors = {sp: cmap(i / max(len(unique_species) - 1, 1))
                      for i, sp in enumerate(unique_species)}

    panel_meta = {
        "text": f"Text (SBERT — {vlm_short.upper()})",
        "image": "Image (ViT-Small-LVD)",
        "combined": f"Combined (ViT + SBERT)",
    }

    for pkey, pshort in panel_meta.items():
        proj = projections[pkey]

        fig, ax = plt.subplots(1, 1, figsize=(8, 7), dpi=dpi)

        for sp in unique_species:
            for orig in unique_origins:
                mask = (species_arr == sp) & (origins_arr == orig)
                if not mask.any():
                    continue
                ax.scatter(
                    proj[mask, 0], proj[mask, 1],
                    c=[species_colors[sp]],
                    marker=ORIGIN_MARKERS[orig],
                    s=point_size,
                    alpha=alpha,
                    edgecolors="none",
                    rasterized=True,
                )

        texts = []
        for sp in unique_species:
            for orig in unique_origins:
                mask = (species_arr == sp) & (origins_arr == orig)
                if not mask.any():
                    continue
                cx, cy = proj[mask].mean(axis=0)
                label = f"{ORIGIN_ABBREV[orig]}-{sp}"
                t = ax.text(
                    cx, cy, label, fontsize=10, fontstyle="italic",
                    weight="bold", ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=0.8, boxstyle="round,pad=0.2"),
                )
                texts.append(t)

        if has_adjust and texts:
            adjust_text(
                texts, ax=ax,
                expand_text=(1.2, 1.4),
                arrowprops=dict(arrowstyle="-", lw=0.4, color="gray", alpha=0.5),
            )

        ax.set_xlabel("UMAP-1", fontsize=18)
        ax.set_ylabel("UMAP-2", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend: species colors + origin markers
        species_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=species_colors[sp], markersize=7,
                   label=sp, markeredgecolor="none")
            for sp in unique_species
        ]
        origin_handles = [
            Line2D([0], [0], marker=ORIGIN_MARKERS[o], color="w",
                   markerfacecolor="0.4", markersize=7,
                   label=f"{ORIGIN_ABBREV[o]}: {o}", markeredgecolor="none")
            for o in unique_origins
        ]
        all_handles = species_handles + origin_handles

        ax.legend(
            handles=all_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(all_handles), 7),
            fontsize=10,
            frameon=True,
            framealpha=0.95,
            edgecolor="0.8",
            columnspacing=0.6,
            handletextpad=0.3,
        )

        plt.tight_layout()

        stem = f"umap_cross_regional_{pkey}_{vlm_short}"
        for ext in ["pdf", "png", "svg"]:
            path = output_dir / f"{stem}.{ext}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close(fig)


def save_metadata(
    data: dict,
    projections: dict,
    output_dir: Path,
    vlm_short: str,
    args,
):
    """Export metadata JSON for reproducibility and figure captions."""
    vit_mode = getattr(args, "vit_mode", "pretrained")
    meta = {
        "timestamp": datetime.now().isoformat(),
        "script": "umap_cross_regional.py",
        "vlm_model": vlm_short,
        "vit_mode": vit_mode,
        "vit_model": f"vit_small_patch14_dinov2.lvd142m ({vit_mode})",
        "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
        "umap_params": {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "seed": args.seed,
        },
        "sampling": {
            "split": "val",
            "max_per_origin": args.max_per_origin,
            "mode": "balanced per (species, origin)",
            "seed": args.seed,
        },
        "total_points": int(data["sbert_embeddings"].shape[0]),
        "species": sorted(set(data["species"])),
        "origins": sorted(set(data["origins"])),
        "per_species_origin_counts": data["counts"],
    }
    path = output_dir / f"umap_cross_regional_{vlm_short}_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-regional UMAP visualization (15 retrieval species)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to data/ directory")
    parser.add_argument("--vlm", type=str, default="gemma4-bf16",
                        choices=["qwen25vl", "qwen3vl", "qwen3-fp8",
                                 "qwen35-fp8", "qwen36-fp8", "gemma4-bf16"],
                        help="VLM caption embeddings to use")
    parser.add_argument("--max_per_origin", type=int, default=60,
                        help="Max samples per (species, origin) bucket")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (300 for print)")
    parser.add_argument("--point_size", type=int, default=12,
                        help="Scatter point size (3-panel)")
    parser.add_argument("--individual", action="store_true",
                        help="Also export individual per-modality panels")
    parser.add_argument("--vit_mode", type=str, default="pretrained",
                        choices=["pretrained", "finetuned"],
                        help="ViT embedding space (pretrained LVD or finetuned)")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    splits_dir = data_root / "04_evaluation" / "splits"
    anchor_dir = PROJECT_ROOT / "03_captioning" / "caption_anchors"
    sbert_dir = data_root / "04_evaluation" / "caption_embeddings" / args.vlm
    if args.vit_mode == "finetuned":
        vit_dir = data_root / "04_evaluation" / "vit_embeddings_finetuned"
    else:
        vit_dir = data_root / "04_evaluation" / "vit_embeddings"
    output_dir = data_root / "04_evaluation" / "results" / "visualization" / args.vit_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-REGIONAL UMAP — 15 Retrieval Species")
    print("=" * 70)
    print(f"  VLM model:         {args.vlm}")
    print(f"  ViT mode:          {args.vit_mode}")
    print(f"  Max per origin:    {args.max_per_origin}")
    print(f"  SBERT dir:         {sbert_dir}")
    print(f"  ViT dir:           {vit_dir}")
    print(f"  Seed:              {args.seed}")
    print(f"  DPI:               {args.dpi}")
    print(f"  Output dir:        {output_dir}")
    print()

    # ── Step 1: Load data ───────────────────────────────────────────
    print("[1/4] Loading val embeddings for 15 cross-regional species...")
    data = load_cross_regional_data(
        splits_dir, sbert_dir, vit_dir, anchor_dir,
        max_per_origin=args.max_per_origin, seed=args.seed,
    )

    n = data["sbert_embeddings"].shape[0]
    unique_sp = sorted(set(data["species"]))
    unique_or = sorted(set(data["origins"]))
    print(f"  Total points:    {n:,}")
    print(f"  Species ({len(unique_sp)}):   {', '.join(unique_sp)}")
    print(f"  Origins ({len(unique_or)}):   {', '.join(unique_or)}")
    print()

    # Per (species, origin) summary
    for sp in unique_sp:
        origins_for = [o for s, o in zip(data["species"], data["origins"]) if s == sp]
        origin_counts = defaultdict(int)
        for o in origins_for:
            origin_counts[o] += 1
        parts = [f"{ORIGIN_ABBREV[o]}={c}" for o, c in sorted(origin_counts.items())]
        print(f"    {sp:15s}  {sum(origin_counts.values()):4d}  ({', '.join(parts)})")

    # ── Step 2: UMAP projections ────────────────────────────────────
    print(f"\n[2/4] Running UMAP projections (3 modalities)...")

    print("  → Text (SBERT)...")
    proj_text = run_umap(data["sbert_embeddings"], seed=args.seed,
                         n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    print("  → Image (ViT)...")
    proj_image = run_umap(data["vit_embeddings"], seed=args.seed,
                          n_neighbors=args.n_neighbors, min_dist=args.min_dist)

    print("  → Combined (α=0.5 late fusion distance, matching retrieval)...")
    combined_dist = make_combined_distance(
        data["sbert_embeddings"], data["vit_embeddings"], alpha=0.5)
    proj_combined = run_umap(combined_dist, seed=args.seed,
                             n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                             metric="precomputed")

    projections = {
        "text": proj_text,
        "image": proj_image,
        "combined": proj_combined,
    }

    # ── Step 3: Create figures ──────────────────────────────────────
    print(f"\n[3/4] Creating 3-panel figure...")
    create_three_panel_figure(
        projections, data["species"], data["origins"],
        output_dir, args.vlm,
        dpi=args.dpi, point_size=args.point_size,
    )

    if args.individual:
        print(f"\n  Creating individual panels...")
        create_individual_panels(
            projections, data["species"], data["origins"],
            output_dir, args.vlm,
            dpi=args.dpi, point_size=args.point_size,
        )

    # ── Step 4: Save metadata ──────────────────────────────────────
    print(f"\n[4/4] Saving metadata...")
    save_metadata(data, projections, output_dir, args.vlm, args)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
