# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Cross-model agreement.

Measures agreement between captions produced by different VLMs for the same
grain. The public-release version keeps the manuscript-facing metrics only:
lexical Jaccard overlap and optional SBERT cosine similarity with a shuffled
negative control.
"""

from collections import defaultdict
from itertools import combinations
from random import Random

import numpy as np

from .constants import HAS_SBERT, HAS_TORCH, MODELS
from .helpers import (
    allocate_sample_budget,
    new_stat,
    pair_caption_entries,
    resolve_dataset,
    run_parallel_tasks,
    safe_model_name,
    stat_add,
    stat_finalize,
    stat_merge,
    tokenize,
)


def _cross_model_slide_worker(args):
    """Process-pool worker for lexical cross-model agreement on one slide."""
    slide, entries_a, entries_b = args
    matched_pairs, unmatched_a, unmatched_b, dup_a, dup_b = pair_caption_entries(entries_a, entries_b)

    jaccard = new_stat()
    scored_pairs = 0

    for _, cap_a, cap_b in matched_pairs:
        toks_a = tokenize(cap_a)
        toks_b = tokenize(cap_b)
        if not toks_a or not toks_b:
            continue
        set_a = set(toks_a)
        set_b = set(toks_b)
        stat_add(jaccard, len(set_a & set_b) / max(len(set_a | set_b), 1))
        scored_pairs += 1

    return {
        "slide": slide,
        "matched_pairs": len(matched_pairs),
        "scored_pairs": scored_pairs,
        "unmatched_a": unmatched_a,
        "unmatched_b": unmatched_b,
        "duplicate_keys_a": dup_a,
        "duplicate_keys_b": dup_b,
        "jaccard": jaccard,
    }


def _select_sbert_device(sbert_device):
    """Resolve SBERT device string, falling back to CPU if needed."""
    if sbert_device is None:
        if HAS_TORCH:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        return "cpu"

    requested = str(sbert_device).strip() or "cpu"
    if requested.startswith("cuda"):
        if HAS_TORCH:
            import torch
            if torch.cuda.is_available():
                return requested
        print("  WARNING: CUDA requested for SBERT but unavailable; falling back to CPU")
        return "cpu"
    return requested


def _compute_sbert_for_pair(
    mk_a, mk_b, model_data, sbert_encoder, sbert_max_pairs, sbert_batch_size, common_slides
):
    """Compute SBERT cosine similarity plus shuffled negative control."""
    rng = Random(42)

    per_slide_counts = {}
    for slide in common_slides:
        pairs, _, _, _, _ = pair_caption_entries(model_data[mk_a][slide], model_data[mk_b][slide])
        per_slide_counts[slide] = len(pairs)

    budget = allocate_sample_budget(per_slide_counts, sbert_max_pairs)
    sim_global = new_stat()
    all_emb_a_list = []
    all_emb_b_list = []

    for slide in common_slides:
        k = budget.get(slide, 0)
        if k <= 0:
            continue
        pairs, _, _, _, _ = pair_caption_entries(model_data[mk_a][slide], model_data[mk_b][slide])
        if not pairs:
            continue
        if k < len(pairs):
            pairs = rng.sample(pairs, k)

        texts_a = [cap_a for _, cap_a, _ in pairs]
        texts_b = [cap_b for _, _, cap_b in pairs]
        emb_a = sbert_encoder.encode(
            texts_a,
            batch_size=sbert_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        emb_b = sbert_encoder.encode(
            texts_b,
            batch_size=sbert_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_emb_a_list.append(emb_a)
        all_emb_b_list.append(emb_b)

        cos = np.sum(emb_a * emb_b, axis=1)
        for value in cos:
            stat_add(sim_global, float(value))

    result = {
        "computed": sim_global["n"] > 0,
        "n_pairs": int(sim_global["n"]),
        "similarity": stat_finalize(sim_global),
    }
    if not result["computed"]:
        result["reason"] = "no pairs encoded after sampling"
        return result

    seeds = [42, 123, 456]
    all_emb_a = np.concatenate(all_emb_a_list, axis=0)
    all_emb_b = np.concatenate(all_emb_b_list, axis=0)
    n_ctrl = len(all_emb_b)
    true_mean = result["similarity"]["mean"]

    per_seed = []
    deltas = []
    for seed in seeds:
        ctrl_rng = np.random.RandomState(seed)
        perm_idx = ctrl_rng.permutation(n_ctrl)
        ctrl_cos = np.sum(all_emb_a * all_emb_b[perm_idx], axis=1)
        ctrl_mean = float(np.mean(ctrl_cos))
        ctrl_std = float(np.std(ctrl_cos, ddof=1)) if n_ctrl > 1 else 0.0
        delta = true_mean - ctrl_mean
        per_seed.append({
            "seed": seed,
            "mean": round(ctrl_mean, 6),
            "std": round(ctrl_std, 6),
            "delta": round(delta, 6),
        })
        deltas.append(delta)

    result["negative_control"] = {
        "strategy": "random_permutation_across_slides",
        "seeds": seeds,
        "n_pairs": int(n_ctrl),
        "mean_ctrl": round(float(np.mean([r["mean"] for r in per_seed])), 6),
        "mean_delta": round(float(np.mean(deltas)), 6),
        "std_delta": round(float(np.std(deltas, ddof=1)), 6) if len(deltas) > 1 else 0.0,
        "per_seed": per_seed,
    }
    return result


def phase_cross_model(
    model_data,
    model_dataset,
    slide_species,
    slide_dataset_truth,
    workers,
    skip_sbert=False,
    sbert_model_name="sentence-transformers/all-MiniLM-L6-v2",
    sbert_max_pairs=0,
    sbert_batch_size=256,
    sbert_device=None,
    sbert_allow_download=False,
):
    """Compute pairwise cross-model agreement matrices."""
    print("\n" + "=" * 60)
    print("Phase 4: Cross-model agreement")
    print("=" * 60)

    available_models = sorted(mk for mk in MODELS if mk in model_data and model_data[mk])
    model_pairs = list(combinations(available_models, 2))
    n_models = len(available_models)

    print(f"  Models available: {n_models} ({', '.join(safe_model_name(m) for m in available_models)})")
    print(f"  Pairwise comparisons: {len(model_pairs)}")

    if n_models < 2:
        print("  WARNING: fewer than 2 models available, skipping cross-model phase")
        return {}

    pair_results = {}
    common_slides_by_pair = {}

    for pair_idx, (mk_a, mk_b) in enumerate(model_pairs):
        pair_key = f"{mk_a}_vs_{mk_b}"
        name_a = safe_model_name(mk_a)
        name_b = safe_model_name(mk_b)
        print(f"\n  [{pair_idx + 1}/{len(model_pairs)}] {name_a} vs {name_b}")

        common_slides = sorted(set(model_data[mk_a].keys()) & set(model_data[mk_b].keys()))
        common_slides_by_pair[pair_key] = common_slides
        print(f"    Common slides: {len(common_slides)}")

        if not common_slides:
            pair_results[pair_key] = {
                "model_a": mk_a,
                "model_b": mk_b,
                "name_a": name_a,
                "name_b": name_b,
                "common_slides": 0,
                "matched_grains": 0,
            }
            continue

        tasks = [(slide, model_data[mk_a][slide], model_data[mk_b][slide]) for slide in common_slides]
        jaccard_global = new_stat()
        per_dataset = defaultdict(lambda: {"jaccard": new_stat(), "matched_pairs": 0})
        total_matched = 0
        total_scored = 0
        total_unmatched_a = 0
        total_unmatched_b = 0

        for slide_result in run_parallel_tasks(tasks, _cross_model_slide_worker, workers, f"    {name_a} vs {name_b}"):
            slide = slide_result["slide"]
            dataset = resolve_dataset(slide, model_dataset[mk_a].get(slide), slide_dataset_truth)
            total_matched += slide_result["matched_pairs"]
            total_scored += slide_result["scored_pairs"]
            total_unmatched_a += slide_result["unmatched_a"]
            total_unmatched_b += slide_result["unmatched_b"]
            stat_merge(jaccard_global, slide_result["jaccard"])
            stat_merge(per_dataset[dataset]["jaccard"], slide_result["jaccard"])
            per_dataset[dataset]["matched_pairs"] += slide_result["matched_pairs"]

        pair_results[pair_key] = {
            "model_a": mk_a,
            "model_b": mk_b,
            "name_a": name_a,
            "name_b": name_b,
            "common_slides": len(common_slides),
            "matched_grains": total_matched,
            "scored_grains": total_scored,
            "unmatched_a": total_unmatched_a,
            "unmatched_b": total_unmatched_b,
            "jaccard": stat_finalize(jaccard_global),
            "per_dataset": {
                dataset: {
                    "dataset": dataset,
                    "matched_pairs": vals["matched_pairs"],
                    "jaccard": stat_finalize(vals["jaccard"]),
                }
                for dataset, vals in sorted(per_dataset.items())
            },
        }
        print(f"    Matched: {total_matched:,}  Jaccard: {stat_finalize(jaccard_global)['mean']:.4f}")

    sbert_computed = False
    selected_sbert_device = None
    if skip_sbert:
        sbert_reason = "disabled by --skip-sbert"
    elif not HAS_SBERT:
        sbert_reason = "sentence-transformers not installed"
    else:
        sbert_reason = None
        selected_sbert_device = _select_sbert_device(sbert_device)
        print(f"\n  Loading SBERT model on {selected_sbert_device}...")
        sbert_encoder = None
        try:
            from sentence_transformers import SentenceTransformer
            sbert_encoder = SentenceTransformer(
                sbert_model_name,
                device=selected_sbert_device,
                local_files_only=not sbert_allow_download,
            )
        except TypeError:
            from sentence_transformers import SentenceTransformer
            sbert_encoder = SentenceTransformer(sbert_model_name, device=selected_sbert_device)
        except Exception as exc:
            sbert_reason = f"failed to load SBERT model: {exc}"
            print(f"  WARNING: {sbert_reason}")

        if sbert_encoder is not None:
            for pair_idx, (mk_a, mk_b) in enumerate(model_pairs):
                pair_key = f"{mk_a}_vs_{mk_b}"
                pr = pair_results[pair_key]
                if pr.get("matched_grains", 0) == 0:
                    pr["sbert"] = {"computed": False, "reason": "no matched pairs"}
                    continue
                print(f"  SBERT [{pair_idx + 1}/{len(model_pairs)}] {safe_model_name(mk_a)} vs {safe_model_name(mk_b)}...")
                sbert_result = _compute_sbert_for_pair(
                    mk_a, mk_b, model_data, sbert_encoder,
                    sbert_max_pairs, sbert_batch_size, common_slides_by_pair[pair_key],
                )
                pr["sbert"] = sbert_result
                sbert_computed = True
                if sbert_result["computed"]:
                    sim_mean = sbert_result["similarity"]["mean"]
                    delta = sbert_result.get("negative_control", {}).get("mean_delta", 0)
                    print(f"    cosine={sim_mean:.4f}  delta={delta:.4f}  n={sbert_result['n_pairs']:,}")
            del sbert_encoder

    model_labels = [safe_model_name(mk) for mk in available_models]
    jaccard_matrix = [[None] * n_models for _ in range(n_models)]
    sbert_matrix = [[None] * n_models for _ in range(n_models)]
    for i in range(n_models):
        jaccard_matrix[i][i] = 1.0
        sbert_matrix[i][i] = 1.0

    for mk_a, mk_b in model_pairs:
        pair_key = f"{mk_a}_vs_{mk_b}"
        pr = pair_results.get(pair_key, {})
        i = available_models.index(mk_a)
        j = available_models.index(mk_b)
        jaccard_val = pr.get("jaccard", {}).get("mean")
        sbert_val = pr.get("sbert", {}).get("similarity", {}).get("mean") if pr.get("sbert", {}).get("computed") else None
        jaccard_matrix[i][j] = jaccard_val
        jaccard_matrix[j][i] = jaccard_val
        sbert_matrix[i][j] = sbert_val
        sbert_matrix[j][i] = sbert_val

    cross_model = {
        "n_models": n_models,
        "n_pairs": len(model_pairs),
        "model_keys": available_models,
        "model_labels": model_labels,
        "pair_key_priority": ["id", "image_path", "mask_index", "row_index"],
        "pairs": pair_results,
        "matrix": {
            "jaccard": jaccard_matrix,
            "sbert_cosine": sbert_matrix,
        },
        "sbert_config": {
            "model": sbert_model_name,
            "computed": sbert_computed,
            "reason": sbert_reason,
            "device": selected_sbert_device if sbert_reason is None else None,
            "allow_download": bool(sbert_allow_download),
            "max_pairs": int(sbert_max_pairs),
        },
    }

    print("\n  === Cross-Model Agreement Matrix ===")
    print(f"  Models: {', '.join(model_labels)}")
    header = "  {:>20s}".format("") + "".join(f"  {lbl:>14s}" for lbl in model_labels)
    print("\n  Jaccard matrix:")
    print(header)
    for i, lbl in enumerate(model_labels):
        row_vals = "".join(
            f"  {jaccard_matrix[i][j]:>14.4f}" if jaccard_matrix[i][j] is not None else f"  {'N/A':>14s}"
            for j in range(n_models)
        )
        print(f"  {lbl:>20s}{row_vals}")

    if sbert_computed:
        print("\n  SBERT cosine matrix:")
        print(header)
        for i, lbl in enumerate(model_labels):
            row_vals = "".join(
                f"  {sbert_matrix[i][j]:>14.4f}" if sbert_matrix[i][j] is not None else f"  {'N/A':>14s}"
                for j in range(n_models)
            )
            print(f"  {lbl:>20s}{row_vals}")

    return cross_model
