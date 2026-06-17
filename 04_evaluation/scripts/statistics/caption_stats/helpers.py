# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Generic statistics helpers, tokenization, and utility functions.

These are pure-compute functions with no side effects, shared across
all phase modules.  All stat accumulators use the Welford-style
running-stat pattern to support streaming / parallel reduction.
"""

import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from tqdm import tqdm

from .constants import TOKEN_RE, MODELS


# ── Running-statistics accumulator ────────────────────────────────────
# Each stat is a dict with keys: n, mean, M2, min, max
# This allows single-pass mean + std + min/max without storing values.


def new_stat():
    """Create a new running-statistics accumulator."""
    return {"n": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}


def stat_add(stat, value):
    """Add a single observation (Welford online algorithm)."""
    stat["n"] += 1
    delta = value - stat["mean"]
    stat["mean"] += delta / stat["n"]
    delta2 = value - stat["mean"]
    stat["M2"] += delta * delta2
    if value < stat["min"]:
        stat["min"] = value
    if value > stat["max"]:
        stat["max"] = value


def stat_merge(dst, src):
    """Merge two running-stat accumulators (parallel-safe reduction)."""
    if src["n"] == 0:
        return
    if dst["n"] == 0:
        dst.update(src)
        return
    n = dst["n"] + src["n"]
    delta = src["mean"] - dst["mean"]
    dst["M2"] += src["M2"] + delta * delta * dst["n"] * src["n"] / n
    dst["mean"] = (dst["mean"] * dst["n"] + src["mean"] * src["n"]) / n
    dst["n"] = n
    dst["min"] = min(dst["min"], src["min"])
    dst["max"] = max(dst["max"], src["max"])


def stat_finalize(stat):
    """Convert running accumulator to a presentation dict."""
    n = stat["n"]
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    variance = stat["M2"] / n if n > 0 else 0.0
    return {
        "n": n,
        "mean": round(stat["mean"], 6),
        "std": round(math.sqrt(max(variance, 0.0)), 6),
        "min": round(stat["min"], 6),
        "max": round(stat["max"], 6),
    }


# ── Convenience helpers ──────────────────────────────────────────────

def pct(count, total):
    """Return percentage rounded to 4 decimal places."""
    return round(100.0 * count / total, 4) if total > 0 else 0.0


def tokenize(text):
    """Lowercase tokenization: sequences of ``[a-z]`` possibly with hyphens."""
    return TOKEN_RE.findall(text.lower())


def compute_hist_percentiles(hist, probs=(5, 25, 50, 75, 95)):
    """Compute percentiles from an integer histogram ``{value: count}``."""
    if not hist:
        return {f"p{p}": 0 for p in probs}

    total = sum(hist.values())
    if total <= 0:
        return {f"p{p}": 0 for p in probs}

    ordered = sorted(hist.items())
    thresholds = {p: int(math.ceil((p / 100.0) * total)) for p in probs}
    out = {}
    csum = 0
    probs_sorted = sorted(probs)
    pi = 0
    for value, count in ordered:
        csum += count
        while pi < len(probs_sorted) and csum >= thresholds[probs_sorted[pi]]:
            out[f"p{probs_sorted[pi]}"] = value
            pi += 1
        if pi >= len(probs_sorted):
            break

    last_val = ordered[-1][0]
    for p in probs:
        out.setdefault(f"p{p}", last_val)
    return out


def safe_model_name(model_key):
    """Return the human-readable model name for a model key."""
    return MODELS.get(model_key, model_key)


# ── Parallel execution ───────────────────────────────────────────────

def run_parallel_tasks(tasks, worker_fn, workers, desc):
    """Run *tasks* through *worker_fn* using a process pool.

    Falls back to ThreadPoolExecutor when process-pool semaphores are blocked
    (e.g. inside certain container runtimes).
    Yields results as they complete.
    """
    if not tasks:
        return []

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker_fn, task) for task in tasks]
            return [
                fut.result()
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False)
            ]
    except (PermissionError, OSError, BrokenProcessPool):
        print(f"  WARNING: ProcessPool unavailable, falling back to ThreadPool.")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_fn, task) for task in tasks]
        return [
            fut.result()
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, leave=False)
        ]


# ── Dataset resolution ───────────────────────────────────────────────

def infer_dataset_from_slide(slide):
    """Guess dataset name from slide filename patterns."""
    if slide.startswith("mediterranean_"):
        return "mediterranean"
    if slide.startswith("hun_") or slide.startswith("Ambrosia-Iva"):
        return "hungarian"

    swedish_patterns = ["_40x_ZS", "_20x_ZS", "layers_40x", "merged_reference", "mm_circle"]
    if any(p in slide for p in swedish_patterns):
        return "swedish"

    return "french"


def resolve_dataset(slide, discovered_dataset, slide_dataset_truth):
    """Resolve dataset for a slide: truth table → discovered → inferred."""
    if slide_dataset_truth and slide in slide_dataset_truth:
        return slide_dataset_truth[slide]
    if discovered_dataset:
        return discovered_dataset
    return infer_dataset_from_slide(slide)


# ── Pair-key extraction (cross-model matching) ───────────────────────

def extract_pair_key(obj, row_idx):
    """Extract a deterministic matching key from a JSONL record.

    Priority: ``id`` → ``image_path`` → ``mask_index`` → ``row_index``.
    Returns a string like ``"id:12345"`` or ``"row:00000001"``.
    """
    for field in ("id", "image_path", "mask_index"):
        value = obj.get(field)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return f"{field}:{value_str}"
    return f"row:{row_idx:08d}"


def pair_caption_entries(entries_a, entries_b):
    """Match caption entries from two models by pre-computed pair key.

    Each entry is ``(caption, pair_key_string)``.

    Returns ``(matched, unmatched_a, unmatched_b, dup_a, dup_b)`` where
    ``matched`` is a **deterministically sorted** list of
    ``(key, caption_a, caption_b)``.
    """
    lookup_a = {}
    lookup_b = {}
    dup_a = 0
    dup_b = 0

    for caption, key in entries_a:
        if key in lookup_a:
            dup_a += 1
        lookup_a[key] = caption

    for caption, key in entries_b:
        if key in lookup_b:
            dup_b += 1
        lookup_b[key] = caption

    keys_a = set(lookup_a.keys())
    keys_b = set(lookup_b.keys())
    matched_keys = sorted(keys_a & keys_b)  # deterministic (codex fix)

    matched = [(k, lookup_a[k], lookup_b[k]) for k in matched_keys]
    unmatched_a = len(keys_a) - len(matched_keys)
    unmatched_b = len(keys_b) - len(matched_keys)

    return matched, unmatched_a, unmatched_b, dup_a, dup_b


# ── Budget allocation ────────────────────────────────────────────────

def allocate_sample_budget(per_slide_counts, max_pairs):
    """Allocate sample counts per slide proportionally."""
    if max_pairs <= 0:
        return {slide: count for slide, count in per_slide_counts.items()}

    total = sum(per_slide_counts.values())
    if total <= max_pairs:
        return {slide: count for slide, count in per_slide_counts.items()}

    exact = {slide: (count / total) * max_pairs for slide, count in per_slide_counts.items()}
    alloc = {slide: int(math.floor(v)) for slide, v in exact.items()}

    remaining = max_pairs - sum(alloc.values())
    order = sorted(exact.keys(), key=lambda s: exact[s] - alloc[s], reverse=True)
    for slide in order:
        if remaining <= 0:
            break
        if per_slide_counts[slide] > alloc[slide]:
            alloc[slide] += 1
            remaining -= 1

    return alloc


def allocate_budget_with_min(counts, total, min_each=0):
    """Allocate integer budget across keys proportionally with optional minimum."""
    alloc = {k: 0 for k in counts}
    keys = [k for k, v in counts.items() if v > 0]
    if total <= 0 or not keys:
        return alloc

    remaining = int(total)

    # Mandatory minimum per non-empty key when feasible.
    if min_each > 0 and remaining > 0:
        for _ in range(min_each):
            candidates = [k for k in keys if alloc[k] < counts[k]]
            if not candidates or remaining <= 0:
                break
            if remaining < len(candidates):
                top = sorted(candidates, key=lambda k: counts[k], reverse=True)[:remaining]
                for k in top:
                    alloc[k] += 1
                remaining = 0
                break
            for k in candidates:
                alloc[k] += 1
                remaining -= 1
                if remaining <= 0:
                    break

    if remaining <= 0:
        return alloc

    rem_capacity = {k: max(counts[k] - alloc[k], 0) for k in keys}
    cap_sum = sum(rem_capacity.values())
    if cap_sum <= 0:
        return alloc

    exact = {k: remaining * rem_capacity[k] / cap_sum for k in keys}
    base_add = {k: min(int(math.floor(exact[k])), rem_capacity[k]) for k in keys}
    for k, v in base_add.items():
        alloc[k] += v
    remaining -= sum(base_add.values())

    if remaining <= 0:
        return alloc

    order = sorted(
        keys,
        key=lambda k: (exact[k] - math.floor(exact[k]), rem_capacity[k]),
        reverse=True,
    )
    oi = 0
    while remaining > 0 and any(alloc[k] < counts[k] for k in keys):
        k = order[oi % len(order)]
        oi += 1
        if alloc[k] >= counts[k]:
            continue
        alloc[k] += 1
        remaining -= 1

    return alloc


# ── Markdown formatting helpers ──────────────────────────────────────

def fmt_num(v, decimals=1, default="-"):
    """Format a numeric value for markdown tables."""
    if v is None:
        return default
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return default


def fmt_int(v, default="-"):
    """Format an integer value for markdown tables."""
    try:
        return f"{int(v):,}"
    except Exception:
        return default
