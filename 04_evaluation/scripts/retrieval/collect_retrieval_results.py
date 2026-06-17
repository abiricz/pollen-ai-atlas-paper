#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Collect Retrieval Results for Paper
====================================

Reads the retrieval JSON outputs and the negative control, then produces:
  1. Console: formatted tables matching paper layout
  2. retrieval_summary.json — structured summary for docs/paper
  3. retrieval_summary.csv — flat table for supplementary data

Tables produced:
  - Table R1: Global results (ALL + FULL × image/text/combined × Gemma4)
  - Table R2: Per-species results (FULL mode, Gemma4 text — distractor-rich cross-slide)
  - Table R3: Negative control comparison (real vs shuffled)

Usage:
    python collect_retrieval_results.py
"""

import json
import sys
import csv
import os
from pathlib import Path
from datetime import datetime

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "data" / "04_evaluation" / "results" / "retrieval"
)
RESULTS_DIR = Path(
    os.environ.get("RETRIEVAL_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))
)

VLM_FILES = {
    "gemma4-bf16": RESULTS_DIR / "retrieval_gemma4-bf16.json",
    "qwen25vl":    RESULTS_DIR / "retrieval_qwen25vl.json",
    "qwen3-fp8":   RESULTS_DIR / "retrieval_qwen3-fp8.json",
    "qwen35-fp8":  RESULTS_DIR / "retrieval_qwen35-fp8.json",
    "qwen36-fp8":  RESULTS_DIR / "retrieval_qwen36-fp8.json",
}
NEG_CTRL_FILE = RESULTS_DIR / "retrieval_negative_control.json"
OUTPUT_JSON   = RESULTS_DIR / "retrieval_summary.json"
OUTPUT_CSV    = RESULTS_DIR / "retrieval_summary.csv"
OUTPUT_QUERY_CSV = RESULTS_DIR / "retrieval_query_provenance.csv"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fmt(v, width=7, precision=3):
    """Format a float metric value."""
    if v is None:
        return " " * width
    return f"{v:{width}.{precision}f}"


def fmt_ci(v, ci_lo, ci_hi, width=7, precision=3):
    """Format a metric value with optional CI."""
    if v is None:
        return " " * width
    base = f"{v:{width}.{precision}f}"
    if ci_lo is not None:
        base += f" [{ci_lo:.2f}-{ci_hi:.2f}]"
    return base


def load_vlm_results(path):
    """Load retrieval JSON for one VLM."""
    with open(path) as f:
        return json.load(f)


def load_negative_control():
    """Load negative control JSON if available."""
    if NEG_CTRL_FILE.exists():
        with open(NEG_CTRL_FILE) as f:
            return json.load(f)
    return None


# ─── Table R1: Global summary ────────────────────────────────────────────────

def print_global_table(all_data):
    """Print global results table (matches paper Table layout)."""
    print("\n" + "=" * 90)
    print("  TABLE R1: GLOBAL RETRIEVAL RESULTS")
    print("  (macro-averaged across 15 cross-regional species)")
    print("=" * 90)

    # Column headers
    metrics = ["P@1", "P@5", "P@10", "P@20", "MRR", "mAP@20"]
    conditions = [
        ("all",  "image",    None,  "ALL / Image"),
        ("all",  "text",     None,  "ALL / Text"),
        ("all",  "combined", 0.50,  "ALL / Combined"),
        ("full", "image",    None,  "FULL / Image"),
        ("full", "text",     None,  "FULL / Text"),
        ("full", "combined", 0.50,  "FULL / Combined"),
        ("cross_regional", "image",    None,  "CROSS-REG / Image"),
        ("cross_regional", "text",     None,  "CROSS-REG / Text"),
        ("cross_regional", "combined", 0.50,  "CROSS-REG / Combined"),
    ]

    # Header
    hdr = f"  {'Condition':<25s}"
    for vlm in all_data:
        for m in metrics:
            hdr += f" {m:>7s}"
        hdr += "  |"
    print(f"\n  {'':25s}", end="")
    for vlm in all_data:
        label = "Gemma-4-31B" if "gemma4" in vlm else vlm
        span = len(metrics) * 8 + 2
        print(f" {label:^{span}s}", end="")
    print()

    print(f"  {'Condition':<25s}", end="")
    for vlm in all_data:
        for m in metrics:
            print(f" {m:>7s}", end="")
        print("  |", end="")
    print()
    print("  " + "-" * (25 + len(all_data) * (len(metrics) * 8 + 4)))
    # Rows
    for mode, modality, alpha, label in conditions:
        key = f"{mode}_{modality}"
        if alpha is not None:
            key += f"_a{alpha:.2f}"

        print(f"  {label:<25s}", end="")
        for vlm, data in all_data.items():
            results = data.get("results", {})
            if key in results:
                g = results[key]["global"]
                ci = results[key].get("global_ci_95", {})
                for m in metrics:
                    val = g.get(m)
                    ci_bounds = ci.get(m)
                    if ci_bounds and m in ("P@20", "MRR", "mAP@20"):
                        print(f" {fmt_ci(val, ci_bounds[0], ci_bounds[1])}", end="")
                    else:
                        print(f" {fmt(val)}", end="")
            else:
                for m in metrics:
                    print(f" {'--':>7s}", end="")
            print("  |", end="")
        print()

    # nQ — show mode-specific query counts
    print()
    query_modes = [
        ("all",            "Queries ALL (img/txt)"),
        ("full",           "Queries FULL (img/txt)"),
        ("cross_regional", "Queries CR (img/txt)"),
    ]
    for qmode, qlabel in query_modes:
        print(f"  {qlabel:<25s}", end="")
        for vlm, data in all_data.items():
            r = data.get("results", {})
            img_q = r.get(f"{qmode}_image", {}).get("config", {}).get("n_queries", "?")
            txt_q = r.get(f"{qmode}_text", {}).get("config", {}).get("n_queries", "?")
            span = len(metrics) * 8 + 2
            print(f" {f'{img_q} / {txt_q}':^{span}s}", end="")
        print()


# ─── Table R2: Per-species (FULL text, Qwen2.5-VL) ──────────────────────────

def print_species_table(data, key="full_text"):
    """Print per-species breakdown for a given retrieval condition."""
    # Determine label from key
    if key.startswith("cross_regional"):
        mode_label = "CROSS-REGIONAL"
        mode_desc = "PRIMARY cross-domain benchmark: leave-origin-out retrieval"
    elif key.startswith("full"):
        mode_label = "FULL"
        mode_desc = "distractor-rich cross-slide retrieval with expert morphological queries"
    else:
        mode_label = key.upper()
        mode_desc = ""

    print("\n" + "=" * 100)
    print(f"  TABLE R2: PER-SPECIES RESULTS — {mode_label} mode, Gemma-4-31B text retrieval")
    if mode_desc:
        print(f"  ({mode_desc})")
    print("=" * 100)

    results = data.get("results", {}).get(key, {})
    if not results:
        print("  [No results for this condition]")
        return

    per_species = results.get("per_species", {})
    per_query = results.get("per_query", [])
    metrics = ["P@1", "P@5", "P@10", "P@20", "MRR", "mAP@20"]

    # Compute mean total_relevant per species from per_query
    sp_nrel = {}
    for q in per_query:
        sp_name = q.get("query_species")
        tr = q.get("total_relevant")
        if sp_name and tr is not None:
            sp_nrel.setdefault(sp_name, []).append(tr)
    sp_mean_nrel = {sp: int(np.mean(vs)) for sp, vs in sp_nrel.items()}

    print(f"\n  {'Species':<22s}", end="")
    for m in metrics:
        print(f" {m:>7s}", end="")
    print(f" {'#Q':>4s}  {'nRel':>8s}  {'Origins'}")
    print("  " + "-" * 100)

    for sp, m in sorted(per_species.items()):
        print(f"  {sp:<22s}", end="")
        for mk in metrics:
            print(f" {fmt(m.get(mk))}", end="")
        ostr = ",".join(o[:3] for o in m.get("origins", []))
        nrel_str = f"{sp_mean_nrel[sp]:>8,}" if sp in sp_mean_nrel else "       -"
        print(f" {m.get('n_queries', 0):>4d}  {nrel_str}  {ostr}")

    # Global row
    g = results.get("global", {})
    ci = results.get("global_ci_95", {})
    print("  " + "-" * 95)
    print(f"  {'MACRO AVERAGE':<22s}", end="")
    for mk in metrics:
        val_str = fmt(g.get(mk))
        if mk in ci:
            val_str += f" [{ci[mk][0]:.3f}-{ci[mk][1]:.3f}]"
        print(f" {val_str}", end="")
    print(f" {g.get('n_queries', 0):>4d}         ({g.get('n_species', 0)} species)")


# ─── Table R3: Negative control comparison ───────────────────────────────────

def print_negative_control_table(neg_ctrl, all_data):
    """Print real vs shuffled comparison."""
    print("\n" + "=" * 90)
    print("  TABLE R3: NEGATIVE CONTROL — Species Label Shuffle")
    print("  Real embeddings, shuffled species labels. Averaged over 3 seeds.")
    print("  Permutation sanity check: validates retrieval measures genuine species signal.")
    print("=" * 90)

    if neg_ctrl is None:
        print("  [No negative control results found]")
        return

    # Support both old format (top-level averaged/vlm_for_text)
    # and new format (per_vlm dict)
    per_vlm = neg_ctrl.get("per_vlm")
    if per_vlm is None:
        # Old format: single VLM
        vlm0 = neg_ctrl.get("vlm_for_text", "qwen25vl")
        per_vlm = {vlm0: {
            "averaged": neg_ctrl.get("averaged", {}),
            "per_seed": neg_ctrl.get("per_seed", {}),
        }}

    conditions = [
        ("all_image",          "ALL / Image"),
        ("all_text",           "ALL / Text"),
        ("all_combined_a0.50", "ALL / Combined"),
        ("full_image",         "FULL / Image"),
        ("full_text",          "FULL / Text"),
        ("full_combined_a0.50","FULL / Combined"),
        ("cross_regional_image",          "CROSS-REG / Image"),
        ("cross_regional_text",           "CROSS-REG / Text"),
        ("cross_regional_combined_a0.50", "CROSS-REG / Combined"),
    ]

    metrics = ["P@1", "P@20", "MRR", "mAP@20"]

    for vlm_neg, vlm_neg_data in per_vlm.items():
        averaged = vlm_neg_data.get("averaged", {})
        real_data = all_data.get(vlm_neg, {}).get("results", {})
        vlm_label = "Gemma-4-31B" if "gemma4" in vlm_neg else vlm_neg

        print(f"\n  --- {vlm_label} ({vlm_neg}) ---")

        print(f"\n  {'Condition':<25s}", end="")
        for label in ["REAL", "SHUFFLED", "Δ (collapse)"]:
            for m in metrics:
                print(f" {m:>7s}", end="")
            print("  |", end="")
        print()
        print("  " + "-" * (25 + 3 * (len(metrics) * 8 + 4)))

        for key, label in conditions:
            print(f"  {label:<25s}", end="")

            # Real result
            real_vals = {}
            if key in real_data:
                g = real_data[key]["global"]
                for m in metrics:
                    v = g.get(m, 0)
                    real_vals[m] = v
                    print(f" {fmt(v)}", end="")
            else:
                for m in metrics:
                    print(f" {'--':>7s}", end="")
            print("  |", end="")

            # Shuffled result
            shuf_vals = {}
            if key in averaged:
                for m in metrics:
                    v = averaged[key].get(m, 0)
                    shuf_vals[m] = v
                    print(f" {fmt(v)}", end="")
            else:
                for m in metrics:
                    print(f" {'--':>7s}", end="")
            print("  |", end="")

            # Delta
            for m in metrics:
                rv = real_vals.get(m)
                sv = shuf_vals.get(m)
                if rv is not None and sv is not None:
                    delta = rv - sv
                    sign = "+" if delta >= 0 else ""
                    print(f" {sign}{delta:>6.3f}", end="")
                else:
                    print(f" {'--':>7s}", end="")
            print("  |")

    print()
    print(f"  Seeds: {neg_ctrl.get('seeds', [])}")
    print(f"  VLMs tested: {neg_ctrl.get('vlms', [neg_ctrl.get('vlm_for_text', '?')])}")


# ─── Build summary JSON ─────────────────────────────────────────────────────

def build_summary(all_data, neg_ctrl):
    """Build structured summary for docs/paper."""
    summary = {
        "experiment": "cross_regional_multimodal_retrieval",
        "version": "2.2",
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Retrieval evaluation: can we find the same pollen species "
            "across a million-scale corpus spanning 4 geographic origins? "
            "Tested with image queries (one-shot exemplar), text queries "
            "(expert morphological descriptors), and late fusion. "
            "Includes negative control (label shuffle) to validate signal."
        ),
        "corpus": {
            "total_samples": None,
            "slides": None,
            "species_tested": 15,
        },
        "global_results": {},
        "per_species": {},
        "negative_control": {},
    }

    # Fill from first VLM data
    for vlm, data in all_data.items():
        vlm_label = "Gemma-4-31B" if "gemma4" in vlm else vlm

        # Corpus info (same for both VLMs)
        if summary["corpus"]["total_samples"] is None:
            ss = data.get("slides_summary", {})
            # total_samples = actual grain count from corpus, NOT slide count.
            # Get from first result config's corpus_size field.
            corpus_size = None
            for key, result in data.get("results", {}).items():
                cs = result.get("config", {}).get("corpus_size")
                if cs is not None:
                    corpus_size = cs
                    break
            summary["corpus"]["total_samples"] = corpus_size
            summary["corpus"]["slides"] = ss.get("corpus_slides")
            summary["corpus"]["species_tested"] = ss.get("cross_regional_species", 15)
            summary["corpus"]["species_list"] = ss.get("species_list", [])

        # Global results
        for key, result in data.get("results", {}).items():
            g = result.get("global", {})
            ci = result.get("global_ci_95", {})
            label = f"{vlm_label} / {key}"
            entry = {
                "P@1": g.get("P@1"),
                "P@5": g.get("P@5"),
                "P@10": g.get("P@10"),
                "P@20": g.get("P@20"),
                "MRR": g.get("MRR"),
                "mAP@20": g.get("mAP@20", g.get("mAP")),
                "n_queries": g.get("n_queries"),
                "n_species": g.get("n_species"),
            }
            if ci:
                entry["ci_95"] = ci
            summary["global_results"][label] = entry

        # Per-species (primary VLM: all modes)
        if vlm == list(all_data.keys())[0]:  # first (primary) VLM
            for key in ["full_text", "full_image", "full_combined_a0.50",
                         "all_text", "all_image", "all_combined_a0.50",
                         "cross_regional_text", "cross_regional_image",
                         "cross_regional_combined_a0.50"]:
                result = data.get("results", {}).get(key, {})
                ps = result.get("per_species", {})
                if ps:
                    summary["per_species"][key] = ps

    # Negative control
    if neg_ctrl:
        # Support both old and new format
        per_vlm = neg_ctrl.get("per_vlm")
        if per_vlm is None:
            vlm0 = neg_ctrl.get("vlm_for_text", "qwen25vl")
            per_vlm = {vlm0: {"averaged": neg_ctrl.get("averaged", {})}}

        summary["negative_control"] = {
            "description": neg_ctrl.get("description"),
            "seeds": neg_ctrl.get("seeds"),
            "vlms": list(per_vlm.keys()),
            "per_vlm_averaged": {
                vlm: vlm_data.get("averaged", {})
                for vlm, vlm_data in per_vlm.items()
            },
            "per_vlm_averaged_ci_95": {
                vlm: vlm_data.get("averaged_ci_95", {})
                for vlm, vlm_data in per_vlm.items()
            },
        }

    return summary


# ─── CSV export ──────────────────────────────────────────────────────────────

def write_csv(all_data, neg_ctrl):
    """Write flat CSV with all results."""
    rows = []
    header = ["vlm", "mode", "modality", "alpha", "metric", "value",
              "ci_lo", "ci_hi",
              "n_queries", "n_species", "source"]

    for vlm, data in all_data.items():
        for key, result in data.get("results", {}).items():
            g = result.get("global", {})
            ci = result.get("global_ci_95", {})
            cfg = result.get("config", {})
            mode = cfg.get("mode", key.split("_")[0])
            modality = cfg.get("modality", key.split("_")[1] if "_" in key else "")
            alpha = cfg.get("alpha")

            for metric_name, metric_val in g.items():
                if isinstance(metric_val, (int, float)):
                    ci_bounds = ci.get(metric_name)
                    ci_lo = f"{ci_bounds[0]:.6f}" if ci_bounds else ""
                    ci_hi = f"{ci_bounds[1]:.6f}" if ci_bounds else ""
                    rows.append([
                        vlm, mode, modality,
                        f"{alpha:.2f}" if alpha else "",
                        metric_name, f"{metric_val:.6f}",
                        ci_lo, ci_hi,
                        g.get("n_queries", ""),
                        g.get("n_species", ""),
                        "real",
                    ])

    # Negative control
    if neg_ctrl:
        # Support both old format (top-level averaged) and new (per_vlm)
        per_vlm = neg_ctrl.get("per_vlm")
        if per_vlm is None:
            vlm0 = neg_ctrl.get("vlm_for_text", "")
            per_vlm = {vlm0: {"averaged": neg_ctrl.get("averaged", {})}}

        for vlm_neg, vlm_neg_data in per_vlm.items():
            averaged = vlm_neg_data.get("averaged", {})
            for key, metrics in averaged.items():
                # Parse mode and modality from key, handling multi-word modes
                if key.startswith("cross_regional_"):
                    mode = "cross_regional"
                    rest = key[len("cross_regional_"):]
                else:
                    parts = key.split("_", 1)
                    mode = parts[0]
                    rest = parts[1] if len(parts) > 1 else ""
                modality = rest.split("_")[0] if rest else ""
                for metric_name, metric_val in metrics.items():
                    if isinstance(metric_val, (int, float)):
                        rows.append([
                            vlm_neg,
                            mode, modality, "",
                            metric_name, f"{metric_val:.6f}",
                            "", "",  # no CIs for shuffled
                            "", "", "shuffled",
                        ])

    with open(OUTPUT_CSV, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"\n  CSV saved: {OUTPUT_CSV}")


def write_query_provenance_csv(all_data):
    """Write one row per evaluated query with provenance and probe pool details."""
    header = [
        "vlm",
        "condition_key",
        "mode",
        "modality",
        "alpha",
        "query_slide",
        "query_species",
        "query_origin",
        "query_image_path",
        "query_text",
        "query_text_source",
        "n_grains",
        "n_corpus_searched",
        "n_probed_slides",
        "n_excluded",
        "total_relevant",
        "excluded_slides",
        "target_slides",
        "probed_against_slides",
        "pool_composition",
        "top10_species",
        "top10_scores",
        "P@1",
        "P@5",
        "P@10",
        "P@20",
        "MRR",
        "mAP@20",
    ]

    rows = []
    for vlm, data in all_data.items():
        for key, result in data.get("results", {}).items():
            cfg = result.get("config", {})
            mode = cfg.get("mode", "")
            modality = cfg.get("modality", "")
            alpha = cfg.get("alpha")

            for q in result.get("per_query", []):
                pool_composition = q.get("pool_composition", [])
                probed_slides = q.get("probed_against_slides")
                if not probed_slides:
                    probed_slides = [str(pc.get("slide")) for pc in pool_composition]
                pool_compact = " || ".join(
                    f"{pc.get('slide')}::{pc.get('species')}::"
                    f"{pc.get('n_grains')}::{int(bool(pc.get('is_target')))}"
                    for pc in pool_composition
                )

                rows.append([
                    vlm,
                    key,
                    mode,
                    modality,
                    f"{alpha:.2f}" if isinstance(alpha, (int, float)) else "",
                    q.get("query_slide", ""),
                    q.get("query_species", ""),
                    q.get("query_origin", ""),
                    q.get("query_image_path", ""),
                    q.get("query_text", q.get("query_text_excerpt", "")),
                    q.get("query_text_source", ""),
                    q.get("n_grains", ""),
                    q.get("n_corpus_searched", ""),
                    q.get("n_probed_slides", len(probed_slides)),
                    q.get("n_excluded", ""),
                    q.get("total_relevant", ""),
                    " || ".join(q.get("excluded_slides", [])),
                    " || ".join(q.get("target_slides", [])),
                    " || ".join(probed_slides),
                    pool_compact,
                    " || ".join(q.get("top10_species", [])),
                    " || ".join(str(v) for v in q.get("top10_scores", [])),
                    q.get("P@1", ""),
                    q.get("P@5", ""),
                    q.get("P@10", ""),
                    q.get("P@20", ""),
                    q.get("MRR", ""),
                    q.get("mAP@20", ""),
                ])

    with open(OUTPUT_QUERY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  Query CSV saved: {OUTPUT_QUERY_CSV}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  RETRIEVAL RESULTS COLLECTOR")
    print("=" * 70)

    # Load VLM results
    all_data = {}
    for vlm, path in VLM_FILES.items():
        if path.exists():
            data = load_vlm_results(path)
            all_data[vlm] = data
            print(f"  Loaded: {path.name} "
                  f"(v{data['experiment'].get('version', '?')}, "
                  f"{data['experiment'].get('timestamp', '?')[:19]})")
        else:
            print(f"  [MISSING] {path.name}")

    if not all_data:
        print("\n  No retrieval results found. Run retrieval_experiments.py first.")
        sys.exit(1)

    # Load negative control
    neg_ctrl = load_negative_control()
    if neg_ctrl:
        print(f"  Loaded: {NEG_CTRL_FILE.name} "
              f"(seeds={neg_ctrl.get('seeds')}, "
              f"{neg_ctrl.get('timestamp', '?')[:19]})")
    else:
        print(f"  [MISSING] {NEG_CTRL_FILE.name} — run with --negative_control")

    # Print tables
    print_global_table(all_data)

    # Per-species for both benchmarks
    # Per-species for the primary VLM
    primary_vlm = "gemma4-bf16" if "gemma4-bf16" in all_data else next(iter(all_data), None)
    if primary_vlm:
        print_species_table(all_data[primary_vlm], key="full_text")
        print_species_table(all_data[primary_vlm], key="cross_regional_text")

    # Negative control
    if neg_ctrl:
        print_negative_control_table(neg_ctrl, all_data)

    # Save summary JSON
    summary = build_summary(all_data, neg_ctrl)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary JSON saved: {OUTPUT_JSON}")

    # Save CSV
    write_csv(all_data, neg_ctrl)
    write_query_provenance_csv(all_data)

    print("\n  Done.")


if __name__ == "__main__":
    main()
