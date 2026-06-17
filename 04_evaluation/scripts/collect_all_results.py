#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Collect ALL evaluation results across 20 experiments × 5 seeds.
Extracts every eval JSON file: TS2, TS1, cross-region (both 'all' and 'overlap' variants),
and self-to-self (in-domain) cross evals.

Outputs:
  - Console: summary tables, cross-region matrices, per-seed appendix
  - data/04_evaluation/results/all_results_summary.json  (full structured data)
  - data/04_evaluation/results/all_results_summary.csv   (flat table, every metric × seed)
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data" / "04_evaluation" / "results"

# ─── Experiment definitions ───────────────────────────────────────────────────
EXPERIMENTS = OrderedDict([
    # Option A: ImageNet normalization
    ("exp01_linear_probe_all",                  {"option": "A", "region": "all",           "norm": "imagenet", "label": "Exp1: All (ImageNet)"}),
    ("exp02_linear_probe_french",               {"option": "A", "region": "french",        "norm": "imagenet", "label": "Exp2: French (ImageNet)"}),
    ("exp03_linear_probe_hungarian",            {"option": "A", "region": "hungarian",     "norm": "imagenet", "label": "Exp3: Hungarian (ImageNet)"}),
    ("exp04_linear_probe_swedish",              {"option": "A", "region": "swedish",       "norm": "imagenet", "label": "Exp4: Swedish (ImageNet)"}),
    ("exp05_linear_probe_mediterranean",        {"option": "A", "region": "mediterranean", "norm": "imagenet", "label": "Exp5: Mediterranean (ImageNet)"}),
    # Option A: Stain normalization
    ("exp06_linear_probe_french_stainnorm",     {"option": "A", "region": "french",        "norm": "stainnorm", "label": "Exp6: French (StainNorm)"}),
    ("exp07_linear_probe_hungarian_stainnorm",  {"option": "A", "region": "hungarian",     "norm": "stainnorm", "label": "Exp7: Hungarian (StainNorm)"}),
    ("exp08_linear_probe_swedish_stainnorm",    {"option": "A", "region": "swedish",       "norm": "stainnorm", "label": "Exp8: Swedish (StainNorm)"}),
    ("exp09_linear_probe_mediterranean_stainnorm", {"option": "A", "region": "mediterranean", "norm": "stainnorm", "label": "Exp9: Mediterranean (StainNorm)"}),
    ("exp10_linear_probe_all_stainnorm",        {"option": "A", "region": "all",           "norm": "stainnorm", "label": "Exp10: All (StainNorm)"}),
    # Option C: LUPI
    ("exp11_lupi_all",                          {"option": "C", "region": "all",           "norm": "imagenet", "label": "Exp11: LUPI All"}),
    ("exp12_lupi_french",                       {"option": "C", "region": "french",        "norm": "imagenet", "label": "Exp12: LUPI French"}),
    ("exp13_lupi_hungarian",                    {"option": "C", "region": "hungarian",     "norm": "imagenet", "label": "Exp13: LUPI Hungarian"}),
    ("exp14_lupi_swedish",                      {"option": "C", "region": "swedish",       "norm": "imagenet", "label": "Exp14: LUPI Swedish"}),
    ("exp15_lupi_mediterranean",                {"option": "C", "region": "mediterranean", "norm": "imagenet", "label": "Exp15: LUPI Mediterranean"}),
    # Option D: Distillation
    ("exp16_distill_all",                       {"option": "D", "region": "all",           "norm": "imagenet", "label": "Exp16: Distill All"}),
    ("exp17_distill_french",                    {"option": "D", "region": "french",        "norm": "imagenet", "label": "Exp17: Distill French"}),
    ("exp18_distill_hungarian",                 {"option": "D", "region": "hungarian",     "norm": "imagenet", "label": "Exp18: Distill Hungarian"}),
    ("exp19_distill_swedish",                   {"option": "D", "region": "swedish",       "norm": "imagenet", "label": "Exp19: Distill Swedish"}),
    ("exp20_distill_mediterranean",             {"option": "D", "region": "mediterranean", "norm": "imagenet", "label": "Exp20: Distill Mediterranean"}),
])

SEEDS = [41, 42, 43, 44, 45]
REGIONS = ["french", "hungarian", "swedish", "mediterranean"]

# All metrics to extract
ALL_METRICS = [
    "top1_accuracy", "top5_accuracy", "macro_f1", "weighted_f1",
    "macro_precision", "macro_recall",
]

# Cross-region file has nested dicts: "all" and "overlap"
CROSS_METRICS = ["top1_accuracy", "top5_accuracy", "macro_f1", "macro_precision", "macro_recall", "num_samples", "num_classes"]


# ─── Loading helpers ──────────────────────────────────────────────────────────

def load_standard_eval(path):
    """Load TS1/TS2 eval JSON → dict of metric: value."""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for m in ALL_METRICS:
        if m in data:
            result[m] = data[m]
    # Also grab valid_samples / total_samples
    result["total_samples"] = data.get("total_samples") or data.get("valid_samples")
    result["num_classes"] = data.get("num_classes_in_test")
    return result


def load_cross_eval(path):
    """Load cross-region eval JSON → {"all": {...}, "overlap": {...}, "meta": {...}}."""
    with open(path) as f:
        data = json.load(f)
    result = {"meta": {}}
    result["meta"]["source"] = data.get("source", "")
    result["meta"]["target"] = data.get("target", "")
    result["meta"]["is_in_domain"] = data.get("is_in_domain", False)
    result["meta"]["overlap_taxa_count"] = data.get("overlap_taxa_count", 0)
    result["meta"]["overlap_species"] = data.get("overlap_species", [])
    for variant in ["all", "overlap"]:
        if variant in data and isinstance(data[variant], dict):
            sub = {}
            for m in CROSS_METRICS:
                if m in data[variant]:
                    sub[m] = data[variant][m]
            result[variant] = sub
    return result


# ─── Collection ───────────────────────────────────────────────────────────────

def collect_results():
    """Collect ALL results into a structured dict."""
    all_results = {}

    for exp_name, meta in EXPERIMENTS.items():
        exp_dir = RESULTS_DIR / exp_name
        if not exp_dir.exists():
            print(f"[SKIP] {exp_name} — not found")
            continue

        exp_data = {
            "meta": dict(meta),
            "seeds": {},
        }

        for seed in SEEDS:
            seed_dir = exp_dir / f"seed_{seed}"
            if not seed_dir.exists():
                continue

            seed_data = {"ts2": None, "ts1": None, "cross": {}}

            # TS2
            ts2_path = seed_dir / "eval_ts2_expert.json"
            if ts2_path.exists():
                seed_data["ts2"] = load_standard_eval(ts2_path)

            # TS1
            ts1_path = seed_dir / "eval_ts1_legacy.json"
            if ts1_path.exists():
                seed_data["ts1"] = load_standard_eval(ts1_path)

            # Cross-region (including self-to-self = in-domain)
            for ef in sorted(seed_dir.glob("eval_cross_*_to_*.json")):
                name = ef.stem  # eval_cross_french_to_hungarian
                seed_data["cross"][name] = load_cross_eval(ef)

            exp_data["seeds"][seed] = seed_data

        all_results[exp_name] = exp_data

    return all_results


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(values):
    """Mean ± SE."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "se": se, "n": n, "values": [float(v) for v in arr]}


def aggregate_standard(all_results, eval_key):
    """Aggregate TS1 or TS2 across seeds → {exp_name: {metric: stats}}."""
    agg = {}
    for exp_name, exp_data in all_results.items():
        metric_lists = defaultdict(list)
        for seed in SEEDS:
            sd = exp_data["seeds"].get(seed)
            if not sd or not sd[eval_key]:
                continue
            for m, v in sd[eval_key].items():
                if v is not None:
                    metric_lists[m].append(v)
        if metric_lists:
            agg[exp_name] = {m: compute_stats(vals) for m, vals in metric_lists.items()}
    return agg


def aggregate_cross(all_results, variant="overlap"):
    """Aggregate cross-region across seeds → {exp_name: {cross_name: {metric: stats}}}."""
    agg = {}
    for exp_name, exp_data in all_results.items():
        cross_agg = defaultdict(lambda: defaultdict(list))
        cross_meta = {}
        for seed in SEEDS:
            sd = exp_data["seeds"].get(seed)
            if not sd:
                continue
            for cross_name, cross_data in sd["cross"].items():
                cross_meta[cross_name] = cross_data.get("meta", {})
                sub = cross_data.get(variant, {})
                for m, v in sub.items():
                    if v is not None:
                        cross_agg[cross_name][m].append(v)
        if cross_agg:
            agg[exp_name] = {
                cn: {"stats": {m: compute_stats(vals) for m, vals in mdict.items()}, "meta": cross_meta.get(cn, {})}
                for cn, mdict in cross_agg.items()
            }
    return agg


# ─── Formatting helpers ──────────────────────────────────────────────────────

def fmt(stats, width=0):
    """Format stats as 'mean±SE'."""
    if not stats:
        return "—".ljust(width) if width else "—"
    s = f"{stats['mean']:.2f}±{stats['se']:.2f}"
    return s.ljust(width) if width else s


def fmt_pct(stats, width=0):
    """Format as percentage string."""
    if not stats:
        return "—".ljust(width) if width else "—"
    s = f"{stats['mean']:.1f}±{stats['se']:.1f}%"
    return s.ljust(width) if width else s


# ─── Console output ──────────────────────────────────────────────────────────

def print_main_table(ts2_agg, ts1_agg):
    """Print TS2 main results: Top-1, Precision, Recall, F1 for all 20 experiments."""
    W = 16
    print("\n" + "=" * 120)
    print("TABLE 1: TS2 Expert — All 20 Experiments (mean±SE, 5 seeds)")
    print("=" * 120)
    hdr = f"{'Exp':<42} {'Top-1':>{W}} {'Prec':>{W}} {'Recall':>{W}} {'F1':>{W}} {'N':>6}"
    print(hdr)
    print("-" * 120)

    for exp_name in EXPERIMENTS.keys():
        ts2 = ts2_agg.get(exp_name, {})
        label = EXPERIMENTS[exp_name]["label"]
        print(f"{label:<42} "
              f"{fmt(ts2.get('top1_accuracy'), W)} "
              f"{fmt(ts2.get('macro_precision'), W)} "
              f"{fmt(ts2.get('macro_recall'), W)} "
              f"{fmt(ts2.get('macro_f1'), W)} "
              f"{ts2.get('total_samples', {}).get('mean', 0):>6.0f}")

    print("=" * 120)


def print_ts1_table(ts1_agg):
    """Print TS1 results."""
    W = 16
    print("\n" + "=" * 120)
    print("TABLE 2: TS1 Legacy — All 20 Experiments (mean±SE, 5 seeds)")
    print("=" * 120)
    hdr = f"{'Exp':<42} {'Top-1':>{W}} {'Prec':>{W}} {'Recall':>{W}} {'F1':>{W}}"
    print(hdr)
    print("-" * 120)

    for exp_name in EXPERIMENTS.keys():
        ts1 = ts1_agg.get(exp_name, {})
        label = EXPERIMENTS[exp_name]["label"]
        print(f"{label:<42} "
              f"{fmt(ts1.get('top1_accuracy'), W)} "
              f"{fmt(ts1.get('macro_precision'), W)} "
              f"{fmt(ts1.get('macro_recall'), W)} "
              f"{fmt(ts1.get('macro_f1'), W)}")

    print("=" * 120)


def print_cross_matrices(cross_agg_overlap, cross_agg_all):
    """Print cross-region matrices for each experiment group."""
    # Group experiments by option+norm
    groups = OrderedDict([
        ("Option A (ImageNet)", ["exp02_linear_probe_french", "exp03_linear_probe_hungarian", "exp04_linear_probe_swedish", "exp05_linear_probe_mediterranean"]),
        ("Option A (StainNorm)", ["exp06_linear_probe_french_stainnorm", "exp07_linear_probe_hungarian_stainnorm", "exp08_linear_probe_swedish_stainnorm", "exp09_linear_probe_mediterranean_stainnorm"]),
        ("Option C (LUPI)", ["exp12_lupi_french", "exp13_lupi_hungarian", "exp14_lupi_swedish", "exp15_lupi_mediterranean"]),
        ("Option D (Distill)", ["exp17_distill_french", "exp18_distill_hungarian", "exp19_distill_swedish", "exp20_distill_mediterranean"]),
    ])

    for variant_label, agg_data in [("OVERLAP taxa only", cross_agg_overlap), ("ALL target taxa", cross_agg_all)]:
        print(f"\n{'=' * 120}")
        print(f"CROSS-REGION MATRICES — {variant_label} (Top-1 mean±SE, 5 seeds)")
        print("=" * 120)

        for group_label, exp_names in groups.items():
            print(f"\n  ── {group_label} ──")
            # Build matrix: source region → target region
            for exp_name in exp_names:
                if exp_name not in agg_data:
                    continue
                meta = EXPERIMENTS[exp_name]
                source = meta["region"]
                cross = agg_data[exp_name]

                # Print each cross-eval for this source
                for target in REGIONS:
                    # Find the matching cross eval key
                    cross_key = f"eval_cross_{source}_to_{target}"
                    if cross_key not in cross:
                        if source == target:
                            tag = "(self)"
                        elif (source == "hungarian" and target == "mediterranean") or \
                             (source == "mediterranean" and target == "hungarian"):
                            tag = "(0 overlap)"
                        else:
                            tag = "(missing)"
                        print(f"    {source:>15} → {target:<15} {tag}")
                        continue
                    
                    cd = cross[cross_key]
                    stats = cd["stats"]
                    cm = cd.get("meta", {})
                    acc = stats.get("top1_accuracy")
                    f1 = stats.get("macro_f1")
                    prec = stats.get("macro_precision")
                    rec = stats.get("macro_recall")
                    n_samp = stats.get("num_samples")
                    n_cls = stats.get("num_classes")

                    is_self = (source == target)
                    marker = " [self]" if is_self else ""
                    n_info = ""
                    if n_samp:
                        n_info = f" n={n_samp['mean']:.0f}"
                    if n_cls:
                        n_info += f" c={n_cls['mean']:.0f}"

                    print(f"    {source:>15} → {target:<15} "
                          f"Acc={fmt(acc, 14)} F1={fmt(f1, 14)} Prec={fmt(prec, 14)} Rec={fmt(rec, 14)}"
                          f"{n_info}{marker}")


def print_per_seed_appendix(all_results):
    """Print per-seed values for every experiment on TS2 and TS1."""
    print(f"\n{'=' * 140}")
    print("APPENDIX: Per-Seed Results — TS2 Expert & TS1 Legacy")
    print("=" * 140)

    for exp_name in EXPERIMENTS.keys():
        exp_data = all_results.get(exp_name)
        if not exp_data:
            continue
        label = EXPERIMENTS[exp_name]["label"]
        print(f"\n  ── {label} ──")
        print(f"  {'Seed':>6}  {'TS2 Top-1':>10}  {'TS2 Prec':>10}  {'TS2 Rec':>10}  {'TS2 F1':>10}  "
              f"{'TS1 Top-1':>10}  {'TS1 Prec':>10}  {'TS1 Rec':>10}  {'TS1 F1':>10}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  "
              f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        vals_ts2 = defaultdict(list)
        vals_ts1 = defaultdict(list)

        for seed in SEEDS:
            sd = exp_data["seeds"].get(seed)
            if not sd:
                continue
            ts2 = sd.get("ts2") or {}
            ts1 = sd.get("ts1") or {}
            
            def g(d, k):
                v = d.get(k)
                if v is not None:
                    return f"{v:>10.2f}"
                return f"{'—':>10}"

            print(f"  {seed:>6}  {g(ts2, 'top1_accuracy')}  {g(ts2, 'macro_precision')}  "
                  f"{g(ts2, 'macro_recall')}  {g(ts2, 'macro_f1')}  "
                  f"{g(ts1, 'top1_accuracy')}  {g(ts1, 'macro_precision')}  "
                  f"{g(ts1, 'macro_recall')}  {g(ts1, 'macro_f1')}")

            for m in ALL_METRICS:
                if m in ts2 and ts2[m] is not None:
                    vals_ts2[m].append(ts2[m])
                if m in ts1 and ts1[m] is not None:
                    vals_ts1[m].append(ts1[m])

        # Mean ± SE row
        def stat_str(vals_dict, key):
            vals = vals_dict.get(key, [])
            if not vals:
                return f"{'—':>10}"
            s = compute_stats(vals)
            return f"{s['mean']:>7.2f}±{s['se']:.2f}"

        print(f"  {'M±SE':>6}  {stat_str(vals_ts2, 'top1_accuracy'):>10}  "
              f"{stat_str(vals_ts2, 'macro_precision'):>10}  {stat_str(vals_ts2, 'macro_recall'):>10}  "
              f"{stat_str(vals_ts2, 'macro_f1'):>10}  "
              f"{stat_str(vals_ts1, 'top1_accuracy'):>10}  {stat_str(vals_ts1, 'macro_precision'):>10}  "
              f"{stat_str(vals_ts1, 'macro_recall'):>10}  {stat_str(vals_ts1, 'macro_f1'):>10}")


def print_cross_seed_appendix(all_results):
    """Print per-seed cross-region values (overlap) for every regional experiment."""
    print(f"\n{'=' * 140}")
    print("APPENDIX: Per-Seed Cross-Region Results — Overlap Taxa Top-1 Accuracy")
    print("=" * 140)

    for exp_name in EXPERIMENTS.keys():
        exp_data = all_results.get(exp_name)
        if not exp_data:
            continue
        meta = EXPERIMENTS[exp_name]
        if meta["region"] == "all":
            continue  # "all" experiments have no cross-region

        label = meta["label"]
        source = meta["region"]
        
        # Collect all cross-region targets
        targets = []
        for target in REGIONS:
            cross_key = f"eval_cross_{source}_to_{target}"
            # Check if any seed has this
            for seed in SEEDS:
                sd = exp_data["seeds"].get(seed)
                if sd and cross_key in sd.get("cross", {}):
                    targets.append(target)
                    break

        if not targets:
            continue

        print(f"\n  ── {label} ──")
        hdr = f"  {'Seed':>6}"
        for t in targets:
            tag = " [self]" if t == source else ""
            hdr += f"  {'→' + t + tag:>20}"
        print(hdr)
        print(f"  {'-'*6}" + "".join(f"  {'-'*20}" for _ in targets))

        cross_vals = {t: [] for t in targets}
        for seed in SEEDS:
            sd = exp_data["seeds"].get(seed)
            if not sd:
                continue
            row = f"  {seed:>6}"
            for t in targets:
                cross_key = f"eval_cross_{source}_to_{t}"
                cd = sd.get("cross", {}).get(cross_key)
                if cd and "overlap" in cd:
                    v = cd["overlap"].get("top1_accuracy")
                    if v is not None:
                        row += f"  {v:>20.2f}"
                        cross_vals[t].append(v)
                    else:
                        row += f"  {'—':>20}"
                else:
                    row += f"  {'—':>20}"
            print(row)

        # Mean ± SE
        row = f"  {'M±SE':>6}"
        for t in targets:
            vals = cross_vals[t]
            if vals:
                s = compute_stats(vals)
                row += f"  {s['mean']:>14.2f}±{s['se']:.2f}"
            else:
                row += f"  {'—':>20}"
        print(row)


# ─── Save ─────────────────────────────────────────────────────────────────────

def build_summary(ts2_agg, ts1_agg, cross_overlap, cross_all):
    """Build the full summary dict for JSON output."""
    summary = {}
    for exp_name, meta in EXPERIMENTS.items():
        entry = {
            "meta": dict(meta),
            "ts2": ts2_agg.get(exp_name, {}),
            "ts1": ts1_agg.get(exp_name, {}),
            "cross_overlap": {},
            "cross_all": {},
        }
        if exp_name in cross_overlap:
            for cn, cd in cross_overlap[exp_name].items():
                entry["cross_overlap"][cn] = {"stats": cd["stats"], "meta": cd.get("meta", {})}
        if exp_name in cross_all:
            for cn, cd in cross_all[exp_name].items():
                entry["cross_all"][cn] = {"stats": cd["stats"], "meta": cd.get("meta", {})}
        summary[exp_name] = entry
    return summary


def save_results(summary, all_results):
    """Save results to JSON and CSV."""
    # JSON
    json_path = RESULTS_DIR / "all_results_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # CSV — every metric × every eval_type × every seed
    csv_path = RESULTS_DIR / "all_results_summary.csv"
    header = ["experiment", "option", "region", "norm", "eval_type", "variant",
              "metric", "mean", "se", "n"] + [f"seed_{s}" for s in SEEDS]
    rows = []

    for exp_name, data in summary.items():
        meta = data["meta"]
        base = [exp_name, meta["option"], meta["region"], meta["norm"]]

        # TS2
        for m, stats in data.get("ts2", {}).items():
            row = base + ["ts2_expert", "—", m, f"{stats['mean']:.4f}", f"{stats['se']:.4f}", str(stats['n'])]
            for v in stats.get("values", []):
                row.append(f"{v:.4f}")
            rows.append(row)

        # TS1
        for m, stats in data.get("ts1", {}).items():
            row = base + ["ts1_legacy", "—", m, f"{stats['mean']:.4f}", f"{stats['se']:.4f}", str(stats['n'])]
            for v in stats.get("values", []):
                row.append(f"{v:.4f}")
            rows.append(row)

        # Cross-region overlap
        for cn, cd in data.get("cross_overlap", {}).items():
            for m, stats in cd.get("stats", {}).items():
                row = base + [cn, "overlap", m, f"{stats['mean']:.4f}", f"{stats['se']:.4f}", str(stats['n'])]
                for v in stats.get("values", []):
                    row.append(f"{v:.4f}")
                rows.append(row)

        # Cross-region all
        for cn, cd in data.get("cross_all", {}).items():
            for m, stats in cd.get("stats", {}).items():
                row = base + [cn, "all", m, f"{stats['mean']:.4f}", f"{stats['se']:.4f}", str(stats['n'])]
                for v in stats.get("values", []):
                    row.append(f"{v:.4f}")
                rows.append(row)

    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"Saved CSV: {csv_path} ({len(rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Collecting ALL evaluation results")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    all_results = collect_results()

    # Aggregations
    ts2_agg = aggregate_standard(all_results, "ts2")
    ts1_agg = aggregate_standard(all_results, "ts1")
    cross_overlap = aggregate_cross(all_results, variant="overlap")
    cross_all = aggregate_cross(all_results, variant="all")

    # Console output
    print_main_table(ts2_agg, ts1_agg)
    print_ts1_table(ts1_agg)
    print_cross_matrices(cross_overlap, cross_all)
    print_per_seed_appendix(all_results)
    print_cross_seed_appendix(all_results)

    # Save
    summary = build_summary(ts2_agg, ts1_agg, cross_overlap, cross_all)
    save_results(summary, all_results)

    print("\nDone! All results collected.")


if __name__ == "__main__":
    main()
