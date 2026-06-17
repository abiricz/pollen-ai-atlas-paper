#!/usr/bin/env python3
# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.
"""
Collect and summarize results from all classification experiments.

This script scans experiment result folders and generates:
1. Training summary table (train_acc, val_acc, epochs, time)
2. Test set evaluation tables (TS1, TS2) with precision, recall, F1
3. Cross-region generalization matrices (detailed by source→target)
4. Comparison: ImageNet vs Stain Normalization
5. Checkpoint analysis (best vs final epoch)

Usage:
    python collect_experiment_results.py
    python collect_experiment_results.py --output results_summary.md
    python collect_experiment_results.py --format json --output results.json
    python collect_experiment_results.py --detailed  # Include per-experiment cross tables
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Default paths
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
RESULTS_DIR = DATA_ROOT / "04_evaluation" / "results"

# Experiment order for display
EXPERIMENT_ORDER = [
    "exp01_linear_probe_all",
    "exp02_linear_probe_french",
    "exp03_linear_probe_hungarian",
    "exp04_linear_probe_swedish",
    "exp05_linear_probe_mediterranean",
    "exp06_linear_probe_french_stainnorm",
    "exp07_linear_probe_hungarian_stainnorm",
    "exp08_linear_probe_swedish_stainnorm",
    "exp09_linear_probe_mediterranean_stainnorm",
    "exp10_linear_probe_all_stainnorm",
]

# Grouped experiments for comparison
IMAGENET_EXPS = ["exp02_linear_probe_french", "exp03_linear_probe_hungarian", 
                 "exp04_linear_probe_swedish", "exp05_linear_probe_mediterranean"]
STAINNORM_EXPS = ["exp06_linear_probe_french_stainnorm", "exp07_linear_probe_hungarian_stainnorm",
                  "exp08_linear_probe_swedish_stainnorm", "exp09_linear_probe_mediterranean_stainnorm"]

# Region mapping
EXP_TO_REGION = {
    "exp01_linear_probe_all": "all",
    "exp02_linear_probe_french": "french",
    "exp03_linear_probe_hungarian": "hungarian",
    "exp04_linear_probe_swedish": "swedish",
    "exp05_linear_probe_mediterranean": "mediterranean",
    "exp06_linear_probe_french_stainnorm": "french",
    "exp07_linear_probe_hungarian_stainnorm": "hungarian",
    "exp08_linear_probe_swedish_stainnorm": "swedish",
    "exp09_linear_probe_mediterranean_stainnorm": "mediterranean",
    "exp10_linear_probe_all_stainnorm": "all",
}

# Short names for display
SHORT_NAMES = {
    "exp01_linear_probe_all": "Exp1: All",
    "exp02_linear_probe_french": "Exp2: French",
    "exp03_linear_probe_hungarian": "Exp3: Hungarian",
    "exp04_linear_probe_swedish": "Exp4: Swedish",
    "exp05_linear_probe_mediterranean": "Exp5: Mediterranean",
    "exp06_linear_probe_french_stainnorm": "Exp6: French+SN",
    "exp07_linear_probe_hungarian_stainnorm": "Exp7: Hungarian+SN",
    "exp08_linear_probe_swedish_stainnorm": "Exp8: Swedish+SN",
    "exp09_linear_probe_mediterranean_stainnorm": "Exp9: Mediterranean+SN",
    "exp10_linear_probe_all_stainnorm": "Exp10: All+SN",
}


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)
        return None


def load_training_log(exp_dir: Path) -> Optional[List[Dict]]:
    """Load training log CSV."""
    log_path = exp_dir / "training_log.csv"
    if not log_path.exists():
        return None
    
    import csv
    rows = []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v not in ('inf', '-inf', '') else None 
                        for k, v in row.items() if k != 'timestamp'})
            rows[-1]['timestamp'] = row.get('timestamp', '')
    return rows


def collect_experiment_results(results_dir: Path) -> Dict[str, Dict]:
    """Collect all results for experiments in the results directory."""
    experiments = {}
    
    for exp_name in EXPERIMENT_ORDER:
        exp_dir = results_dir / exp_name
        if not exp_dir.exists():
            continue
        
        exp_data = {
            "name": exp_name,
            "short_name": SHORT_NAMES.get(exp_name, exp_name),
            "path": str(exp_dir),
        }
        
        # Load experiment metadata
        metadata = load_json(exp_dir / "experiment_metadata.json")
        if metadata:
            exp_data["metadata"] = metadata
            exp_data["git_commit"] = metadata.get("git", {}).get("commit", "unknown")
            exp_data["started_at"] = metadata.get("experiment", {}).get("started_at", "")
            exp_data["completed_at"] = metadata.get("experiment", {}).get("completed_at", "")
            exp_data["status"] = metadata.get("experiment", {}).get("status", "unknown")
            exp_data["train_samples"] = metadata.get("data", {}).get("train_samples", 0)
            exp_data["val_samples"] = metadata.get("data", {}).get("val_samples", 0)
        
        # Load training results summary
        summary = load_json(exp_dir / "results_summary.json")
        if summary:
            exp_data["training"] = {
                "best_val_acc": summary.get("best_val_accuracy", 0),
                "final_val_acc": summary.get("final_val_accuracy", 0),
                "final_train_acc": summary.get("final_train_accuracy", 0),
                "final_val_top5": summary.get("final_val_top5_accuracy", 0),
                "epochs": summary.get("epochs_completed", 0),
                "time_hours": summary.get("training_time_hours", 0),
            }
            # Check if best == final (indicating convergence)
            exp_data["training"]["best_is_final"] = abs(
                summary.get("best_val_accuracy", 0) - summary.get("final_val_accuracy", 0)
            ) < 0.001
        
        # Load training log for detailed epoch info
        training_log = load_training_log(exp_dir)
        if training_log:
            exp_data["training_log"] = training_log
            # Find best epoch
            best_epoch = max(range(len(training_log)), 
                           key=lambda i: training_log[i].get('val_acc', 0) or 0)
            exp_data["training"]["best_epoch"] = best_epoch + 1
        
        # Load TS1 eval
        ts1_eval = load_json(exp_dir / "eval_ts1_legacy.json")
        if ts1_eval:
            exp_data["ts1"] = {
                "top1": ts1_eval.get("top1_accuracy", 0),
                "top5": ts1_eval.get("top5_accuracy", 0),
                "macro_f1": ts1_eval.get("macro_f1", 0),
                "weighted_f1": ts1_eval.get("weighted_f1", 0),
                "macro_precision": ts1_eval.get("macro_precision", 0),
                "macro_recall": ts1_eval.get("macro_recall", 0),
                "samples": ts1_eval.get("total_samples", 0),
            }
        
        # Load TS2 eval
        ts2_eval = load_json(exp_dir / "eval_ts2_expert.json")
        if ts2_eval:
            exp_data["ts2"] = {
                "top1": ts2_eval.get("top1_accuracy", 0),
                "top5": ts2_eval.get("top5_accuracy", 0),
                "macro_f1": ts2_eval.get("macro_f1", 0),
                "weighted_f1": ts2_eval.get("weighted_f1", 0),
                "macro_precision": ts2_eval.get("macro_precision", 0),
                "macro_recall": ts2_eval.get("macro_recall", 0),
                "samples": ts2_eval.get("total_samples", 0),
            }
        
        # Load cross-region summary AND individual cross files
        source_region = EXP_TO_REGION.get(exp_name, "unknown")
        cross_results = {}
        
        # Try loading individual cross-region files first (more detailed)
        for target in ["french", "hungarian", "swedish", "mediterranean"]:
            cross_file = exp_dir / f"eval_cross_{source_region}_to_{target}.json"
            if cross_file.exists():
                data = load_json(cross_file)
                if data and "overlap" in data:
                    overlap_data = data["overlap"]
                    cross_results[target] = {
                        "accuracy": overlap_data.get("top1_accuracy", 0),
                        "top5_accuracy": overlap_data.get("top5_accuracy", 0),
                        "macro_f1": overlap_data.get("macro_f1", 0),
                        "macro_precision": overlap_data.get("macro_precision", 0),
                        "macro_recall": overlap_data.get("macro_recall", 0),
                        "samples": overlap_data.get("num_samples", 0),
                        "num_overlapping_classes": data.get("overlap_taxa_count", "-"),
                        "is_in_domain": data.get("is_in_domain", False),
                    }
        
        # Also try the summary file for any missing data
        cross_summary = None
        cross_file = exp_dir / f"eval_cross_{source_region}_summary.json"
        if cross_file.exists():
            cross_summary = load_json(cross_file)
        
        if cross_summary and "results" in cross_summary:
            # Parse the cross-region results format
            for key, data in cross_summary["results"].items():
                if isinstance(data, dict) and "overlap" in data:
                    target = data.get("target", key.split("_to_")[-1])
                    if target not in cross_results:  # Don't overwrite detailed data
                        overlap_data = data["overlap"]
                        cross_results[target] = {
                            "accuracy": overlap_data.get("top1_accuracy", 0),
                            "top5_accuracy": overlap_data.get("top5_accuracy", 0),
                            "macro_f1": overlap_data.get("macro_f1", 0),
                            "macro_precision": overlap_data.get("macro_precision", 0),
                            "macro_recall": overlap_data.get("macro_recall", 0),
                            "samples": overlap_data.get("num_samples", 0),
                            "num_overlapping_classes": data.get("overlap_taxa_count", "-"),
                            "is_in_domain": data.get("is_in_domain", False),
                        }
        
        if cross_results:
            exp_data["cross_region"] = cross_results
            exp_data["source_region"] = source_region
        
        experiments[exp_name] = exp_data
    
    return experiments


def format_markdown_tables(experiments: Dict[str, Dict]) -> str:
    """Format experiment results as markdown tables."""
    lines = []
    lines.append("# Classification Experiment Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Results directory:** `{RESULTS_DIR}`")
    lines.append("")
    
    # SIDE-BY-SIDE COMPARISON TABLE (Main table - what the user wants to see first)
    lines.append("## ImageNet vs Stain Normalization: Side-by-Side Comparison")
    lines.append("")
    lines.append("Side-by-side comparison of each regional model with and without stain normalization.")
    lines.append("")
    lines.append("### TS2 Expert (10,218 samples, 45 taxa)")
    lines.append("")
    lines.append("| Region | ImageNet Top-1 | SN Top-1 | Δ | ImageNet F1 | SN F1 | Δ |")
    lines.append("|--------|----------------|----------|---|-------------|-------|---|")
    
    pairs = [
        ("French", "exp02_linear_probe_french", "exp06_linear_probe_french_stainnorm"),
        ("Hungarian", "exp03_linear_probe_hungarian", "exp07_linear_probe_hungarian_stainnorm"),
        ("Swedish", "exp04_linear_probe_swedish", "exp08_linear_probe_swedish_stainnorm"),
        ("Mediterranean", "exp05_linear_probe_mediterranean", "exp09_linear_probe_mediterranean_stainnorm"),
        ("All", "exp01_linear_probe_all", "exp10_linear_probe_all_stainnorm"),
    ]
    
    for region, img_exp, sn_exp in pairs:
        img = experiments.get(img_exp, {}).get("ts2", {})
        sn = experiments.get(sn_exp, {}).get("ts2", {})
        
        img_top1 = img.get("top1", 0) if img else 0
        sn_top1 = sn.get("top1", 0) if sn else 0
        img_f1 = img.get("macro_f1", 0) if img else 0
        sn_f1 = sn.get("macro_f1", 0) if sn else 0
        
        if not img and not sn:
            continue
        
        delta_top1 = sn_top1 - img_top1 if sn else 0
        delta_f1 = sn_f1 - img_f1 if sn else 0
        
        d1 = f"+{delta_top1:.1f}" if delta_top1 > 0 else f"{delta_top1:.1f}" if sn else "—"
        d2 = f"+{delta_f1:.1f}" if delta_f1 > 0 else f"{delta_f1:.1f}" if sn else "—"
        
        sn_top1_str = f"{sn_top1:.1f}%" if sn else "—"
        sn_f1_str = f"{sn_f1:.1f}%" if sn else "—"
        
        lines.append(f"| {region} | {img_top1:.1f}% | {sn_top1_str} | **{d1}** | "
                    f"{img_f1:.1f}% | {sn_f1_str} | **{d2}** |")
    
    lines.append("")
    
    lines.append("### TS1 Legacy (5,065 samples, 11 taxa)")
    lines.append("")
    lines.append("| Region | ImageNet Top-1 | SN Top-1 | Δ | ImageNet F1 | SN F1 | Δ |")
    lines.append("|--------|----------------|----------|---|-------------|-------|---|")
    
    for region, img_exp, sn_exp in pairs:
        img = experiments.get(img_exp, {}).get("ts1", {})
        sn = experiments.get(sn_exp, {}).get("ts1", {})
        
        img_top1 = img.get("top1", 0) if img else 0
        sn_top1 = sn.get("top1", 0) if sn else 0
        img_f1 = img.get("macro_f1", 0) if img else 0
        sn_f1 = sn.get("macro_f1", 0) if sn else 0
        
        if not img and not sn:
            continue
        
        delta_top1 = sn_top1 - img_top1 if sn else 0
        delta_f1 = sn_f1 - img_f1 if sn else 0
        
        d1 = f"+{delta_top1:.1f}" if delta_top1 > 0 else f"{delta_top1:.1f}" if sn else "—"
        d2 = f"+{delta_f1:.1f}" if delta_f1 > 0 else f"{delta_f1:.1f}" if sn else "—"
        
        sn_top1_str = f"{sn_top1:.1f}%" if sn else "—"
        sn_f1_str = f"{sn_f1:.1f}%" if sn else "—"
        
        lines.append(f"| {region} | {img_top1:.1f}% | {sn_top1_str} | **{d1}** | "
                    f"{img_f1:.1f}% | {sn_f1_str} | **{d2}** |")
    
    lines.append("")
    
    # Cross-region comparison (side by side)
    lines.append("### Cross-Region Generalization (Overlapping Taxa)")
    lines.append("")
    lines.append("Each row compares ImageNet vs SN for the same source→target pair.")
    lines.append("")
    
    cross_pairs = [
        ("French", "exp02_linear_probe_french", "exp06_linear_probe_french_stainnorm"),
        ("Hungarian", "exp03_linear_probe_hungarian", "exp07_linear_probe_hungarian_stainnorm"),
        ("Swedish", "exp04_linear_probe_swedish", "exp08_linear_probe_swedish_stainnorm"),
        ("Mediterranean", "exp05_linear_probe_mediterranean", "exp09_linear_probe_mediterranean_stainnorm"),
    ]
    
    for source, img_exp, sn_exp in cross_pairs:
        img_cross = experiments.get(img_exp, {}).get("cross_region", {})
        sn_cross = experiments.get(sn_exp, {}).get("cross_region", {})
        
        if not img_cross and not sn_cross:
            continue
        
        lines.append(f"#### {source} → Others")
        lines.append("")
        lines.append("| Target | #Taxa | ImageNet Top-1 | SN Top-1 | Δ | ImageNet F1 | SN F1 | Δ |")
        lines.append("|--------|-------|----------------|----------|---|-------------|-------|---|")
        
        # Get all targets from both
        all_targets = set(img_cross.keys()) | set(sn_cross.keys())
        for target in sorted(all_targets):
            img_data = img_cross.get(target, {})
            sn_data = sn_cross.get(target, {})
            
            img_top1 = img_data.get("accuracy", 0) if img_data else 0
            sn_top1 = sn_data.get("accuracy", 0) if sn_data else 0
            img_f1 = img_data.get("macro_f1", 0) if img_data else 0
            sn_f1 = sn_data.get("macro_f1", 0) if sn_data else 0
            n_taxa = img_data.get("num_overlapping_classes", "-") if img_data else sn_data.get("num_overlapping_classes", "-")
            
            delta_top1 = sn_top1 - img_top1
            delta_f1 = sn_f1 - img_f1
            
            d1 = f"+{delta_top1:.1f}" if delta_top1 > 0 else f"{delta_top1:.1f}"
            d2 = f"+{delta_f1:.1f}" if delta_f1 > 0 else f"{delta_f1:.1f}"
            
            marker = " ⭐" if delta_top1 > 10 else " ❌" if delta_top1 < -10 else ""
            
            lines.append(f"| {target.capitalize()} | {n_taxa} | {img_top1:.1f}% | {sn_top1:.1f}% | **{d1}**{marker} | "
                        f"{img_f1:.1f}% | {sn_f1:.1f}% | **{d2}** |")
        
        lines.append("")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Training summary table (detailed, for reference)
    lines.append("## Training Performance (Detailed)")
    lines.append("")
    lines.append("| Exp | Region | Train Acc | Val Acc | Best Epoch | Epochs | Time (h) | Best=Final |")
    lines.append("|-----|--------|-----------|---------|------------|--------|----------|------------|")
    
    for exp_name in EXPERIMENT_ORDER:
        if exp_name not in experiments:
            continue
        exp = experiments[exp_name]
        if "training" not in exp:
            continue
        t = exp["training"]
        short = exp["short_name"].replace("Exp", "").split(":")[0].strip()
        region = exp["short_name"].split(":")[-1].strip()
        best_is_final = "✓" if t.get("best_is_final", False) else "✗"
        lines.append(f"| {short} | {region} | {t.get('final_train_acc', 0):.1f}% | "
                    f"{t.get('best_val_acc', 0):.1f}% | {t.get('best_epoch', '-')} | "
                    f"{t.get('epochs', 0)} | {t.get('time_hours', 0):.2f} | {best_is_final} |")
    
    lines.append("")
    
    # TS2 Expert evaluation table
    lines.append("## Test Set Evaluation: TS2 Expert (10,335 grains)")
    lines.append("")
    lines.append("| Exp | Region | Top-1 | Macro Prec | Macro Recall | Macro F1 | Samples |")
    lines.append("|-----|--------|-------|------------|--------------|----------|---------|")
    
    for exp_name in EXPERIMENT_ORDER:
        if exp_name not in experiments:
            continue
        exp = experiments[exp_name]
        if "ts2" not in exp:
            continue
        ts2 = exp["ts2"]
        short = exp["short_name"].replace("Exp", "").split(":")[0].strip()
        region = exp["short_name"].split(":")[-1].strip()
        lines.append(f"| {short} | {region} | {ts2.get('top1', 0):.1f}% | "
                    f"{ts2.get('macro_precision', 0):.1f}% | {ts2.get('macro_recall', 0):.1f}% | "
                    f"{ts2.get('macro_f1', 0):.1f}% | {ts2.get('samples', 0):,} |")
    
    lines.append("")
    
    # TS1 Legacy evaluation table (if available)
    has_ts1 = any("ts1" in experiments.get(e, {}) for e in EXPERIMENT_ORDER)
    if has_ts1:
        lines.append("## Test Set Evaluation: TS1 Legacy (6,723 grains)")
        lines.append("")
        lines.append("| Exp | Region | Top-1 | Macro Prec | Macro Recall | Macro F1 | Samples |")
        lines.append("|-----|--------|-------|------------|--------------|----------|---------|")
        
        for exp_name in EXPERIMENT_ORDER:
            if exp_name not in experiments:
                continue
            exp = experiments[exp_name]
            if "ts1" not in exp:
                continue
            ts1 = exp["ts1"]
            short = exp["short_name"].replace("Exp", "").split(":")[0].strip()
            region = exp["short_name"].split(":")[-1].strip()
            lines.append(f"| {short} | {region} | {ts1.get('top1', 0):.1f}% | "
                        f"{ts1.get('macro_precision', 0):.1f}% | {ts1.get('macro_recall', 0):.1f}% | "
                        f"{ts1.get('macro_f1', 0):.1f}% | {ts1.get('samples', 0):,} |")
        
        lines.append("")
    
    # Cross-region summary
    has_cross = any("cross_region" in experiments.get(e, {}) for e in EXPERIMENT_ORDER)
    if has_cross:
        lines.append("## Cross-Region Generalization")
        lines.append("")
        lines.append("Results from training on one region and evaluating on others (overlapping taxa only).")
        lines.append("")
        
        for exp_name in EXPERIMENT_ORDER:
            if exp_name not in experiments:
                continue
            exp = experiments[exp_name]
            if "cross_region" not in exp:
                continue
            
            lines.append(f"### {exp['short_name']}")
            lines.append("")
            lines.append("| Target | Overlap Taxa | Samples | Top-1 | Top-5 | Macro F1 |")
            lines.append("|--------|--------------|---------|-------|-------|----------|")
            
            cross = exp["cross_region"]
            # Order targets: self-domain first, then alphabetical
            source = exp.get("source_region", "unknown")
            ordered_targets = [source] + sorted([t for t in cross.keys() if t != source])
            
            for target in ordered_targets:
                if target not in cross:
                    continue
                data = cross[target]
                if isinstance(data, dict) and "accuracy" in data:
                    marker = " (self)" if target == source else ""
                    lines.append(f"| {target.capitalize()}{marker} | {data.get('num_overlapping_classes', '-')} | "
                               f"{data.get('samples', 0):,} | {data.get('accuracy', 0):.1f}% | "
                               f"{data.get('top5_accuracy', 0):.1f}% | {data.get('macro_f1', 0):.1f}% |")
            
            lines.append("")
    
    # Comparison: ImageNet vs Stain Normalization
    lines.append("## Comparison: ImageNet vs Stain Normalization")
    lines.append("")
    lines.append("Side-by-side comparison of baseline (ImageNet norm) vs stain normalized experiments.")
    lines.append("")
    
    # TS2 Comparison Table
    lines.append("### TS2 Expert Results")
    lines.append("")
    lines.append("| Region | ImageNet Top-1 | Stainnorm Top-1 | Δ | ImageNet F1 | Stainnorm F1 | Δ |")
    lines.append("|--------|----------------|-----------------|---|-------------|--------------|---|")
    
    pairs = [
        ("French", "exp02_linear_probe_french", "exp06_linear_probe_french_stainnorm"),
        ("Hungarian", "exp03_linear_probe_hungarian", "exp07_linear_probe_hungarian_stainnorm"),
        ("Swedish", "exp04_linear_probe_swedish", "exp08_linear_probe_swedish_stainnorm"),
        ("Mediterranean", "exp05_linear_probe_mediterranean", "exp09_linear_probe_mediterranean_stainnorm"),
    ]
    
    for region, img_exp, sn_exp in pairs:
        img = experiments.get(img_exp, {}).get("ts2", {})
        sn = experiments.get(sn_exp, {}).get("ts2", {})
        
        if img and sn:
            img_top1 = img.get("top1", 0)
            sn_top1 = sn.get("top1", 0)
            delta_top1 = sn_top1 - img_top1
            img_f1 = img.get("macro_f1", 0)
            sn_f1 = sn.get("macro_f1", 0)
            delta_f1 = sn_f1 - img_f1
            
            d1 = f"+{delta_top1:.1f}" if delta_top1 > 0 else f"{delta_top1:.1f}"
            d2 = f"+{delta_f1:.1f}" if delta_f1 > 0 else f"{delta_f1:.1f}"
            
            lines.append(f"| {region} | {img_top1:.1f}% | {sn_top1:.1f}% | **{d1}pp** | "
                        f"{img_f1:.1f}% | {sn_f1:.1f}% | **{d2}pp** |")
    
    lines.append("")
    
    # TS1 Comparison Table
    lines.append("### TS1 Legacy Results")
    lines.append("")
    lines.append("| Region | ImageNet Top-1 | Stainnorm Top-1 | Δ | ImageNet F1 | Stainnorm F1 | Δ |")
    lines.append("|--------|----------------|-----------------|---|-------------|--------------|---|")
    
    for region, img_exp, sn_exp in pairs:
        img = experiments.get(img_exp, {}).get("ts1", {})
        sn = experiments.get(sn_exp, {}).get("ts1", {})
        
        if img and sn:
            img_top1 = img.get("top1", 0)
            sn_top1 = sn.get("top1", 0)
            delta_top1 = sn_top1 - img_top1
            img_f1 = img.get("macro_f1", 0)
            sn_f1 = sn.get("macro_f1", 0)
            delta_f1 = sn_f1 - img_f1
            
            d1 = f"+{delta_top1:.1f}" if delta_top1 > 0 else f"{delta_top1:.1f}"
            d2 = f"+{delta_f1:.1f}" if delta_f1 > 0 else f"{delta_f1:.1f}"
            
            lines.append(f"| {region} | {img_top1:.1f}% | {sn_top1:.1f}% | **{d1}pp** | "
                        f"{img_f1:.1f}% | {sn_f1:.1f}% | **{d2}pp** |")
    
    lines.append("")
    
    # Insights
    lines.append("## Key Insights")
    lines.append("")
    
    # Find best models
    best_val = max((e for e in experiments.values() if "training" in e),
                   key=lambda x: x["training"].get("best_val_acc", 0), default=None)
    best_ts2 = max((e for e in experiments.values() if "ts2" in e),
                   key=lambda x: x["ts2"].get("top1", 0), default=None)
    
    if best_val:
        lines.append(f"- **Best validation accuracy:** {best_val['short_name']} "
                    f"({best_val['training']['best_val_acc']:.1f}%)")
    if best_ts2:
        lines.append(f"- **Best TS2 test accuracy:** {best_ts2['short_name']} "
                    f"({best_ts2['ts2']['top1']:.1f}%)")
    
    # Convergence analysis
    converged = [e for e in experiments.values() 
                 if "training" in e and e["training"].get("best_is_final", False)]
    lines.append(f"- **Models converged (best=final):** {len(converged)}/{len([e for e in experiments.values() if 'training' in e])}")
    
    lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                       help="Path to results directory")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Output format")
    args = parser.parse_args()
    
    # Collect results
    experiments = collect_experiment_results(args.results_dir)
    
    if not experiments:
        print("No experiment results found!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(experiments)} experiments", file=sys.stderr)
    
    # Format output
    if args.format == "json":
        output = json.dumps(experiments, indent=2, default=str)
    else:
        output = format_markdown_tables(experiments)
    
    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
