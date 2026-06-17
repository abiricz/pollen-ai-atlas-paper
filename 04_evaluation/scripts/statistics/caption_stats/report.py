# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Markdown report generation for final caption statistics.
"""

from .constants import MODELS
from .helpers import fmt_int, fmt_num, safe_model_name


def generate_markdown(stats, output_path):
    """Generate a concise markdown summary."""
    lines = []
    add = lines.append

    add("# Caption Statistics Report")
    add("")
    add(f"**Generated:** {stats.get('metadata', {}).get('generated_at', '-')}")
    add(f"**Runtime:** {fmt_num(stats.get('metadata', {}).get('runtime_seconds', 0), 1)}s")
    add(f"**Workers:** {fmt_int(stats.get('metadata', {}).get('n_workers', 0))}")
    add("")

    basic = stats.get("basic_stats", {}).get("per_model", {})
    model_keys = sorted(basic.keys()) if basic else sorted(MODELS.keys())
    model_labels = [safe_model_name(mk) for mk in model_keys]

    header = "| Metric | " + " | ".join(model_labels) + " |"
    sep = "|--------" + "|".join(["-" * (len(lbl) + 2) for lbl in model_labels]) + "|"

    def bm(model_key, *keys):
        value = basic.get(model_key, {})
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    add("## Overview")
    add("")
    add(header)
    add(sep)
    add("| Total captions | " + " | ".join(fmt_int(bm(mk, "total_captions")) for mk in model_keys) + " |")
    add("| Mean words/caption | " + " | ".join(
        f"{fmt_num(bm(mk, 'word_count', 'mean'), 2)} +/- {fmt_num(bm(mk, 'word_count', 'std'), 2)}"
        for mk in model_keys
    ) + " |")
    add("| Word count P5/P50/P95 | " + " | ".join(
        f"{fmt_num(bm(mk, 'word_count_percentiles', 'p5'), 0)}/"
        f"{fmt_num(bm(mk, 'word_count_percentiles', 'p50'), 0)}/"
        f"{fmt_num(bm(mk, 'word_count_percentiles', 'p95'), 0)}"
        for mk in model_keys
    ) + " |")
    add("| Vocabulary size | " + " | ".join(fmt_int(bm(mk, "vocabulary_size")) for mk in model_keys) + " |")
    add("| Word range compliance (60-80) | " + " | ".join(
        f"{fmt_num(bm(mk, 'prompt_compliance', 'word_range_60_80_pct'), 2)}%"
        for mk in model_keys
    ) + " |")
    add("| Digit incidence | " + " | ".join(
        f"{fmt_num(bm(mk, 'prompt_compliance', 'numeric_leakage_pct'), 2)}%"
        for mk in model_keys
    ) + " |")
    add("| Size-number leakage | " + " | ".join(
        f"{fmt_num(bm(mk, 'prompt_compliance', 'size_numeric_leakage_pct'), 2)}%"
        for mk in model_keys
    ) + " |")

    morph = stats.get("morphological_coverage", {})
    if morph:
        add("")
        add("## Morphological Coverage")
        add("")
        add(header)
        add(sep)
        add("| Global vocab coverage | " + " | ".join(
            f"{fmt_num(morph.get(mk, {}).get('global_vocab_coverage_pct'), 2)}%"
            for mk in model_keys
        ) + " |")
        add("| Anchor-term recall (per caption) | " + " | ".join(
            fmt_num(morph.get(mk, {}).get('global_anchor_overlap', {}).get('recall', {}).get('mean'), 4)
            for mk in model_keys
        ) + " |")
        add("| Anchor-term F1 (per caption) | " + " | ".join(
            fmt_num(morph.get(mk, {}).get('global_anchor_overlap', {}).get('f1', {}).get('mean'), 4)
            for mk in model_keys
        ) + " |")

    cross = stats.get("cross_model", {})
    if cross and cross.get("matrix"):
        add("")
        add("## Cross-Model Agreement Matrix")
        add("")
        matrix_labels = cross.get("model_labels", [])
        n = len(matrix_labels)
        for metric_name, metric_key in [("Jaccard", "jaccard"), ("SBERT Cosine", "sbert_cosine")]:
            mat = cross["matrix"].get(metric_key)
            if not mat:
                continue
            add(f"### {metric_name}")
            add("")
            add("| | " + " | ".join(matrix_labels) + " |")
            add("|---" + "|".join(["---"] * n) + "|")
            for i, lbl in enumerate(matrix_labels):
                row_vals = " | ".join(
                    fmt_num(mat[i][j], 4) if mat[i][j] is not None else "N/A"
                    for j in range(n)
                )
                add(f"| {lbl} | {row_vals} |")
            add("")

    audit = stats.get("expert_audit_sample", {})
    if audit and audit.get("enabled"):
        add("")
        add("## Expert Audit Sample")
        add("")
        add(f"- Models: {fmt_int(audit.get('n_models', 2))}")
        add(f"- Sample size: {fmt_int(audit.get('actual_sample_size'))}")
        add(f"- Datasets: {fmt_int(audit.get('n_datasets'))}")
        add(f"- Species: {fmt_int(audit.get('n_species'))}")
        add(f"- Slides: {fmt_int(audit.get('n_slides'))}")
        add(f"- JSONL: `{audit.get('paths', {}).get('jsonl', '-')}`")
        add(f"- CSV: `{audit.get('paths', {}).get('csv', '-')}`")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
