# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
caption_stats - Modular caption evaluation for the Pollen AI Atlas.

Retained public-release phases:
  phase_basic.py      -> caption length, leakage, and prompt-compliance stats
  phase_morphology.py -> anchor-derived morphological vocabulary coverage
  phase_cross_model.py-> pairwise Jaccard and optional SBERT agreement
  phase_audit.py      -> expert audit sample export
"""

from .io import (
    build_slide_term_sets,
    discover_jsonl_files,
    load_anchor_texts,
    load_cross_dataset_slide_map,
    phase_read_captions,
)
from .phase_audit import phase_export_expert_audit_sample
from .phase_basic import phase_basic_stats
from .phase_cross_model import phase_cross_model
from .phase_morphology import phase_morphological_coverage
from .report import generate_markdown

__all__ = [
    "build_slide_term_sets",
    "discover_jsonl_files",
    "load_anchor_texts",
    "load_cross_dataset_slide_map",
    "phase_read_captions",
    "phase_basic_stats",
    "phase_morphological_coverage",
    "phase_cross_model",
    "phase_export_expert_audit_sample",
    "generate_markdown",
]
