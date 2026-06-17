# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Constants, paths, compiled regexes, and optional dependency flags.
"""

import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STATISTICS_DIR = SCRIPT_DIR.parent
REPO_ROOT = STATISTICS_DIR.parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "03_captioning"
ANCHOR_DIR = REPO_ROOT / "03_captioning" / "caption_anchors"
EXCLUSION_YAML = REPO_ROOT / "03_captioning" / "slide_exclusions.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "04_evaluation" / "results" / "caption_statistics"
VOCAB_PATH = OUTPUT_DIR / "anchor_vocabulary.json"
CROSS_DATASET_MATRIX_PATH = REPO_ROOT / "04_evaluation" / "results" / "cross_dataset_matrix.json"

MODELS = {
    "production_qwen25vl_final": "Qwen2.5-VL-32B",
    "production_qwen3-fp8_final": "Qwen3-VL-30B-FP8",
    "production_qwen35-fp8_final": "Qwen3.5-27B-FP8",
    "production_qwen36-fp8_final": "Qwen3.6-27B-FP8",
    "production_gemma4-bf16_final": "Gemma4-31B-BF16",
}

DATASETS = ["french", "hungarian", "mediterranean", "swedish"]

TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")
NUMBER_RE = re.compile(r"\d")
SIZE_NUMBER_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:µm|um|micron(?:s)?|micrometer(?:s)?|mm|x)\b",
    flags=re.IGNORECASE,
)

DEBRIS_OPENER = "debris, dust or artifact detected; no matching pollen grain present"
DEFAULT_PROMPT_MARKERS = [
    "hint or exemplar inconsistent",
    DEBRIS_OPENER,
]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
