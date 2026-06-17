# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

# Shared library modules for pollen-ai-atlas
# 
# Contains:
#   - utils.py: WSIPipeline, mining functions, attention reranking
#   - model.py: ViT backbones, model_selection
#   - loader.py: WSI data loading, datasets
#   - classifier.py: Pollen classifier for filtering candidates

from lib.model import model_selection, ViTBackbone
from lib.loader import WSITileDataset

# Optional imports (may require heavy dependencies)
try:
    from lib.utils import WSIPipeline
except ImportError:
    WSIPipeline = None

try:
    from lib.classifier import PollenClassifier, load_classifier
except ImportError:
    PollenClassifier = None
    load_classifier = None
