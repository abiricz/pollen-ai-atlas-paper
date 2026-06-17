# Shared library

`lib/` contains reusable Python modules used across mining, filtering, captioning support, and evaluation. These modules keep WSI I/O, ViT backbone construction, SAM-2 helpers, class mapping, and image preprocessing in one importable package.

## Module index

| Module | Purpose |
| --- | --- |
| `loader.py` | PyTorch datasets and WSI tile/crop loading helpers, including stain-normalization support and batched statistics utilities. |
| `model.py` | ViT model selection, tiled feature-map reconstruction, and `ViTBackbone` wrappers. |
| `utils.py` | `WSIPipeline`, SAM-2 integration, ViT similarity functions, NMS helpers, convergence checks, prompt construction, and mask measurement utilities. |
| `classifier.py` | Pollen/background classifier wrapper, fixed class map for the filtering checkpoint, and batch prediction helpers. |
| `species_mapping.py` | HITL species mapping loaded from `03_captioning/caption_anchors/*_species.txt`; used by evaluation to build trainable class IDs. |
| `preprocessing.py` | Annotation loading, H5 writing, clustering, geometry, image filtering, and helper functions used during initialization and prefiltering. |
| `__init__.py` | Lightweight package exports with guarded imports for heavy dependencies. |

Scripts import this package by adding the repository root to `sys.path`. Run commands from the repository root or from the documented script folders so relative paths resolve consistently.
