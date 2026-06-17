# Caption statistics package

This directory is the implementation package behind `../compute_caption_stats.py`. It keeps each analysis phase in a small module so the driver can run the final caption-statistics workflow in a fixed order.

## Module index

| Module | Purpose |
| --- | --- |
| `constants.py` | Shared paths, model names, feature flags, and prompt markers. |
| `io.py` | Caption JSONL discovery, loading, cross-dataset metadata loading, and anchor text loading. |
| `helpers.py` | Common formatting, token, species, and safety helpers. |
| `phase_basic.py` | Caption counts, lengths, prompt-compliance markers, and slide/model summaries. |
| `phase_morphology.py` | Morphology vocabulary coverage against anchor-derived terms. |
| `phase_cross_model.py` | Jaccard and SBERT agreement across caption models. |
| `phase_audit.py` | Expert-audit sample export. |
| `report.py` | Markdown report generation from computed statistics. |
| `__init__.py` | Public imports for the driver. |

The package reads from `data/03_captioning/`, `03_captioning/caption_anchors/`, and `data/04_evaluation/results/cross_dataset_matrix.json`; it writes through the driver to `data/04_evaluation/results/caption_statistics/`.
