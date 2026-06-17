# Evaluation scripts

This folder is the execution hub for the manuscript evaluation. The subfolders divide the work by purpose: preparing data, training classifiers, running retrieval, plotting embeddings, computing statistics, and aggregating outputs.

## Entry-point index

| File or folder | Purpose |
| --- | --- |
| `collect_all_results.py` | Collects every classification evaluation JSON across Options A/C/D, 20 experiment definitions, and five seeds; writes generated summaries under `data/04_evaluation/results/`. |
| `data_preparation/` | Builds split files, verifies split/test-set separation, curates QuPath annotations, and computes stain-normalization references. |
| `experiments/` | Runs the classification experiments defined in `experiments/experiment_config.yaml`. |
| `retrieval/` | Builds ViT image embeddings, runs image/text/combined retrieval, runs negative controls, and collects retrieval summaries. |
| `visualization/` | Produces UMAP figures from validation-split SBERT and ViT embeddings. |
| `statistics/` | Computes caption statistics, detection metrics, morphology vocabulary, and expert-audit samples. |
| `analysis/` | Builds the cross-dataset matrix and collects Option A-focused experiment summaries. |

## Shared inputs

All scripts use the repository root as the path anchor. Generated data lives under `DATA_ROOT` when it is set, otherwise under the repository-local `data/` directory:

```text
data/03_captioning/
data/04_evaluation/splits/
data/04_evaluation/results/
04_evaluation/annotations/
03_captioning/caption_anchors/
```

## Standard sequence

1. Build and verify split files in `data_preparation/`.
2. Train and evaluate Options A, C, and D in `experiments/`.
3. Aggregate classification outputs with `collect_all_results.py`.
4. Run retrieval and visualization from their subfolders.
5. Run caption and detection statistics from `statistics/`.

Each shell runner activates `.venv` when present and logs outputs under `data/04_evaluation/results/`.
