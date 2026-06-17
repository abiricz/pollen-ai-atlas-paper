# 04_evaluation

This folder contains the manuscript evaluation workflows. It starts from filtered detections and caption JSONL files, prepares train/validation/test inputs, runs the classification and retrieval experiments, generates embedding visualizations, computes statistics, and collects result tables.

## Subtree index

| Path | Purpose |
| --- | --- |
| `scripts/README.md` | Entry-point map for the evaluation scripts. |
| `scripts/data_preparation/` | Split generation, QuPath GeoJSON curation, test-set disjointness checks, and stain-normalization references. |
| `scripts/experiments/` | Classification experiments: Option A image-only, Option C LUPI, and Option D caption-to-image distillation. |
| `scripts/retrieval/` | Cross-regional multimodal retrieval experiments and result collection. |
| `scripts/visualization/` | UMAP embedding figures from precomputed caption and image embeddings. |
| `scripts/statistics/` | Caption statistics, detection metrics, vocabulary extraction, and expert-audit export. |
| `scripts/analysis/` | Result aggregation utilities and cross-dataset species-overlap matrix creation. |
| `scripts/collect_all_results.py` | Main collector for all 20 classification experiments across seeds. |

## Data contract

Evaluation reads these inputs:

```text
data/03_captioning/{dataset}/filtered/*_filtered.h5
data/03_captioning/{dataset}/production_*_final/*_captions.jsonl
data/04_evaluation/splits/{train,val}/
data/04_evaluation/caption_embeddings/
data/04_evaluation/vit_embeddings/
04_evaluation/annotations/{ts1_legacy,ts2_expert}/*_curated.geojson
```

Evaluation writes these products:

```text
data/04_evaluation/splits/
data/04_evaluation/normalization/
data/04_evaluation/results/
data/04_evaluation/caption_embeddings/
data/04_evaluation/vit_embeddings/
```

Set `DATA_ROOT` before running scripts that read or write generated data.

## Run order

```bash
cd 04_evaluation/scripts/data_preparation
bash run_train_val_splitter.sh
python verify_all_splits.py
python verify_test_sets_disjointness.py

cd ../experiments/option_A
bash train_all.sh 1-5
bash eval_all.sh 1-5

cd ../option_C
bash embed_gemma4.sh
bash train_all.sh 11-15
bash eval_all.sh 11-15

cd ../option_D
bash train_all.sh all
bash eval_all.sh all

cd ../..
python collect_all_results.py

cd retrieval
bash retrieval_experiments.sh
python collect_retrieval_results.py

cd ../visualization
bash run_umap_cross_regional.sh

cd ../statistics
bash run_caption_stats.sh --skip-sbert
```

Review GPU IDs, paths, batch sizes, and seed arguments in each runner before launching full jobs.
