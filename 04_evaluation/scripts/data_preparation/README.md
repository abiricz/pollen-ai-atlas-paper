# Data preparation

This folder prepares the evaluation inputs used by classification, retrieval, visualization, and statistics. It converts annotation files into a common form, creates train/validation splits, verifies disjointness, and builds stain-normalization references for the stain-normalized Option A experiments.

## File index

| File | Purpose |
| --- | --- |
| `curate_qupath_annotations.py` | Merges raw and expert-confirmed QuPath GeoJSON files into curated evaluation GeoJSON files with consistent feature structure. |
| `train_val_splitter.py` | Builds train/validation split files from caption JSONL files while removing samples that overlap TS1 or TS2 test regions. |
| `run_train_val_splitter.sh` | Shell runner for `train_val_splitter.py`; forwards all CLI arguments. |
| `verify_all_splits.py` | Verifies train, validation, TS1, and TS2 separation and writes a split verification report. |
| `verify_test_sets_disjointness.py` | Verifies TS1 and TS2 slide/region disjointness and writes a disjointness report. |
| `compute_stainnorm_reference.py` | Samples training patches, builds Macenko reference images, and computes per-region channel statistics. |
| `compute_stainnorm_reference.sh` | Shell runner for all-region or per-region stain-normalization reference creation. |

## Data contract

Inputs:

```text
data/03_captioning/{dataset}/filtered/*_filtered.h5
data/03_captioning/{dataset}/production_*_final/*_captions.jsonl
04_evaluation/annotations/{ts1_legacy,ts2_expert}/*_curated.geojson
data/00_raw_wsi/{dataset}/
```

Outputs:

```text
data/04_evaluation/splits/manifest.json
data/04_evaluation/splits/train/*_train.json
data/04_evaluation/splits/val/*_val.json
data/04_evaluation/normalization/*_reference.npy
data/04_evaluation/normalization/*_stainnorm_stats.json
data/04_evaluation/results/*verification*.json
```

## Usage

```bash
cd 04_evaluation/scripts/data_preparation
bash run_train_val_splitter.sh
python verify_all_splits.py
python verify_test_sets_disjointness.py
bash compute_stainnorm_reference.sh all
```

Run split creation before any classification training. Run stain-normalization reference creation before experiments 6-10.
