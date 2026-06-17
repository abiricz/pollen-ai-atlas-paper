# 01_initialization

This folder contains the bootstrapping assets used before large-scale mining. It connects expert-selected query crops, OWL-ViT tile annotations, clustering-based prefiltering, and the released ViT-Small-LVD checkpoint that downstream stages reuse.

## File index

| File or folder | Purpose |
| --- | --- |
| `query_images/` | Expert-selected PNG crops used as one-shot query images for OWL-ViT initialization, WSI mining, and retrieval image queries. |
| `Tif_tile_annotator.py` | Runs OWL-ViT over WSI tiles with query images and writes tile-level detections to H5. |
| `Tif_tile_annotator.sh` | Batch runner for `Tif_tile_annotator.py`; reads `DATASET`, `DATA_ROOT`, `WSI_ROOT`, GPU assignments, and the model list from the script header. |
| `Tif_annotation_clustering.py` | Converts OWL-ViT detections into clustered global annotations, filters by shape and score, and prepares image/H5 outputs for embedder training. |
| `Tif_annotation_clustering_prefiltering.sh` | Batch runner for clustering and prefiltering selected slides. |
| `Tif_clustering_embedder_finetuning.ipynb` | Notebook record for the ViT-Small-LVD embedder finetuning run. |
| `history_vit_small_lvd_20250620_0312.csv` | Training history for the released embedder checkpoint. |
| `weights_vit_small_lvd_20250620_0312.pth` | Finetuned ViT-Small-LVD weights used by mining and filtering. |

## Data contract

Inputs come from the raw WSI tree:

```text
data/00_raw_wsi/{french,hungarian,mediterranean,swedish}/
```

The scripts write generated annotations, clustered crops, and training products under `DATA_ROOT/01_initialization/`. The released checkpoint and training history stay in this folder because later stages reference them directly.

## Usage

```bash
cd 01_initialization
DATA_ROOT=/path/to/pollen_ai_atlas_workdir bash Tif_tile_annotator.sh
DATA_ROOT=/path/to/pollen_ai_atlas_workdir bash Tif_annotation_clustering_prefiltering.sh
```

Set `DATASET`, `DATA_ROOT`, `WSI_ROOT`, and the GPU list before launching a batch. The shell runners are organized by slide lists so a public user can process one taxon, one dataset, or a full batch with the same entry points.
