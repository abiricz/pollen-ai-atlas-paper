# Embedding visualization

This folder creates UMAP visualizations of pollen grain embeddings for the 15 cross-regional retrieval species. The figures compare text embeddings, image embeddings, and alpha=0.5 late-fusion distances using validation-split samples.

## File index

| File | Purpose |
| --- | --- |
| `umap_cross_regional.py` | Loads validation-split SBERT and ViT embeddings, balances samples per species and origin, computes UMAP layouts, and writes three-panel plus individual figures. |
| `run_umap_cross_regional.sh` | Runs Gemma4-BF16 UMAP generation for pretrained and finetuned ViT modes, with an option to also render Qwen2.5-VL figures. |

## Data contract

Inputs:

```text
data/04_evaluation/splits/val/*_val.json
data/04_evaluation/caption_embeddings/{vlm}/*_embeddings.h5
data/04_evaluation/vit_embeddings/*_embeddings.h5
03_captioning/caption_anchors/*_species.txt
```

Outputs:

```text
data/04_evaluation/results/visualization/pretrained/umap_cross_regional_*.{pdf,png,svg,json}
data/04_evaluation/results/visualization/finetuned/umap_cross_regional_*.{pdf,png,svg,json}
```

## Figure design

| Panel | Space | Source | Metric |
| --- | --- | --- | --- |
| `a` | Text | SBERT caption embeddings | Cosine distance. |
| `b` | Image | ViT-Small-LVD image embeddings | Cosine distance. |
| `c` | Combined | Alpha=0.5 late fusion | Precomputed fused distance. |

Points are colored by species and shaped by geographic origin. Metadata JSON files record UMAP parameters, species counts, origin counts, and sampling settings.

## Usage

```bash
cd 04_evaluation/scripts/visualization
bash run_umap_cross_regional.sh
bash run_umap_cross_regional.sh --both
```

Use direct Python calls for custom sample caps, VLMs, ViT modes, UMAP parameters, or output DPI.
