# Query images

This folder contains the expert-selected pollen crop images used to initialize the pipeline. Each PNG is named by the slide or reference label it represents, and that basename is reused by mining, caption anchors, retrieval configuration, and species mapping.

## File pattern

| Pattern | Meaning |
| --- | --- |
| `{slide}.png` | Query crop for a French, Hungarian, Swedish, or Mediterranean slide. |
| `mediterranean_pollen_*_reference.png` | Mediterranean reference crop used for the named taxon or family. |
| `hun_*_edf.png` | Hungarian reference crop for the named taxon. |
| Swedish long-form slide names | Query crops aligned to Swedish reference slide identifiers. |

## Consumers

`01_initialization/Tif_tile_annotator.py` uses these images for OWL-ViT query-based detection. `02_mining/miner.py` uses the same files to seed ViT similarity mining. `04_evaluation/scripts/retrieval/retrieval_experiments.py` embeds them as image queries for the retrieval experiments.

Keep the image basename synchronized with the WSI slide basename and the matching files in `03_captioning/caption_anchors/`.
