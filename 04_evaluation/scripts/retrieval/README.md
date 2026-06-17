# Retrieval

This folder runs the cross-regional multimodal retrieval experiments. The retrieval task asks whether a one-shot image query, an expert text query, or a late-fused combination retrieves same-taxon pollen grains from the captioned corpus across regions and scanner setups.

## File index

| File | Purpose |
| --- | --- |
| `retrieval_config.yaml` | Defines retrieval species, image/text query sources, VLM models, modes, metrics, and independent expert descriptors. |
| `extract_vit_embeddings.py` | Extracts pretrained ViT-Small-LVD image embeddings for captioned crops into per-slide H5 files. |
| `retrieval_experiments.py` | Runs image-only, text-only, combined, cross-regional, full-corpus, and negative-control retrieval evaluations. |
| `retrieval_experiments.sh` | Batch runner for main retrieval, label-shuffle controls, and result collection. |
| `collect_retrieval_results.py` | Converts retrieval JSON outputs into summary JSON/CSV tables and query provenance files. |

## Data contract

Inputs:

```text
01_initialization/query_images/
03_captioning/caption_anchors/*_species.txt
data/03_captioning/{dataset}/production_*_final/*_captions.jsonl
data/04_evaluation/vit_embeddings/*_embeddings.h5
data/04_evaluation/caption_embeddings/{vlm}/*_embeddings.h5
```

Outputs:

```text
data/04_evaluation/results/retrieval/retrieval_*.json
data/04_evaluation/results/retrieval/retrieval_summary.json
data/04_evaluation/results/retrieval/retrieval_summary.csv
data/04_evaluation/results/retrieval/retrieval_query_provenance.csv
```

## Retrieval modes

| Mode | Corpus rule |
| --- | --- |
| `all` | Searches the full corpus. |
| `full` | Removes the query slide and scanner-step siblings. |
| `cross_regional` | Removes all slides from the query origin. |

## Usage

```bash
cd 04_evaluation/scripts/retrieval
python extract_vit_embeddings.py --resume
bash retrieval_experiments.sh --main-only
bash retrieval_experiments.sh --control-only
python collect_retrieval_results.py
```

`retrieval_experiments.sh` runs all configured VLM caption corpora and writes logs next to the retrieval outputs.
