# Classification experiments

This folder contains the classification experiment system used in the paper. `experiment_config.yaml` defines the shared data paths, seeds, regions, test sets, model settings, and experiment names for Options A, C, and D.

## File index

| File or folder | Purpose |
| --- | --- |
| `experiment_config.yaml` | Central configuration for data paths, model backbones, normalization, regions, test sets, training defaults, and experiment definitions. |
| `option_A/` | Image-only ViT-Small-LVD linear-probe experiments, including ImageNet-normalized and stain-normalized variants. |
| `option_C/` | LUPI training with Gemma4 caption embeddings as privileged information during training. |
| `option_D/` | Caption-to-image distillation that trains an image-only student from a caption-aware teacher. |

## Experiment groups

| Range | Option | Description |
| --- | --- | --- |
| `1-5` | A | Image-only linear probes for all regions and each individual region with ImageNet normalization. |
| `6-10` | A | Image-only linear probes with Macenko stain normalization references. |
| `11-15` | C | LUPI classifiers trained with image embeddings plus Gemma4 SBERT caption embeddings. |
| `16-20` | D | Distilled image-only students trained from caption-aware teachers. |

## Data contract

Inputs:

```text
data/04_evaluation/splits/{train,val}/
data/03_captioning/{dataset}/filtered/*_filtered.h5
data/03_captioning/{dataset}/production_gemma4-bf16_final/*_captions.jsonl
data/04_evaluation/caption_embeddings/gemma4-bf16/*_embeddings.h5
04_evaluation/annotations/{ts1_legacy,ts2_expert}/*_curated.geojson
03_captioning/caption_anchors/*_species.txt
```

Outputs:

```text
data/04_evaluation/results/exp*/
data/04_evaluation/results/logs/
```

## Run order

```bash
cd 04_evaluation/scripts/experiments/option_A
bash train_all.sh 1-5 --seeds 41,42,43,44,45
bash eval_all.sh 1-5 --seeds 41,42,43,44,45

cd ../option_C
bash embed_gemma4.sh
bash train_all.sh 11-15 --seeds 41,42,43,44,45
bash eval_all.sh 11-15 --seeds 41,42,43,44,45

cd ../option_D
bash train_all.sh all --seeds 41,42,43,44,45
bash eval_all.sh all --seeds 41,42,43,44,45
```

Use `--debug`, `--max-samples`, `--epochs`, `--device`, `--ddp`, `--seed`, `--seeds`, and `--run-name` as documented in the shell runner headers.
