# Option C: LUPI caption-augmented classification

Option C trains classifiers with image embeddings plus Gemma4 caption embeddings as privileged information during training. Evaluation uses image-only inference through the Option C adapter, so reported metrics measure an image model trained with caption supervision.

## File index

| File | Purpose |
| --- | --- |
| `embed_captions.py` | Encodes VLM caption JSONL files into SBERT H5 embeddings keyed by sample ID. |
| `embed_gemma4.sh` | Runs `embed_captions.py` for `production_gemma4-bf16_final` captions. |
| `train_lupi.py` | Trains one LUPI experiment from `../experiment_config.yaml`. |
| `evaluate_lupi.py` | Evaluates LUPI checkpoints with zeroed text features at test time while reusing Option A evaluation logic. |
| `train_all.sh` | Batch runner for experiments `11-15`, selected regions, or direct experiment names. |
| `eval_all.sh` | Batch runner for intra-region and cross-region LUPI evaluation. |

## Experiment names

| Selector | Experiment |
| --- | --- |
| `exp11`, `lupi_all`, `combined` | All-region LUPI training. |
| `exp12`, `lupi_french`, `french` | French-region LUPI training. |
| `exp13`, `lupi_hungarian`, `hungarian` | Hungarian-region LUPI training. |
| `exp14`, `lupi_swedish`, `swedish` | Swedish-region LUPI training. |
| `exp15`, `lupi_mediterranean`, `mediterranean` | Mediterranean-region LUPI training. |
| `11-15` or `all` | Runs all Option C experiments. |

## Usage

```bash
cd 04_evaluation/scripts/experiments/option_C
bash embed_gemma4.sh
bash train_all.sh 11-15 --seeds 41,42,43,44,45
bash eval_all.sh 11-15 --seeds 41,42,43,44,45
```

The embedding step writes `data/04_evaluation/caption_embeddings/gemma4-bf16/*_embeddings.h5`, which `train_lupi.py` reads by sample ID.
