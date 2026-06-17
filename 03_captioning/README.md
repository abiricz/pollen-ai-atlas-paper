# 03_captioning

This folder turns mined candidates into filtered pollen detections and VLM captions. It holds the filtering script, high-throughput captioning script, vLLM launcher, calibration files, slide-quality controls, and expert text anchors used in the caption prompts.

## File index

| File or folder | Purpose |
| --- | --- |
| `filter_candidates.py` | Filters raw mining H5 files with ViT attention reranking, medoid prototype refinement, NMS, and the pollen/background classifier. |
| `filter_candidates_production.sh` | Batch runner for filtering selected slides and writing filtered H5 plus QuPath test-region GeoJSON metadata. |
| `caption_production_concurrent.py` | Async captioning engine for vLLM-compatible OpenAI endpoints; reads filtered H5 detections, crops WSI regions, builds morphology prompts, and writes JSONL captions. |
| `caption_production_concurrent.sh` | Batch runner for production captioning across filtered slides. |
| `start_vllm_concurrent.sh` | Starts one or more local vLLM servers for Qwen and Gemma VLM variants. |
| `pixel_config.yaml` | Per-dataset and per-slide microns-per-pixel calibration used for physical grain measurements. |
| `slide_exclusions.yaml` | Dataset-level slide gating for caption production. |
| `species_thresholds.yaml` | Pollen/background classifier thresholds and NMS defaults for filtering. |
| `caption_anchors/` | Expert species, family, hint, and anchor text files keyed by slide basename. |

## Data contract

Inputs:

```text
data/02_mining/{dataset}/*_detections.h5
data/00_raw_wsi/{dataset}/
01_initialization/weights_vit_small_lvd_20250620_0312.pth
03_captioning/caption_anchors/
```

SAM-2 paths are configured with `SAM2_ROOT` and `SAM2_CKPT`. vLLM model paths are configured with variables such as `GEMMA4_BF16_PATH`, `QWEN25_AWQ_PATH`, and `VLLM_MODEL_ROOT`.

Outputs:

```text
data/03_captioning/{dataset}/filtered/*_filtered.h5
data/03_captioning/{dataset}/production_gemma4-bf16_final/*_captions.jsonl
data/03_captioning/{dataset}/production_qwen25vl_final/*_captions.jsonl
data/03_captioning/{dataset}/production_qwen3-fp8_final/*_captions.jsonl
data/03_captioning/{dataset}/production_qwen35-fp8_final/*_captions.jsonl
data/03_captioning/{dataset}/production_qwen36-fp8_final/*_captions.jsonl
```

Gemma4 BF16 is the primary caption source for the manuscript workflow. The Qwen caption folders support cross-model caption agreement and retrieval comparisons.

## Usage

```bash
cd 03_captioning
DATA_ROOT=/path/to/pollen_ai_atlas_workdir bash filter_candidates_production.sh
VLLM_MODEL_ROOT=/path/to/vlm_models bash start_vllm_concurrent.sh gemma4-bf16 cluster
DATA_ROOT=/path/to/pollen_ai_atlas_workdir bash caption_production_concurrent.sh
```

The filtering step runs before captioning. The captioning runner discovers filtered H5 files, anchor files, pixel calibration, and slide gating from this folder.
