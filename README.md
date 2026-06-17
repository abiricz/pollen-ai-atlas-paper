# Pollen AI Atlas - paper code repository

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.17809-b31b1b.svg)](https://doi.org/10.48550/arXiv.2606.17809)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.20690944.svg)](https://zenodo.org/records/20690944)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository is the code companion for the official Pollen AI Atlas preprint and its Zenodo processed-data release. The released corpus contains 1,511,390 multimodal pollen-grain records from 80 captioned slides, derived from 85 processed pure-species bright-field whole-slide images spanning four geographic origins, four scanner settings, 46 taxon labels, and 31 botanical families.

The code covers query-image initialization, token-level mining, candidate filtering, VLM caption production, split construction, detection and caption diagnostics, classification, retrieval, embedding visualization, and final statistics.

The official preprint DOI is [10.48550/arXiv.2606.17809](https://doi.org/10.48550/arXiv.2606.17809). The live Zenodo data DOI is [10.5281/zenodo.20690944](https://doi.org/10.5281/zenodo.20690944), published as Version v1 on 2026-06-17.

## Zenodo data record

The Zenodo package is the processed data release used by the manuscript. The public archive is `pollen_ai_atlas.zip`, 42.8 GB, and its deposited structure is:

```text
pollen_ai_atlas/
|- README.md
|- LICENSE
|- filtered_detections/
|  |- french/                           # Per-origin HDF5 files
|  |- hungarian/
|  |- mediterranean/
|  `- swedish/                         # 85 HDF5 files total, all processed slides
|- captions/                            # Five model-specific JSONL caption corpora
|- pollen_grain_crops/
|  |- french/
|  |- hungarian/
|  |- mediterranean/
|  `- swedish/                         # 1,511,390 PNG crops plus per-slide CSV manifests
|- ground_truth/
|  |- ts2_expert/                       # 84 GeoJSON files
|  `- ts1_legacy/                       # 21 GeoJSON files
|- ground_truth_crops/                  # 17,318 PNG crops plus CSV manifests
|- splits/                              # 80 train JSON, 80 validation JSON, 1 manifest
|- model/
|  `- weights_vit_small_lvd_20250620_0312.pth
`- metadata/
   |- slide_manifest.json               # 85 slide entries
   |- caption_anchors/                  # 400 text files
   `- query_images/                     # 85 matched exemplar PNGs
```

| Deposited path | Role | Package size / note |
| --- | --- | --- |
| `filtered_detections/` | Per-origin folders of per-slide HDF5 records with masks, boxes, prompts, confidence scores, classifier outputs, and token-index metadata. | 2.7 GB |
| `captions/` | Five grain-level caption JSONL corpora: Gemma4, Qwen2.5-VL, Qwen3-VL, Qwen3.5, and Qwen3.6. Gemma4 is the primary workflow set. | See Zenodo file listing for per-corpus sizes |
| `pollen_grain_crops/` | Per-origin released crop image corpus and crop manifests. | 35.6 GB |
| `ground_truth/` | Expert TS2 and legacy TS1 GeoJSON annotations. | 14 MB |
| `ground_truth_crops/` | Validated pollen crops extracted from TS1 and TS2 annotations, plus manifests. | 465.7 MB |
| `splits/` | Model-agnostic train/validation split files and split manifest. | 44 MB |
| `metadata/` | Slide manifest, caption-anchor files, and exemplar query images. | 6.8 MB |
| `model/` | Fine-tuned LVD-ViT-S checkpoint used in downstream processing. | 88.4 MB |

## Repository map

| Path | Role | Main files |
| --- | --- | --- |
| `01_initialization/` | Query-image bootstrapping and ViT embedder initialization. | `Tif_tile_annotator.py`, `Tif_annotation_clustering.py`, `Tif_clustering_embedder_finetuning.ipynb`, `query_images/`, released ViT checkpoint. |
| `02_mining/` | Token-level WSI mining with ViT similarity and SAM-2 masks. | `miner.py`, `miner.sh`. |
| `03_captioning/` | Candidate filtering, calibration, caption anchors, and VLM caption production. | `filter_candidates.py`, `caption_production_concurrent.py`, `start_vllm_concurrent.sh`, `pixel_config.yaml`, `slide_exclusions.yaml`, `species_thresholds.yaml`, `caption_anchors/`. |
| `04_evaluation/` | Evaluation workflows. | `scripts/data_preparation/`, `scripts/experiments/`, `scripts/retrieval/`, `scripts/visualization/`, `scripts/statistics/`, `scripts/collect_all_results.py`. |
| `lib/` | Shared Python code used by mining, filtering, captioning, and evaluation. | WSI loaders, ViT wrappers, SAM-2 pipeline helpers, classifier utilities, species mapping. |
| `requirements.txt` | Python package requirements for the public workflow. | Core scientific Python, image I/O, ML, plotting, retrieval, and evaluation packages. |
| `setup_env.sh` | Environment bootstrapper. | Creates `.venv`, installs requirements, checks local SAM-2, and prints vLLM setup guidance. |

Each folder has its own README with the scripts, expected inputs, generated outputs, and role in the workflow.

## Usage

### Reuse the processed data record

For local reuse, set `DATA_ROOT` to the extracted Zenodo package root:

```bash
export DATA_ROOT=/path/to/pollen_ai_atlas
```

The extracted Zenodo tree is suitable for inspecting and reusing the deposited HDF5, JSONL, GeoJSON, crop, split, metadata, and checkpoint records directly.

### Rerun from raw whole-slide inputs

Full slide-level reproduction uses the 85 original processed pyramidal TIFF whole-slide images, 148.9 GB / 138.7 GiB, which are not included in the Zenodo package. With those inputs, mirror the raw-input layout below under `DATA_ROOT` and run the initialization, mining, filtering, and captioning stages in order:

```text
DATA_ROOT/
|-- 00_raw_wsi/{french,hungarian,mediterranean,swedish}/
|-- 02_mining/{dataset}/*_detections.h5
|-- 03_captioning/{dataset}/filtered/*_filtered.h5
|-- 03_captioning/{dataset}/production_*_final/*_captions.jsonl
`-- 04_evaluation/
    |-- annotations/{ts1_legacy,ts2_expert}/
    |-- splits/{train,val}/
    |-- caption_embeddings/
    |-- vit_embeddings/
    `-- results/
```

Use this layout when rerunning from raw WSI inputs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SAM-2 and vLLM are GPU-dependent components. Use `SAM2_ROOT`, `SAM2_CKPT`, `VLLM_MODEL_ROOT`, and the model-specific environment variables documented in the stage READMEs to point scripts at the local installations and model checkpoints.

## Reproducibility workflow

Run stages in numeric order: initialize query images and the ViT embedder, mine WSI candidates, filter and caption detections, then run the evaluation scripts for splits, classification, retrieval, visualization, statistics, and result aggregation.

Shell runners expose dataset, path, GPU, seed, and output variables at the top of each file. Set those variables directly or override them with environment variables before launching long GPU jobs.

## Citation

If you use this repository, cite the Pollen AI Atlas paper and the associated Zenodo data record:

```text
Biricz, A., Gedda, B., Magyar, D., Spanu, A., Fillinger, J., Pollner, P., & Csabai, I. (2026). Million-scale multimodal pollen microscopy with expert-guided foundation models. arXiv. https://doi.org/10.48550/arXiv.2606.17809
```

```text
Biricz, A., Gedda, B., Magyar, D., Spanu, A., Fillinger, J., Pollner, P., & Csabai, I. (2026). Pollen AI Atlas: million-scale multimodal pollen microscopy with expert-guided foundation models (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20690944
```

This repository also builds on the annotation-initialization work from the preceding paper:

```text
Biricz, A., et al. (2025). Efficient and scalable training set generation for automated pollen monitoring with Hirst-type samplers. Scientific Reports. https://doi.org/10.1038/s41598-025-31646-2
```

## Related works

- **SAM-2:** Ravi et al. (2024). [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714).
- **DINOv2:** Oquab et al. (2023). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- **Vision Transformer:** Dosovitskiy et al. (2020). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
- **OWL-ViT:** Minderer et al. (2022). [Simple Open-Vocabulary Object Detection with Vision Transformers](https://arxiv.org/abs/2205.06230).
- **Qwen2.5-VL:** Bai et al. (2025). [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923).

## Third-party components and licenses

| Component | Used for | Upstream source | License |
| --- | --- | --- | --- |
| SAM-2 | Single-click mask generation for mining and candidate filtering. | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | [Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| DINOv2 | Self-supervised ViT feature foundation. | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) |
| `vit_small_patch14_dinov2.lvd142m` | Default ViT backbone for embeddings, mining, retrieval, and classifiers. | [Hugging Face model card](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) | [Apache 2.0](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) |
| timm | ViT model creation and pretrained checkpoint loading. | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | [Apache 2.0](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE) |
| OWL-ViT | Query-image annotation initialization. | [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) | [Apache 2.0](https://huggingface.co/google/owlvit-base-patch32) |
| Hugging Face Transformers | OWL-ViT and model utility APIs. | [huggingface/transformers](https://github.com/huggingface/transformers) | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| PyTorch and torchvision | Deep learning, transforms, NMS, training, and inference. | [pytorch/pytorch](https://github.com/pytorch/pytorch), [pytorch/vision](https://github.com/pytorch/vision) | [BSD-style](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| vLLM | Local OpenAI-compatible VLM serving for caption production. | [vllm-project/vllm](https://github.com/vllm-project/vllm) | [Apache 2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE) |
| OpenAI Python SDK | Client for local vLLM chat-completions endpoints. | [openai/openai-python](https://github.com/openai/openai-python) | [Apache 2.0](https://github.com/openai/openai-python/blob/main/LICENSE) |
| Qwen2.5-VL-32B-Instruct-AWQ | Qwen captioning model served through vLLM. | [Qwen/Qwen2.5-VL-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ) | [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ) |
| Gemma4 31B | Gemma captioning model served through vLLM. | [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) | [Apache 2.0](https://ai.google.dev/gemma/docs/core/model_card_4) |
| SentenceTransformers | Caption and query text embeddings. | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) | [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE) |
| OpenSlide Python and OpenSlide | Whole-slide image reading. | [openslide-python](https://pypi.org/project/openslide-python/) | LGPL-2.1-only, BSD-3-Clause, MIT, and public-domain components, as listed by the package metadata. |
| tiffslide | Alternative WSI reader used by evaluation scripts. | [Bayer-Group/tiffslide](https://github.com/Bayer-Group/tiffslide) | [BSD-3-Clause / New BSD](https://github.com/Bayer-Group/tiffslide/blob/main/LICENSE) |
| TIAToolbox | Macenko stain normalization experiments. | [TissueImageAnalytics/tiatoolbox](https://github.com/TissueImageAnalytics/tiatoolbox) | [BSD-3-Clause](https://github.com/TissueImageAnalytics/tiatoolbox/blob/develop/LICENSE) |

The captioning scripts also support local Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, and Gemma4 checkpoint variants through `VLLM_MODEL_ROOT` and model-specific environment variables. Each local checkpoint remains governed by its model card.

## License

Copyright 2026 András Biricz.

The code in this repository is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE). The Zenodo data package is licensed separately under CC BY 4.0.
