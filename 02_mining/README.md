# 02_mining

This folder mines pollen candidates from whole-slide images. The miner combines query-image ViT similarity, SAM-2 mask proposal, non-maximum suppression, entropy/convergence checks, and H5 persistence.

## File index

| File | Purpose |
| --- | --- |
| `miner.py` | Main mining program. It reads one WSI and one query image, ranks ViT token similarities, asks SAM-2 for masks, filters masks by geometry and confidence, and writes detections to H5. |
| `miner.sh` | Batch runner for selected slides. It defines dataset paths, query-image paths, ViT/SAM-2 checkpoints, GPU scheduling, and per-slide command assembly. |

## Data contract

Inputs:

```text
data/00_raw_wsi/{dataset}/{slide}.tif
01_initialization/query_images/{slide}.png
01_initialization/weights_vit_small_lvd_20250620_0312.pth
SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt
```

Outputs:

```text
data/02_mining/{dataset}/{slide}_detections.h5
```

Each detection H5 becomes the input to `03_captioning/filter_candidates.py`.

## Usage

```bash
cd 02_mining
DATA_ROOT=/path/to/pollen_ai_atlas_workdir SAM2_ROOT=/path/to/sam2 SAM2_CKPT=/path/to/sam2/checkpoints/sam2.1_hiera_large.pt bash miner.sh
```

Edit the `slides` array and GPU settings in `miner.sh` for the machine that will run the mining job. The Python script can also be called directly for a single slide when tuning thresholds or SAM-2 settings.
