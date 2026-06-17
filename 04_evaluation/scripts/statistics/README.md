# Statistics

This folder computes final caption and detection statistics for the paper code release. It reads caption JSONL files, caption anchors, cross-dataset metadata, and curated GeoJSON annotations, then writes machine-readable summaries and audit exports.

## File index

| File or folder | Purpose |
| --- | --- |
| `extract_anchor_vocabulary.py` | Extracts morphology vocabulary and per-slide anchor metadata from `03_captioning/caption_anchors/`. |
| `compute_caption_stats.py` | Runs caption statistics phases: read captions, basic compliance, morphology coverage, cross-model agreement, and expert-audit sample export. |
| `compute_detection_metrics.py` | Computes precision, recall, F1, and grouped detection metrics from raw and curated GeoJSON annotations. |
| `run_caption_stats.sh` | Runs anchor vocabulary extraction and caption statistics in order. |
| `caption_stats/` | Modular implementation package used by `compute_caption_stats.py`. |

## Data contract

Inputs:

```text
data/03_captioning/{dataset}/production_*_final/*_captions.jsonl
03_captioning/caption_anchors/*_{anchor,hint,species,family}.txt
03_captioning/slide_exclusions.yaml
data/04_evaluation/results/cross_dataset_matrix.json
04_evaluation/annotations/{ts1_legacy,ts2_expert}/
```

Outputs:

```text
data/04_evaluation/results/caption_statistics/
data/04_evaluation/results/detection_metrics/
```

## Usage

```bash
cd 04_evaluation/scripts/statistics
bash run_caption_stats.sh --skip-sbert
python compute_detection_metrics.py
```

Remove `--skip-sbert` when a compatible `sentence-transformers` model is available for semantic cross-model agreement.
