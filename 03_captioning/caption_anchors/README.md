# Caption anchors

This folder contains the expert-curated text files that bind each slide basename to taxonomy and morphology. The captioning, species-mapping, retrieval, and caption-statistics scripts read these files directly, so the filename contract is part of the public data model.

## File pattern

| Pattern | Meaning |
| --- | --- |
| `{slide}_species.txt` | Species or genus label used for class mapping and caption metadata. |
| `{slide}_species_verified.txt` | Verification marker for slides with confirmed species labels. |
| `{slide}_family.txt` | Taxonomic family label used in prompts and detection summaries. |
| `{slide}_hint.txt` | Short taxon hint used by caption prompting. |
| `{slide}_anchor.txt` | Longer expert morphology description used by caption prompting and vocabulary extraction. |

## Consumers

`03_captioning/caption_production_concurrent.py` inserts the species, family, hint, and anchor text into morphology prompts. `lib/species_mapping.py` builds slide-to-class mappings from `*_species.txt`. `04_evaluation/scripts/statistics/extract_anchor_vocabulary.py` extracts the morphology vocabulary from `*_anchor.txt` and `*_hint.txt`. `04_evaluation/scripts/retrieval/retrieval_config.yaml` defines independent retrieval descriptions while keeping these anchors as the authoritative taxonomy source.

Add or edit anchor files by slide basename, then keep the matching query image in `01_initialization/query_images/` synchronized with that basename.
