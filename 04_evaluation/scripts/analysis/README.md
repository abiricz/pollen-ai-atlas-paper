# Analysis and result collection

This folder contains reusable aggregation scripts for experiment outputs and cross-dataset species overlap. The scripts support audit, reporting, and downstream manuscript assembly.

## File index

| File | Purpose |
| --- | --- |
| `collect_experiment_results.py` | Collects Option A training and evaluation summaries, including TS1/TS2 metrics, cross-region matrices, and ImageNet-versus-stain-normalization comparisons. |
| `create_cross_dataset_matrix.py` | Builds the species-overlap matrix used to decide valid cross-region evaluations and retrieval species coverage. |

## Main classification collector

The full paper classification summary across Options A/C/D is collected from the parent scripts folder:

```bash
cd 04_evaluation/scripts
python collect_all_results.py
```

`create_cross_dataset_matrix.py` writes `data/04_evaluation/results/cross_dataset_matrix.json` and a readable text companion.
