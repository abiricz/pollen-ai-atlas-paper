# Option D: caption-to-image distillation

Option D trains image-only student models using soft targets from caption-aware teachers. The resulting student checkpoint has the same inference architecture as Option A and is evaluated with the Option A evaluation functions.

## File index

| File | Purpose |
| --- | --- |
| `train_distill.py` | Runs teacher-to-student training, caches teacher logits, and writes a drop-in image-only student checkpoint. |
| `evaluate_distill.py` | Evaluates distilled student checkpoints on TS1, TS2, and cross-region test sets. |
| `train_all.sh` | Batch runner for all Option D experiments or a selected distillation experiment. |
| `eval_all.sh` | Batch runner for Option D intra-region and cross-region evaluation. |

## Training modes

| Mode | Command flag | Meaning |
| --- | --- | --- |
| Reuse teacher | default | Discovers the matching Option C teacher checkpoint and trains the student. |
| Explicit teacher | `--teacher-checkpoint PATH` | Uses a specific teacher checkpoint. |
| Full two-stage | `--train-teacher` | Trains the teacher and then trains the student. |
| Cached logits | `--cache-logits` | Stores teacher logits for faster student training. |

## Usage

```bash
cd 04_evaluation/scripts/experiments/option_D
bash train_all.sh all --seeds 41,42,43,44,45
bash eval_all.sh all --seeds 41,42,43,44,45
```

Run Option C caption embedding and teacher training before the default Option D workflow.
