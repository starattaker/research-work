# Research Log — DenPAR Bone Loss Severity Replication

Paper: *AI-assisted radiographic analysis in detecting alveolar bone-loss severity and patterns* (Sci Rep 2026, DOI [10.1038/s41598-026-38061-1](https://doi.org/10.1038/s41598-026-38061-1))

**Scope:** tooth detection → keypoint detection → severity (ICC). No segmentation or pattern classification.

| Step | Doc | Status |
|------|-----|--------|
| Environment | [01_environment.md](01_environment.md) | Done |
| Dataset | [02_dataset.md](02_dataset.md) | Done |
| Preprocessing | [03_preprocessing.md](03_preprocessing.md) | Done (v1–v4); stats in [preprocessing_comparison.md](preprocessing_comparison.md) |
| Hyperparameters | [04_hyperparameters.md](04_hyperparameters.md) | Done |
| Detection training | [05_detection_training.md](05_detection_training.md) | Done (test mAP50 **0.873**) |
| Preproc comparison | [preprocessing_comparison.md](preprocessing_comparison.md) | **v1–v4** rebuilt on friend GPU (2026-08-20) |
| Keypoint training | [06_keypoint_training.md](06_keypoint_training.md) | v1 CEJ; **v2 all 3 trained**; v3/v4 data ready, training next |
| Experiment registry | [experiments/README.md](experiments/README.md) | **v2 recorded** in `paper_table.json` |
| Severity / ICC | *(pending)* | Not started |

**Resume here:** [CHECKPOINT.md](CHECKPOINT.md)

**LaTeX progress paper:** [../paper/replication_progress.tex](../paper/replication_progress.tex)

**Skill:** say **"Make a checkpoint"** to sync this log after changes.
