# Research Log — DenPAR Bone Loss Severity Replication

Paper: *AI-assisted radiographic analysis in detecting alveolar bone-loss severity and patterns* (Sci Rep 2026, DOI [10.1038/s41598-026-38061-1](https://doi.org/10.1038/s41598-026-38061-1))

**Scope:** tooth detection → keypoint detection → severity (ICC). No segmentation or pattern classification.

| Step | Doc | Status |
|------|-----|--------|
| Environment | [01_environment.md](01_environment.md) | Done |
| Dataset | [02_dataset.md](02_dataset.md) | Done |
| Preprocessing | [03_preprocessing.md](03_preprocessing.md) | Done (v1–v6); **v6 selected** |
| Hyperparameters | [04_hyperparameters.md](04_hyperparameters.md) | Done |
| Detection training | [05_detection_training.md](05_detection_training.md) | Done (test mAP50 **0.873**) |
| Preproc comparison | [preprocessing_comparison.md](preprocessing_comparison.md) | v1–v4 stats |
| Keypoint training | [06_keypoint_training.md](06_keypoint_training.md) | **v6 best** (int OKS **0.894**) |
| Severity / ICC | [07_severity_icc.md](07_severity_icc.md) | **In progress** — test ICC ~0.50–0.57 (paper 0.801) |
| Experiment registry | [experiments/README.md](experiments/README.md) | v2–v4 in `paper_table.json` |

**Resume here:** [CHECKPOINT.md](CHECKPOINT.md)

**LaTeX progress paper:** [../paper/replication_progress.tex](../paper/replication_progress.tex)

**Skill:** say **"Make a checkpoint"** to sync this log after changes.
