# 02 — Dataset (DenPAR)

**Source:** [Zenodo 16645076](https://doi.org/10.5281/zenodo.16645076)  
**Raw path:** `data/DenPAR/Dataset/`

## Splits (IOPA radiographs)

| Split | Images | Teeth (annotated) |
|-------|--------|-------------------|
| Training | 650 | 2,883 |
| Validation | 150 | 655 |
| Testing | 200 | 864 |
| **Total** | **1,000** | **4,402** |

Counts from `data/processed/preprocess_summary.json` after conversion.

## Annotation types used (severity pipeline)

- Tooth bounding boxes + root type (single / double)
- CEJ points, apex points (ground truth in JSON)
- Bone level lines + tooth masks → intersection points (derived at preprocess)
- Bone loss severity labels (for ICC evaluation, not used in training)

## Not used in this replication

Segmentation masks (YOLO-seg / Mask R-CNN), bone loss pattern labels.
