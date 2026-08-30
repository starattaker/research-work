# 07 — Severity inference + ICC

**Status:** Honest test ICC **~0.50–0.57** vs paper **0.801**  
**Updated:** 2026-08-31  
**Data:** `data/processed_v6` (GT sides = **PCA slots**, not paper x-sort)

## What we learned

1. **NMS 0.6** is box-NMS inside each Keypoint R-CNN, not cross-model fusion.
2. **GT severity is not a file** — Eq. 1 from v6 keypoint JSON.
3. **v6 GT ≠ paper x-sort** (ICC ≈ 0 between them). Always `--gt-slot-convention pca`.
4. **Pairing 3 models** is the ICC bottleneck (oracle 0.79 vs honest ~0.57).
5. **CEJ-nearest pairing** is not always better than slot-index (`lr` 0.41 vs 0.56).
6. **Train ICC ~0.02** with test 0.50 is a red flag (outliers / invalid combos). Eq. 1 now clips to **[0, 100]** and rejects intersection not between CEJ and apex.

## Paper vs us

| Stage | Ours | Paper |
|-------|------|------:|
| YOLO mAP50 | 0.873 | 0.963 |
| CEJ OKS | 0.927 | 0.954 |
| INT OKS | 0.894 | 0.912 |
| Apex OKS | 0.871 | 0.815 |
| Severity ICC (test) | **~0.50–0.57** | **0.801** |

## Command

```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_friend_full.sh
```
