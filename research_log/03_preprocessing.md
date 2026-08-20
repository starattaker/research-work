# 03 — Preprocessing (chapter log)

**Script:** `src/preprocess/prepare_dataset.py`  
**Validation:** `scripts/validate_preprocessed.py`  
**QA visuals:** `scripts/visualize_yolo_labels.py`, `scripts/visualize_intersections.py`  
**Output:** `data/processed/`

This file tracks preprocessing **versions**, issues found, and numerics for the methods section of the replication paper.

---

## Outputs (all versions)

### YOLO detection (`yolo_detection/`)

- Classes: `0 = single`, `1 = double` (inferred — not in DenPAR raw JSON)
- Splits: `train/`, `val/`, `test/` with `images/` + `labels/`

### Keypoint R-CNN (`keypoints/{cej,intersection,apex}/`)

Per-image JSON: `bboxes`, `labels` (1=single, 2=double), `keypoints` (2 × `[x,y,vis]`, sorted by x)

---

## Chapter 1 — Initial preprocessing (v1, 2026-08-12)

### Strategy

| Step | Rule |
|------|------|
| CEJ / apex → tooth | Point inside bbox with **8 px margin**; if multiple boxes, nearest center; if none, nearest center globally |
| Root type | **Not in DenPAR** — inferred: ≥2 visible apex → double (`label=2`, YOLO class 1), else single |
| Intersection | Bone line × tooth mask contour segment intersection; **fallback:** nearest contour point to **line midpoint** |
| Invalid keypoints | `(0,0,0)` visibility 0 |

**Rationale for 8 px margin (v1):** heuristic to keep apex/CEJ near tooth edge when expert click sits just outside the DenPAR bbox (derived from mask). **Not from paper** — arbitrary default.

### v1 numerics (full dataset, 4,402 teeth)

| Metric | Value |
|--------|-------|
| Images (train / val / test) | 650 / 150 / 200 |
| Teeth | 4,402 |
| Single / double | 3,332 / 1,070 (75.7% / 24.3%) |
| 0 apex / 1 apex / ≥2 apex | 1,134 / 2,198 / 1,070 |
| Apex outside strict bbox (margin=0) | 1,347 points |
| Apex in multiple bboxes (margin=8) | 362 points |

Per split (root labels):

| Split | Teeth | Single | Double |
|-------|-------|--------|--------|
| train | 2,883 | 2,204 | 679 |
| val | 655 | 494 | 161 |
| test | 864 | 634 | 230 |

### v1 issues found (2026-08-13 QA)

1. **Root type not in dataset** — we infer from apex count; authors' repo expects pre-built `labels` but publishes no prep script.
2. **Overlapping bboxes steal apices** — e.g. test image **1194**: DenPAR has 4 apex / 4 teeth; v1 assigns 2 apex to tooth 0 → double, 0 to tooth 1 → single (overlap + nearest-center rule).
3. **8 px margin** expands bbox — points strictly outside raw box can still match; user rejected this for replication fidelity.
4. **Intersection fallback (midpoint)** — when bone line polyline misses contour, nearest point to **line center** can land on wrong tooth surface (visible in `intersection_qa/` early figures).

---

## Chapter 2 — QA tooling (2026-08-13)

| Tool | Output | Purpose |
|------|--------|---------|
| `visualize_yolo_labels.py` | `research_log/figures/yolo_qa/` | Bbox color + apex dots + single/double tag |
| `visualize_intersections.py` | `research_log/figures/intersection_qa/` | Bone lines, contours, intersection points |

8-image YOLO QA (test): **0 label mismatches** vs apex rule (internally consistent, not necessarily clinically correct).

10-image intersection QA (test): generated `test_270`, `1227`, `431`, `808`, `1065`, `1082`, `622`, `1112`, `361`, `682`.

---

## Chapter 3 — Preprocessing revision (v2, 2026-08-13)

### Changes

| Component | v1 | v2 |
|-----------|----|----|
| Bbox point match | 8 px margin | **Strict raw DenPAR bbox** (margin = 0) |
| Intersection hit | Polyline segment × contour | Same (unchanged) |
| Intersection miss | Nearest contour to **midpoint** | **Extend bone-line endpoint** (tooth-side end) along line direction until ray hits contour |
| Intersection last resort | (midpoint nearest) | Nearest contour to **that endpoint** |

**Endpoint selection:** bone-line endpoint closer to the tooth bbox center; extension direction = tangent at that end toward the tooth.

### v2 numerics (2026-08-13, margin=0 + endpoint intersection fallback)

| Metric | Value |
|--------|-------|
| Single / double | **3,181 / 1,221** (72.3% / 27.7%) |
| 0 apex / 1 apex / ≥2 apex | **1,292 / 1,889 / 1,221** |
| Apex outside strict bbox | **1,347** (point-level; unchanged from v1) |
| Apex in multiple bboxes | **226** (strict bbox; was 362 with margin=8) |

Comparison vs v1:

| Metric | v1 | v2 | Δ |
|--------|----|----|---|
| Single / double | 3,332 / 1,070 | 3,181 / 1,221 | −151 / +151 |
| 0 apex / 1 apex / ≥2 apex | 1,134 / 2,198 / 1,070 | 1,292 / 1,889 / 1,221 | +158 / −309 / +151 |
| Apex in multiple bboxes | 362 | 226 | −136 |

Per split (single / double):

| Split | v1 | v2 |
|-------|----|----|
| train | 2,204 / 679 | 2,100 / 783 |
| val | 494 / 161 | 479 / 176 |
| test | 634 / 230 | 602 / 262 |

**Why so many 0-apex teeth (v2: 1,292 ≈ 29%)?** Not missing teeth in DenPAR — apex points exist in the image JSON but fail assignment:
- Apex click is **outside** the raw tooth bbox (1,347 apex *points* fall outside some bbox; after nearest-center assignment some teeth still end up with 0).
- **Overlap / stealing:** neighbor tooth wins nearest-center tie → donor tooth gets 0 apex → labeled **single** by rule.
- **Strict bbox (v2):** more points excluded than v1 margin=8 → **+158** teeth with 0 apex vs v1.

**Note:** YOLO weights in `runs/detect/.../best.pt` were trained on **v1 labels**. Re-train detection on v2 labels before comparing to paper mAP.

### Re-run

```powershell
$env:PYTHONPATH='.'
.\venv\Scripts\python.exe -m src.preprocess.prepare_dataset
.\venv\Scripts\python.exe scripts\validate_preprocessed.py
.\venv\Scripts\python.exe scripts\visualize_intersections.py --split test --n 10 --seed 7
```

---

## Chapter 4 — Mask + KNN + grace (v3, 2026-08-14) — **implemented**

**Status:** Implemented in `prepare_dataset.py` (`--strategy v3`). Run comparison: `python scripts/compare_preprocessing.py`.

### Passes

| Pass | Rule |
|------|------|
| 1 | Assign CEJ/apex if point lies **on tooth segmentation mask** |
| 2 | Else nearest mask pixel within **grace_px** (default 4) |
| 3 | Beyond grace → **drop** point (no assignment) |
| 4 | Tie-break overlapping teeth: nearest **mask centroid** (not bbox center) |
| 5 | Root type from apex count after cleaning; intersection same as v2 |

### Re-run

```bash
python -m src.preprocess.prepare_dataset --strategy v3 --output-root data/processed_v3 --grace-px 4
python scripts/compare_preprocessing.py
```

See [preprocessing_comparison.md](preprocessing_comparison.md) for v1/v2/v3 table.

---

## Chapter 5 — Planned v3 cleaning (mask + KNN + grace region) — **superseded by Chapter 4**

*(Original planning notes retained in git history.)*

- FDI-based root type from metadata vs apex-count inference
- Re-train YOLO on v2/v3 labels when root labels stabilize

---

## DenPAR fields used vs created

| Field | Source |
|-------|--------|
| `bboxes`, `CEJ_Points`, `Apex_Points` | DenPAR JSON |
| `Bone_Lines` | DenPAR bone JSON |
| Tooth masks | DenPAR `Masks (Tooth-wise)/` |
| **single / double** | **Created by us** |
| **intersection keypoints** | **Created by us** |
