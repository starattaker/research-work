# ICC replication — compact context (Aug 2026)

## Paper target
- Test ICC **0.801** (train 0.851, val 0.824) — Sci Rep 2026 DenPAR

## NMS (what it is)
- **Non-max suppression on detection boxes** @ IoU 0.6 inside each Keypoint R-CNN forward pass.
- Removes duplicate tooth detections; **not** cross-model keypoint fusion.
- After NMS: pick detection with max IoU to YOLO box → 2 keypoints per model.

## Combining 3 models (no masks)
1. CEJ / INT / APEX models run on same YOLO ROI (`inference_mode=roi`, paper path).
2. Each outputs 2 keypoints (tensor rows arbitrary).
3. **Paper rule** (`dataset.py`): `sorted(keypoints, key=lambda x: x[0])` → slot0=left, slot1=right.
4. Severity per side: CEJ[i] + INT[i] + APEX[i] → Eq. 1.
5. ICC rows: **both sides** when both valid (`--severity-protocol both_sides`).

## Expert / GT severity — where is it?
- **Not stored** as a severity file.
- Computed on-the-fly from `data/processed_v6/keypoints/{cej,intersection,apex}/{split}/annotations/*.json`.
- `--gt-slot-convention paper_x` matches paper; `pca` = v6 preprocess slots.

## Commands (friend GPU)
```bash
cd ~/faraz/Test_work/research-work && source venv/bin/activate && export PYTHONPATH=.
bash scripts/run_icc_all_splits.sh
python scripts/sweep_icc_combine.py --cej-weights runs/keypoints/v6_cej/best.pt \
  --intersection-weights runs/keypoints/v6_intersection/best.pt \
  --apex-weights runs/keypoints/v6_apex/best.pt
python scripts/diagnose_severity_icc.py --cej-weights ... (same three)
```

## full vs roi
- **roi**: YOLO box → RoI Align heads only (paper Fig. emphasis).
- **full**: full RPN forward, then match detection to YOLO box (OKS eval path).
- Default ICC script now uses **roi + paper_x**.

## Root cause of low ICC (~0.05)
- Tensor slot order ≠ anatomical side; oracle 8-combo ~0.78 ceiling.
- Fix: paper x-sort combine (not masks).

## Paper repo (`external/paper_repo`)
- Cloned; severity notebooks only; no published ICC script.
- Confirms x-sort in `Keypoint Detection/dataset.py`.
