# ICC — compact notes (2026-08-31)

## Target
Test ICC **0.801**. Honest now **~0.50–0.57**. Oracle **~0.79**.

## Do not use
- `gt_slot_convention=paper_x` with processed_v6 (GT is PCA).
- Oracle 8-combo at inference (cheats with GT severity).

## Guards in Eq. 1
Clip [0, 100]; drop INT not between CEJ and apex (cross-side combos).

## Combine modes
`hungarian` (assign INT/APEX to CEJs by distance) · `tensor` · `lr` · `mask_pca` · `paper_x`

## One command
```bash
cd ~/faraz/Test_work/research-work && bash scripts/run_icc_friend_full.sh
```
