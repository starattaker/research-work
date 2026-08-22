# Intersection logic — bug analysis (2026-08-20)

**Script:** `scripts/visualize_intersection_logic.py`, `scripts/debug_intersection_cases.py`  
**Figures:** `figures/intersection_logic/`, `figures/intersection_debug/`

## Algorithm (v2+, unchanged in v3/v4)

For each expert bone line:

1. **Pair teeth** — x-mean of polyline falls between adjacent bbox centers (left, right).
2. **Per tooth in pair** — point on that tooth's mask contour:
   - **Direct (D):** polyline segment ∩ contour segment → pick hit nearest bone-line midpoint.
   - **Ray (R):** no direct hit → from tooth-side endpoint, cast ray along **polyline tangent** until contour hit; pick nearest to anchor.
   - **Nearest (N):** no ray hit → closest contour point to anchor.

Test set method mix (1,300 intersection points): **455 D / 790 R / 55 N** (~61% ray).

## Reported cases

### test_51 — T1 spurious R

| Bone line | Pair | Method | Result | Issue |
|-----------|------|--------|--------|-------|
| B0 | T0,T1 | **RAY** | (578, 361) | Line spans x=8→179 (entirely on T0 side). No direct hit on T1. Ray from anchor p9 (179,431) shoots +x for **405 px** → hits T1 crown, not crest. |
| B1 | T0,T1 | D | (569, 432) | Correct — line segment crosses T1 contour. |
| B2 | T1,T2 | D | (700, 414) | Correct. |

**Root cause:** B0 is paired with T1 because line midpoint x≈94 lies between T0/T1 bbox centers, but geometry is only near T0. Ray tangent continues horizontally into T1 at wrong height.

**Debug crop:** `figures/intersection_debug/debug_51_T1_B0.jpg`

### test_5 — T1 spurious R (B3)

| Bone line | Pair | Method | Result | Issue |
|-----------|------|--------|--------|-------|
| B0 | T0,T1 | R | (474, 778) | Ray dist 0.7 px — OK (near anchor). |
| B1 | T1,T2 | D | (634, 730) | OK. |
| B3 | T1,T2 | **RAY** | (614, 670) | Short line at y≈590 between T1/T2. Ray from anchor shoots **190 px** left-down → hits T1 mesial wall, not near bone line. |

**Debug crop:** `figures/intersection_debug/debug_5_T1_B3.jpg`

## Root causes (summary)

1. **Pairing by line midpoint** — one long bone polyline can sit entirely over one tooth while still pairing two teeth by x-range.
2. **Ray along polyline tangent** — when direct miss, ray follows bone-line direction (not toward tooth center or perpendicular to crest), so it can travel hundreds of pixels and hit an unrelated contour segment.
3. **Multiple intersections per tooth** — each adjacent bone line adds a point; `pad_keypoints(..., 2)` keeps leftmost two by x, which may retain spurious ray hits and drop valid ones.

## Overloaded teeth (>3 raw keypoints)

Count on **test** (CEJ + apex + intersection, v2 bbox assign): **526 teeth** (of 864).

Typical pattern: 1 CEJ + 1 apex + **2 intersections** (from two bone lines) = 4 → truncated to 2 per type after pad.

| Type | Count in overloaded set |
|------|-------------------------|
| total=4 | majority (2 int + 1 CEJ + 1 apex) |
| total≥5 | multi-CEJ, multi-apex, or 3+ intersections |

**Figures:** `figures/overloaded_teeth/` (one image per overloaded tooth)  
**JSON:** `figures/overloaded_teeth/summary_test.json`, `figures/intersection_debug/overloaded_teeth.json`

## Possible fixes (not implemented)

| Fix | Idea |
|-----|------|
| A | Skip ray if anchor farther than N px from tooth bbox |
| B | Ray toward **tooth mask centroid**, not polyline tangent |
| C | Pair bone line only to teeth whose bbox **intersects** line bounding box (or within margin) |
| D | One intersection per tooth: pick best bone line by distance to tooth |
| E | Clip ray max length (e.g. 30 px) before fallback to nearest |

## Commands

```powershell
python scripts/visualize_intersection_logic.py --split test --all
python scripts/debug_intersection_cases.py
python scripts/visualize_overloaded_teeth.py --split test
```
