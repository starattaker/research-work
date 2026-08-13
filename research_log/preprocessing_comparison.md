# Preprocessing comparison (v1 / v2 / v3)

**BBox count is identical across strategies** — all use raw DenPAR tooth boxes (4,402 teeth). Differences are **apex/CEJ assignment** → single/double labels and 0-apex teeth.

| Strategy | Teeth | Single | Double | 0 apex | 1 apex | ≥2 apex | % 0 apex |
|----------|------:|-------:|-------:|-------:|-------:|--------:|---------:|
| **v1** (8 px margin) | 4,402 | 3,332 | 1,070 | 1,134 | 2,198 | 1,070 | 25.8% |
| **v2** (strict bbox) | 4,402 | 3,181 | 1,221 | 1,292 | 1,889 | 1,221 | 29.4% |
| **v3** (mask + 4 px grace) | 4,402 | *run script* | — | — | — | — | — |

Refresh v3 row:

```bash
python scripts/compare_preprocessing.py
```

## What each version changes

| | Apex/CEJ assignment | Intersection fallback |
|--|---------------------|------------------------|
| **v1** | Point inside bbox **+ 8 px margin** | Contour hit, else nearest to **line midpoint** |
| **v2** | **Strict** DenPAR bbox (margin 0) | Contour hit, else **endpoint extension** |
| **v3** | Point on **tooth mask**, else nearest mask within **4 px** | Same as v2 |

## Interpretation

- **v2 vs v1:** Removing the 8 px margin adds **+158 teeth with 0 apex** (apex clicks fall outside strict bbox). Double-root count rises (+151) because fewer apex points attach per tooth under strict rules.
- **v3 goal:** Recover valid near-mask apex/CEJ without v1’s loose margin; drop orphan points beyond 4 px grace.

## Re-preprocess commands (separate folders — do not overwrite)

```bash
python -m src.preprocess.prepare_dataset --strategy v1 --output-root data/processed_v1
python -m src.preprocess.prepare_dataset --strategy v2 --output-root data/processed_v2
python -m src.preprocess.prepare_dataset --strategy v3 --output-root data/processed_v3 --grace-px 4
```
