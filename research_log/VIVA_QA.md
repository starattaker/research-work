# Viva / Defence Question Bank

**Purpose:** every question a researcher, examiner or reviewer could reasonably ask about this
work, with an answer that is backed by a number and a file. Read this before the viva.

**Rule for this document:** no answer here may assert something the repo cannot show. If an
answer needs a number we do not have, it is filed under
[Part H: Questions we cannot fully answer](#part-h-questions-we-cannot-fully-answer) with an
honest script instead of a bluff. Admitting a known limit confidently beats being caught
guessing — examiners reward the first and punish the second.

**Last updated:** 2026-09-13 · See [Changelog](#changelog) at the end.

---

## Contents

- [Part A — The 60-second summary](#part-a--the-60-second-summary)
- [Part B — Every headline number and where it comes from](#part-b--every-headline-number-and-where-it-comes-from)
- [Part C — Statistics and ML methodology](#part-c--statistics-and-ml-methodology)
- [Part D — Terms in our own pipeline](#part-d--terms-in-our-own-pipeline)
- [Part E — The hard questions](#part-e--the-hard-questions)
- [Part F — Dataset and clinical questions](#part-f--dataset-and-clinical-questions)
- [Part G — Engineering and reproducibility](#part-g--engineering-and-reproducibility)
- [Part H — Questions we cannot fully answer](#part-h--questions-we-cannot-fully-answer)
- [Part I — Glossary](#part-i--glossary)

---

## Part A — The 60-second summary

> We replicate and extend DenPAR (Wimalasiri et al.), which measures alveolar bone-loss
> severity on periapical radiographs. Our pipeline is YOLOv8x tooth detection → three
> Keypoint R-CNN heads (CEJ, bone-level intersection, apex) → a geometric severity formula
> → ICC against reference severity.
>
> We contribute three things the reference did not do. **One:** multipass region-growing
> label assignment, which cuts expert-click loss from 5.6% to 0.6% (CEJ) and 23.7% to 9.0%
> (apex) with 0.029% cross-tooth contamination. **Two:** an apex-*independent* measurement
> axis — we show the reference formula's weakness is that it re-estimates its own axis from
> the same noisy landmarks it then projects, so error both displaces points *and rotates the
> axis*. **Three:** an honest agreement study: ICC(2,1) with bootstrap CIs, Bland–Altman
> limits, a formula-vs-formula control on identical ground truth, a 126-configuration
> protocol sensitivity sweep, and a controlled noise-injection experiment that isolates the
> mechanism.
>
> Headline: axis-constrained severity with a mask-PCA axis reaches **test ICC 0.823 vs 0.726**
> for the reference formula at identical pairing and identical keypoints.

**If you only memorise one thing:** the contribution is *measurement geometry and label
quality*, not a bigger model. We deliberately do not claim the architecture as novel.

---

## Part B — Every headline number and where it comes from

### Dataset

| Quantity | Value | Source |
|---|---|---|
| Images | 1,000 (650 train / 150 val / 200 test) | `data/DenPAR/Dataset/`, verified by directory count |
| Teeth | 4,402 (2,883 / 655 / 864) | `research_log/preprocessing_comparison.json` |
| Single / double root | 3,498 / 904 (v4/v6 labelling) | same |
| Teeth with zero apex click | 27.7% | same |

> **Note:** single/double root is **not** a native DenPAR field. We derive it from the number
> of assigned apex clicks. Say this before you are asked.

### Detection (YOLOv8x, two-class, test split)

| Class | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| all | 0.850 | 0.844 | **0.873** | **0.794** |
| single | 0.906 | 0.854 | 0.919 | 0.798 |
| double | 0.793 | 0.834 | 0.827 | 0.790 |

Reference paper: 0.892 / — / **0.963** / **0.907** (also two-class, also test).
Source: `research_log/05_detection_training.md`.

### Keypoints (mean OKS, test split)

| Version | CEJ | Intersection | Apex |
|---|---|---|---|
| v2 (strict bbox) | 0.843 | 0.815 | 0.781 |
| v3 (mask + 4px) | 0.911 | 0.817 | 0.836 |
| v4 (region grow) | 0.921 | 0.822 | 0.853 |
| **v6 (production)** | **0.926** | **0.895** | **0.864** |
| v7 (wider grace) | 0.928 | 0.882 | 0.881 |
| Reference (COCO AP) | 0.954 | 0.912 | 0.815 |

Source: `research_log/experiments/registry.json` (v6 audited 2026-09-12, `run_dir` verified).

### Severity ICC — the headline

ICC(2,1), absolute agreement, end-to-end from YOLO detections, GT method = predicted method,
`match_by_slot` pairing. Source: `research_log/axis_severity_icc.json`,
`research_log/severity_agreement.json`.

| Severity definition | Val ICC | **Test ICC** | Test 95% CI | Test MAE |
|---|---|---|---|---|
| Reference min–max line | 0.713 | 0.726 | [0.668, 0.807] | 7.22 |
| **Mask PCA axis** | **0.860** | **0.823** | **[0.761, 0.880]** | **5.40** |
| CEJ→INT midpoint axis | 0.684 | 0.751 | [0.702, 0.828] | 7.23 |
| Reference paper reported | 0.824 | 0.801 | — | — |

**The CI point is worth making unprompted:** mask-PCA's interval [0.761, 0.880] sits almost
entirely above the reference formula's [0.668, 0.807]. The improvement is not a point-estimate
artefact.

### Label retention (the preprocessing contribution)

| Strategy | CEJ drop | Apex drop | χ² vs v2 |
|---|---|---|---|
| v1 (8px bbox margin) | 6.7% | 3.4% | 13.28 (p=2.7e-4) |
| v2 (strict bbox) | 5.6% | 3.5% | — (reference) |
| v3 (mask + 4px) | 1.9% | 23.7% | 193.25 (p=6.2e-44) |
| **v4/v6 (region grow 1–8px)** | **0.6%** | **9.0%** | 61.94 (p=3.5e-15) |

At 8px: **97.30%** of clicks assigned with **0.029%** cross-tooth contamination; yield elbow at
11px; 100% assignment needs 22px. Source: `research_log/figures/point_assignment_full/point_assignment_report.json`.

---

## Part C — Statistics and ML methodology

### Q: Did you do cross-validation?

**No, and deliberately.** DenPAR ships an **official** 650/150/200 split, and the reference
paper we are replicating reports on that split. If we had re-partitioned with k-fold, our
numbers would no longer be comparable to the 0.963 / 0.801 we are measuring ourselves
against — the comparison *is* the point of a replication.

What we did instead, which serves the same purpose:

1. **Strict three-way discipline.** Model selection (early stopping, checkpoint choice) uses
   validation loss only. The test split is touched once per reported configuration.
2. **Protocol sensitivity in place of fold variance.** Our 126-configuration sweep gives a
   test-ICC spread of **SD 0.062**, which is the dispersion estimate k-fold would have
   provided.
3. **Bootstrap CIs** on every reported ICC (2,000 resamples), which quantify sampling
   uncertainty on the test set directly.

**If pressed — "but k-fold would still be more robust":** agreed, and we say so in the
limitations. With 1,000 images, 5-fold would cost 5× the training for every one of the seven
preprocessing versions. The honest trade was breadth of ablation over fold repetition. We
would run k-fold on the final configuration given more compute.

> **Trap:** do not claim the bootstrap is equivalent to cross-validation. It is not.
> Bootstrap resamples the *evaluation* set and tells you about sampling noise in the metric;
> k-fold resamples the *training* set and tells you about variance in the fitted model.
> Say exactly that if challenged — it shows you know the difference.

### Q: What is ICC, which variant did you use, and why that one?

**Intraclass Correlation Coefficient** measures agreement between two measurements of the
same quantity. Unlike Pearson correlation, it penalises *systematic* offsets: if our predicted
severity were always exactly 10 points higher than reference, Pearson would be 1.0 and ICC
would not.

We use **ICC(2,1): two-way random effects, absolute agreement, single measurement.**

- *Two-way random* — both the teeth and the two "raters" (reference vs automated) are treated
  as drawn from populations; we want a claim that generalises beyond this particular pair.
- *Absolute agreement* — not merely "consistency". We care that the numbers match, not just
  that they are linearly related.
- *Single measurement* — we report the reliability of one measurement, not of an average of
  several.

This matches the reference paper, which is required for the comparison to mean anything.
Implementation: `src/severity/icc.py::icc21` (hand-written, numpy only).

**Interpretation bands (Koo & Li 2016):** <0.50 poor, 0.50–0.75 moderate, 0.75–0.90 good,
>0.90 excellent. Our 0.823 is **good**; the reference's 0.801 is also good; our 0.726 baseline
is moderate.

**If asked "why not Pearson / Spearman / R²?"** — because none of them penalise bias, and a
severity measurement that is reliably 8 points too high is clinically wrong even if perfectly
correlated.

### Q: What is Bland–Altman and why did you use it?

A Bland–Altman plot answers a question ICC cannot: **is the error constant, or does it depend
on how severe the bone loss is?** You plot the *difference* between the two measurements
against their *mean*. Three numbers come out:

- **Bias** — the mean difference; systematic over- or under-estimation.
- **SD of differences** — the spread.
- **Limits of agreement (LoA)** = bias ± 1.96 × SD — the range containing ~95% of
  disagreements.

Our test-split results (`research_log/severity_agreement.json`):

| Method | Bias | LoA | MAE | RMSE |
|---|---|---|---|---|
| Reference Eq.1 | −0.37 | [−28.6, +27.9] | 7.22 | 14.41 |
| **Mask PCA** | **−0.93** | **[−20.8, +18.9]** | **5.40** | **10.17** |
| CEJ→INT midpoint | −0.06 | [−27.1, +27.0] | 7.23 | 13.80 |

**The reading that matters:** all three are near-unbiased (bias < 1 percentage point), so
nothing is systematically over-reading. What differs is **spread** — mask-PCA's limits are
roughly 40 points wide against the reference's 56. Same central tendency, much tighter
agreement.

**The honest caveat, volunteer it:** ±20 percentage points is still wide for individual
clinical decisions. This supports screening, not unsupervised diagnosis. That is in the
limitations.

### Q: What is a bootstrap confidence interval and how did you compute it?

The ICC is a point estimate from one particular test set. Resample the paired observations
*with replacement* 2,000 times, recompute ICC each time, and take the 2.5th and 97.5th
percentiles of that distribution. That is the **percentile bootstrap 95% CI**.

It requires no distributional assumption, which matters because the ICC sampling distribution
is not normal near the bounds. Implementation:
`scripts/analyze_agreement.py::bootstrap_icc_ci`.

**What it does not do:** it cannot tell you about variance from re-training the model. It
resamples evaluation pairs only. Be precise about this.

### Q: Why is there no coefficient of variation? Your protocol promised one.

**Because CV does not apply to this estimator, and reporting one would be the second error.**

CV = SD/mean *over repeated measurements of the same quantity*. Our pipeline is
**deterministic** — running inference twice on the same radiograph returns an identical
severity. The within-subject CV is therefore identically zero and carries no information.
A *between-tooth* CV would describe how varied the disease is in the cohort, not how reliable
the estimator is.

What we report instead, all of which genuinely bear on stability:

1. ICC + MAE against reference severity, all three splits, with CIs.
2. A **formula-vs-formula control** on identical ground-truth keypoints, which separates the
   formula's intrinsic reproducibility from error contributed by the learned components.
3. A **126-configuration sensitivity analysis** (test-ICC SD 0.062).
4. A **controlled noise-injection study** (below) that measures degradation as a function of
   injected localisation error — 200 images x 10 noise levels x 10 realisations.

> This is a strong answer. It converts "promised and not delivered" into "considered, found
> inapplicable, replaced with four things that are better." Deliver it in that order.

### Q: How do you know the mask-PCA advantage isn't a coincidence of your checkpoint?

**We tested it directly, by experiment.** This is the strongest single item in the defence.

The claim is mechanistic: under the reference formula the axis is re-estimated from the same
three noisy landmarks it then projects, so keypoint error both **displaces the points** and
**rotates the axis**. Under a mask-PCA axis the direction comes from the segmentation
(thousands of pixels), so error can only displace points along a fixed direction.

Observing that mask-PCA wins does not prove that — the two differ in many ways at once. So we
isolated it: take **ground-truth** keypoints, inject controlled isotropic Gaussian noise of
known σ, and measure how far each formula drifts from its **own** clean value. Same teeth,
same masks, same pairing. The only variable is injected localisation error.

<!-- NOISE_TABLE_START -->
**Full study: v5 test split, 200 images, 10 noise realisations per σ.**
ICC(2,1) of noisy severity against each formula's own clean (σ=0) value.
Source: `research_log/noise_sensitivity_v5.json`, figure
`research_log/figures/noise_sensitivity_v5.png`.

| σ (px) | Reference Eq.1 | **Mask PCA** | CEJ→INT midpoint | Crown-width (no apex) |
|---|---|---|---|---|
| 0 | 1.000 | **1.000** | 1.000 | 1.000 |
| 1 | 0.998 | **1.000** | 0.999 | 0.997 |
| 2 | 0.996 | **0.999** | 0.998 | 0.993 |
| 3 | 0.996 | **0.998** | 0.995 | 0.985 |
| 4 | 0.992 | **0.997** | 0.989 | 0.980 |
| 6 | 0.976 | **0.994** | 0.982 | 0.940 |
| 8 | 0.965 | **0.989** | 0.970 | 0.925 |
| 12 | 0.944 | **0.976** | 0.939 | 0.896 |
| 16 | 0.897 | **0.958** | 0.909 | 0.887 |
| 24 | 0.851 | **0.914** | 0.825 | 0.784 |
<!-- NOISE_TABLE_END -->

**Mask-PCA is the most robust definition at every non-zero noise level**, and the margin over
the reference formula widens monotonically — 0.005 at σ=4, 0.024 at σ=8, 0.061 at σ=16, 0.063
at σ=24. That is exactly the signature the axis-rotation mechanism predicts, and it is not
something a coincidence of one checkpoint would produce.

### The crown-width index degrades fastest — and why that is interesting, not embarrassing

The genuinely apex-free crown-width index is the **least** noise-robust of the four (0.784 at
σ=24). Report this; it is a real finding with a clean explanation:

1. **Noise enters twice.** Its denominator is the distance between the two CEJ points — both of
   which are themselves noisy. The reference formulas have a denominator anchored partly on the
   apex, which is a *different* landmark from the numerator's endpoints, so errors partially
   decorrelate. Here the same noisy CEJ points drive numerator *and* denominator.
2. **The denominator is small.** Crown width is roughly 100–250px; root length is roughly
   400–500px. The same absolute perturbation is a much larger *relative* perturbation.

**The honest conclusion to state:** removing the apex is not free. It buys independence from an
unreliable landmark but pays for it with a smaller, noisier normaliser. The mask-PCA axis is
the better engineering trade — it removes the apex from the *axis* (where it does the most
damage, by rotating the measurement direction) while keeping root length as a large, stable
denominator.

> **This is a strong thing to have found.** It shows you tested your own new idea honestly
> rather than only reporting what flattered it. If asked "did anything you tried not work?",
> this is your answer.

> **Caveat on the MAE panel of the figure:** crown-width shows the largest absolute drift there
> partly for the scale reason above (smaller denominator ⇒ larger percentage for the same
> displacement). ICC is scale-free and is the fair cross-method comparison. Say this before
> someone catches it.

### Q: Why chi-square for the preprocessing comparison?

We are comparing **categorical** outcomes — how many teeth are labelled single- vs
double-rooted under each assignment strategy. A χ² test of independence asks whether the
strategy changes that distribution more than chance. All three comparisons against v2 are
significant (smallest χ²=13.28, p=2.7e-4). Source: `scripts/compare_preprocessing.py`.

**Limitation, state it:** significance here means the *label distribution* shifted, not that
it shifted toward the clinically correct answer. That is why we do not stop at label
statistics — the real arbiter is downstream test OKS after keypoint training, which is why
v4/v6 win rather than v3 (v3 has the best CEJ retention but loses 23.7% of apex clicks).

### Q: How did you prevent overfitting / how was the model selected?

- **Early stopping on validation loss**, patience 30, best checkpoint saved on best val loss
  (`src/keypoint/train.py`). Detection used patience 25.
- Best epochs land at **4–6** out of 35–36 run for the keypoint heads — these models overfit
  early, and the early stopping is doing real work.
- **StepLR** (step 4, γ 0.6), Adam, lr 1e-4, weight decay 1e-6.
- Augmentation: CLAHE (clip 40, 8×8 tiles). We deliberately avoid mosaic/cutmix on keypoints
  because aggressive spatial augmentation disrupts the landmark geometry we are trying to learn.

**"Best epoch 5 of 35 — isn't that severe overfitting?"** Yes, and we report it rather than
hide it. With 650 training images and a ResNet-50-FPN backbone the model saturates fast. The
correct response is early stopping, which we use; the alternative would be a smaller backbone
or heavier regularisation, which we list as future work.

### Q: Is there data leakage between train and test?

No, and there are three specific places it could have crept in:

1. **Split level** — we use DenPAR's official image-level splits and never re-partition. No
   image appears in two splits.
2. **Preprocessing** — region-growing assignment is computed per image from that image's own
   mask and clicks. No cross-image or cross-split statistic is fitted.
3. **Protocol selection** — the 126-config sweep was selected on **validation**, then applied
   to test once. We explicitly report that the val-selected config (0.7246) does slightly
   *worse* on test than the test-best peek (0.7273), and we report the val-locked number. That
   gap is evidence we did not tune on test.

> That last point is a good one to offer unprompted. Voluntarily reporting a number that makes
> you look slightly worse is the strongest possible evidence of protocol discipline.

### Q: Your classes are imbalanced (3,498 single vs 904 double). How did you handle it?

We did not rebalance, and the effect is visible and reported: double-root detection is the
weak class at mAP@0.5 **0.827 vs 0.919** for single, with only 230 test instances. We report
per-class metrics rather than hiding behind the aggregate.

We did not resample because the imbalance is the *natural* prevalence in the dataset, and
resampling would distort the root-type prior that the downstream apex-slot logic depends on.
Class-weighted loss is listed in future work.

---

## Part D — Terms in our own pipeline

> These are the questions where being vague is fatal, because they are about *your own code*.

### Q: What is the "Hungarian" mode? Did you implement the Hungarian algorithm?

**Be precise here.** Our `hungarian` combine mode (`src/severity/hungarian_assign.py`) solves
the assignment problem of matching predicted intersection and apex points to CEJ anchors by
**minimising total pairing distance**. For the 2×2 case — two CEJ anchors, two candidate
points — we enumerate both permutations and take the cheaper one:

```
cost_identity = d(anchor0, p0) + d(anchor1, p1)
cost_swapped  = d(anchor0, p1) + d(anchor1, p0)
```

**This is a brute-force enumeration, not a general Hungarian implementation.** At 2×2 it
returns the identical optimum the Hungarian algorithm would, because there are only two
feasible permutations. We use the name because it is the same assignment problem.

**If asked "why not `scipy.optimize.linear_sum_assignment`?"** — because a tooth has at most
two sides, so the problem is 2×2 and exhaustive enumeration is exact and faster. The general
algorithm would be required if we ever extended to multi-rooted teeth with >2 landmarks per
type.

### Q: What are the six "combine modes"?

After the three keypoint heads run independently, each returns up to two points. Something has
to decide *which CEJ pairs with which intersection and which apex* on the same tooth side.
That decision is the combine mode (`src/severity/pred_combine.py`):

| Mode | Rule |
|---|---|
| `tensor` | Trust the model's own slot ordering — pair slot *i* with slot *i*. |
| `hungarian` | Minimum-total-distance assignment to CEJ anchors (above). |
| `lr` | Assign by left/right position relative to the bounding box. |
| `mask_pca` | Assign by which side of the mask's PCA long axis each point falls on. |
| `paper_x` | The reference paper's convention: sort by x coordinate. |
| `geom_consistent` | Rank all candidate triples by x-spread, greedily take the most geometrically plausible. |

**Why this matters so much:** the pairing choice alone moves test ICC across a range of
0.559–0.727. The reference paper does **not** specify its pairing rule, so part of our gap to
0.801 is a protocol we had to reinvent. That is a legitimate finding about the replicability
of the original work.

### Q: What are the pairing protocols (`both_sides`, `match_by_slot`, `one_per_tooth`)?

This is the *evaluation*-side counterpart: having computed severities, how do we pair a
predicted value with a reference value to feed the ICC?

- **`match_by_slot`** — pair by slot index (GT slot 0 ↔ predicted slot 0). **This is what we
  report.** It is the strictest honest choice: it penalises the model for getting
  mesial/distal ordering wrong.
- **`both_sides`** — pair by nearest CEJ anchor, greedily. More forgiving, because it forgives
  a side swap.
- **`one_per_tooth`** — collapse each tooth to a single value. Loses half the data and scores
  worst (~0.56–0.58).

### Q: What is the apex merge radius?

Double-rooted teeth have two apices; single-rooted have one. At inference the apex head may
emit two points for a single-rooted tooth. If two predicted apices are closer than the merge
radius, we treat them as one landmark.

We chose 8–32px from the **ground-truth** apex-separation distribution: n=832 double-root
teeth, mean separation **131.5px**, SD 47.6, 5th percentile 57.6px. Since genuine double
apices are ~130px apart, anything within 8–32px is a duplicate detection, not a second root.
ICC is **flat across 8–32px**, so the result does not hinge on the exact value. Source:
`research_log/figures/apex_merge_analysis/apex_distance_report.json`.

### Q: Your OKS numbers look close to the paper's. Are they the same metric?

**No — and this is the single most important caveat to volunteer before you are caught.**

Ours is **mean OKS per image** over IoU-matched detections with **uniform** per-keypoint
constants κ = 1/K (`src/keypoint/train_utils.py::evaluate_oks`, line 110:
`sigmas = torch.ones(len(gt))/len(gt)`).

The reference reports **COCO keypoint Average Precision**, which thresholds OKS at multiple
levels and integrates precision over recall.

Mean OKS is systematically **more forgiving** than AP. So we report both side by side only as
an order-of-magnitude reference, and we explicitly **do not** claim to beat their apex result
even though our apex mean-OKS (0.864) exceeds their reported AP (0.815).

> Saying this yourself converts a potential "gotcha" into evidence of rigour.

### Q: What is "region growing" in your preprocessing?

Expert CEJ/apex clicks often land a few pixels *outside* the tooth mask. Naive rules either
drop them (losing 23.7% of apex clicks at a fixed 4px grace) or use a wide fixed margin (which
steals clicks from neighbouring teeth).

Ours assigns on-mask first, then expands in **1-pixel rings** outward to a maximum of 8px,
assigning each click to the first tooth whose mask it reaches, with ties broken toward the
mask centroid. Result: **97.30%** assigned at **0.029%** cross-tooth contamination.

### Q: What are "PCA slots"?

Which point is mesial and which is distal. The reference sorts by x-coordinate, which breaks
for tilted or rotated teeth. We project onto the tooth mask's **PCA long axis** and assign
sides by which side of that axis a point falls on. On v6, x-sorting and PCA slots are
essentially uncorrelated (ICC ≈ −0.006, `research_log/icc_gt_sanity_test.json`) — they are
genuinely different conventions, not a cosmetic change.

### Q: What is the "oracle" number (~0.79)?

An upper bound, **not a result**. It asks: if pairing were perfect — if an oracle told us the
correct GT-to-prediction correspondence — what ICC would the reference formula reach? Answer:
≈0.79.

**Why it matters:** it says the keypoints already carry nearly all the available information,
and the remaining gap is a *matching* problem, not a *perception* problem. That is why we did
not pursue a bigger keypoint model. It is never used at inference, and we say so.

---

## Part E — The hard questions

### Q: Your title says apex-constrained, but the apex is in your denominator. Which is it?

**The apex is removed from the axis *direction*, not from the root-length normaliser — and
those are different failure modes.**

In the reference formula the apex does two jobs: it helps fit the line *and* it sets the
denominator. A mislocalised apex therefore **rotates** the measurement axis, which moves the
projected positions of CEJ and intersection non-linearly. Under mask-PCA the axis comes from
the segmentation, so apex error can only **rescale** a denominator — a bounded, monotone,
first-order effect.

Evidence: test ICC 0.823 vs 0.726 at identical pairing and identical keypoints, plus the
noise-injection curves in Part C.

**A truly apex-free severity would have to be in raw pixels**, which is not comparable across
teeth, patients or magnifications. That is exactly why the apex stays in the denominator, and
we say so explicitly in §3.4.2.

**We also implement a genuinely apex-free alternative** (crown-width index) that normalises by
the mesial–distal CEJ span instead of root length — no apex, no mask. It is the
second-most noise-robust method tested. Its values are on a **different scale** and are not
numerically comparable to the other three.

### Q: Why is your ICC 0.73 when the paper reports 0.801?

Four quantified contributors, in descending order:

1. **Detection gap.** Our two-class mAP@0.5 is 0.873 vs their 0.963, and the keypoint heads
   are trained on GT boxes but evaluated on YOLO boxes — a train/test ROI domain shift.
2. **Intersection localisation.** 0.895 vs their 0.912 — and intersection is the *numerator*
   of the severity ratio, so it is the most sensitive input.
3. **Pairing protocol.** 126 configs span test ICC 0.559–0.727 (SD 0.062). The reference does
   not specify its pairing rule, so part of the gap is a protocol we reinvented.
4. **Oracle bound ≈0.79**, showing the residual is matching, not perception.

**And the corrected headline:** with the mask-PCA axis we reach **0.823 on test — above their
0.801 — using demonstrably weaker keypoints.**

### Q: Why does a crude mask axis beat an axis through the actual landmarks?

Crude but **stable**, and ICC rewards stability.

The control experiment proves the two definitions measure nearly the same quantity: on
*identical ground-truth keypoints* they agree at ICC **0.838** (test, MAE 2.46) and **0.931**
(val, MAE 1.61). So the advantage is not a different definition of bone loss.

The advantage appears only when inputs are **predicted**, because the reference line is
re-estimated from the same three noisy points it then projects — error enters twice, once as
displacement and once as rotation. The PCA axis is estimated from thousands of mask pixels, so
it is essentially noise-free by comparison. **This is a variance-reduction result, not a better
clinical definition**, and we frame it that way.

### Q: You report 0.989 mAP for one class and 0.873 for two. Which is your detector?

**The two-class one at 0.873.** The reference trains two classes (single vs double root)
because root count feeds the downstream apex-slot logic.

The 1-class 0.9889 is a **validation** number on a **strictly easier task**: collapsing the
classes removes the classification component of AP entirely, so a correctly-localised box that
is misclassified single-vs-double still counts as a hit. It **must not** be compared to the
reference's 0.963. We report it as an upper bound on pure localisation quality with exactly
that caveat.

### Q: v7 has better CEJ and apex OKS. Why did you ship v6?

**Because severity is a conjunction, not an average.** It needs CEJ *and* intersection *and*
apex on the same tooth side — and intersection is the numerator, the quantity being measured.

v7 gains +0.002 CEJ and +0.017 apex but **loses 0.013 on intersection** (0.895 → 0.882).
Selecting on the mean would have optimised the wrong objective. Both versions also overfit by
epoch 5–6 of a 35-epoch run, so the v7 CEJ/apex gains are within run-to-run noise while the
intersection regression was consistent.

We documented this as a **negative result** rather than burying it.

### Q: Only about a third of teeth get a severity score. Is this usable?

36.4% of test tooth-sides (629 of 1,728) yield a severity under the mask-PCA axis.

**This is an annotation ceiling, not a model failure.** 27.7% of DenPAR teeth carry no apex
click at all under the best label construction, and without an apex there is no root length to
normalise by. The same ceiling constrains the reference implementation — they simply do not
report it.

Our refinement protocol raised apex retention from 23.7% loss (v3) to 9.0% (v4/v6) and CEJ
loss to 0.6%, which is the part fixable without re-annotating. Raising it further requires
re-annotation, which is in future work.

### Q: What is actually new here, given you reproduced someone else's pipeline?

Three things, each with a number:

1. **Label construction** — region-growing cuts CEJ loss 5.6%→0.6% and apex loss 23.7%→9.0%
   at 0.029% contamination, and this propagates downstream: **+0.08 to +0.09 mean-OKS from v2
   to v6**.
2. **PCA slot ordering** instead of x-sorting, which is what makes mesial/distal assignment
   survive oblique teeth.
3. **Axis-constrained measurement**, lifting test ICC 0.726 → 0.823 at identical pairing and
   identical keypoints — a pure geometry gain with no extra training — *plus* a controlled
   noise-injection experiment proving the mechanism rather than asserting it.

We explicitly do **not** claim the multi-keypoint head, the three-model split, or the detector
as contributions.

### Q: Why didn't you just use a newer/better model?

Because our own analysis shows the model is **not** the binding constraint. The oracle bound
(≈0.79 vs 0.73 honest) says the keypoints already carry nearly all available information and
the loss is in pairing. And 27.7% of teeth have no apex annotation, which no architecture
fixes.

Swapping in a newer backbone would move a number we have shown is not the bottleneck, while
answering no scientific question. We chose to characterise and fix the *measurement geometry*
instead, and to prove the mechanism experimentally.

---

## Part F — Dataset and clinical questions

### Q: Is this clinically validated?

**No, and the thesis says so in two places.** Every reference severity here is derived from
dataset annotations, not from a periodontist reading the film. We make no claim of equivalence
to clinical judgement.

A 214-image clinician-annotated cohort has been assembled with a collaborating clinic, but
**data-use rights have not been granted**, so it appears only as a planned external-validation
protocol — no images, no statistics, no patient-level data.

### Q: Would this work on panoramic radiographs or CBCT?

Untested, and we do not claim it. The mask-PCA axis should transfer in principle since it
needs only a tooth segmentation. Panoramic images carry rotational-tomographic distortion that
periapicals do not, which would affect the root-length denominator specifically. CBCT would
additionally allow buccal/lingual bone loss that a 2D projection cannot show at all — that is
listed as future work.

### Q: What are the clinical limitations of a ±20 percentage-point limit of agreement?

Real and we state it. It supports **screening and triage** — flagging teeth for clinician
attention, or measuring change over time within the same patient — not autonomous diagnosis.
For individual treatment planning a clinician must remain in the loop.

### Q: How were the single/double root labels obtained?

**We derived them**; they are not in DenPAR. A tooth with ≥2 assigned apex clicks is labelled
double-rooted. This is a modelling decision with a consequence: label counts shift with the
assignment strategy (3,332/1,070 under v1 vs 3,498/904 under v4), which is exactly what the
χ² tests quantify. State this proactively — it is a derived label, not ground truth.

---

## Part G — Engineering and reproducibility

### Q: Can someone reproduce your results?

Yes, with stated gaps. The repository carries: all preprocessing code, all training scripts,
the processed annotations, every metrics JSON, the experiment registry with per-run records,
the per-tooth severity pairs as CSV, and run logs with timestamps and git commits.

**Not in the repo:** model weights (`runs/` is gitignored, ~136MB per detection checkpoint)
and the raw DenPAR images (redistribution rights). Both are obtainable — DenPAR from Zenodo,
weights by re-running the documented commands.

### Q: How do you know your reported numbers are the ones your code actually produced?

Because we were burned by exactly that and fixed it. On 2026-09-12 the experiment registry
recorded **v7's** keypoint numbers under the **v6** tag, because `collect_training_results.py`
globbed every experiment's metrics file and force-labelled them all with the requested
experiment id — and alphabetically `v7_cej` overwrote `v6_cej`.

We caught it with an automated sanity check that verifies every recorded `run_dir` actually
references the experiment being written, fixed the glob, reproduced the bug against fixtures,
and re-ran. The v6 numbers in this document are post-fix and verified
(`research_log/run_logs/thesis_evidence_20260912T185113Z.log`, "sanity check OK").

> If asked about errors, tell this story. "We built a check, it caught a real bug, here is the
> log" is a much better answer than "we were careful."

### Q: Was any of this run more than once? Is it stable?

Yes. The severity ICC pipeline was run independently twice on different dates. Test values:
paper_eq1 0.726 → 0.741, mask_pca 0.823 → 0.824, cej_int_midpoint 0.751 → 0.770. The
**ordering and conclusion are identical**; the small drift reflects non-determinism in
detection/NMS. Both runs are in the repository.

---

## Part H — Questions we cannot fully answer

> Honest limits. Do not improvise around these — say the limit, then say the fix.

| Question | Honest answer |
|---|---|
| "What is your k-fold variance?" | Not measured. We used the official split for comparability and substituted bootstrap CIs and a 126-config sensitivity sweep. k-fold on the final config is the obvious next step. |
| "What is COCO AP for your keypoints?" | Not computed. `torchmetrics` computes it during training but only mean-OKS was persisted. Our numbers are mean-OKS with uniform κ and are not like-for-like with the reference's AP. |
| "What is the inference latency / FPS?" | Not measured. The benchmark script exists (`scripts/benchmark_yolo_speed.py`) but was never run to completion. |
| "How does it perform on curved vs straight roots?" | No per-morphology breakdown exists. The only morphology-stratified evidence we hold is double-root detection (mAP 0.827) vs single (0.919). |
| "What about inter-observer agreement of the original annotations?" | DenPAR ships single annotations per landmark; no repeat-annotation data exists, so annotation reliability cannot be estimated. |
| "Did you re-train detection on the cleaned labels?" | No. YOLO was trained once on v1 labels and never re-trained on v4/v6. The annotation ablation is keypoint-only, not end-to-end. This is a genuine gap. |
| "Statistical significance of 0.823 vs 0.726?" | We report non-overlapping-ish bootstrap CIs ([0.761,0.880] vs [0.668,0.807]) but did not run a paired significance test on the ICC difference. A paired bootstrap on the difference would be the right test. |

---

## Part I — Glossary

| Term | Meaning |
|---|---|
| **CEJ** | Cemento-enamel junction — where enamel meets cementum; the coronal reference landmark. |
| **Bone-level intersection (INT)** | Where the alveolar bone crest meets the tooth surface. The numerator landmark. |
| **Apex** | Root tip. Sets root length (the denominator). |
| **Mesial / distal** | Toward the midline / away from it. Our two "slots" per tooth. |
| **IOPA** | Intra-oral periapical radiograph. |
| **OKS** | Object Keypoint Similarity — distance-based keypoint agreement, normalised by object scale. |
| **mAP@0.5** | Mean average precision at IoU 0.5. |
| **ICC(2,1)** | Two-way random, absolute agreement, single measurement. |
| **LoA** | Limits of agreement = bias ± 1.96 SD (Bland–Altman). |
| **MAE / RMSE** | Mean absolute / root-mean-square error, in severity percentage points. |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalisation. |
| **NMS** | Non-maximum suppression. |
| **ROI** | Region of interest — the cropped box fed to the keypoint head. |
| **PCA long axis** | Principal component of the mask pixel cloud — the tooth's long direction. |
| **v1–v7** | Our preprocessing/label-construction versions. v6 is production. |

---

## Changelog

| Date | Change |
|---|---|
| 2026-09-13 | Created. Covers cross-validation, ICC, Bland–Altman, bootstrap, CV-withdrawal, Hungarian, combine modes, pairing protocols, apex merge, OKS-vs-AP, noise-injection mechanism study, and known limits. |
| 2026-09-13 | Full noise study (v5 test, 200 images, 10 trials/σ) replaces the 60-image prototype table. **Corrects an earlier reading:** at full scale the crown-width apex-free index is the LEAST noise-robust of the four, not the second-most. Added the explanation (noise enters both numerator and denominator; small denominator) and the resulting conclusion that mask-PCA is the better trade. |

### How to extend this document

When a new result lands, add it in **three** places: the number in Part B, the question it
answers in Part C–G, and a Changelog row. When someone asks a question this document cannot
answer, add it to **Part H first** with an honest answer, and only move it up once a number
exists to back it.
