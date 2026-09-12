"""Controlled noise-injection study: how fast does each severity formula break?

The central claim of the axis-constrained severity work is a MECHANISM claim:

    Under the paper's min-max line, the measurement axis is re-estimated from
    the same three noisy predicted landmarks it then projects, so keypoint
    error both DISPLACES the points and ROTATES the axis. Under a mask-PCA
    axis the direction comes from the segmentation (thousands of pixels), so
    keypoint error can only displace points along a fixed direction.

Observing that mask-PCA wins end-to-end does not prove that mechanism - the
two differ in many ways at once. This script isolates it. We take GROUND
TRUTH keypoints, inject controlled isotropic Gaussian noise of known sigma,
and measure how far each formula's severity drifts from its own clean value.

Everything else is held fixed: same teeth, same masks, same pairing, same
clean reference. The ONLY thing that varies is the injected localisation
error. If the mechanism claim is right, the paper formula must degrade
faster than the mask-PCA formula as sigma grows - and the gap must widen
monotonically.

This needs no model, no GPU, and no training: it runs on GT annotations
plus tooth masks.

Usage:
    python scripts/noise_sensitivity_severity.py --data-root data/processed_v6 \
        --split test --sigmas 0 1 2 4 8 12 16 --trials 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.denpar_paths import DEFAULT_DENPAR_ROOT, denpar_mask_path, resolve_denpar_root
from src.preprocess.prepare_dataset import load_tooth_mask
from src.severity.axis_severity import (
    AxisSeverityMethod,
    severities_apex_free_both_sides,
    severities_both_sides,
)
from src.severity.gt_labels import load_gt_annotations
from src.severity.icc import icc21

METHODS = (
    AxisSeverityMethod.PAPER_EQ1,
    AxisSeverityMethod.MASK_PCA,
    AxisSeverityMethod.CEJ_INT_MIDPOINT,
)
APEX_FREE = "cej_width_no_apex"


def jitter(points: list, sigma: float, rng: np.random.Generator) -> list:
    """Add isotropic Gaussian noise to every VISIBLE keypoint, preserving flags."""
    out = []
    for kp in points:
        if len(kp) > 2 and int(kp[2]) == 0:
            out.append(list(kp))
            continue
        x, y = float(kp[0]), float(kp[1])
        if x == 0.0 and y == 0.0:
            out.append(list(kp))
            continue
        out.append([x + rng.normal(0, sigma), y + rng.normal(0, sigma), kp[2] if len(kp) > 2 else 2])
    return out


def severities_for(cej, inter, apex, mask, bbox) -> dict[str, dict[int, float]]:
    """All four severity definitions for one tooth, keyed by method then slot."""
    out: dict[str, dict[int, float]] = {}
    for m in METHODS:
        out[m.value] = dict(severities_both_sides(cej, inter, apex, m, mask=mask, bbox=bbox))
    out[APEX_FREE] = dict(severities_apex_free_both_sides(cej, inter, bbox=bbox))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Noise-injection sensitivity of severity formulas")
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--sigmas", type=float, nargs="*", default=[0, 1, 2, 4, 8, 12, 16])
    p.add_argument("--trials", type=int, default=5, help="Noise realisations per sigma")
    p.add_argument("--limit", type=int, default=0, help="Cap number of images (0 = all)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("research_log/noise_sensitivity.json"))
    p.add_argument("--fig", type=Path, default=Path("research_log/figures/noise_sensitivity.png"))
    args = p.parse_args()

    raw_root = resolve_denpar_root(args.raw_root)
    ann_dir = args.data_root / "keypoints" / "cej" / args.split / "annotations"
    stems = sorted(q.stem for q in ann_dir.glob("*.json"))
    if args.limit:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"No annotations under {ann_dir}")

    # Cache clean annotations + masks once.
    cache = []
    for stem in tqdm(stems, desc="loading"):
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        masks = []
        for i in range(len(merged["bboxes"])):
            mp = denpar_mask_path(raw_root, args.split, stem, i)
            masks.append(load_tooth_mask(mp) if mp.exists() else None)
        cache.append((stem, merged, masks))

    method_names = [m.value for m in METHODS] + [APEX_FREE]
    results: dict = {
        "data_root": args.data_root.as_posix(),
        "split": args.split,
        "trials": args.trials,
        "n_images": len(cache),
        "note": (
            "Reference is each formula's OWN clean (sigma=0) severity, so every "
            "curve starts at ICC=1.0 and measures only that formula's robustness "
            "to keypoint localisation error - not its agreement with any other."
        ),
        "curves": {name: [] for name in method_names},
    }

    rng_master = np.random.default_rng(args.seed)

    # Clean reference severities, per (stem, tooth, slot, method).
    clean: dict[str, dict[tuple, float]] = {name: {} for name in method_names}
    for stem, merged, masks in cache:
        for i, bbox in enumerate(merged["bboxes"]):
            sev = severities_for(
                merged["cej"][i], merged["intersection"][i], merged["apex"][i], masks[i], bbox
            )
            for name in method_names:
                for slot, val in sev[name].items():
                    clean[name][(stem, i, slot)] = val

    for sigma in args.sigmas:
        per_method_icc = {name: [] for name in method_names}
        per_method_mae = {name: [] for name in method_names}

        trials = 1 if sigma == 0 else args.trials
        for _ in range(trials):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            paired = {name: ([], []) for name in method_names}

            for stem, merged, masks in cache:
                for i, bbox in enumerate(merged["bboxes"]):
                    sev = severities_for(
                        jitter(merged["cej"][i], sigma, rng),
                        jitter(merged["intersection"][i], sigma, rng),
                        jitter(merged["apex"][i], sigma, rng),
                        masks[i],
                        bbox,
                    )
                    for name in method_names:
                        for slot, val in sev[name].items():
                            ref = clean[name].get((stem, i, slot))
                            if ref is None:
                                continue
                            paired[name][0].append(ref)
                            paired[name][1].append(val)

            for name in method_names:
                ref_vals, noisy_vals = paired[name]
                if len(ref_vals) < 3:
                    continue
                mat = np.column_stack([ref_vals, noisy_vals])
                val = icc21(mat)
                if val is not None and np.isfinite(val):
                    per_method_icc[name].append(float(val))
                per_method_mae[name].append(float(np.mean(np.abs(mat[:, 0] - mat[:, 1]))))

        for name in method_names:
            iccs = per_method_icc[name]
            maes = per_method_mae[name]
            results["curves"][name].append(
                {
                    "sigma_px": sigma,
                    "icc_vs_clean": float(np.mean(iccs)) if iccs else None,
                    "icc_std": float(np.std(iccs)) if len(iccs) > 1 else 0.0,
                    "mae_pct": float(np.mean(maes)) if maes else None,
                    "n_pairs": len(paired[name][0]) if name in paired else 0,
                }
            )
        line = "  ".join(
            f"{name}={results['curves'][name][-1]['icc_vs_clean']:.3f}"
            if results["curves"][name][-1]["icc_vs_clean"] is not None else f"{name}=n/a"
            for name in method_names
        )
        print(f"sigma={sigma:5.1f}px   {line}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    # ---- figure -------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    styles = {
        "paper_eq1": ("#e63946", "o", "Paper Eq.1 (min-max line)"),
        "mask_pca": ("#2a9d8f", "s", "Mask PCA axis"),
        "cej_int_midpoint": ("#457b9d", "^", "CEJ->INT midpoint axis"),
        APEX_FREE: ("#f4a261", "D", "Crown-width index (no apex)"),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    for name in method_names:
        rows = [r for r in results["curves"][name] if r["icc_vs_clean"] is not None]
        if not rows:
            continue
        xs = [r["sigma_px"] for r in rows]
        color, marker, label = styles[name]
        ax1.plot(xs, [r["icc_vs_clean"] for r in rows], marker=marker, color=color, label=label, lw=1.8, ms=5)
        ax2.plot(xs, [r["mae_pct"] for r in rows], marker=marker, color=color, label=label, lw=1.8, ms=5)

    ax1.set_xlabel("Injected keypoint noise $\\sigma$ (px)")
    ax1.set_ylabel("ICC(2,1) vs. own clean severity")
    ax1.set_title("Robustness to landmark localisation error", fontsize=10)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5)

    ax2.set_xlabel("Injected keypoint noise $\\sigma$ (px)")
    ax2.set_ylabel("MAE vs. own clean severity (pct. points)")
    ax2.set_title("Severity drift induced by the same noise", fontsize=10)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7.5)

    fig.tight_layout()
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.fig}")


if __name__ == "__main__":
    main()
