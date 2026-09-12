"""Agreement analysis from dumped severity pairs: ICC CIs + Bland-Altman.

Consumes the CSVs written by ``dump_severity_pairs.py`` and produces the
statistics the thesis needs but never had:

* ICC(2,1) with a bootstrap 95% confidence interval (the existing reports
  carry point estimates only).
* Mean absolute error and root-mean-square error in percentage points.
* Bland-Altman agreement: bias, SD of differences, and 95% limits of
  agreement, with a difference-vs-mean figure per split/method.

No GPU and no model are required - this reads CSV only. The ICC point
estimates recomputed here must match ``research_log/axis_severity_icc.json``;
any mismatch means the dump and the original run disagree and should be
investigated before the numbers go into the thesis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import scripts._bootstrap  # noqa: F401

from src.severity.icc import icc21

METHOD_ORDER = ["paper_eq1", "mask_pca", "cej_int_midpoint"]
GT_PAIR_COLUMNS = [
    ("paper_eq1", "mask_pca"),
    ("paper_eq1", "cej_int_midpoint"),
    ("mask_pca", "cej_int_midpoint"),
]


def bootstrap_icc_ci(
    gt: np.ndarray,
    pred: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for ICC(2,1), resampling pairs with replacement."""
    if len(gt) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    n = len(gt)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = icc21(np.column_stack([gt[idx], pred[idx]]))
        if val is not None and np.isfinite(val):
            stats.append(val)
    if len(stats) < max(20, n_boot // 10):
        return None, None
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def agreement_stats(
    gt: np.ndarray,
    pred: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """ICC + CI, error magnitudes, and Bland-Altman agreement limits."""
    if len(gt) < 3:
        return {"n_pairs": int(len(gt)), "icc": None}

    diff = pred - gt
    mean = (pred + gt) / 2.0
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    lo, hi = bootstrap_icc_ci(gt, pred, n_boot, seed)

    return {
        "n_pairs": int(len(gt)),
        "icc": icc21(np.column_stack([gt, pred])),
        "icc_ci95_low": lo,
        "icc_ci95_high": hi,
        "mae_pct": float(np.mean(np.abs(diff))),
        "rmse_pct": float(np.sqrt(np.mean(diff**2))),
        "bland_altman": {
            "bias_pct": bias,
            "sd_of_differences_pct": sd_diff,
            "loa_lower_pct": bias - 1.96 * sd_diff,
            "loa_upper_pct": bias + 1.96 * sd_diff,
            "mean_severity_pct": float(np.mean(mean)),
        },
    }


def bland_altman_plot(gt: np.ndarray, pred: np.ndarray, title: str, out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib unavailable, skipping figure {out_path.name}")
        return

    diff = pred - gt
    mean = (pred + gt) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    ax.scatter(mean, diff, s=9, alpha=0.35, edgecolors="none", color="#2a6f97")
    ax.axhline(bias, color="#b5179e", lw=1.4, label=f"bias {bias:+.2f}")
    for loa, tag in ((bias + 1.96 * sd, "+1.96 SD"), (bias - 1.96 * sd, "-1.96 SD")):
        ax.axhline(loa, color="#6c757d", lw=1.0, ls="--", label=f"{tag} {loa:+.2f}")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.4)
    ax.set_xlabel("Mean of reference and predicted severity (pct.)")
    ax.set_ylabel("Predicted - reference (pct. points)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  figure -> {out_path}")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except ValueError:
        return None
    return val if np.isfinite(val) else None


def analyze_end_to_end(rows: list[dict], args, fig_dir: Path) -> dict:
    grouped: dict[tuple[str, str], tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for row in rows:
        gt = as_float(row.get("gt_severity"))
        pred = as_float(row.get("pred_severity"))
        if gt is None or pred is None:
            continue
        gt_list, pred_list = grouped[(row["split"], row["method"])]
        gt_list.append(gt)
        pred_list.append(pred)

    out: dict = {}
    for split in ("train", "val", "test"):
        for method in METHOD_ORDER:
            key = (split, method)
            if key not in grouped:
                continue
            gt = np.asarray(grouped[key][0], dtype=float)
            pred = np.asarray(grouped[key][1], dtype=float)
            stats = agreement_stats(gt, pred, args.n_boot, args.seed)
            out.setdefault(split, {})[method] = stats
            icc = stats.get("icc")
            icc_s = f"{icc:.4f}" if icc is not None else "n/a"
            print(f"[{split}] {method:20s} ICC={icc_s}  n={stats['n_pairs']}")
            if split == "test":
                bland_altman_plot(
                    gt,
                    pred,
                    f"Bland-Altman: {method} (test, n={len(gt)})",
                    fig_dir / f"bland_altman_{method}_test.png",
                )
    return out


def analyze_gt_only(rows: list[dict], args) -> dict:
    grouped: dict[tuple[str, str], tuple[list[float], list[float]]] = defaultdict(
        lambda: ([], [])
    )
    for row in rows:
        for col_a, col_b in GT_PAIR_COLUMNS:
            val_a = as_float(row.get(col_a))
            val_b = as_float(row.get(col_b))
            if val_a is None or val_b is None:
                continue
            a_list, b_list = grouped[(row["split"], f"{col_a}_vs_{col_b}")]
            a_list.append(val_a)
            b_list.append(val_b)

    out: dict = {}
    for (split, pair), (a_vals, b_vals) in sorted(grouped.items()):
        stats = agreement_stats(
            np.asarray(a_vals, dtype=float),
            np.asarray(b_vals, dtype=float),
            args.n_boot,
            args.seed,
        )
        out.setdefault(split, {})[pair] = stats
        icc = stats.get("icc")
        icc_s = f"{icc:.4f}" if icc is not None else "n/a"
        print(f"[{split}] {pair:34s} ICC={icc_s}  n={stats['n_pairs']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Agreement analysis from severity pair CSVs")
    p.add_argument("--pairs-dir", type=Path, default=Path("research_log/severity_pairs"))
    p.add_argument("--out", type=Path, default=Path("research_log/severity_agreement.json"))
    p.add_argument("--fig-dir", type=Path, default=Path("research_log/figures/agreement"))
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for the ICC CI")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    report: dict = {
        "source": args.pairs_dir.as_posix(),
        "icc_model": "ICC(2,1) two-way random, absolute agreement, single measurement",
        "ci_method": f"percentile bootstrap over pairs, n_boot={args.n_boot}, seed={args.seed}",
        "end_to_end": {},
        "gt_only": {},
    }

    e2e = read_rows(args.pairs_dir / "severity_pairs_endtoend.csv")
    if e2e:
        print("=" * 66)
        print("END-TO-END (GT vs predicted severity)")
        print("=" * 66)
        report["end_to_end"] = analyze_end_to_end(e2e, args, args.fig_dir)
    else:
        print("No end-to-end CSV found; skipping (run dump_severity_pairs.py on the GPU box).")

    gt_only = read_rows(args.pairs_dir / "severity_pairs_gtonly.csv")
    if gt_only:
        print()
        print("=" * 66)
        print("GT-ONLY (formula vs formula, identical keypoints)")
        print("=" * 66)
        report["gt_only"] = analyze_gt_only(gt_only, args)
    else:
        print("No GT-only CSV found; skipping.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
