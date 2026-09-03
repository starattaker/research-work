"""Paper figures: axis-constrained severity on multiple test images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import scripts._bootstrap  # noqa: F401

from src.denpar_paths import DEFAULT_DENPAR_ROOT, denpar_mask_path, resolve_denpar_root
from src.preprocess.prepare_dataset import load_tooth_mask
from src.severity.axis_severity import (
    AxisSeverityMethod,
    axis_cej_int_midpoint,
    axis_mask_pca,
    project_scalar_along_axis,
    severities_both_sides,
)
from src.severity.gt_labels import point_from_slot
from src.severity.inference_pipeline import load_gt_annotations


def draw_axis(ax, origin, direction, length: float, color: str, label: str):
    ox, oy = origin
    dx, dy = direction
    ax.plot(
        [ox - dx * length, ox + dx * length],
        [oy - dy * length, oy + dy * length],
        color=color,
        linewidth=2,
        label=label,
    )


def visualize_tooth(
    img: np.ndarray,
    merged: dict,
    tooth_idx: int,
    mask: np.ndarray | None,
    out_path: Path,
):
    bbox = merged["bboxes"][tooth_idx]
    cej = merged["cej"][tooth_idx]
    inter = merged["intersection"][tooth_idx]
    apex = merged["apex"][tooth_idx]

    pts = []
    labels = []
    colors = []
    for slot in (0, 1):
        for name, arr, c in (("CEJ", cej, "cyan"), ("INT", inter, "yellow"), ("APEX", apex, "magenta")):
            p = point_from_slot(arr, slot)
            if p != (0.0, 0.0):
                pts.append(p)
                labels.append(f"{name}{slot}")
                colors.append(c)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x1, y1, x2, y2 = map(int, bbox)
    pad = 40
    h, w = img.shape[:2]
    crop = img[
        max(0, y1 - pad) : min(h, y2 + pad),
        max(0, x1 - pad) : min(w, x2 + pad),
    ]
    ox_off, oy_off = max(0, x1 - pad), max(0, y1 - pad)

    for ax, method, title in zip(
        axes,
        (
            AxisSeverityMethod.PAPER_EQ1,
            AxisSeverityMethod.MASK_PCA,
            AxisSeverityMethod.CEJ_INT_MIDPOINT,
        ),
        ("Paper Eq.1 (min-max line)", "Mask PCA axis", "CEJ–INT midpoint axis"),
    ):
        ax.imshow(crop, cmap="gray")
        for p, lab, col in zip(pts, labels, colors):
            ax.scatter(p[0] - ox_off, p[1] - oy_off, c=col, s=40, edgecolors="white", linewidths=0.5)
            ax.text(p[0] - ox_off + 3, p[1] - oy_off - 3, lab, color=col, fontsize=7)

        visible_cej = [point_from_slot(cej, i) for i in range(2)]
        visible_cej = [p for p in visible_cej if p != (0.0, 0.0)]
        visible_int = [point_from_slot(inter, i) for i in range(2)]
        visible_int = [p for p in visible_int if p != (0.0, 0.0)]

        if method == AxisSeverityMethod.MASK_PCA:
            axis = axis_mask_pca(mask)
        elif method == AxisSeverityMethod.CEJ_INT_MIDPOINT:
            axis = axis_cej_int_midpoint(visible_cej, visible_int, bbox)
        else:
            axis = None

        if axis is not None:
            origin, direction = axis
            draw_axis(
                ax,
                (origin[0] - ox_off, origin[1] - oy_off),
                direction,
                length=120,
                color="lime",
                label="axis",
            )

        sides = severities_both_sides(cej, inter, apex, method, mask=mask, bbox=bbox)
        sev_txt = "  ".join(f"s{s}={v:.1f}%" for s, v in sides)
        ax.set_title(f"{title}\n{sev_txt}", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"Tooth {tooth_idx} — axis severity comparison", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data/processed_v6"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_DENPAR_ROOT)
    p.add_argument("--split", default="test")
    p.add_argument("--stems", nargs="*", default=["431", "5", "100", "240", "622"])
    p.add_argument("--out-dir", type=Path, default=Path("paper/figures/axis_severity"))
    args = p.parse_args()

    raw = resolve_denpar_root(args.raw_root)
    img_dir = args.data_root / "yolo_detection" / args.split / "images"
    if not img_dir.exists():
        img_dir = args.data_root / "keypoints" / "cej" / args.split / "images"

    manifest = []
    for stem in args.stems:
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        merged = load_gt_annotations(args.data_root, args.split, stem)
        if merged is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for tooth_idx in range(min(2, len(merged["bboxes"]))):
            mask_path = denpar_mask_path(raw, args.split, stem, tooth_idx)
            mask = load_tooth_mask(mask_path) if mask_path.exists() else None
            out = args.out_dir / f"{stem}_tooth{tooth_idx}.png"
            visualize_tooth(img, merged, tooth_idx, mask, out)
            manifest.append(out.as_posix())
            print(f"Wrote {out}")

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
