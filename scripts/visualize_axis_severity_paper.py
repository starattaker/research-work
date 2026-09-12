"""Paper figures: axis-constrained severity, one tooth-side at a time.

Each output image has FOUR panels, one per severity definition:

  1. Paper Eq.1      - min-max line through {CEJ, INT, APEX} sorted by x.
                       Uses the apex for BOTH the axis direction and the
                       root-length denominator.
  2. Mask PCA axis   - direction = major PCA component of the tooth mask.
                       Apex used ONLY as the root-length denominator.
  3. CEJ->INT axis   - direction = CEJ-midpoint -> INT-midpoint.
                       Apex used ONLY as the root-length denominator.
  4. Crown-width     - GENUINELY apex-free. Normalises the CEJ-to-INT
     (no apex)         distance by the mesial-distal CEJ span (crown width)
                       instead of by root length. No apex keypoint, and no
                       tooth mask, is used anywhere in this computation.

Every panel draws the SAME visual grammar so the four definitions are
directly comparable at a glance:

  - the tooth mask contour, for spatial context
  - raw landmark points (CEJ cyan, INT yellow, APEX magenta)
  - the axis actually used, drawn long enough to span the tooth
  - dotted drop-lines from each raw point to its projection on the axis
  - a projected marker (square) at each projection
  - the DENOMINATOR / NORMALISER segment: gray dashed
  - the NUMERATOR / bone-loss segment: bold red, with the severity % labelled
    directly on it

One image is written per (stem, tooth, side that has a valid severity under
at least one method): "{stem}_tooth{idx}_slot{slot}.png". The first valid
slot for each tooth is additionally saved at the legacy path
"{stem}_tooth{idx}.png" so existing LaTeX \\figfile references keep working.
"""

from __future__ import annotations

import argparse
import json
import math
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
    crown_width,
    mask_axis_extent,
    project_scalar_along_axis,
    severities_apex_free_both_sides,
    severities_both_sides,
)
from src.severity.bone_loss import minimax_line_params, project_point_to_line
from src.severity.gt_labels import point_from_slot
from src.severity.inference_pipeline import load_gt_annotations

CEJ_COLOR = "#00e5ff"
INT_COLOR = "#ffee00"
APEX_COLOR = "#ff00ff"
AXIS_COLOR = "#39ff14"
DENOM_COLOR = "#b0b0b0"
NUM_COLOR = "#ff3b30"
WIDTH_COLOR = "#ffb703"


# --------------------------------------------------------------------- geometry ---


def mask_contour_px(mask: np.ndarray | None, ox_off: float, oy_off: float) -> np.ndarray | None:
    if mask is None:
        return None
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts[:, 0] -= ox_off
    pts[:, 1] -= oy_off
    return pts


def draw_mask_contour(ax, mask_px: np.ndarray | None) -> None:
    if mask_px is None:
        return
    closed = np.vstack([mask_px, mask_px[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="#4cc9f0", lw=1.0, alpha=0.55, zorder=1)
    ax.fill(closed[:, 0], closed[:, 1], color="#4cc9f0", alpha=0.06, zorder=0)


def draw_point(ax, p_px: tuple[float, float], color: str, label: str) -> None:
    ax.scatter([p_px[0]], [p_px[1]], c=color, s=42, edgecolors="black", linewidths=0.6, zorder=7)
    ax.annotate(
        label,
        p_px,
        xytext=(4, 4),
        textcoords="offset points",
        color=color,
        fontsize=7,
        fontweight="bold",
        zorder=8,
    )


def draw_projection(ax, raw_px: tuple[float, float], proj_px: tuple[float, float], color: str) -> None:
    ax.plot(
        [raw_px[0], proj_px[0]],
        [raw_px[1], proj_px[1]],
        color=color,
        linestyle=":",
        linewidth=1.1,
        alpha=0.85,
        zorder=4,
    )
    ax.scatter(
        [proj_px[0]], [proj_px[1]], marker="s", s=30, facecolors=color,
        edgecolors="black", linewidths=0.6, zorder=6,
    )


def draw_axis_span(
    ax,
    origin: tuple[float, float],
    direction: tuple[float, float],
    t_lo: float,
    t_hi: float,
    ox_off: float,
    oy_off: float,
    margin_frac: float = 0.10,
    min_margin: float = 8.0,
) -> None:
    span = max(t_hi - t_lo, 1e-6)
    pad = max(span * margin_frac, min_margin)
    t0, t1 = t_lo - pad, t_hi + pad
    ox, oy = origin
    dx, dy = direction
    p0 = (ox + t0 * dx - ox_off, oy + t0 * dy - oy_off)
    p1 = (ox + t1 * dx - ox_off, oy + t1 * dy - oy_off)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=AXIS_COLOR, linewidth=2.3, alpha=0.95, zorder=3)


def annotate_segment(
    ax,
    a_px: tuple[float, float],
    b_px: tuple[float, float],
    text: str,
    offset: tuple[float, float] = (10, -10),
) -> None:
    mid = ((a_px[0] + b_px[0]) / 2.0, (a_px[1] + b_px[1]) / 2.0)
    ax.annotate(
        text,
        mid,
        xytext=offset,
        textcoords="offset points",
        color=NUM_COLOR,
        fontsize=8.5,
        fontweight="bold",
        zorder=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=NUM_COLOR, alpha=0.85, lw=0.8),
    )


# ---------------------------------------------------------------------- panels ---


def panel_axis_method(
    ax,
    title: str,
    cej: tuple[float, float],
    inter: tuple[float, float],
    apex: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
    axis_t_lo: float,
    axis_t_hi: float,
    ox_off: float,
    oy_off: float,
    severity: float | None,
) -> None:
    """Shared rendering for MASK_PCA and CEJ_INT_MIDPOINT: apex sets only the
    denominator, so the axis line is drawn from the mask/landmark extent, and
    all three landmarks are dropped onto it."""

    def to_px(p):
        return (p[0] - ox_off, p[1] - oy_off)

    draw_axis_span(ax, origin, direction, axis_t_lo, axis_t_hi, ox_off, oy_off)

    t_cej = project_scalar_along_axis(cej, origin, direction)
    t_int = project_scalar_along_axis(inter, origin, direction)
    t_apex = project_scalar_along_axis(apex, origin, direction)

    def proj_px(t):
        return (origin[0] + t * direction[0] - ox_off, origin[1] + t * direction[1] - oy_off)

    cej_proj, int_proj, apex_proj = proj_px(t_cej), proj_px(t_int), proj_px(t_apex)

    # Denominator (root length): CEJ_proj -> APEX_proj, drawn first (background)
    ax.plot(
        [cej_proj[0], apex_proj[0]], [cej_proj[1], apex_proj[1]],
        color=DENOM_COLOR, linestyle="--", linewidth=2.0, alpha=0.9, zorder=2,
        label="root length (denominator)",
    )
    # Numerator (bone loss): CEJ_proj -> INT_proj, drawn on top
    ax.plot(
        [cej_proj[0], int_proj[0]], [cej_proj[1], int_proj[1]],
        color=NUM_COLOR, linewidth=3.2, alpha=0.95, zorder=5,
        label="bone loss (numerator)",
    )

    draw_projection(ax, to_px(cej), cej_proj, CEJ_COLOR)
    draw_projection(ax, to_px(inter), int_proj, INT_COLOR)
    draw_projection(ax, to_px(apex), apex_proj, APEX_COLOR)

    draw_point(ax, to_px(cej), CEJ_COLOR, "CEJ")
    draw_point(ax, to_px(inter), INT_COLOR, "INT")
    draw_point(ax, to_px(apex), APEX_COLOR, "APEX")

    sev_txt = f"{severity:.1f}%" if severity is not None else "n/a"
    annotate_segment(ax, cej_proj, int_proj, f"severity = {sev_txt}")
    ax.set_title(title, fontsize=9.5)


def panel_paper_eq1(
    ax,
    cej: tuple[float, float],
    inter: tuple[float, float],
    apex: tuple[float, float],
    ox_off: float,
    oy_off: float,
    severity: float | None,
) -> None:
    def to_px(p):
        return (p[0] - ox_off, p[1] - oy_off)

    points = sorted([cej, inter, apex], key=lambda p: p[0])
    m, c = minimax_line_params(*points)
    cej_proj = project_point_to_line(cej[0], cej[1], m, c)
    int_proj = project_point_to_line(inter[0], inter[1], m, c)
    apex_proj = project_point_to_line(apex[0], apex[1], m, c)

    direction = (0.0, 1.0) if math.isinf(m) else (1.0 / math.hypot(1, m), m / math.hypot(1, m))
    ts = [
        (p[0] - cej_proj[0]) * direction[0] + (p[1] - cej_proj[1]) * direction[1]
        for p in (cej_proj, int_proj, apex_proj)
    ]
    draw_axis_span(ax, cej_proj, direction, min(ts), max(ts), ox_off, oy_off)

    def px(p):
        return (p[0] - ox_off, p[1] - oy_off)

    cej_p, int_p, apex_p = px(cej_proj), px(int_proj), px(apex_proj)

    ax.plot(
        [cej_p[0], apex_p[0]], [cej_p[1], apex_p[1]],
        color=DENOM_COLOR, linestyle="--", linewidth=2.0, alpha=0.9, zorder=2,
    )
    ax.plot(
        [cej_p[0], int_p[0]], [cej_p[1], int_p[1]],
        color=NUM_COLOR, linewidth=3.2, alpha=0.95, zorder=5,
    )

    draw_projection(ax, to_px(cej), cej_p, CEJ_COLOR)
    draw_projection(ax, to_px(inter), int_p, INT_COLOR)
    draw_projection(ax, to_px(apex), apex_p, APEX_COLOR)

    draw_point(ax, to_px(cej), CEJ_COLOR, "CEJ")
    draw_point(ax, to_px(inter), INT_COLOR, "INT")
    draw_point(ax, to_px(apex), APEX_COLOR, "APEX")

    sev_txt = f"{severity:.1f}%" if severity is not None else "n/a"
    annotate_segment(ax, cej_p, int_p, f"severity = {sev_txt}")
    ax.set_title("Paper Eq.1 (min-max line)\napex sets axis AND denominator", fontsize=9.5)


def panel_apex_free(
    ax,
    cej_slot: tuple[float, float],
    int_slot: tuple[float, float],
    cej0: tuple[float, float],
    cej1: tuple[float, float],
    origin: tuple[float, float],
    direction: tuple[float, float],
    ox_off: float,
    oy_off: float,
    severity: float | None,
    width_px: float | None,
) -> None:
    """Genuinely apex-free: no apex point drawn, no mask used. Numerator is the
    projected CEJ->INT distance for this side; denominator is the raw crown
    width (distance between the two CEJ points)."""

    def to_px(p):
        return (p[0] - ox_off, p[1] - oy_off)

    t_cej = project_scalar_along_axis(cej_slot, origin, direction)
    t_int = project_scalar_along_axis(int_slot, origin, direction)

    def proj_px(t):
        return (origin[0] + t * direction[0] - ox_off, origin[1] + t * direction[1] - oy_off)

    cej_proj, int_proj = proj_px(t_cej), proj_px(t_int)
    draw_axis_span(ax, origin, direction, min(t_cej, t_int), max(t_cej, t_int), ox_off, oy_off, margin_frac=0.35)

    # Crown-width normaliser: the raw segment between the two CEJ points.
    cej0_px, cej1_px = to_px(cej0), to_px(cej1)
    ax.plot(
        [cej0_px[0], cej1_px[0]], [cej0_px[1], cej1_px[1]],
        color=WIDTH_COLOR, linestyle="--", linewidth=2.2, alpha=0.95, zorder=2,
    )
    w_txt = f"{width_px:.0f}px" if width_px else "n/a"
    annotate_segment(ax, cej0_px, cej1_px, f"crown width W={w_txt}", offset=(0, 10))

    # Numerator: projected CEJ(side) -> INT(side) distance.
    ax.plot(
        [cej_proj[0], int_proj[0]], [cej_proj[1], int_proj[1]],
        color=NUM_COLOR, linewidth=3.2, alpha=0.95, zorder=5,
    )
    draw_projection(ax, to_px(cej_slot), cej_proj, CEJ_COLOR)
    draw_projection(ax, to_px(int_slot), int_proj, INT_COLOR)
    draw_point(ax, to_px(cej_slot), CEJ_COLOR, "CEJ")
    draw_point(ax, to_px(int_slot), INT_COLOR, "INT")
    if cej0_px != to_px(cej_slot):
        draw_point(ax, cej0_px if to_px(cej0) != to_px(cej_slot) else cej1_px, "#0077b6", "CEJ(other side)")

    sev_txt = f"{severity:.1f}%" if severity is not None else "n/a"
    annotate_segment(ax, cej_proj, int_proj, f"severity = {sev_txt}\n(no apex)", offset=(10, -22))
    ax.set_title("Crown-width index (no apex)\nno mask, no root-length landmark", fontsize=9.5)


# ------------------------------------------------------------------------ main ---


def visualize_tooth_side(
    img: np.ndarray,
    merged: dict,
    tooth_idx: int,
    slot: int,
    mask: np.ndarray | None,
    out_path: Path,
) -> bool:
    bbox = merged["bboxes"][tooth_idx]
    cej_pts = merged["cej"][tooth_idx]
    inter_pts = merged["intersection"][tooth_idx]
    apex_pts = merged["apex"][tooth_idx]

    cej = point_from_slot(cej_pts, slot)
    inter = point_from_slot(inter_pts, slot)
    apex = point_from_slot(apex_pts, slot)
    if cej == (0.0, 0.0) or inter == (0.0, 0.0) or apex == (0.0, 0.0):
        return False

    visible_cej = [point_from_slot(cej_pts, i) for i in range(2)]
    visible_cej_only = [p for p in visible_cej if p != (0.0, 0.0)]
    visible_int = [point_from_slot(inter_pts, i) for i in range(2)]
    visible_int_only = [p for p in visible_int if p != (0.0, 0.0)]

    sev = {
        m.value: dict(severities_both_sides(cej_pts, inter_pts, apex_pts, m, mask=mask, bbox=bbox)).get(slot)
        for m in (AxisSeverityMethod.PAPER_EQ1, AxisSeverityMethod.MASK_PCA, AxisSeverityMethod.CEJ_INT_MIDPOINT)
    }
    sev["cej_width_no_apex"] = dict(
        severities_apex_free_both_sides(cej_pts, inter_pts, bbox=bbox)
    ).get(slot)

    x1, y1, x2, y2 = map(int, bbox)
    pad = 45
    h, w = img.shape[:2]
    crop = img[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)]
    ox_off, oy_off = max(0, x1 - pad), max(0, y1 - pad)
    mask_px = mask_contour_px(mask, ox_off, oy_off)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))

    # Panel 1: paper Eq.1
    ax = axes[0]
    ax.imshow(crop, cmap="gray")
    draw_mask_contour(ax, mask_px)
    panel_paper_eq1(ax, cej, inter, apex, ox_off, oy_off, sev["paper_eq1"])
    ax.axis("off")

    # Panel 2: mask PCA (axis spans the FULL mask extent, not just landmarks)
    ax = axes[1]
    ax.imshow(crop, cmap="gray")
    draw_mask_contour(ax, mask_px)
    axis = axis_mask_pca(mask)
    if axis is not None:
        origin, direction = axis
        extent = mask_axis_extent(mask, origin, direction)
        if extent is not None:
            t_lo, t_hi = extent
        else:
            ts = [project_scalar_along_axis(p, origin, direction) for p in (cej, inter, apex)]
            t_lo, t_hi = min(ts), max(ts)
        panel_axis_method(
            ax, "Mask PCA axis\napex sets denominator only",
            cej, inter, apex, origin, direction, t_lo, t_hi, ox_off, oy_off, sev["mask_pca"],
        )
    else:
        ax.set_title("Mask PCA axis\n(no mask available)", fontsize=9.5)
    ax.axis("off")

    # Panel 3: CEJ -> INT midpoint axis
    ax = axes[2]
    ax.imshow(crop, cmap="gray")
    draw_mask_contour(ax, mask_px)
    axis = axis_cej_int_midpoint(visible_cej_only, visible_int_only, bbox)
    if axis is not None:
        origin, direction = axis
        ts = [project_scalar_along_axis(p, origin, direction) for p in (cej, inter, apex)]
        panel_axis_method(
            ax, "CEJ→INT midpoint axis\napex sets denominator only",
            cej, inter, apex, origin, direction, min(ts), max(ts), ox_off, oy_off, sev["cej_int_midpoint"],
        )
    else:
        ax.set_title("CEJ→INT midpoint axis\n(unavailable)", fontsize=9.5)
    ax.axis("off")

    # Panel 4: genuinely apex-free crown-width index
    ax = axes[3]
    ax.imshow(crop, cmap="gray")
    draw_mask_contour(ax, mask_px)
    axis_apex_free = axis_cej_int_midpoint(visible_cej_only, visible_int_only, bbox)
    if len(visible_cej_only) == 2 and axis_apex_free is not None:
        origin, direction = axis_apex_free
        panel_apex_free(
            ax, cej, inter, visible_cej_only[0], visible_cej_only[1],
            origin, direction, ox_off, oy_off, sev["cej_width_no_apex"],
            crown_width(cej_pts),
        )
    else:
        ax.set_title("Crown-width index (no apex)\n(needs both CEJ points)", fontsize=9.5)
    ax.axis("off")

    fig.suptitle(f"Tooth {tooth_idx}, side {slot} — severity by axis definition", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


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

            first_written = None
            for slot in (0, 1):
                out = args.out_dir / f"{stem}_tooth{tooth_idx}_slot{slot}.png"
                ok = visualize_tooth_side(img, merged, tooth_idx, slot, mask, out)
                if ok:
                    manifest.append(out.as_posix())
                    print(f"Wrote {out}")
                    if first_written is None:
                        first_written = out

            if first_written is not None:
                legacy = args.out_dir / f"{stem}_tooth{tooth_idx}.png"
                legacy.write_bytes(first_written.read_bytes())
                manifest.append(legacy.as_posix())
                print(f"Wrote {legacy} (legacy alias of {first_written.name})")

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
