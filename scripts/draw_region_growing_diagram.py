"""Paper-quality v4 region-growing diagram: real masks, two teeth, point assignment animation."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import (
    SPLIT_ALIASES,
    distance_to_mask,
    load_tooth_mask,
)


# BGR
C_BG = (28, 28, 32)
C_MASK_FILL = (220, 235, 255)
C_MASK_EDGE = (255, 210, 60)
C_RING = (200, 90, 255)
C_RING_FAINT = (160, 70, 200)
C_UNASSIGNED = (160, 160, 160)
C_DROP = (80, 80, 255)
C_TOOTH_COLORS = [(80, 200, 80), (80, 180, 255)]  # green, amber
C_NEW = (0, 255, 255)


@dataclass
class PointSpec:
    pt: list[float]
    label: str  # CEJ / Apex
    assigned_at: int | None = None  # ring step
    tooth: int | None = None
    dropped: bool = False


def grace_band(mask: np.ndarray, inner: float, outer: float) -> np.ndarray:
    inv = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    if inner <= 0:
        return ((dist > 0) & (dist <= outer)).astype(np.uint8)
    return ((dist > inner) & (dist <= outer)).astype(np.uint8)


def cumulative_ring(mask: np.ndarray, max_r: int) -> np.ndarray:
    if max_r <= 0:
        return np.zeros_like(mask, dtype=np.uint8)
    return grace_band(mask, 0.0, float(max_r))


def load_scene(
    raw_root: Path,
    split: str,
    stem: str,
    tooth_indices: tuple[int, int],
    margin: int = 48,
) -> tuple[np.ndarray, list[np.ndarray], list[list[float]], list[PointSpec]]:
    split_raw = next(k for k, v in SPLIT_ALIASES.items() if v == split)
    img = cv2.imread(str(raw_root / split_raw / "Images" / f"{stem}.jpg"))
    if img is None:
        img = cv2.imread(str(Path("data/processed/yolo_detection") / split / "images" / f"{stem}.jpg"))
    if img is None:
        raise FileNotFoundError(f"image {stem}")

    kp = json.loads((raw_root / split_raw / "Key Points Annotations" / f"{stem}.json").read_text())
    mask_dir = raw_root / split_raw / "Masks (Tooth-wise)" / stem
    mask_paths = sorted(mask_dir.glob("*.png"))
    bboxes = kp["bboxes"]

    masks = []
    for i in tooth_indices:
        m = load_tooth_mask(mask_paths[i])
        if m is None:
            raise ValueError(f"missing mask {i}")
        masks.append(m)

    xs, ys = [], []
    for i in tooth_indices:
        x1, y1, x2, y2 = bboxes[i]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    x1, x2 = int(min(xs)) - margin, int(max(xs)) + margin
    y1, y2 = int(min(ys)) - margin, int(max(ys)) + margin
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    crop = img[y1:y2, x1:x2].copy()
    crop_masks = [m[y1:y2, x1:x2].copy() for m in masks]
    crop_bboxes = [[b[0] - x1, b[1] - y1, b[2] - x1, b[3] - y1] for b in (bboxes[i] for i in tooth_indices)]

    # Real CEJ + apex near these teeth (crop coords)
    points: list[PointSpec] = []
    for tag, key in (("CEJ", "CEJ_Points"), ("Apex", "Apex_Points")):
        for pt in kp.get(key, []):
            px, py = pt[0] - x1, pt[1] - y1
            if not (0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]):
                continue
            # keep points near our two teeth
            near = False
            for m in crop_masks:
                if distance_to_mask(px, py, m) <= 12.0:
                    near = True
                    break
            if near:
                points.append(PointSpec([px, py], tag))

    if len(points) < 4:
        # fallback synthetic near mask boundary
        for mi, m in enumerate(crop_masks):
            ys2, xs2 = np.where(m > 0)
            if len(xs2) == 0:
                continue
            top_i = int(np.argmin(ys2))
            bot_i = int(np.argmax(ys2))
            points.append(PointSpec([float(xs2[top_i]), float(ys2[top_i] - 5)], "CEJ"))
            points.append(PointSpec([float(xs2[bot_i]), float(ys2[bot_i] + 6)], "Apex"))

    return crop, crop_masks, crop_bboxes, points


def simulate_assignment(
    points: list[PointSpec],
    masks: list[np.ndarray],
    max_r: int = 8,
) -> list[list[PointSpec]]:
    """Per-frame snapshots: assignments visible up to ring r."""
    frames: list[list[PointSpec]] = []
    pts_raw = [[p.pt[0], p.pt[1]] for p in points]

    for r in range(0, max_r + 1):
        state = [PointSpec(p.pt[:], p.label) for p in points]
        assigned_map: dict[tuple[float, float], tuple[int, int]] = {}
        pending = [list(pt) for pt in pts_raw]
        step_px = 1

        for ring in range(0, r + 1):
            if not pending:
                break
            next_pending = []
            for pt in pending:
                cands = []
                for i, m in enumerate(masks):
                    d = distance_to_mask(pt[0], pt[1], m)
                    if ring == 0:
                        if d == 0.0:
                            cands.append(i)
                    else:
                        inner = float(ring - step_px)
                        if d <= float(ring) and d > inner:
                            cands.append(i)
                if cands:
                    assigned_map[(round(pt[0], 1), round(pt[1], 1))] = (cands[0], ring)
                else:
                    next_pending.append(pt)
            pending = next_pending

        for s in state:
            key = (round(s.pt[0], 1), round(s.pt[1], 1))
            if key in assigned_map:
                s.tooth, s.assigned_at = assigned_map[key]
            elif r == max_r:
                s.dropped = True
        frames.append(state)
    return frames


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def render_frame(
    xray: np.ndarray,
    masks: list[np.ndarray],
    points: list[PointSpec],
    ring_r: int,
    scale: int,
    caption: str,
) -> np.ndarray:
    h, w = xray.shape[:2]
    base = cv2.addWeighted(xray, 0.55, np.full_like(xray, C_BG), 0.45, 0)

    overlay = base.copy()
    for mi, m in enumerate(masks):
        overlay[m > 0] = C_MASK_FILL
        if ring_r > 0:
            band = cumulative_ring(m, ring_r)
            overlay[band > 0] = C_RING_FAINT if mi else C_RING

    for mi, m in enumerate(masks):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, C_MASK_EDGE, 2, cv2.LINE_AA)
        if ring_r > 0:
            outer = cumulative_ring(m, ring_r)
            oc, _ = cv2.findContours(outer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if oc:
                cv2.drawContours(overlay, oc, -1, C_RING, 1, cv2.LINE_AA)

    blend = cv2.addWeighted(base, 0.35, overlay, 0.65, 0)

    for p in points:
        x, y = int(round(p.pt[0])), int(round(p.pt[1]))
        if p.dropped:
            cv2.drawMarker(blend, (x, y), C_DROP, cv2.MARKER_TILTED_CROSS, 18, 3, cv2.LINE_AA)
        elif p.tooth is not None:
            col = C_TOOTH_COLORS[p.tooth % 2]
            if p.assigned_at == ring_r:
                col = C_NEW
            cv2.circle(blend, (x, y), 10, col, -1, cv2.LINE_AA)
            cv2.circle(blend, (x, y), 12, (20, 20, 20), 2, cv2.LINE_AA)
        else:
            cv2.circle(blend, (x, y), 9, C_UNASSIGNED, -1, cv2.LINE_AA)
            cv2.circle(blend, (x, y), 11, (240, 240, 240), 2, cv2.LINE_AA)

    # labels for points
    for p in points:
        x, y = int(round(p.pt[0])), int(round(p.pt[1]))
        cv2.putText(blend, p.label[0], (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(blend, p.label[0], (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    banner_h = 56
    legend_h = 72
    canvas = np.full((h + banner_h + legend_h, w, 3), 255, dtype=np.uint8)
    canvas[banner_h : banner_h + h, :] = blend
    cv2.putText(canvas, caption, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    leg_y = banner_h + h + 28
    for i, (txt, col) in enumerate(
        [
            ("Gray = unassigned", C_UNASSIGNED),
            ("Green/blue = assigned to tooth", C_TOOTH_COLORS[0]),
            ("Yellow = assigned this ring", C_NEW),
            ("Red X = dropped (>8 px)", C_DROP),
        ]
    ):
        x0 = 12 + i * 280
        cv2.circle(canvas, (x0, leg_y - 6), 8, col, -1)
        cv2.putText(canvas, txt, (x0 + 16, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

    return upscale(canvas, scale)


def build_key_strip(frames: list[np.ndarray], keys: list[int]) -> np.ndarray:
    tiles = [frames[k] for k in keys if k < len(frames)]
    gap = 12
    h = max(t.shape[0] for t in tiles)
    w = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    row = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for t in tiles:
        row[0 : t.shape[0], x : x + t.shape[1]] = t
        x += t.shape[1] + gap
    return row


def write_gif(paths: list[Path], out: Path, ms: int = 550) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    imgs = [Image.open(p).convert("RGB") for p in paths]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=ms, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    p.add_argument("--split", default="test")
    p.add_argument("--stem", default="431")
    p.add_argument("--teeth", default="1,2", help="two adjacent tooth indices")
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("research_log/figures/region_growing"))
    p.add_argument("--paper-figures", type=Path, default=Path("paper/figures"))
    args = p.parse_args()

    tidx = tuple(int(x) for x in args.teeth.split(","))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.paper_figures.mkdir(parents=True, exist_ok=True)

    xray, masks, bboxes, point_specs = load_scene(args.raw_root, args.split, args.stem, tidx)
    frame_states = simulate_assignment(point_specs, masks, max_r=8)

    rendered: list[np.ndarray] = []
    anim_dir = args.out_dir / "animation_frames_v2"
    anim_dir.mkdir(exist_ok=True)
    paths: list[Path] = []

    for r, states in enumerate(frame_states):
        cap = f"Ring r = {r} px" + (" — assign points on mask" if r == 0 else f" — expand +{r} px grace")
        img = render_frame(xray, masks, states, r, args.scale, cap)
        rendered.append(img)
        path = anim_dir / f"frame_{r:02d}.png"
        cv2.imwrite(str(path), img)
        paths.append(path)

    strip = build_key_strip(rendered, [0, 1, 2, 4, 6, 8])
    flow = np.vstack(
        [
            np.full((40, strip.shape[1], 3), 255, dtype=np.uint8),
            strip,
        ]
    )
    cv2.putText(
        flow,
        "v4 region growing on real tooth masks (test 431): cumulative rings recover off-mask CEJ/apex clicks",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    out_strip = args.out_dir / "region_growing_schematic_grid.png"
    out_flow = args.out_dir / "region_growing_flow.png"
    out_gif = args.out_dir / "region_growing_steps.gif"
    cv2.imwrite(str(out_strip), strip)
    # vertical flow: all rings
    flow_parts = rendered
    gap = 8
    fh = sum(f.shape[0] for f in flow_parts) + gap * (len(flow_parts) - 1) + 20
    fw = max(f.shape[1] for f in flow_parts)
    flow_canvas = np.full((fh, fw, 3), 255, dtype=np.uint8)
    y = 10
    for f in flow_parts:
        flow_canvas[y : y + f.shape[0], 0 : f.shape[1]] = f
        y += f.shape[0] + gap
    cv2.imwrite(str(out_flow), flow_canvas)
    cv2.imwrite(str(args.out_dir / "region_growing_paper_single.png"), rendered[-1])
    write_gif(paths, out_gif)

    for name in ("region_growing_schematic_grid.png", "region_growing_flow.png", "region_growing_steps.gif", "region_growing_paper_single.png"):
        src = args.out_dir / name
        if src.exists():
            shutil.copy2(src, args.paper_figures / name)

    meta = {
        "stem": args.stem,
        "teeth": list(tidx),
        "scale": args.scale,
        "frames": [str(x) for x in paths],
        "strip": str(out_strip),
        "gif": str(out_gif),
    }
    (args.out_dir / "manifest_v2.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
