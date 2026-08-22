"""Label-based severity ICC: expert-derived vs each preprocessing export (not model inference)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import scripts._bootstrap  # noqa: F401

from src.preprocess.prepare_dataset import SPLIT_ALIASES, build_tooth_records
from src.severity.bone_loss import compute_bone_loss_severity


def icc21(matrix: np.ndarray) -> float | None:
    y = np.asarray(matrix, dtype=np.float64)
    n, k = y.shape
    if n < 3:
        return None
    m = y.mean(axis=1, keepdims=True)
    r = y.mean(axis=0, keepdims=True)
    g = y.mean()
    ss_b = k * np.sum((m - g) ** 2)
    ss_r = n * np.sum((r - g) ** 2)
    ss_t = np.sum((y - g) ** 2)
    ss_e = ss_t - ss_b - ss_r
    ms_b = ss_b / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_b + (k - 1) * ms_e + (k / n) * (ms_r - ms_e)
    if abs(denom) < 1e-12:
        return None
    return float((ms_b - ms_e) / denom)


def vis_pt(kps: list, i: int = 0) -> tuple[float, float]:
    p = kps[i]
    if len(p) > 2 and int(p[2]) == 0:
        return (0.0, 0.0)
    return (float(p[0]), float(p[1]))


def sev_strategy(raw_root: Path, split_raw: str, stem: str, strategy: str) -> list[float | None]:
    kp_p = raw_root / split_raw / "Key Points Annotations" / f"{stem}.json"
    bone_p = raw_root / split_raw / "Bone Level Annotations" / f"{stem}.json"
    mask_dir = raw_root / split_raw / "Masks (Tooth-wise)" / stem
    kp = json.loads(kp_p.read_text())
    bone = json.loads(bone_p.read_text()) if bone_p.exists() else None
    recs = build_tooth_records(kp, bone, mask_dir if mask_dir.exists() else None, strategy=strategy)
    out = []
    for r in recs:
        c = tuple(r.cej[0]) if r.cej[0] != [0.0, 0.0] else (0.0, 0.0)
        t = tuple(r.intersection[0]) if r.intersection[0] != [0.0, 0.0] else (0.0, 0.0)
        a = tuple(r.apex[0]) if r.apex[0] != [0.0, 0.0] else (0.0, 0.0)
        out.append(compute_bone_loss_severity(c, t, a))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/DenPAR/Dataset"))
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--out", type=Path, default=Path("research_log/severity_icc_label_based.json"))
    args = p.parse_args()

    versions = {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4"}
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    pairs: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k in versions}

    for split in splits:
        split_raw = {v: k for k, v in SPLIT_ALIASES.items()}[split]
        kp_dir = args.raw_root / split_raw / "Key Points Annotations"
        for kp_p in tqdm(sorted(kp_dir.glob("*.json")), desc=split):
            stem = kp_p.stem
            exp = sev_strategy(args.raw_root, split_raw, stem, "v2")
            if not exp:
                continue
            for vid, strat in versions.items():
                pred = sev_strategy(args.raw_root, split_raw, stem, strat)
                if len(pred) != len(exp):
                    continue
                for e, pr in zip(exp, pred):
                    if e is None or pr is None:
                        continue
                    pairs[vid][0].append(e)
                    pairs[vid][1].append(pr)

    results = {
        "note": "Label ICC vs expert (strict bbox + v2 intersection). v2=1.0 by definition. NOT model/YOLO ICC (paper 0.801).",
        "paper_target": 0.801,
        "splits": splits,
        "icc": {},
        "mae_pct": {},
        "n_pairs": {},
    }
    for vid, (exp, pr) in pairs.items():
        n = len(exp)
        results["n_pairs"][vid] = n
        if n < 3:
            results["icc"][vid] = None
            continue
        mat = np.column_stack([exp, pr])
        results["icc"][vid] = icc21(mat)
        results["mae_pct"][vid] = float(np.mean(np.abs(mat[:, 0] - mat[:, 1])))

    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
