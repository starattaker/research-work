"""Resolve DenPAR raw dataset layout (friend GPU: data/DenPAR/Dataset/)."""

from __future__ import annotations

from pathlib import Path

SPLIT_TO_FOLDER = {
    "train": "Training",
    "val": "Validation",
    "test": "Testing",
    "training": "Training",
    "validation": "Validation",
    "testing": "Testing",
}

# Repo-relative default on friend machine
DEFAULT_DENPAR_ROOT = Path("data/DenPAR/Dataset")


def resolve_denpar_root(raw_root: Path | None = None) -> Path:
    """Return directory that directly contains Training/, Validation/, Testing/."""
    candidates: list[Path] = []
    if raw_root is not None:
        candidates.extend([raw_root, raw_root / "Dataset"])
    candidates.extend(
        [
            DEFAULT_DENPAR_ROOT,
            Path("data/DenPAR/Dataset"),
            Path("data/DenPAR"),
        ]
    )
    seen: set[str] = set()
    for base in candidates:
        key = str(base.resolve()) if base.exists() else str(base)
        if key in seen:
            continue
        seen.add(key)
        if (base / "Testing" / "Key Points Annotations").is_dir():
            return base
    if raw_root is not None:
        return raw_root
    return DEFAULT_DENPAR_ROOT


def denpar_split_name(split: str) -> str:
    return SPLIT_TO_FOLDER.get(split.lower(), split)


def denpar_mask_path(
    raw_root: Path | None,
    split: str,
    stem: str,
    tooth_idx: int,
) -> Path:
    root = resolve_denpar_root(raw_root)
    split_dir = denpar_split_name(split)
    return root / split_dir / "Masks (Tooth-wise)" / stem / f"mask{tooth_idx + 1}.png"


def denpar_keypoints_dir(raw_root: Path | None, split: str) -> Path:
    root = resolve_denpar_root(raw_root)
    return root / denpar_split_name(split) / "Key Points Annotations"
