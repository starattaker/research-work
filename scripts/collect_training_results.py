"""Collect keypoint (and optional YOLO) training metrics into research_log/.

Run on the training machine after CEJ / intersection / apex finish:

  python scripts/collect_training_results.py
  python scripts/collect_training_results.py --push-log   # also git add + commit summary

Reads: runs/keypoints/*/metrics.json, history.json
Writes: research_log/training_results_summary.md
        research_log/training_results_summary.json
        research_log/metrics_snapshot.txt   (paste-friendly for chat)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import scripts._bootstrap  # noqa: F401

PAPER_OKS = {"cej": 0.954, "intersection": 0.912, "apex": 0.815}


def find_keypoint_runs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    dirs = []
    for metrics in runs_root.glob("**/metrics.json"):
        if metrics.parent.name.startswith("_"):
            continue
        dirs.append(metrics.parent)
    return sorted(set(dirs), key=lambda p: str(p))


def load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    history_path = run_dir / "history.json"
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    elif "history" in data:
        history = data["history"]

    best_epoch = None
    best_val = None
    if history:
        row = min(history, key=lambda r: r.get("val_loss", float("inf")))
        best_epoch = row.get("epoch")
        best_val = row.get("val_loss")

    kpt_type = data.get("keypoint_type") or run_dir.name.split("_")[-1]
    return {
        "run_dir": str(run_dir.as_posix()),
        "keypoint_type": kpt_type,
        "test_oks": data.get("test_oks"),
        "val_oks": data.get("val_oks"),
        "train_oks": data.get("train_oks"),
        "test_map_50": (data.get("test_map") or {}).get("map_50"),
        "test_map": (data.get("test_map") or {}).get("map"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "paper_oks_target": PAPER_OKS.get(kpt_type),
    }


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Run folder | Model | test_oks | Paper OKS | best epoch | epochs |",
        "|------------|-------|---------:|----------:|-----------:|-------:|",
    ]
    for r in rows:
        paper = r.get("paper_oks_target")
        paper_s = f"{paper:.3f}" if paper else "—"
        oks = r.get("test_oks")
        oks_s = f"{oks:.3f}" if oks is not None else "—"
        lines.append(
            f"| `{Path(r['run_dir']).name}` | {r['keypoint_type']} | {oks_s} | {paper_s} | "
            f"{r.get('best_epoch') or '—'} | {r.get('epochs_run') or 0} |"
        )
    return "\n".join(lines)


def snapshot_text(rows: list[dict], generated: str) -> str:
    lines = [f"# Training metrics snapshot — {generated}", ""]
    for r in rows:
        lines.append(f"[{r['keypoint_type']}] {r['run_dir']}")
        lines.append(f"  test_oks: {r.get('test_oks')}")
        lines.append(f"  paper:   {r.get('paper_oks_target')}")
        lines.append(f"  best_epoch: {r.get('best_epoch')}  epochs_run: {r.get('epochs_run')}")
        lines.append("")
    lines.append("Paste this file into chat, or say: Make a checkpoint")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs/keypoints"))
    parser.add_argument("--out-dir", type=Path, default=Path("research_log"))
    parser.add_argument(
        "--push-log",
        action="store_true",
        help="git add summary files and commit (does not push)",
    )
    args = parser.parse_args()

    run_dirs = find_keypoint_runs(args.runs_root)
    rows = []
    for d in run_dirs:
        row = load_run(d)
        if row:
            rows.append(row)

    if not rows:
        print(f"No metrics.json found under {args.runs_root}", file=sys.stderr)
        print("Expected e.g. runs/keypoints/cej/metrics.json", file=sys.stderr)
        sys.exit(1)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    payload = {"generated": generated, "runs": rows}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md_path = args.out_dir / "training_results_summary.md"
    json_path = args.out_dir / "training_results_summary.json"
    txt_path = args.out_dir / "metrics_snapshot.txt"

    md = f"""# Keypoint training results

Generated: {generated}

{markdown_table(rows)}

## Files per run

Each folder under `runs/keypoints/` should contain:
- `best.pt` — weights (keep locally; do not git push large files)
- `metrics.json` — test_oks, test_map, history
- `history.json` — loss per epoch
- `loss_curve.png`, `tensorboard/` — optional viz

## Compare to paper

Use **test_oks** vs paper OKS targets (not test_map).

Run again after each model: `python scripts/collect_training_results.py`
"""
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(snapshot_text(rows, generated), encoding="utf-8")

    print(md)
    print(f"\nWrote:\n  {md_path}\n  {json_path}\n  {txt_path}")

    if args.push_log:
        subprocess.run(
            ["git", "add", str(md_path), str(json_path), str(txt_path)],
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Collect keypoint training metrics ({generated})."],
            check=True,
        )
        print("Committed summary to git. Run: git push")


if __name__ == "__main__":
    main()
