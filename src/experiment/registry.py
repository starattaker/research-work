"""Append-only experiment metrics registry for research paper tables."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PAPER_OKS = {"cej": 0.954, "intersection": 0.912, "apex": 0.815}
KPT_TYPES = ("cej", "intersection", "apex")

EXPERIMENTS_ROOT = Path("research_log/experiments")
RECORDS_DIR = EXPERIMENTS_ROOT / "records"
REGISTRY_PATH = EXPERIMENTS_ROOT / "registry.json"
PAPER_TABLE_PATH = EXPERIMENTS_ROOT / "paper_table.json"
SUMMARY_DIR = EXPERIMENTS_ROOT / "summaries"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def infer_experiment_id(output_dir: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    name = output_dir.name
    m = re.match(r"^(v\d+)_(cej|intersection|apex)$", name)
    if m:
        return m.group(1)
    if name in KPT_TYPES:
        return "v1"
    return name.split("_")[0] if "_" in name else "v1"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"latest": {}, "record_index": []}


def save_registry(reg: dict) -> None:
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    # Keep index bounded (metadata only; full records stay on disk)
    reg["record_index"] = reg.get("record_index", [])[-200:]
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def summarize_run(output_dir: Path, keypoint_type: str) -> dict:
    metrics_path = output_dir / "metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    history_path = output_dir / "history.json"
    history = []
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    elif "history" in data:
        history = data["history"]

    best_epoch = None
    best_val = None
    if history:
        best = min(history, key=lambda r: r.get("val_loss", float("inf")))
        best_epoch = best.get("epoch")
        best_val = best.get("val_loss")

    test_map = data.get("test_map") or {}
    return {
        "keypoint_type": keypoint_type,
        "run_dir": output_dir.as_posix(),
        "test_oks": data.get("test_oks"),
        "val_oks": data.get("val_oks"),
        "test_map_50": test_map.get("map_50"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "paper_oks_target": PAPER_OKS.get(keypoint_type),
    }


def record_keypoint_run(
    output_dir: Path,
    experiment_id: str | None = None,
    keypoint_type: str | None = None,
) -> dict:
    """Append timestamped record; update latest pointer (no overwrite of old records)."""
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    kpt = keypoint_type or data.get("keypoint_type") or output_dir.name.split("_")[-1]
    exp = infer_experiment_id(output_dir, experiment_id)

    summary = summarize_run(output_dir, kpt)
    record = {
        "recorded_at": _utc_iso(),
        "experiment_id": exp,
        **summary,
    }

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    record_name = f"{_utc_stamp()}_{exp}_{kpt}.json"
    record_path = RECORDS_DIR / record_name
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    reg = load_registry()
    reg.setdefault("latest", {}).setdefault(exp, {})[kpt] = {
        "record_file": record_path.as_posix(),
        **{k: record[k] for k in summary if k != "run_dir"},
        "run_dir": summary["run_dir"],
    }
    reg.setdefault("record_index", []).append(
        {"file": record_path.as_posix(), "experiment_id": exp, "keypoint_type": kpt, "at": record["recorded_at"]}
    )
    save_registry(reg)
    rebuild_derived_artifacts()
    return record


def experiment_complete(experiment_id: str) -> bool:
    reg = load_registry()
    latest = reg.get("latest", {}).get(experiment_id, {})
    return all(k in latest for k in KPT_TYPES)


def rebuild_derived_artifacts() -> None:
    reg = load_registry()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    paper_rows = []
    for exp_id in sorted(reg.get("latest", {})):
        latest = reg["latest"][exp_id]
        row = {"experiment_id": exp_id, "models": {}}
        for kpt in KPT_TYPES:
            if kpt in latest:
                row["models"][kpt] = {
                    "test_oks": latest[kpt].get("test_oks"),
                    "paper_target": PAPER_OKS[kpt],
                    "best_epoch": latest[kpt].get("best_epoch"),
                    "epochs_run": latest[kpt].get("epochs_run"),
                    "run_dir": latest[kpt].get("run_dir"),
                }
        paper_rows.append(row)

    PAPER_TABLE_PATH.write_text(
        json.dumps({"updated": _utc_iso(), "experiments": paper_rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Experiment metrics (auto-generated)",
        "",
        f"Updated: {_utc_iso()}",
        "",
        "## Paper table source (`paper_table.json`)",
        "",
        "| Experiment | CEJ OKS | Inter OKS | Apex OKS | Paper CEJ | Paper Int | Paper Apex |",
        "|------------|--------:|----------:|---------:|----------:|----------:|-----------:|",
    ]
    for row in paper_rows:
        m = row["models"]
        def oks(k):
            v = m.get(k, {}).get("test_oks")
            return f"{v:.3f}" if v is not None else "—"

        lines.append(
            f"| **{row['experiment_id']}** | {oks('cej')} | {oks('intersection')} | {oks('apex')} | "
            f"0.954 | 0.912 | 0.815 |"
        )
    (SUMMARY_DIR / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for exp_id in reg.get("latest", {}):
        latest = reg["latest"][exp_id]
        exp_lines = [f"# Keypoint results — {exp_id}", "", f"Updated: {_utc_iso()}", ""]
        exp_lines.append("| Model | test_oks | Paper | best epoch | run_dir |")
        exp_lines.append("|-------|---------:|------:|-----------:|---------|")
        for kpt in KPT_TYPES:
            if kpt not in latest:
                continue
            e = latest[kpt]
            oks = e.get("test_oks")
            oks_s = f"{oks:.3f}" if oks is not None else "—"
            exp_lines.append(
                f"| {kpt} | {oks_s} | {PAPER_OKS[kpt]} | {e.get('best_epoch')} | `{e.get('run_dir')}` |"
            )
        (SUMMARY_DIR / f"{exp_id}_keypoints.md").write_text("\n".join(exp_lines) + "\n", encoding="utf-8")


def update_checkpoint_and_paper(experiment_id: str) -> None:
    reg = load_registry()
    latest = reg.get("latest", {}).get(experiment_id, {})
    if not latest:
        return

    # --- CHECKPOINT.md (short status block) ---
    ckpt_path = Path("research_log/CHECKPOINT.md")
    status = f"Keypoint training **{experiment_id} complete** ({_utc_iso()})."
    oks_bits = []
    for kpt in KPT_TYPES:
        if kpt in latest:
            v = latest[kpt].get("test_oks")
            if v is not None:
                oks_bits.append(f"{kpt} OKS={v:.3f}")
    metrics_line = "; ".join(oks_bits)

    ckpt = ckpt_path.read_text(encoding="utf-8") if ckpt_path.exists() else ""
    if "**Status:**" in ckpt:
        ckpt = re.sub(r"\*\*Status:\*\*[^\n]*", f"**Status:** {status}", ckpt, count=1)
    else:
        ckpt = f"**Status:** {status}\n\n" + ckpt
    marker = "## Latest metrics (auto)"
    block = f"{marker}\n\n- **{experiment_id}:** {metrics_line}\n- Registry: `research_log/experiments/paper_table.json`\n"
    if marker in ckpt:
        ckpt = re.sub(r"## Latest metrics \(auto\)[\s\S]*?(?=\n## |\Z)", block + "\n", ckpt)
    else:
        ckpt = ckpt.rstrip() + "\n\n" + block + "\n"
    ckpt_path.write_text(ckpt, encoding="utf-8")

    # --- 06_keypoint_training.md append section if missing ---
    doc06 = Path("research_log/06_keypoint_training.md")
    section_title = f"## {experiment_id} keypoints (auto)"
    rows = []
    for kpt in KPT_TYPES:
        if kpt not in latest:
            continue
        e = latest[kpt]
        rows.append(
            f"| {kpt} | {e.get('test_oks', '—')} | {PAPER_OKS[kpt]} | {e.get('best_epoch')} | `{e.get('run_dir')}` |"
        )
    section = (
        f"{section_title}\n\n"
        f"Recorded: {_utc_iso()}\n\n"
        f"| Model | test_oks | Paper | best epoch | run_dir |\n"
        f"|-------|----------:|------:|-----------:|---------|\n"
        + "\n".join(rows)
        + "\n"
    )
    existing = doc06.read_text(encoding="utf-8") if doc06.exists() else "# 06 — Keypoint R-CNN\n\n"
    if section_title in existing:
        existing = re.sub(
            rf"{re.escape(section_title)}[\s\S]*?(?=\n## |\Z)",
            section,
            existing,
        )
    else:
        existing = existing.rstrip() + "\n\n" + section
    doc06.write_text(existing, encoding="utf-8")

    # --- paper fragment for \input ---
    frag = Path("paper/generated_keypoint_table.tex")
    frag.parent.mkdir(parents=True, exist_ok=True)
    tex_rows = []
    for kpt in KPT_TYPES:
        if kpt not in latest:
            continue
        v = latest[kpt].get("test_oks")
        if v is None:
            continue
        tex_rows.append(f"{kpt} & {v:.3f} & {PAPER_OKS[kpt]:.3f} \\\\")
    tex = (
        "% Auto-generated — do not edit by hand\n"
        f"% experiment: {experiment_id}  updated: {_utc_iso()}\n"
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{Keypoint test OKS — {experiment_id}.}}\n"
        "\\begin{tabular}{lrr}\n\\toprule\n"
        "Model & Ours (OKS) & Paper \\\\\n\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    frag.write_text(tex, encoding="utf-8")


def git_sync_logs(experiment_id: str, push: bool = True) -> None:
    paths = [
        "research_log/experiments/",
        "research_log/CHECKPOINT.md",
        "research_log/06_keypoint_training.md",
        "paper/generated_keypoint_table.tex",
    ]
    try:
        subprocess.run(["git", "add", *paths], check=True, capture_output=True, text=True)
        msg = f"Auto-log keypoint experiment {experiment_id} ({_utc_iso()})"
        r = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True)
        if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
            print(f"git commit note: {r.stderr or r.stdout}")
        if push:
            p = subprocess.run(["git", "push"], capture_output=True, text=True)
            if p.returncode != 0:
                print(f"git push failed (commit is local): {p.stderr or p.stdout}")
            else:
                print("git push: research_log/experiments synced")
    except FileNotFoundError:
        print("git not found — skipped auto push")


def finalize_experiment(experiment_id: str, auto_push: bool = True) -> None:
    if not experiment_complete(experiment_id):
        return
    rebuild_derived_artifacts()
    update_checkpoint_and_paper(experiment_id)
    git_sync_logs(experiment_id, push=auto_push)
    print(f"Experiment {experiment_id} finalized (registry + checkpoint + paper fragment).")


def after_training(
    output_dir: Path,
    experiment_id: str | None,
    keypoint_type: str,
    auto_push: bool = True,
) -> None:
    record_keypoint_run(output_dir, experiment_id=experiment_id, keypoint_type=keypoint_type)
    exp = infer_experiment_id(Path(output_dir), experiment_id)
    if experiment_complete(exp):
        finalize_experiment(exp, auto_push=auto_push)
    else:
        missing = [k for k in KPT_TYPES if k not in load_registry().get("latest", {}).get(exp, {})]
        print(f"Recorded {exp}/{keypoint_type}. Still waiting for: {', '.join(missing)}")
