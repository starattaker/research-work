"""Collect GT/pred severity pairs for ICC under different protocols."""

from __future__ import annotations


def collect_severity_pairs(rows: list[dict], protocol: str) -> tuple[list[float], list[float]]:
    gt_out: list[float] = []
    pred_out: list[float] = []
    for row in rows:
        if protocol in ("match_by_slot", "both_sides"):
            pred_map = {s: v for s, v in row.get("pred_sides", [])}
            for slot, gt_sev in row.get("gt_sides", []):
                pred_sev = pred_map.get(slot)
                if pred_sev is not None:
                    gt_out.append(gt_sev)
                    pred_out.append(pred_sev)
        else:
            gt_sev = row.get("gt_severity")
            pred_sev = row.get("pred_severity")
            if gt_sev is not None and pred_sev is not None:
                gt_out.append(gt_sev)
                pred_out.append(pred_sev)
    return gt_out, pred_out
