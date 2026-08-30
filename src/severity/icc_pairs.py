"""Collect GT/pred severity pairs for ICC under different protocols."""

from __future__ import annotations

from src.severity.side_details import (
    SideDetail,
    pair_sides_by_cej,
    pair_sides_by_slot_index,
)


def collect_severity_pairs(rows: list[dict], protocol: str) -> tuple[list[float], list[float]]:
    gt_out: list[float] = []
    pred_out: list[float] = []
    for row in rows:
        gt_details: list[SideDetail] = row.get("gt_side_details", [])
        pred_details: list[SideDetail] = row.get("pred_side_details", [])

        if protocol in ("both_sides", "match_by_slot") and gt_details and pred_details:
            if protocol == "match_by_slot":
                pairs = pair_sides_by_slot_index(gt_details, pred_details)
            else:
                pairs = pair_sides_by_cej(gt_details, pred_details)
        else:
            pairs = []
            gt_sev = row.get("gt_severity")
            pred_sev = row.get("pred_severity")
            if gt_sev is not None and pred_sev is not None:
                pairs = [(gt_sev, pred_sev)]

        for g, p in pairs:
            gt_out.append(g)
            pred_out.append(p)

    return gt_out, pred_out
