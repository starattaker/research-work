"""Collect metrics — wrapper around experiment registry (manual / backfill)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scripts._bootstrap  # noqa: F401

from src.experiment.registry import (
    after_training,
    finalize_experiment,
    infer_experiment_id,
    load_registry,
    rebuild_derived_artifacts,
)


def main():
    parser = argparse.ArgumentParser(description="Backfill experiment registry from runs/keypoints/")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/keypoints"))
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--finalize", type=str, default=None, help="Finalize experiment id if complete")
    parser.add_argument("--push", action="store_true", help="Git push after finalize")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    # The glob itself must stay broad: v1's runs live in BARE directories
    # (runs/keypoints/cej, not runs/keypoints/v1_cej), so a glob scoped to
    # "{experiment_id}_*/metrics.json" silently matches nothing for v1.
    #
    # What must be scoped is which matches get RECORDED under the requested
    # id: only keep a match if infer_experiment_id(run_dir) actually agrees
    # with --experiment-id. Forcing every match to the requested id
    # regardless of its own directory's identity is the bug that corrupted
    # the v6 registry entry with v7's numbers on 2026-09-12 - sorted() walks
    # alphabetically, so v7_cej was force-relabelled "v6" and clobbered the
    # real v6_cej entry. This check is what prevents that, for every id.
    found = 0
    for metrics in sorted(args.runs_root.glob("**/metrics.json")):
        run_dir = metrics.parent
        if run_dir.name.startswith("_"):
            continue
        inferred = infer_experiment_id(run_dir)
        if args.experiment_id and inferred != args.experiment_id:
            continue
        exp = args.experiment_id or inferred
        kpt = json.loads(metrics.read_text(encoding="utf-8")).get("keypoint_type")
        if not kpt:
            kpt = run_dir.name.split("_")[-1]
        after_training(run_dir, exp, kpt, auto_push=False)
        found += 1

    if found == 0:
        print(f"No metrics under {args.runs_root}", file=sys.stderr)
        sys.exit(1)

    rebuild_derived_artifacts()
    reg = load_registry()
    print(f"Registry updated: {len(reg.get('record_index', []))} records")

    if args.finalize:
        finalize_experiment(args.finalize, auto_push=args.push and not args.no_push)


if __name__ == "__main__":
    main()
