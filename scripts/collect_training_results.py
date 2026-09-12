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

    # When --experiment-id is given, scope the glob to that experiment's own
    # directories only ("{exp}_*/metrics.json"). Globbing everything under
    # runs-root and then force-labelling every match with the same id is a
    # bug: it silently re-tags every OTHER experiment's metrics.json as this
    # one too, and since sorted() walks alphabetically, whichever directory
    # sorts last for a given keypoint (e.g. v7_cej after v6_cej) clobbers the
    # correct entry in the registry's "latest" pointer. This exact bug
    # corrupted the v6 registry entry with v7's numbers on 2026-09-12.
    pattern = f"{args.experiment_id}_*/metrics.json" if args.experiment_id else "**/metrics.json"
    found = 0
    for metrics in sorted(args.runs_root.glob(pattern)):
        run_dir = metrics.parent
        if run_dir.name.startswith("_"):
            continue
        exp = args.experiment_id or infer_experiment_id(run_dir)
        if args.experiment_id and infer_experiment_id(run_dir) != args.experiment_id:
            print(
                f"skip {run_dir}: directory name does not match --experiment-id {args.experiment_id}",
                file=sys.stderr,
            )
            continue
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
