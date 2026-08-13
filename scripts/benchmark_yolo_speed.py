"""Quick YOLO throughput benchmark — NOT for replication training.

Uses fraction=0.15 and val=False for speed testing only. Does not change mosaic
or other augmentations (mosaic stays at Ultralytics default 1.0). Never use
benchmark settings for the real paper replication run.
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

DATA = Path("data/processed/yolo_detection/data.yaml")
BENCH_DIR = Path("runs/bench_speed")
FRACTION = 0.15  # ~98 train images — enough to stress VRAM, fast to run


def run_config(batch: int, workers: int) -> dict:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    name = f"b{batch}_w{workers}"
    out = BENCH_DIR / name
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    model = YOLO("yolov8x.pt")
    t0 = time.perf_counter()
    try:
        model.train(
            data=str(DATA),
            epochs=1,
            imgsz=640,
            batch=batch,
            workers=workers,
            device=0,
            optimizer="Adam",
            lr0=1e-4,
            cos_lr=True,
            amp=True,
            fraction=FRACTION,
            project=str(BENCH_DIR),
            name=name,
            exist_ok=True,
            plots=False,
            save=False,
            val=False,
            verbose=False,
            patience=999,
        )
        elapsed = time.perf_counter() - t0
        n_imgs = int(650 * FRACTION)
        return {
            "batch": batch,
            "workers": workers,
            "ok": True,
            "elapsed_s": round(elapsed, 1),
            "images_per_s": round(n_imgs / elapsed, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 0),
        }
    except Exception as exc:
        return {
            "batch": batch,
            "workers": workers,
            "ok": False,
            "error": str(exc)[:240],
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 0),
        }
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for batch in (1, 2, 3):
        any_ok = False
        for workers in (4, 2, 0):
            print(f"Testing batch={batch} workers={workers}...", flush=True)
            r = run_config(batch, workers)
            results.append(r)
            print(json.dumps(r), flush=True)
            if r["ok"]:
                any_ok = True
        if not any_ok:
            break

    ok = [r for r in results if r["ok"]]
    best = max(ok, key=lambda x: x["images_per_s"]) if ok else None
    summary = {"best": best, "all": results}
    (BENCH_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nBEST:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
