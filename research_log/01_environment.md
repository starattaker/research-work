# 01 — Environment

**Date:** 2026-08-12

## Python / hardware

| Item | Value |
|------|-------|
| OS | Windows 10 (build 26200) |
| Python | 3.12 (venv at `venv/`) |
| PyTorch | 2.6.0+cu124 |
| CUDA | Available |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |

## Key packages

`ultralytics`, `torchvision`, `albumentations`, `torchmetrics`, `opencv-python`, `pycocotools`, `scikit-learn`

Install: `pip install -r requirements.txt` then PyTorch cu124 wheel.

## Reference code

Official repo cloned to `official_repo/` (authors' scripts; some files incomplete). Custom pipeline in `src/` aligned to paper methods.
