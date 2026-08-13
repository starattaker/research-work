# Clone & train (Linux — e.g. RTX 5070)

## 1. Clone private repo

```bash
git clone -b denpar-severity-replication git@github.com:YOUR_USERNAME/research-work.git
cd research-work
```

Or HTTPS:

```bash
git clone -b denpar-severity-replication https://github.com/YOUR_USERNAME/research-work.git
```

## 2. Environment

```bash
python3.11 -m venv venv || python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 3. What is NOT in the repo (download / regenerate)

| Item | How to get it |
|------|----------------|
| Raw DenPAR | https://doi.org/10.5281/zenodo.16645076 → extract to `data/DenPAR/Dataset/` |
| Preprocessed data | **Included** in repo under `data/processed/` |
| `yolov8x.pt` | Auto-downloads on first YOLO train |
| Checkpoints | Not in repo — train fresh or copy `last.pt` separately |

Re-run preprocessing only if needed:

```bash
python -m src.preprocess.prepare_dataset
python scripts/validate_preprocessed.py
```

## 4. Train (paper params — RTX 5070)

**YOLOv8x** (batch/workers only hardware tweaks):

```bash
python scripts/train_detection.py --batch 4 --workers 8
```

**Keypoint R-CNN** (after YOLO, one at a time):

```bash
python -m src.keypoint.train --data-root data/processed/keypoints/cej --keypoint-type cej --output-dir runs/keypoints/cej --batch-size 8
```

Do not change augmentations or other paper hyperparameters.
