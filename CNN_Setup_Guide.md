# Lung Cancer Detection CNN — Setup & Training Guide

**Student:** 26164741 Huu Thuc Tran  
**Dataset:** IQ-OTH/NCCD Lung Cancer Dataset  
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)  
**CUDA:** 12.7 (driver) / 12.4 (PyTorch build)

---

## Dataset Overview

| Class | Folder | Images |
|-------|--------|--------|
| Benign | `Data/Bengin cases/` | 120 |
| Malignant | `Data/Malignant cases/` | 561 |
| Normal | `Data/Normal cases/` | 416 |
| **Total** | | **1,097** |

All images are **512 × 512 px, RGB JPEG**.  
The dataset is moderately imbalanced — `WeightedRandomSampler` is used during training to compensate.

---

## Step 1 — Install Miniconda (already done)

If Miniconda is not yet installed, download and install it from:  
https://docs.conda.io/en/latest/miniconda.html

Verify:
```
conda --version   # should print conda 26.x.x
```

---

## Step 2 — Create the Conda Environment

Open **Anaconda Prompt** (or any terminal where `conda` is on PATH):

```bash
conda create -n lung_cancer python=3.11 -y
conda activate lung_cancer
```

---

## Step 3 — Install PyTorch with CUDA 12.4

Your driver supports CUDA 12.7; PyTorch's latest compatible build is **cu124**:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

This installs:
- `torch 2.6.0+cu124`
- `torchvision 0.21.0+cu124`
- `torchaudio 2.6.0+cu124`

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True
# Expected: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## Step 4 — Install Additional Dependencies

```bash
pip install scikit-learn matplotlib pillow tqdm
```

| Package | Purpose |
|---------|---------|
| `scikit-learn` | Confusion matrix, classification report |
| `matplotlib` | Training curves, plots |
| `pillow` | Image loading |
| `tqdm` | Progress bars |

---

## Step 5 — Project Structure

```
3. Fuzzy Logic & Neural Network/
├── Data/
│   ├── Bengin cases/       # 120 CT scan JPEGs
│   ├── Malignant cases/    # 561 CT scan JPEGs
│   └── Normal cases/       # 416 CT scan JPEGs
├── lung_cancer_cnn.py      # Main training script
├── cnn_output/             # Created automatically
│   ├── best_resnet18.pth   # Best model checkpoint
│   ├── training_curves.png # Loss & accuracy curves
│   ├── confusion_matrix.png
│   └── training_summary.txt
└── CNN_Setup_Guide.md      # This file
```

---

## Step 6 — Model Architecture

The script uses **transfer learning** with a pretrained **ResNet-18** backbone (ImageNet weights):

```
Input: 224×224 RGB image
  └─ ResNet-18 backbone (pretrained on ImageNet)
       └─ Dropout(0.4)
            └─ Linear(512 → 3)   # 3 classes: Benign / Malignant / Normal
```

Key design choices:
- **Transfer learning**: ImageNet features transfer well to CT scan texture patterns
- **Dropout(0.4)**: Regularisation to prevent overfitting on the small dataset
- **WeightedRandomSampler**: Oversamples benign (minority class) during training
- **Class-weighted CrossEntropyLoss**: Further penalises misclassification of rare classes
- **AdamW optimizer** + **StepLR scheduler**: LR halved every 10 epochs

Other available backbones (set `BACKBONE` in the script):
- `"resnet50"` — deeper, ~2× more parameters
- `"efficientnet_b0"` — more efficient, similar accuracy

---

## Step 7 — Data Augmentation

Training augmentations applied to prevent overfitting:

| Augmentation | Value |
|--------------|-------|
| Resize | 224 × 224 |
| Random Horizontal Flip | p = 0.5 |
| Random Vertical Flip | p = 0.3 |
| Random Rotation | ±15° |
| Color Jitter | brightness ±0.2, contrast ±0.2 |
| Normalize | ImageNet mean/std |

Validation / Test: only Resize + Normalize (no random augmentations).

---

## Step 8 — Run Training

Activate the environment and run from the project directory:

```bash
conda activate lung_cancer
cd "E:\3. Fuzzy Logic & Neural Network"
python lung_cancer_cnn.py
```

Expected console output:
```
============================================================
  Lung Cancer Detection CNN
  IQ-OTH/NCCD Dataset  |  3-class classification
============================================================
Dataset split — Train: 822  Val: 164  Test: 110
Using device: cuda
  GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  VRAM: 8.2 GB

Model: resnet18  |  Params: 11,181,827  |  Trainable: 11,181,827

Starting training for 30 epochs...

Epoch  Train Loss  Train Acc   Val Loss   Val Acc        LR    Time
--------------------------------------------------------------------
    1      0.6423     72.14%     0.5812    74.39%  0.000100   12.3s
    ...
*** Best model saved (val_acc=88.41%) ***
```

Training takes approximately **5–8 minutes** on RTX 4060 Laptop GPU for 30 epochs.

---

## Step 9 — Understanding the Output

After training completes, `cnn_output/` contains:

### `training_curves.png`
Loss and accuracy curves for train/val sets across all epochs.  
Look for:
- Decreasing and converging loss curves (no divergence)
- Accuracy curves that plateau (good generalisation)

### `confusion_matrix.png`
Shows predicted vs. true class for the test set.  
Look for:
- High values on the diagonal (correct predictions)
- Off-diagonal values show where the model confuses classes

### `training_summary.txt`
Quick summary: best val accuracy, final test accuracy, hyperparameters.

---

## Step 10 — Hyperparameter Tuning (Optional)

Edit these constants in `lung_cancer_cnn.py`:

```python
BACKBONE    = "resnet18"   # try "resnet50" or "efficientnet_b0"
NUM_EPOCHS  = 30           # increase to 50 for potentially better accuracy
BATCH_SIZE  = 32           # reduce to 16 if OOM errors occur
LR          = 1e-4         # try 5e-5 for finer tuning
LR_STEP     = 10           # reduce LR every N epochs
LR_GAMMA    = 0.5          # LR reduction factor
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `BATCH_SIZE` to 16 or 8 |
| `ModuleNotFoundError: torch` | Run `conda activate lung_cancer` first |
| Training very slow | Confirm `torch.cuda.is_available()` returns `True` |
| Low accuracy (<70%) | Increase `NUM_EPOCHS` to 50, try `resnet50` backbone |
| `num_workers` warning | Set `num_workers=0` (Windows multiprocessing limitation) |

---

## Alternative: Using pip (without Conda)

If you prefer using the system Python directly:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install scikit-learn matplotlib pillow tqdm
python lung_cancer_cnn.py
```

> **Note:** This installs to the global Python environment. Conda environments keep dependencies isolated and reproducible — recommended for research projects.
