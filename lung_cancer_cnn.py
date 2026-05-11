"""
Lung Cancer Detection CNN
IQ-OTH/NCCD Dataset — 3-class classification: benign / malignant / normal
Author: 26164741 Huu Thuc Tran
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Force stdout flush (important for Windows background processes)
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR    = Path("Data")
OUTPUT_DIR  = Path("cnn_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Map folder names → class labels (alphabetical order used by ImageFolder)
# Bengin cases → 0, Malignant cases → 1, Normal cases → 2
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

IMG_SIZE    = 224          # resize to 224×224 (standard for pretrained backbones)
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
LR          = 1e-4
LR_STEP     = 10           # reduce LR every N epochs
LR_GAMMA    = 0.5
WEIGHT_DECAY = 1e-4
VAL_SPLIT   = 0.15         # 15% validation
TEST_SPLIT  = 0.10         # 10% test
SEED        = 42
NUM_CLASSES = 3
BACKBONE    = "resnet18"   # "resnet18" | "resnet50" | "efficientnet_b0"


# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
# DATASET SPLIT  (train / val / test)
# ─────────────────────────────────────────────────────────────
def make_splits(data_dir, val_split=0.15, test_split=0.10, seed=42):
    """
    Split each class folder into train/val/test indices.
    Returns three ImageFolder-compatible datasets.
    """
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    targets = np.array(full_dataset.targets)
    n = len(full_dataset)

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(datasets.ImageFolder(str(data_dir), transform=train_transform), train_idx)
    val_ds   = Subset(datasets.ImageFolder(str(data_dir), transform=val_transform),  val_idx)
    test_ds  = Subset(datasets.ImageFolder(str(data_dir), transform=val_transform),  test_idx)

    print(f"Dataset split — Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # Class counts for display
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        labels = [full_dataset.targets[i] for i in ds.indices]
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        print(f"  {name}: {dict(zip(CLASS_NAMES, counts))}")

    return train_ds, val_ds, test_ds, full_dataset.targets, train_idx


def make_weighted_sampler(dataset, all_targets, subset_indices):
    """Oversample minority classes to handle class imbalance."""
    subset_labels = [all_targets[i] for i in subset_indices]
    class_counts = np.bincount(subset_labels, minlength=NUM_CLASSES).astype(float)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in subset_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int) -> nn.Module:
    """Load pretrained backbone and replace the classifier head."""
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


# ─────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrix(preds, labels, class_names, save_path):
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print("\nClassification Report (Test Set):")
    print(report)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix (Test Set)')
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")
    return report


# ─────────────────────────────────────────────────────────────
# TRAINING CURVES PLOT
# ─────────────────────────────────────────────────────────────
def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], 'b-o', ms=4, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   'r-o', ms=4, label='Val Loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', ms=4, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   'r-o', ms=4, label='Val Acc')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Training History — {BACKBONE}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Lung Cancer Detection CNN")
    print("  IQ-OTH/NCCD Dataset  |  3-class classification")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    train_ds, val_ds, test_ds, all_targets, train_idx = make_splits(
        DATA_DIR, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED
    )

    sampler = make_weighted_sampler(train_ds, all_targets, train_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────
    model = build_model(BACKBONE, NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {BACKBONE}  |  Params: {total_params:,}  |  Trainable: {trainable:,}")

    # ── Loss & Optimiser ──────────────────────────────────────
    # Compute class weights from training subset
    train_labels = [all_targets[i] for i in train_idx]
    counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
    class_weights = torch.tensor(1.0 / (counts / counts.sum()), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    # ── Training loop ─────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_path = OUTPUT_DIR / f"best_{BACKBONE}.pth"

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
    print("-" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader,   criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
              f"{val_loss:>9.4f}  {val_acc*100:>7.2f}%  {lr_now:>8.6f}  {elapsed:>5.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': CLASS_NAMES,
                'backbone': BACKBONE,
            }, best_model_path)
            print(f"         *** Best model saved (val_acc={val_acc*100:.2f}%) ***")

    # ── Test evaluation ───────────────────────────────────────
    print("\nLoading best model for final test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc*100:.2f}%")

    # ── Save outputs ──────────────────────────────────────────
    plot_curves(history, OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(
        test_preds, test_labels, CLASS_NAMES,
        OUTPUT_DIR / "confusion_matrix.png"
    )

    # Save summary
    summary_path = OUTPUT_DIR / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {BACKBONE}\n")
        f.write(f"Best Val Accuracy: {best_val_acc*100:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_acc*100:.2f}%\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LR}\n")
    print(f"\nSummary saved: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
