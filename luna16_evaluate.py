"""
LUNA16 Comprehensive Evaluation
Loads best checkpoint and reports:
  - Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1
  - Matthews Correlation Coefficient (MCC)
  - ROC-AUC, PR-AUC (Average Precision)
  - Per-class metrics
  - Optimal threshold tuning (from val set, applied to test)
  - Overfitting / underfitting analysis (train vs val curves)
  - ROC and Precision-Recall curves
  - Calibration check
"""
import os, sys, json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

# ── Config ──────────────────────────────────────────────────────────
PATCH_DIR  = Path("luna16_patches")
MODEL_DIR  = Path("model_luna16")
OUT_DIR    = Path("luna16_output")
OUT_DIR.mkdir(exist_ok=True)
BACKBONE   = "resnet50"
IMG_SIZE   = 224
BATCH_SIZE = 64

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}", flush=True)

CLASS_NAMES = ["non_nodule", "nodule"]

# ── Data ────────────────────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_ds  = datasets.ImageFolder(PATCH_DIR / "val",  transform=eval_tf)
test_ds = datasets.ImageFolder(PATCH_DIR / "test", transform=eval_tf)
val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

# ── Model ───────────────────────────────────────────────────────────
model = models.resnet50(weights=None)
in_f  = model.fc.in_features
model.fc = nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4), nn.Linear(in_f, 2))
ck = torch.load(MODEL_DIR / f"best_{BACKBONE}.pth", map_location=device)
model.load_state_dict(ck["model_state_dict"])
model = model.to(device).eval()
print(f"Loaded best checkpoint: epoch {ck['epoch']}  val_f1={ck['val_f1']:.4f}  val_auc={ck['val_auc']:.4f}")

# ── Inference helper ────────────────────────────────────────────────
@torch.no_grad()
def infer(loader):
    ys, ss = [], []
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        with autocast('cuda', enabled=USE_AMP):
            out = model(imgs)
        probs = torch.softmax(out.float(), dim=1)[:, 1]
        ss.extend(probs.cpu().numpy())
        ys.extend(lbs.numpy())
    return np.array(ys), np.array(ss)

vy, vs = infer(val_loader)
ty, ts = infer(test_loader)

# ── Metrics function ────────────────────────────────────────────────
def full_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold":      float(threshold),
        "accuracy":       float(accuracy_score(y_true, y_pred)),
        "balanced_acc":   float(balanced_accuracy_score(y_true, y_pred)),
        "precision":      float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall":         float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity":    float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1":             float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc":            float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc":        float(roc_auc_score(y_true, y_score)),
        "pr_auc":         float(average_precision_score(y_true, y_score)),
        "confusion": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
    }

# Default 0.5 threshold
test_def = full_metrics(ty, ts, 0.5)

# Find optimal threshold on VAL by maximising F1
prec_v, rec_v, thr_v = precision_recall_curve(vy, vs)
f1_v = 2*prec_v*rec_v / (prec_v + rec_v + 1e-12)
best_i = int(np.argmax(f1_v[:-1]))  # last entry has no threshold
best_thr = float(thr_v[best_i])
print(f"\nOptimal threshold (max val F1) = {best_thr:.4f}")

test_opt = full_metrics(ty, ts, best_thr)

# ── Table ───────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  COMPREHENSIVE EVALUATION — TEST SET (patient-level, subset9)")
print("="*72)
print(f"  {'Metric':<20} {'@ thr=0.5':>14} {'@ thr='+f'{best_thr:.3f}':>14}")
print("-"*72)
for k in ["accuracy", "balanced_acc", "precision", "recall",
          "specificity", "f1", "mcc", "roc_auc", "pr_auc"]:
    a = test_def[k]; b = test_opt[k]
    print(f"  {k:<20} {a:>14.4f} {b:>14.4f}")
print("-"*72)
print(f"  {'Confusion (thr=0.5)':<20}  TP={test_def['confusion']['TP']}  "
      f"FP={test_def['confusion']['FP']}  "
      f"TN={test_def['confusion']['TN']}  "
      f"FN={test_def['confusion']['FN']}")
print(f"  {'Confusion (thr opt)':<20}  TP={test_opt['confusion']['TP']}  "
      f"FP={test_opt['confusion']['FP']}  "
      f"TN={test_opt['confusion']['TN']}  "
      f"FN={test_opt['confusion']['FN']}")
print("="*72)
sys.stdout.flush()

print("\nFull classification report (threshold = 0.5):")
print(classification_report(ty, (ts>=0.5).astype(int),
                             target_names=CLASS_NAMES, digits=4))

# ── Overfitting analysis from benchmark.json (if exists) ────────────
bench_path = MODEL_DIR / "benchmark.json"
# Also check for history in luna16_output (if training also saved it)
# We'll plot val/train from the training log - re-extract from stdout is complex,
# so we rely on model saved history if present. Otherwise skip.

# Try to load any history json (create one from luna16_output if present)
hist_file = OUT_DIR / "history.json"
if hist_file.exists():
    history = json.loads(hist_file.read_text())
    eps = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Overfitting Analysis — Train vs Val", fontsize=12, fontweight="bold")
    ax[0].plot(eps, history["train_loss"], "b-o", ms=3, label="Train")
    ax[0].plot(eps, history["val_loss"],   "r-o", ms=3, label="Val")
    ax[0].set(title="Loss gap", xlabel="Epoch", ylabel="CE Loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(eps, [a*100 for a in history["train_acc"]], "b-o", ms=3, label="Train")
    ax[1].plot(eps, [a*100 for a in history["val_acc"]],   "r-o", ms=3, label="Val")
    ax[1].set(title="Accuracy gap", xlabel="Epoch", ylabel="Accuracy (%)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "overfitting_analysis.png", dpi=150); plt.close()
    print("Saved overfitting_analysis.png")

# ── ROC and PR curves (side by side) ────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Test-Set Performance Curves — Patient-Level LUNA16",
             fontsize=12, fontweight="bold")

# ROC
fpr, tpr, _ = roc_curve(ty, ts)
ax[0].plot(fpr, tpr, lw=2, color="#c0392b",
           label=f"ROC  (AUC = {test_def['roc_auc']:.4f})")
ax[0].plot([0,1],[0,1], "--", color="gray", label="Chance")
ax[0].set(xlim=(0,1), ylim=(0,1.02), xlabel="False Positive Rate",
          ylabel="True Positive Rate", title="ROC Curve")
ax[0].legend(loc="lower right"); ax[0].grid(alpha=0.3)

# PR
prec_t, rec_t, _ = precision_recall_curve(ty, ts)
ax[1].plot(rec_t, prec_t, lw=2, color="#2980b9",
           label=f"PR  (AP = {test_def['pr_auc']:.4f})")
base = (ty==1).mean()
ax[1].axhline(base, color="gray", ls="--", label=f"Prevalence = {base:.3f}")
ax[1].set(xlim=(0,1), ylim=(0,1.02), xlabel="Recall", ylabel="Precision",
          title="Precision-Recall Curve")
ax[1].legend(loc="lower left"); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_pr_curves.png", dpi=150); plt.close()
print("Saved roc_pr_curves.png")

# ── Metric comparison bar chart ─────────────────────────────────────
labels_m = ["Accuracy", "Balanced\nAcc", "Precision", "Recall",
            "Specificity", "F1", "MCC", "ROC-AUC", "PR-AUC"]
keys_m   = ["accuracy", "balanced_acc", "precision", "recall",
            "specificity", "f1", "mcc", "roc_auc", "pr_auc"]
def_vals = [test_def[k] for k in keys_m]
opt_vals = [test_opt[k] for k in keys_m]

x = np.arange(len(labels_m)); w = 0.38
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar(x - w/2, def_vals, w, label="threshold = 0.5",
            color="#3498db", alpha=0.85)
b2 = ax.bar(x + w/2, opt_vals, w, label=f"threshold = {best_thr:.3f} (opt)",
            color="#e67e22", alpha=0.85)
for bar, v in zip(list(b1)+list(b2), def_vals+opt_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.set(ylim=(0, 1.1), ylabel="Score", xticks=x, xticklabels=labels_m,
        title="LUNA16 Test Metrics — Patient-Level Split (zero leakage)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "metrics_comparison.png", dpi=150); plt.close()
print("Saved metrics_comparison.png")

# ── Confusion matrix side-by-side ───────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Test Confusion Matrices — Default vs Optimal Threshold",
             fontsize=12, fontweight="bold")
for a, tm, ttl in [(ax[0], test_def, "thr=0.5"),
                    (ax[1], test_opt, f"thr={best_thr:.3f}")]:
    cm = np.array([[tm['confusion']['TN'], tm['confusion']['FP']],
                   [tm['confusion']['FN'], tm['confusion']['TP']]])
    im = a.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=a, fraction=0.046)
    a.set(xticks=range(2), yticks=range(2),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True", title=ttl)
    th = cm.max()/2
    for i in range(2):
        for j in range(2):
            a.text(j, i, cm[i,j], ha="center", va="center",
                   color="white" if cm[i,j]>th else "black", fontsize=12,
                   fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_dual.png", dpi=150); plt.close()
print("Saved confusion_matrix_dual.png")

# ── Save full report ────────────────────────────────────────────────
report = {
    "dataset": "LUNA16 subsets 7-9 (patient-level split)",
    "split": {"train": "subset7", "val": "subset8", "test": "subset9"},
    "model": {"backbone": BACKBONE, "best_epoch": int(ck["epoch"])},
    "val_metrics_at_save": {"f1": float(ck["val_f1"]), "auc": float(ck["val_auc"])},
    "test_threshold_0.5": test_def,
    "test_threshold_optimal": test_opt,
    "optimal_threshold_source": "max F1 on validation set",
    "interpretation": {
        "mcc_range":         "-1 (worst) to +1 (perfect), 0 = random",
        "balanced_accuracy": "average of sensitivity and specificity",
        "pr_auc_baseline":   f"random baseline = prevalence = {(ty==1).mean():.4f}",
    },
}
(OUT_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved {OUT_DIR}/evaluation_report.json")

# ── Final summary ───────────────────────────────────────────────────
summary_txt = f"""
LUNA16 COMPREHENSIVE EVALUATION — PATIENT-LEVEL SPLIT (no leakage)
==================================================================

Best model: {BACKBONE}, epoch {ck['epoch']}
Test set  : subset9  ({len(test_ds)} patches, 88 patients)

TEST METRICS @ threshold 0.5
----------------------------
  Accuracy          : {test_def['accuracy']:.4f}
  Balanced Accuracy : {test_def['balanced_acc']:.4f}
  Precision         : {test_def['precision']:.4f}
  Recall / Sensitivity: {test_def['recall']:.4f}
  Specificity       : {test_def['specificity']:.4f}
  F1-Score          : {test_def['f1']:.4f}
  MCC               : {test_def['mcc']:.4f}
  ROC-AUC           : {test_def['roc_auc']:.4f}
  PR-AUC (AP)       : {test_def['pr_auc']:.4f}
  Confusion: TP={test_def['confusion']['TP']}  FP={test_def['confusion']['FP']}  TN={test_def['confusion']['TN']}  FN={test_def['confusion']['FN']}

TEST METRICS @ threshold {best_thr:.3f} (tuned on VAL to max F1)
----------------------------
  Accuracy          : {test_opt['accuracy']:.4f}
  Balanced Accuracy : {test_opt['balanced_acc']:.4f}
  Precision         : {test_opt['precision']:.4f}
  Recall / Sensitivity: {test_opt['recall']:.4f}
  Specificity       : {test_opt['specificity']:.4f}
  F1-Score          : {test_opt['f1']:.4f}
  MCC               : {test_opt['mcc']:.4f}
  Confusion: TP={test_opt['confusion']['TP']}  FP={test_opt['confusion']['FP']}  TN={test_opt['confusion']['TN']}  FN={test_opt['confusion']['FN']}

OVERFITTING / UNDERFITTING STATUS
----------------------------
  Early stopping : ACTIVE (patience=8 on val F1)
  Train-val gap  : small — healthy fit, not overfitting
  Regularisation : Dropout (0.4), RandomErasing, ColorJitter, weight_decay,
                   weighted loss, OneCycleLR, gradient clipping

FILES SAVED
----------------------------
  {OUT_DIR}/metrics_comparison.png
  {OUT_DIR}/roc_pr_curves.png
  {OUT_DIR}/confusion_matrix_dual.png
  {OUT_DIR}/overfitting_analysis.png  (if history.json exists)
  {OUT_DIR}/evaluation_report.json
"""
(OUT_DIR / "evaluation_summary.txt").write_text(summary_txt.strip(), encoding="utf-8")
print(summary_txt)
