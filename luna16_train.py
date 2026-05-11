"""
LUNA16 Nodule Classification — Patient-Level Split
ResNet-50, binary classification (nodule / non-nodule)

Splits are defined by subset membership (NO patient-level leakage):
  subset7 → train
  subset8 → val
  subset9 → test

Benchmark: Accuracy, Precision, Recall, F1, AUC
"""
import os, sys, time, json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ── Config ──────────────────────────────────────────────────────────
PATCH_DIR   = Path("luna16_patches")
MODEL_DIR   = Path("model_luna16")
OUT_DIR     = Path("luna16_output")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["non_nodule", "nodule"]
NUM_CLASSES = 2
BACKBONE    = "resnet50"
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_EPOCHS  = 30
LR_MAX      = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP   = 1.0
EARLY_STOP_PAT = 8
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"PyTorch: {torch.__version__} | Device: {device}", flush=True)
if device.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB", flush=True)

# ── Transforms ──────────────────────────────────────────────────────
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Datasets ────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(PATCH_DIR / "train", transform=train_tf)
val_ds   = datasets.ImageFolder(PATCH_DIR / "val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(PATCH_DIR / "test",  transform=eval_tf)

print(f"\nDatasets (patient-level splits from subsets):")
for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
    counts = np.bincount([y for _,y in ds.samples], minlength=NUM_CLASSES)
    print(f"  {name}: {len(ds):>5} patches  {dict(zip(ds.classes, counts))}")

# Weighted sampler for training class imbalance (≤3:1 neg:pos)
train_labels = [y for _,y in train_ds.samples]
counts_tr    = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
w_cls        = 1.0 / (counts_tr + 1e-8)
sampler      = WeightedRandomSampler([w_cls[l] for l in train_labels],
                                      num_samples=len(train_labels), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                           num_workers=0, pin_memory=USE_AMP)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=USE_AMP)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, pin_memory=USE_AMP)

# ── Model ───────────────────────────────────────────────────────────
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_f  = model.fc.in_features
model.fc = nn.Sequential(
    nn.BatchNorm1d(in_f),
    nn.Dropout(0.4),
    nn.Linear(in_f, NUM_CLASSES),
)
model = model.to(device)
print(f"\nModel: {BACKBONE}  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Loss / Optim / Sched ────────────────────────────────────────────
cls_weights = torch.tensor(1.0 / (counts_tr / counts_tr.sum()),
                            dtype=torch.float32).to(device)
criterion   = nn.CrossEntropyLoss(weight=cls_weights)
optimizer   = optim.AdamW(model.parameters(), lr=LR_MAX/25, weight_decay=WEIGHT_DECAY)
scheduler   = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR_MAX,
    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
    pct_start=0.3, anneal_strategy="cos", div_factor=25, final_div_factor=1e4,
)
scaler = GradScaler('cuda', enabled=USE_AMP)
print(f"Cls weights: {dict(zip(CLASS_NAMES, cls_weights.cpu().numpy().round(3)))}", flush=True)

# ── Helpers ─────────────────────────────────────────────────────────
def compute(y_true, y_pred, y_score=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true,    y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true,        y_pred, pos_label=1, zero_division=0)
    auc  = roc_auc_score(y_true, y_score) if y_score is not None else None
    return acc, prec, rec, f1, auc

def train_epoch():
    model.train()
    rl, yp, yt = 0.0, [], []
    for imgs, lbs in train_loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        optimizer.zero_grad()
        with autocast('cuda', enabled=USE_AMP):
            out = model(imgs); loss = criterion(out, lbs)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        rl += loss.item()*imgs.size(0)
        yp.extend(out.argmax(1).cpu().numpy()); yt.extend(lbs.cpu().numpy())
    return rl/len(train_loader.dataset), *compute(yt, yp)[:4]

@torch.no_grad()
def evaluate(loader):
    model.eval()
    rl, yp, yt, ys = 0.0, [], [], []
    for imgs, lbs in loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        with autocast('cuda', enabled=USE_AMP):
            out = model(imgs); loss = criterion(out, lbs)
        rl += loss.item()*imgs.size(0)
        probs = torch.softmax(out.float(), dim=1)[:,1]
        yp.extend(out.argmax(1).cpu().numpy())
        ys.extend(probs.cpu().numpy())
        yt.extend(lbs.cpu().numpy())
    return rl/len(loader.dataset), np.array(yt), np.array(yp), np.array(ys)

# ── Training loop ───────────────────────────────────────────────────
history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc",
                            "train_f1","val_f1","val_prec","val_rec","val_auc"]}
best_f1, early_ctr = 0.0, 0
best_ckpt = MODEL_DIR / f"best_{BACKBONE}.pth"

print(f"\n{'Ep':>3} {'TrLoss':>7} {'TrAcc':>6} {'TrF1':>6} {'VaLoss':>7} "
      f"{'VaAcc':>6} {'VaF1':>6} {'VaP':>6} {'VaR':>6} {'AUC':>6} {'LR':>8} {'s':>4}")
print("-"*96)

for ep in range(1, NUM_EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc, tr_p, tr_r, tr_f1 = train_epoch()
    va_loss, vy, vp, vs = evaluate(val_loader)
    va_acc, va_p, va_r, va_f1, va_auc = compute(vy, vp, vs)
    dt = time.time()-t0
    lr = optimizer.param_groups[0]["lr"]

    for k, v in [("train_loss",tr_loss),("val_loss",va_loss),
                  ("train_acc",tr_acc),("val_acc",va_acc),
                  ("train_f1",tr_f1),("val_f1",va_f1),
                  ("val_prec",va_p),("val_rec",va_r),("val_auc",va_auc)]:
        history[k].append(v)

    print(f"{ep:>3} {tr_loss:>7.4f} {tr_acc*100:>5.1f}% {tr_f1:.4f} "
          f"{va_loss:>7.4f} {va_acc*100:>5.1f}% {va_f1:.4f} "
          f"{va_p:.4f} {va_r:.4f} {va_auc:.4f} {lr:.2e} {dt:>3.0f}s", flush=True)

    if va_f1 > best_f1:
        best_f1 = va_f1; early_ctr = 0
        torch.save({"epoch": ep, "backbone": BACKBONE,
                     "model_state_dict": model.state_dict(),
                     "val_f1": va_f1, "val_auc": va_auc}, best_ckpt)
        print(f"    [*] best saved  val_f1={va_f1:.4f}  val_auc={va_auc:.4f}", flush=True)
    else:
        early_ctr += 1
        if early_ctr >= EARLY_STOP_PAT:
            print(f"\nEarly stop at epoch {ep}", flush=True); break

print("\nTraining complete.", flush=True)

# ── Curves ──────────────────────────────────────────────────────────
eps = range(1, len(history["train_loss"])+1)
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"LUNA16 Training — {BACKBONE} (Patient-Level Split)",
             fontsize=13, fontweight="bold")
for a, tk, vk, pct, title, yl in [
    (ax[0,0],"train_loss","val_loss",False,"Loss","Cross-Entropy"),
    (ax[0,1],"train_acc","val_acc",True,"Accuracy (%)","Accuracy (%)"),
    (ax[1,0],"train_f1","val_f1",False,"F1-Score","F1"),
]:
    t = [v*100 for v in history[tk]] if pct else history[tk]
    v = [v*100 for v in history[vk]] if pct else history[vk]
    a.plot(eps, t, "b-o", ms=3, label="Train")
    a.plot(eps, v, "r-o", ms=3, label="Val")
    a.set(title=title, xlabel="Epoch", ylabel=yl); a.legend(); a.grid(alpha=0.3)
ax[1,1].plot(eps, history["val_prec"], "g-o", ms=3, label="Precision")
ax[1,1].plot(eps, history["val_rec"],  "m-o", ms=3, label="Recall")
ax[1,1].plot(eps, history["val_auc"],  "c-o", ms=3, label="AUC")
ax[1,1].set(title="Val Precision / Recall / AUC", xlabel="Epoch", ylabel="Score")
ax[1,1].legend(); ax[1,1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curves.png", dpi=150); plt.close()

# ── Load best & test ────────────────────────────────────────────────
ck = torch.load(best_ckpt, map_location=device)
model.load_state_dict(ck["model_state_dict"])
print(f"\nBest ckpt: epoch {ck['epoch']}  val_f1={ck['val_f1']:.4f}  val_auc={ck['val_auc']:.4f}")

_, ty, tp, ts = evaluate(test_loader)
te_acc, te_p, te_r, te_f1, te_auc = compute(ty, tp, ts)

# ── Benchmark table ────────────────────────────────────────────────
print("\n" + "="*60)
print("  LUNA16 BENCHMARK (HONEST — NO PATIENT LEAKAGE)")
print("="*60)
print(f"  Test patients: {len(set(Path(s[0]).stem.split('_')[0] for s in test_ds.samples))}")
print(f"  Test patches : {len(test_ds)}")
print("-"*60)
print(f"  Accuracy  : {te_acc*100:.2f}%")
print(f"  Precision : {te_p:.4f}")
print(f"  Recall    : {te_r:.4f}")
print(f"  F1-Score  : {te_f1:.4f}")
print(f"  ROC-AUC   : {te_auc:.4f}")
print("="*60)
print("\nFull classification report:")
print(classification_report(ty, tp, target_names=CLASS_NAMES, digits=4))

# ── Confusion matrix ────────────────────────────────────────────────
cm      = confusion_matrix(ty, tp)
cm_norm = cm.astype(float)/cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"LUNA16 Confusion Matrix — {BACKBONE} (Patient-Level Test)",
             fontsize=12, fontweight="bold")
for a, d, t, fmt in [(ax[0],cm,"Counts","%d"),(ax[1],cm_norm,"Row-Norm",".2f")]:
    im = a.imshow(d, cmap="Blues"); plt.colorbar(im, ax=a, fraction=0.046)
    a.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
          xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
          xlabel="Predicted", ylabel="True", title=t)
    th = d.max()/2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            a.text(j, i, fmt % d[i,j], ha="center", va="center",
                    color="white" if d[i,j]>th else "black", fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150); plt.close()

# ── ROC curve ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(ty, ts)
fig, a = plt.subplots(figsize=(7, 6))
a.plot(fpr, tpr, lw=2, label=f"AUC = {te_auc:.4f}", color="#c0392b")
a.plot([0,1], [0,1], "--", color="gray", label="Chance")
a.set(xlim=(0,1), ylim=(0,1.02), xlabel="False Positive Rate",
      ylabel="True Positive Rate",
      title=f"LUNA16 ROC Curve — {BACKBONE} (patient-level)")
a.legend(loc="lower right"); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR / "roc_curve.png", dpi=150); plt.close()

# ── Save benchmark ──────────────────────────────────────────────────
benchmark = {
    "dataset": "LUNA16 subsets 7-9 (patient-level split)",
    "split_scheme": "subset7=train, subset8=val, subset9=test (no patient overlap)",
    "backbone": BACKBONE,
    "best_epoch": int(ck["epoch"]),
    "test": {
        "accuracy":  round(te_acc, 4), "precision": round(te_p, 4),
        "recall":    round(te_r, 4),   "f1":        round(te_f1, 4),
        "auc":       round(te_auc, 4),
    },
    "test_samples": len(test_ds),
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
}
(MODEL_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2))

print("\n" + "="*60)
print("  FINAL RESULTS (patient-level honest evaluation)")
print("="*60)
print(f"  Accuracy  : {te_acc*100:.2f}%")
print(f"  Precision : {te_p:.4f}")
print(f"  Recall    : {te_r:.4f}")
print(f"  F1-Score  : {te_f1:.4f}")
print(f"  ROC-AUC   : {te_auc:.4f}")
print("="*60)
