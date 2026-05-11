"""LUNA16 Ablation Study — isolate each v2 improvement on top of v1.

Variants trained (each = v1 baseline + ONE change):
  v1 + CBAM     : add Channel+Spatial attention after layer4
  v1 + Focal    : replace weighted CE with FocalLoss(alpha=0.25, gamma=2)
  v1 + MixUp    : enable MixUp augmentation (beta=0.4, 50% of batches)
  v1 + TTA      : re-evaluate existing v1 checkpoint with 5-view TTA

Reused from disk (no retrain):
  v1            : baseline checkpoint (luna16_runs_full/v1_full/best.pth)
  v2            : full bundle (luna16_runs_full/v2_full/best.pth)

All training config matches v1: 30 epochs, OneCycleLR max_lr=5e-4, AdamW wd=1e-4,
AMP, gradient clip 1.0, early stop on val F1 (patience=10), Option B split.
"""
import os, sys, time, json, random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix,
)

random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

PATCH_DIR = Path("luna16_patches_full")
RUNS_DIR  = Path("luna16_runs_ablation"); RUNS_DIR.mkdir(exist_ok=True)
PRIOR_DIR = Path("luna16_runs_full")  # for v1, v2 prior checkpoints

CLASS_NAMES = ["non_nodule", "nodule"]
NUM_CLASSES = 2
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_EPOCHS  = 30
LR_MAX      = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP   = 1.0
EARLY_STOP_PAT = 10
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
MIXUP_ALPHA = 0.4
MIXUP_PROB  = 0.5

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"PyTorch: {torch.__version__}  Device: {device}", flush=True)

# ── Modules (identical to v2 build) ──
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
        )
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x))) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return self.sig(self.conv(torch.cat([avg, mx], 1))) * x

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

class ResNet50_Plain(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        b = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = b.fc.in_features
        b.fc = nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        self.net = b
    def forward(self, x): return self.net(x)

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        b = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem   = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
        self.layer1, self.layer2 = b.layer1, b.layer2
        self.layer3, self.layer4 = b.layer3, b.layer4
        self.cbam   = CBAM(2048, reduction=16, kernel_size=7)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.BatchNorm1d(2048), nn.Dropout(dropout), nn.Linear(2048, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(self.cbam(x)))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()

def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

# ── Data loaders ──
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE+16, IMG_SIZE+16)),
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

train_ds = datasets.ImageFolder(PATCH_DIR/"train", transform=train_tf)
val_ds   = datasets.ImageFolder(PATCH_DIR/"val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(PATCH_DIR/"test",  transform=eval_tf)
train_labels = [y for _, y in train_ds.samples]
counts_tr    = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
w_cls        = 1.0 / (counts_tr + 1e-8)
sampler      = WeightedRandomSampler([w_cls[l] for l in train_labels],
                                      num_samples=len(train_labels), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=USE_AMP)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=0, pin_memory=USE_AMP)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,  num_workers=0, pin_memory=USE_AMP)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}", flush=True)

# Loss for v1-style (weighted CE)
cls_weights = torch.tensor(1.0 / (counts_tr / counts_tr.sum()), dtype=torch.float32).to(device)
ce_loss     = nn.CrossEntropyLoss(weight=cls_weights)

# ── Helpers ──
def compute(y_true, y_pred, y_score=None):
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            recall_score(y_true,    y_pred, pos_label=1, zero_division=0),
            f1_score(y_true,        y_pred, pos_label=1, zero_division=0),
            roc_auc_score(y_true, y_score) if y_score is not None else None)

def full_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision":    float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall":       float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity":  float(tn/(tn+fp)) if (tn+fp) else 0.0,
        "f1":           float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc":          float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc":      float(roc_auc_score(y_true, y_score)),
        "pr_auc":       float(average_precision_score(y_true, y_score)),
        "confusion":    {"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)},
    }

@torch.no_grad()
def single_infer(model, loader):
    model.eval(); ys, ss = [], []
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        with autocast('cuda', enabled=USE_AMP):
            out = model(imgs)
        probs = torch.softmax(out.float(), dim=1)[:, 1]
        ss.extend(probs.cpu().numpy()); ys.extend(lbs.numpy())
    return np.array(ys), np.array(ss)

@torch.no_grad()
def tta_infer(model, loader):
    model.eval(); ys, ss = [], []
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        views = [imgs, torch.flip(imgs, [3]), torch.flip(imgs, [2]),
                 torch.rot90(imgs, 1, [2,3]), torch.rot90(imgs, 3, [2,3])]
        all_p = []
        for v in views:
            with autocast('cuda', enabled=USE_AMP):
                out = model(v)
            all_p.append(torch.softmax(out.float(), dim=1)[:, 1])
        probs = torch.stack(all_p).mean(0)
        ss.extend(probs.cpu().numpy()); ys.extend(lbs.numpy())
    return np.array(ys), np.array(ss)

def train_run(name, model, criterion, use_mixup):
    """Train ONE ablation variant with v1's recipe except for the noted change."""
    save_dir = RUNS_DIR / name; save_dir.mkdir(exist_ok=True, parents=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX/25, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS, pct_start=0.3, anneal_strategy="cos",
        div_factor=25, final_div_factor=1e4)
    scaler = GradScaler('cuda', enabled=USE_AMP)
    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc",
                                "train_f1","val_f1","val_prec","val_rec","val_auc"]}
    best_f1, early_ctr = 0.0, 0
    ckpt_path = save_dir / "best.pth"
    print(f"\n=== Training {name} ({NUM_EPOCHS} epochs, mixup={use_mixup}) ===", flush=True)
    print(f"{'Ep':>3} {'TrLoss':>7} {'TrAcc':>6} {'TrF1':>6} {'VaLoss':>7} "
          f"{'VaAcc':>6} {'VaF1':>6} {'VaP':>6} {'VaR':>6} {'AUC':>6} {'LR':>8} {'s':>4}", flush=True)
    print("-"*96, flush=True)
    for ep in range(1, NUM_EPOCHS+1):
        t0 = time.time()
        model.train(); rl, yp, yt = 0.0, [], []
        for imgs, lbs in train_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            use_m = use_mixup and (np.random.rand() < MIXUP_PROB)
            if use_m:
                imgs, ya, yb, lam = mixup(imgs, lbs)
            optimizer.zero_grad()
            with autocast('cuda', enabled=USE_AMP):
                out = model(imgs)
                if use_m:
                    loss = lam*criterion(out, ya) + (1-lam)*criterion(out, yb)
                else:
                    loss = criterion(out, lbs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            rl += loss.item()*imgs.size(0)
            yp.extend(out.argmax(1).cpu().numpy())
            yt.extend((ya if use_m else lbs).cpu().numpy())
        tr_loss = rl/len(train_loader.dataset)
        tr_acc, tr_p, tr_r, tr_f1, _ = compute(yt, yp)
        # val
        model.eval(); rl, yp, yt, ys = 0.0, [], [], []
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(device), lbs.to(device)
                with autocast('cuda', enabled=USE_AMP):
                    out = model(imgs); loss = criterion(out, lbs)
                rl += loss.item()*imgs.size(0)
                probs = torch.softmax(out.float(), dim=1)[:, 1]
                yp.extend(out.argmax(1).cpu().numpy())
                ys.extend(probs.cpu().numpy())
                yt.extend(lbs.cpu().numpy())
        va_loss = rl/len(val_loader.dataset)
        va_acc, va_p, va_r, va_f1, va_auc = compute(yt, yp, ys)
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
            torch.save({"epoch": ep, "run": name,
                         "model_state_dict": model.state_dict(),
                         "val_f1": va_f1, "val_auc": va_auc}, ckpt_path)
            print(f"    [*] best saved  val_f1={va_f1:.4f}  val_auc={va_auc:.4f}", flush=True)
        else:
            early_ctr += 1
            if early_ctr >= EARLY_STOP_PAT:
                print(f"\nEarly stop at epoch {ep}", flush=True); break
    (save_dir/"history.json").write_text(json.dumps(history, indent=2))
    return history, ckpt_path

# ── Run the three new ablations ──
results = {}

# 1) v1 + CBAM
m = ResNet50_CBAM(NUM_CLASSES, dropout=0.4).to(device)
hist, ckpt = train_run("v1_cbam",  m, ce_loss,            use_mixup=False)
m.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
ty, ts = single_infer(m, test_loader)
results["v1_cbam"]  = full_metrics(ty, ts, 0.5)
print(f"\nv1+CBAM test: F1={results['v1_cbam']['f1']:.4f}  AUC={results['v1_cbam']['roc_auc']:.4f}", flush=True)

# 2) v1 + Focal
m = ResNet50_Plain(NUM_CLASSES, dropout=0.4).to(device)
hist, ckpt = train_run("v1_focal", m, FocalLoss(FOCAL_ALPHA, FOCAL_GAMMA), use_mixup=False)
m.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
ty, ts = single_infer(m, test_loader)
results["v1_focal"] = full_metrics(ty, ts, 0.5)
print(f"\nv1+Focal test: F1={results['v1_focal']['f1']:.4f}  AUC={results['v1_focal']['roc_auc']:.4f}", flush=True)

# 3) v1 + MixUp
m = ResNet50_Plain(NUM_CLASSES, dropout=0.4).to(device)
hist, ckpt = train_run("v1_mixup", m, ce_loss,            use_mixup=True)
m.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
ty, ts = single_infer(m, test_loader)
results["v1_mixup"] = full_metrics(ty, ts, 0.5)
print(f"\nv1+MixUp test: F1={results['v1_mixup']['f1']:.4f}  AUC={results['v1_mixup']['roc_auc']:.4f}", flush=True)

# 4) v1 + TTA — re-evaluate prior v1 checkpoint with TTA
print(f"\n=== v1 + TTA — re-evaluate {PRIOR_DIR/'v1_full/best.pth'} ===", flush=True)
m = ResNet50_Plain(NUM_CLASSES, dropout=0.4).to(device)
m.load_state_dict(torch.load(PRIOR_DIR/"v1_full"/"best.pth", map_location=device)["model_state_dict"])
ty, ts = tta_infer(m, test_loader)
results["v1_tta"] = full_metrics(ty, ts, 0.5)
print(f"v1+TTA test: F1={results['v1_tta']['f1']:.4f}  AUC={results['v1_tta']['roc_auc']:.4f}", flush=True)

# Reload v1 and v2 baseline metrics from disk for the report
prior = json.loads((PRIOR_DIR/"comparison_report.json").read_text())
results["v1"] = prior["v1_full"]["test"]
results["v2"] = prior["v2_full"]["test_tta"]

# ── Comparison report ──
report = {
    "split": prior["split_scheme"],
    "train_size": prior["train_size"], "val_size": prior["val_size"], "test_size": prior["test_size"],
    "ablations": results,
    "training": {
        "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "lr_max": LR_MAX,
        "early_stop_patience": EARLY_STOP_PAT, "seed": 42,
    },
}
(RUNS_DIR/"ablation_report.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved {RUNS_DIR/'ablation_report.json'}", flush=True)

# Final ablation table
print("\n" + "="*108, flush=True)
print("  ABLATION STUDY — each row = v1 baseline + ONE change", flush=True)
print("="*108, flush=True)
order = [("v1","v1 (baseline)"), ("v1_cbam","v1 + CBAM"), ("v1_focal","v1 + Focal"),
         ("v1_mixup","v1 + MixUp"), ("v1_tta","v1 + TTA"), ("v2","v2 (all four)")]
hdr = f"  {'Variant':<22}"
ms = ["accuracy","balanced_acc","precision","recall","specificity","f1","mcc","roc_auc","pr_auc"]
hdr += "".join(f"{m[:7]:>9}" for m in ms)
print(hdr, flush=True); print("-"*108, flush=True)
for k, label in order:
    if k not in results: continue
    r = results[k]
    line = f"  {label:<22}"
    line += "".join(f"{r[m]:>9.4f}" for m in ms)
    print(line, flush=True)
print("="*108, flush=True)
