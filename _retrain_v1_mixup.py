"""Retrain v1 + MixUp from scratch (full 30 epochs) — earlier run truncated at ep 20.

Same hyperparameters as v1 baseline, only difference is MixUp enabled."""
import json, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix)

random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

PATCH_DIR = Path("luna16_patches_full")
ABL_DIR   = Path("luna16_runs_ablation")
PRIOR_DIR = Path("luna16_runs_full")
SAVE_DIR  = ABL_DIR / "v1_mixup"; SAVE_DIR.mkdir(exist_ok=True, parents=True)

NUM_CLASSES = 2; IMG_SIZE = 224; BATCH_SIZE = 64
NUM_EPOCHS = 30; LR_MAX = 5e-4; WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0; EARLY_STOP_PAT = 10
MIXUP_ALPHA = 0.4; MIXUP_PROB = 0.5

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}", flush=True)

class ResNet50_Plain(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = b.fc.in_features
        b.fc = nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4), nn.Linear(in_f, NUM_CLASSES))
        self.net = b
    def forward(self, x): return self.net(x)

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
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_AMP)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_AMP)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}", flush=True)

cls_weights = torch.tensor(1.0 / (counts_tr / counts_tr.sum()), dtype=torch.float32).to(device)
criterion   = nn.CrossEntropyLoss(weight=cls_weights)

model = ResNet50_Plain().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR_MAX/25, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS, pct_start=0.3, anneal_strategy="cos",
    div_factor=25, final_div_factor=1e4)
scaler = GradScaler('cuda', enabled=USE_AMP)

def compute(yt, yp, ys=None):
    return (accuracy_score(yt, yp),
            precision_score(yt, yp, pos_label=1, zero_division=0),
            recall_score(yt, yp, pos_label=1, zero_division=0),
            f1_score(yt, yp, pos_label=1, zero_division=0),
            roc_auc_score(yt, ys) if ys is not None else None)

history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc",
                            "train_f1","val_f1","val_prec","val_rec","val_auc"]}
best_f1, early_ctr = 0.0, 0
ckpt_path = SAVE_DIR / "best.pth"

print(f"\n=== Re-training v1+MixUp ({NUM_EPOCHS} epochs, fresh) ===", flush=True)
print(f"{'Ep':>3} {'TrLoss':>7} {'TrAcc':>6} {'TrF1':>6} {'VaLoss':>7} "
      f"{'VaAcc':>6} {'VaF1':>6} {'VaP':>6} {'VaR':>6} {'AUC':>6} {'LR':>8} {'s':>4}", flush=True)
print("-"*96, flush=True)

for ep in range(1, NUM_EPOCHS+1):
    t0 = time.time()
    model.train(); rl, yp, yt = 0.0, [], []
    for imgs, lbs in train_loader:
        imgs, lbs = imgs.to(device), lbs.to(device)
        use_m = np.random.rand() < MIXUP_PROB
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
        torch.save({"epoch": ep, "run": "v1_mixup",
                     "model_state_dict": model.state_dict(),
                     "val_f1": va_f1, "val_auc": va_auc}, ckpt_path)
        print(f"    [*] best saved  val_f1={va_f1:.4f}  val_auc={va_auc:.4f}", flush=True)
    else:
        early_ctr += 1
        if early_ctr >= EARLY_STOP_PAT:
            print(f"\nEarly stop at epoch {ep}", flush=True); break

(SAVE_DIR/"history.json").write_text(json.dumps(history, indent=2))
print(f"\nTraining done. Best val_f1 = {best_f1:.4f}", flush=True)

# ── Test eval ──
@torch.no_grad()
def single_infer(model):
    model.eval(); ys, ss = [], []
    for imgs, lbs in test_loader:
        imgs = imgs.to(device)
        with autocast('cuda', enabled=USE_AMP):
            out = model(imgs)
        probs = torch.softmax(out.float(), dim=1)[:, 1]
        ss.extend(probs.cpu().numpy()); ys.extend(lbs.numpy())
    return np.array(ys), np.array(ss)

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

ck = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
ty, ts = single_infer(model)
m_new = full_metrics(ty, ts, 0.5)
m_new["best_epoch"] = int(ck["epoch"])

print(f"\nv1+MixUp (full 30 ep) test:  F1={m_new['f1']:.4f}  AUC={m_new['roc_auc']:.4f}  MCC={m_new['mcc']:.4f}", flush=True)

# ── Update ablation_report.json with the new v1_mixup numbers ──
report_path = ABL_DIR / "ablation_report.json"
report = json.loads(report_path.read_text())
old = report["ablations"]["v1_mixup"]
report["ablations"]["v1_mixup"] = m_new
report["note"] = "v1_mixup retrained from scratch — full 30 epochs."
report_path.write_text(json.dumps(report, indent=2))
print(f"Updated {report_path}", flush=True)

# ── Print master table again ──
prior = json.loads((PRIOR_DIR/"comparison_report.json").read_text())
all_results = report["ablations"]
print("\n" + "="*112, flush=True)
print("  ABLATION STUDY (UPDATED) — each row = v1 baseline + ONE change", flush=True)
print("="*112, flush=True)
order = [("v1","v1 (baseline)"), ("v1_cbam","v1 + CBAM"),
         ("v1_focal","v1 + Focal"), ("v1_mixup","v1 + MixUp"),
         ("v1_tta","v1 + TTA"), ("v2","v2 (all four)")]
mlist = ["accuracy","balanced_acc","precision","recall","specificity","f1","mcc","roc_auc","pr_auc"]
hdr = f"  {'Variant':<22}" + "".join(f"{k[:8]:>10}" for k in mlist)
print(hdr, flush=True); print("-"*112, flush=True)
v1ref = all_results["v1"]
for k, label in order:
    if k not in all_results: continue
    r = all_results[k]
    line = f"  {label:<22}" + "".join(f"{r[m]:>10.4f}" for m in mlist)
    print(line, flush=True)
print("="*112, flush=True)

print("\n  Δ vs v1 baseline (positive = improvement):", flush=True)
print("-"*112, flush=True)
for k, label in order:
    if k == "v1" or k not in all_results: continue
    line = f"  {label:<22}" + "".join(f"{all_results[k][m]-v1ref[m]:>+10.4f}" for m in mlist)
    print(line, flush=True)
print("="*112, flush=True)
