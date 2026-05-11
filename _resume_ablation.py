"""Recover ablation results after the cuDNN crash.

Loads each saved checkpoint, runs test inference, computes the full 9-metric suite,
and writes the final ablation_report.json + master table."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix)

PATCH_DIR  = Path("luna16_patches_full")
ABL_DIR    = Path("luna16_runs_ablation")
PRIOR_DIR  = Path("luna16_runs_full")
NUM_CLASSES = 2; IMG_SIZE = 224; BATCH_SIZE = 64
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}", flush=True)

# ── modules ──
class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(c, c//r, 1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(c//r, c, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x))) * x

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k//2, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return self.sig(self.conv(torch.cat([avg, mx], 1))) * x

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c); self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ResNet50_Plain(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.resnet50(weights=None)
        in_f = b.fc.in_features
        b.fc = nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(0.4), nn.Linear(in_f, 2))
        self.net = b
    def forward(self, x): return self.net(x)

class ResNet50_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.resnet50(weights=None)
        self.stem   = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
        self.layer1, self.layer2 = b.layer1, b.layer2
        self.layer3, self.layer4 = b.layer3, b.layer4
        self.cbam   = CBAM(2048)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(nn.Flatten(), nn.BatchNorm1d(2048),
                                     nn.Dropout(0.4), nn.Linear(2048, 2))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(self.cbam(x)))

# ── data (test only) ──
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds  = datasets.ImageFolder(PATCH_DIR/"test", transform=eval_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=USE_AMP)

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

@torch.no_grad()
def tta_infer(model):
    model.eval(); ys, ss = [], []
    for imgs, lbs in test_loader:
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

results = {}

def eval_ckpt(name, model_class, ckpt_path, infer_fn=single_infer):
    print(f"  Evaluating {name} ...", end=" ", flush=True)
    m = model_class().to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    ty, ts = infer_fn(m)
    r = full_metrics(ty, ts, 0.5)
    r["best_epoch"] = int(ck["epoch"])
    results[name] = r
    print(f"epoch={r['best_epoch']}  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}  MCC={r['mcc']:.4f}", flush=True)

eval_ckpt("v1_cbam",  ResNet50_CBAM,  ABL_DIR/"v1_cbam"/"best.pth")
eval_ckpt("v1_focal", ResNet50_Plain, ABL_DIR/"v1_focal"/"best.pth")
eval_ckpt("v1_mixup", ResNet50_Plain, ABL_DIR/"v1_mixup"/"best.pth")
eval_ckpt("v1_tta",   ResNet50_Plain, PRIOR_DIR/"v1_full"/"best.pth", infer_fn=tta_infer)

# Reload v1 + v2 from prior comparison
prior = json.loads((PRIOR_DIR/"comparison_report.json").read_text())
results["v1"] = prior["v1_full"]["test"]
results["v2"] = prior["v2_full"]["test_tta"]

# ── Save report ──
report = {
    "split": prior["split_scheme"],
    "train_size": prior["train_size"],
    "val_size":   prior["val_size"],
    "test_size":  prior["test_size"],
    "ablations": results,
    "note": "v1_mixup early-stopped at epoch 20 due to a cuDNN stream error during epoch 23; epoch-20 best checkpoint used (val_f1=0.9805).",
}
(ABL_DIR/"ablation_report.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved {ABL_DIR/'ablation_report.json'}", flush=True)

# ── Master table ──
print("\n" + "="*112, flush=True)
print("  ABLATION STUDY — each row = v1 baseline + ONE change", flush=True)
print("="*112, flush=True)
order = [("v1","v1 (baseline)"), ("v1_cbam","v1 + CBAM"),
         ("v1_focal","v1 + Focal"), ("v1_mixup","v1 + MixUp"),
         ("v1_tta","v1 + TTA"), ("v2","v2 (all four)")]
mlist = ["accuracy","balanced_acc","precision","recall","specificity","f1","mcc","roc_auc","pr_auc"]
hdr = f"  {'Variant':<22}" + "".join(f"{k[:8]:>10}" for k in mlist)
print(hdr, flush=True); print("-"*112, flush=True)
v1ref = results["v1"]
for k, label in order:
    if k not in results: continue
    r = results[k]
    line = f"  {label:<22}" + "".join(f"{r[m]:>10.4f}" for m in mlist)
    print(line, flush=True)
print("="*112, flush=True)

# ── Δ vs v1 table ──
print("\n  Δ vs v1 baseline (positive = improvement):", flush=True)
print("-"*112, flush=True)
for k, label in order:
    if k == "v1" or k not in results: continue
    line = f"  {label:<22}" + "".join(f"{results[k][m]-v1ref[m]:>+10.4f}" for m in mlist)
    print(line, flush=True)
print("="*112, flush=True)
