"""Sanity audit triggered by tutor feedback.

Checks:
  1. No patient-level leakage between train / val / test.
  2. Class index mapping — what does y=1 actually correspond to?
  3. Re-compute v1, v2, v3 metrics with EXPLICIT pos_label = nodule.
  4. Compare to the previously reported numbers.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix)

PATCH_DIR = Path("luna16_patches_full")
RUNS_DIR  = Path("luna16_runs_full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}\n", flush=True)

# ─────────────────────────────────────────────────────────
# 1. Patient-level leakage check
# ─────────────────────────────────────────────────────────
print("="*70)
print("  1.  PATIENT-LEVEL LEAKAGE CHECK")
print("="*70)

def get_series_uids(split):
    uids = set()
    for cls in ["nodule", "non_nodule"]:
        for f in (PATCH_DIR/split/cls).glob("*.png"):
            parts = f.stem.split("_")
            # filename: {seriesUID}_{x}_{y}_{z}.png
            #   seriesUID is a 1.3.6.1... style ID with no underscores in our pipeline
            uids.add(parts[0])
    return uids

train_uids = get_series_uids("train")
val_uids   = get_series_uids("val")
test_uids  = get_series_uids("test")
print(f"  train patients: {len(train_uids):>4}")
print(f"  val   patients: {len(val_uids):>4}")
print(f"  test  patients: {len(test_uids):>4}")
print(f"\n  train ∩ test:  {len(train_uids & test_uids)} (must be 0)")
print(f"  train ∩ val:   {len(train_uids & val_uids)} (must be 0)")
print(f"  val ∩ test:    {len(val_uids & test_uids)} (must be 0)")
leakage_clean = not (train_uids & test_uids or train_uids & val_uids or val_uids & test_uids)
print(f"\n  ✓ NO leakage detected" if leakage_clean else "\n  ✗ LEAKAGE DETECTED")

# ─────────────────────────────────────────────────────────
# 2. Class index mapping
# ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  2.  CLASS INDEX MAPPING")
print("="*70)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds = datasets.ImageFolder(PATCH_DIR/"test", transform=eval_tf)
print(f"  ImageFolder class_to_idx: {test_ds.class_to_idx}")
print(f"  ImageFolder classes:      {test_ds.classes}")
print()
# Count labels in the test set
labels_arr = np.array([y for _, y in test_ds.samples])
n0 = (labels_arr == 0).sum()
n1 = (labels_arr == 1).sum()
print(f"  Patches with label=0:  {n0}   (folder = '{test_ds.classes[0]}')")
print(f"  Patches with label=1:  {n1}   (folder = '{test_ds.classes[1]}')")

if test_ds.class_to_idx.get("nodule") == 0:
    print(f"\n  ⚠  ImageFolder assigns nodule → 0 and non_nodule → 1")
    print(f"     But every metric in our reports used pos_label=1.")
    print(f"     → Reported 'recall' was actually the recall of the NON-NODULE class.")
else:
    print(f"\n  ✓ ImageFolder assigns nodule → 1 (matches our pos_label=1)")

# ─────────────────────────────────────────────────────────
# 3. Re-evaluate each saved checkpoint with EXPLICIT positive class = nodule
# ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  3.  RE-EVALUATION WITH POS_LABEL = NODULE")
print("="*70)

# Models
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
        avg = x.mean(dim=1, keepdim=True); mx = x.max(dim=1, keepdim=True).values
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
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(self.cbam(x)))

class DenseNet121_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.densenet121(weights=None)
        self.features = b.features; self.relu = nn.ReLU(inplace=True)
        in_f = b.classifier.in_features
        self.cbam = CBAM(in_f); self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.BatchNorm1d(in_f),
                                   nn.Dropout(0.4), nn.Linear(in_f, 2))
    def forward(self, x):
        x = self.relu(self.features(x))
        return self.head(self.pool(self.cbam(x)))

test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=USE_AMP)
# Get label of each test patch (under ImageFolder's mapping)
ys_full = np.array([y for _, y in test_ds.samples])
# Build a corrected "nodule = 1" label vector
nod_idx = test_ds.class_to_idx["nodule"]
y_nodule = (ys_full == nod_idx).astype(int)
print(f"\n  Using nodule-as-positive labels (count = {y_nodule.sum()} of {len(y_nodule)})")

@torch.no_grad()
def infer(model, tta=False):
    """Return probability of NODULE class for each test patch."""
    model.eval(); ss = []
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        if tta:
            views = [imgs, torch.flip(imgs, [3]), torch.flip(imgs, [2]),
                     torch.rot90(imgs, 1, [2,3]), torch.rot90(imgs, 3, [2,3])]
            all_p = []
            for v in views:
                with autocast('cuda', enabled=USE_AMP):
                    out = model(v)
                # ImageFolder's nodule index → take that softmax column
                all_p.append(torch.softmax(out.float(), dim=1)[:, nod_idx])
            probs = torch.stack(all_p).mean(0)
        else:
            with autocast('cuda', enabled=USE_AMP):
                out = model(imgs)
            probs = torch.softmax(out.float(), dim=1)[:, nod_idx]
        ss.extend(probs.cpu().numpy())
    return np.array(ss)

def full_metrics_nodule(y_score, threshold=0.5):
    y_true = y_nodule
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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

runs = {
    "v1_full":           ResNet50_Plain,
    "v2_full":           ResNet50_CBAM,
    "v3_full":           DenseNet121_CBAM,
}
results = {}
for name, ModelClass in runs.items():
    ck_path = RUNS_DIR/name/"best.pth"
    if not ck_path.exists():
        print(f"  [skip] {name}: checkpoint missing")
        continue
    m = ModelClass().to(device)
    m.load_state_dict(torch.load(ck_path, map_location=device, weights_only=False)["model_state_dict"])
    if name == "v1_full":
        probs = infer(m, tta=False)
        results[name] = full_metrics_nodule(probs)
        # also v1+TTA
        probs_tta = infer(m, tta=True)
        results["v1_full_tta"] = full_metrics_nodule(probs_tta)
    else:
        probs = infer(m, tta=True)
        results[name] = full_metrics_nodule(probs)
    print(f"  {name}: F1 = {results[name]['f1']:.4f}  Recall = {results[name]['recall']:.4f}  AUC = {results[name]['roc_auc']:.4f}")

# ─────────────────────────────────────────────────────────
# 4. Compare to previously reported numbers
# ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  4.  PREVIOUSLY REPORTED  vs  CORRECTED (nodule = positive)")
print("="*70)
prior = json.loads((RUNS_DIR/"comparison_report.json").read_text())
prior_v1 = prior["v1_full"]["test"]
prior_v2 = prior["v2_full"]["test_tta"]
prior_v3 = prior["v3_full"]["test_tta"]

def row(label, old, new):
    keys = ["accuracy","precision","recall","specificity","f1","mcc","roc_auc","pr_auc"]
    print(f"\n  {label}")
    print(f"    {'metric':<14} {'reported':>10} {'corrected':>11} {'Δ':>9}")
    for k in keys:
        a = old[k]; b = new[k]; d = b-a
        sign = "+" if d>=0 else ""
        print(f"    {k:<14} {a:>10.4f} {b:>11.4f} {sign}{d:>8.4f}")
    if "confusion" in new:
        print(f"    confusion:  TP={new['confusion']['TP']}  FP={new['confusion']['FP']}  TN={new['confusion']['TN']}  FN={new['confusion']['FN']}")

row("v1 (single pass)",       prior_v1, results["v1_full"])
row("v2 (with TTA)",          prior_v2, results["v2_full"])
row("v3 (with TTA)",          prior_v3, results["v3_full"])

# Save corrected report
corrected = {
    "note": "Re-computed with pos_label = nodule (the medically relevant positive class).",
    "test_size": len(y_nodule),
    "n_nodules_in_test": int(y_nodule.sum()),
    "n_non_nodules_in_test": int(len(y_nodule) - y_nodule.sum()),
    "results": results,
}
out = RUNS_DIR / "comparison_report_corrected.json"
out.write_text(json.dumps(corrected, indent=2))
print(f"\nSaved {out}")
