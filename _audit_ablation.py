"""Re-evaluate ablation checkpoints with corrected pos_label = nodule."""
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
ABL_DIR   = Path("luna16_runs_ablation")
RUNS_DIR  = Path("luna16_runs_full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds = datasets.ImageFolder(PATCH_DIR/"test", transform=eval_tf)
loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=USE_AMP)
nod_idx = test_ds.class_to_idx["nodule"]
ys = np.array([y for _, y in test_ds.samples])
y_nodule = (ys == nod_idx).astype(int)
print(f"Nodule positive labels: {y_nodule.sum()} of {len(y_nodule)}", flush=True)

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

@torch.no_grad()
def infer(model, tta=False):
    model.eval(); ss = []
    for imgs, _ in loader:
        imgs = imgs.to(device)
        if tta:
            views = [imgs, torch.flip(imgs, [3]), torch.flip(imgs, [2]),
                     torch.rot90(imgs, 1, [2,3]), torch.rot90(imgs, 3, [2,3])]
            all_p = []
            for v in views:
                with autocast('cuda', enabled=USE_AMP):
                    out = model(v)
                all_p.append(torch.softmax(out.float(), dim=1)[:, nod_idx])
            probs = torch.stack(all_p).mean(0)
        else:
            with autocast('cuda', enabled=USE_AMP):
                out = model(imgs)
            probs = torch.softmax(out.float(), dim=1)[:, nod_idx]
        ss.extend(probs.cpu().numpy())
    return np.array(ss)

def full_metrics(score, threshold=0.5):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_nodule, pred, labels=[0, 1]).ravel()
    return {
        "accuracy":     float(accuracy_score(y_nodule, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_nodule, pred)),
        "precision":    float(precision_score(y_nodule, pred, pos_label=1, zero_division=0)),
        "recall":       float(recall_score(y_nodule, pred, pos_label=1, zero_division=0)),
        "specificity":  float(tn/(tn+fp)) if (tn+fp) else 0.0,
        "f1":           float(f1_score(y_nodule, pred, pos_label=1, zero_division=0)),
        "mcc":          float(matthews_corrcoef(y_nodule, pred)),
        "roc_auc":      float(roc_auc_score(y_nodule, score)),
        "pr_auc":       float(average_precision_score(y_nodule, score)),
        "confusion":    {"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn)},
    }

variants = [
    ("v1",       ResNet50_Plain, RUNS_DIR/"v1_full"/"best.pth",       False),
    ("v1_cbam",  ResNet50_CBAM,  ABL_DIR/"v1_cbam"/"best.pth",        False),
    ("v1_focal", ResNet50_Plain, ABL_DIR/"v1_focal"/"best.pth",       False),
    ("v1_mixup", ResNet50_Plain, ABL_DIR/"v1_mixup"/"best.pth",       False),
    ("v1_tta",   ResNet50_Plain, RUNS_DIR/"v1_full"/"best.pth",       True),
    ("v2",       ResNet50_CBAM,  RUNS_DIR/"v2_full"/"best.pth",       True),
]
results = {}
for name, ModelClass, ck_path, tta in variants:
    print(f"Evaluating {name} (tta={tta})...", flush=True)
    m = ModelClass().to(device)
    m.load_state_dict(torch.load(ck_path, map_location=device, weights_only=False)["model_state_dict"])
    s = infer(m, tta=tta)
    results[name] = full_metrics(s)

# Print master table
print("\n" + "="*120, flush=True)
print("  ABLATION + 3-WAY (CORRECTED — nodule = positive)", flush=True)
print("="*120, flush=True)
order = [("v1","v1 (baseline)"), ("v1_cbam","v1 + CBAM"),
         ("v1_focal","v1 + Focal"), ("v1_mixup","v1 + MixUp"),
         ("v1_tta","v1 + TTA"), ("v2","v2 (all four, +TTA)")]
mlist = ["accuracy","balanced_acc","precision","recall","specificity","f1","mcc","roc_auc","pr_auc"]
hdr = f"  {'Variant':<22}" + "".join(f"{k[:9]:>10}" for k in mlist)
print(hdr, flush=True); print("-"*120, flush=True)
v1ref = results["v1"]
for k, label in order:
    r = results[k]
    line = f"  {label:<22}" + "".join(f"{r[m]:>10.4f}" for m in mlist)
    print(line, flush=True)
print("="*120, flush=True)

print("\n  Δ vs v1 (positive = improvement):", flush=True)
print("-"*120, flush=True)
for k, label in order:
    if k == "v1": continue
    r = results[k]
    line = f"  {label:<22}" + "".join(f"{r[m]-v1ref[m]:>+10.4f}" for m in mlist)
    print(line, flush=True)
print("="*120, flush=True)

print("\n  Confusion matrices (nodule positive):", flush=True)
for k, label in order:
    c = results[k]["confusion"]
    print(f"    {label:<22}  TP={c['TP']:>3}  FP={c['FP']:>3}  TN={c['TN']:>5}  FN={c['FN']:>3}", flush=True)

# Save
out_path = ABL_DIR / "ablation_report_corrected.json"
out_path.write_text(json.dumps({
    "note": "Corrected: pos_label = nodule.",
    "test_size": len(y_nodule),
    "n_nodules": int(y_nodule.sum()),
    "n_non_nodules": int(len(y_nodule) - y_nodule.sum()),
    "results": results,
}, indent=2))
print(f"\nSaved {out_path}", flush=True)
