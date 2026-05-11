"""Generate a 60-second MP4 demo for Seminar 2.
Frames:
  - 5s  title card
  - 6 × 8s  test-patch + CBAM attention with prediction label (last is a wrong example)
  - 7s  summary frame with overall test metrics
Output: demo.mp4
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import imageio.v2 as iio          # imageio.v2 keeps the simple API
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PATCH_DIR  = Path("luna16_patches_full")
MODEL_PATH = Path("luna16_runs_full/v2_full/best.pth")  # the Option-B-trained v2 (matches deck metrics)
OUT_PATH   = Path("demo.mp4")

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print(f"Device: {device}", flush=True)

# ── Model definition (must match v2 build) ──
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
    def forward_map(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return self.sig(self.conv(torch.cat([avg, mx], 1)))
    def forward(self, x):
        return self.forward_map(x) * x

class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c); self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        b = models.resnet50(weights=None)
        self.stem   = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
        self.layer1, self.layer2 = b.layer1, b.layer2
        self.layer3, self.layer4 = b.layer3, b.layer4
        self.cbam   = CBAM(2048)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(nn.Flatten(), nn.BatchNorm1d(2048),
                                     nn.Dropout(dropout), nn.Linear(2048, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.head(self.pool(self.cbam(x)))
    def predict_with_attention(self, x):
        feats = self.stem(x)
        feats = self.layer1(feats); feats = self.layer2(feats)
        feats = self.layer3(feats); feats = self.layer4(feats)
        ca    = self.cbam.ca(feats)
        sa_map = self.cbam.sa.forward_map(ca)              # (B, 1, 7, 7)
        refined = sa_map * ca
        logits  = self.head(self.pool(refined))
        return logits, sa_map

# ── Load model ──
model = ResNet50_CBAM(num_classes=2, dropout=0.4).to(device).eval()
ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
print(f"Loaded {MODEL_PATH} (epoch {ck['epoch']})", flush=True)

# ── Test set ──
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE = 224
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds = datasets.ImageFolder(PATCH_DIR/"test", transform=eval_tf)
loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0,
                      pin_memory=USE_AMP)
print(f"Test set: {len(test_ds)} patches", flush=True)

# ── Run inference + collect attention maps for ALL patches ──
print("Running inference on entire test set...", flush=True)
all_probs   = []
all_labels  = []
all_attn    = []     # raw 7×7 spatial attention maps
with torch.no_grad():
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        with autocast('cuda', enabled=USE_AMP):
            logits, sa = model.predict_with_attention(imgs)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(lbs.numpy())
        all_attn.append(sa.float().cpu().numpy())

probs  = np.concatenate(all_probs)
labels = np.concatenate(all_labels)
attns  = np.concatenate(all_attn)         # (N, 1, 7, 7)
preds  = (probs >= 0.5).astype(int)
print(f"Predictions complete. Acc={(preds==labels).mean():.4f}", flush=True)

# ── Pick 6 patches for the demo ──
# Indices for nodule (label=1) and non_nodule (label=0)
idx_nod = np.where(labels == 1)[0]
idx_non = np.where(labels == 0)[0]

def pick(condition_array, sort_key, n=1, descending=True):
    """Return n indices into condition_array sorted by sort_key."""
    if len(condition_array) == 0: return []
    keys = sort_key[condition_array]
    order = np.argsort(-keys) if descending else np.argsort(keys)
    return condition_array[order][:n].tolist()

# 1) high-confidence correct nodule
correct_nod = idx_nod[(preds[idx_nod] == 1)]
hi_conf_nod = pick(correct_nod, probs, n=1, descending=True)[0]

# 2) high-confidence correct non-nodule
correct_non = idx_non[(preds[idx_non] == 0)]
hi_conf_non = pick(correct_non, probs, n=1, descending=False)[0]

# 3) low-confidence-but-correct nodule (hard case)
low_conf_nod = pick(correct_nod, probs, n=1, descending=False)[0]

# 4) low-confidence-but-correct non-nodule
hi_conf_non2 = pick(correct_non, probs, n=2, descending=False)[1]

# 5) another correct nodule (mid confidence)
mid_conf_nod_candidates = correct_nod[np.abs(probs[correct_nod] - 0.85) < 0.05]
mid_conf_nod = mid_conf_nod_candidates[0] if len(mid_conf_nod_candidates) else correct_nod[len(correct_nod)//2]

# 6) ONE wrong example — false positive (model said nodule, was non-nodule)
fp_idx = idx_non[(preds[idx_non] == 1)]
if len(fp_idx) == 0:
    fp_idx = idx_nod[(preds[idx_nod] == 0)]   # fallback to false negative
wrong_pick = pick(fp_idx, probs, n=1, descending=True)[0]

selected = [
    (hi_conf_nod,  True),
    (hi_conf_non,  True),
    (low_conf_nod, True),
    (hi_conf_non2, True),
    (mid_conf_nod, True),
    (wrong_pick,   False),
]
print("Selected demo patches:", flush=True)
for i, (idx, correct) in enumerate(selected, 1):
    p = float(probs[idx])
    pred_class = "Nodule" if preds[idx]==1 else "Non-nodule"
    truth      = "Nodule" if labels[idx]==1 else "Non-nodule"
    mark       = "✓" if correct else "✗"
    print(f"  {i}. idx={idx}  truth={truth:<10}  pred={pred_class:<10}  "
          f"conf={p:.3f}  {mark}", flush=True)

# ── Frame rendering helpers ──
WIDTH, HEIGHT = 1280, 720
DPI           = 100
FPS           = 10
FIG_SIZE      = (WIDTH/DPI, HEIGHT/DPI)

NAVY  = "#0F2C4F"
RED   = "#D71920"
CREAM = "#F5F2EA"
GREEN = "#2E7D32"
GRAY  = "#5F6470"

def fig_to_array(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]   # RGB
    plt.close(fig)
    return buf

def render_title():
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_facecolor(NAVY)
    # Vertical accent bar on left
    ax.add_patch(mpatches.Rectangle((0, 0), 0.04, 1, transform=ax.transAxes,
                                    facecolor=RED, edgecolor='none'))
    ax.text(0.5, 0.62, "Lung Cancer Classification",
            ha="center", va="center", fontsize=44, color="white",
            fontweight="bold", family="serif", transform=ax.transAxes)
    ax.text(0.5, 0.50, "Live Demonstration  ·  ResNet-50 + CBAM",
            ha="center", va="center", fontsize=20, color="#C0CADC",
            style="italic", family="serif", transform=ax.transAxes)
    ax.add_patch(mpatches.Rectangle((0.45, 0.42), 0.10, 0.005,
                                    transform=ax.transAxes,
                                    facecolor=RED, edgecolor='none'))
    ax.text(0.5, 0.32, "Test set: subsets 8 + 9 (1 528 patches, 176 unseen patients)",
            ha="center", va="center", fontsize=15, color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.10, "Seminar 2  ·  12 May 2026  ·  Huu Thuc Tran",
            ha="center", va="center", fontsize=12, color="#C0CADC",
            transform=ax.transAxes)
    return fig_to_array(fig)

def render_patch_frame(idx, is_correct, frame_num=None, total_frames=None):
    img_path, true_lbl = test_ds.samples[idx]
    raw = np.array(Image.open(img_path).convert("L"))   # original 64×64 grayscale
    sa  = attns[idx, 0]                                 # 7×7 attention
    sa_up = np.array(Image.fromarray(sa).resize((raw.shape[1], raw.shape[0]),
                                                 Image.BILINEAR))
    p     = float(probs[idx])
    pred  = int(preds[idx])
    truth = int(labels[idx])
    pred_str  = "Nodule" if pred  == 1 else "Non-nodule"
    truth_str = "Nodule" if truth == 1 else "Non-nodule"
    conf      = p if pred == 1 else (1 - p)

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=CREAM)

    # Top-bar with section info
    ax_bar = fig.add_axes([0, 0.93, 1, 0.07]); ax_bar.set_axis_off()
    ax_bar.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax_bar.transAxes,
                                        facecolor=NAVY, edgecolor='none'))
    ax_bar.text(0.02, 0.5, "LIVE INFERENCE  ·  ResNet-50 + CBAM",
                transform=ax_bar.transAxes, ha="left", va="center",
                fontsize=11, color="white", family="sans-serif",
                fontweight="bold")
    if frame_num is not None:
        ax_bar.text(0.98, 0.5, f"Patch {frame_num} / {total_frames}",
                    transform=ax_bar.transAxes, ha="right", va="center",
                    fontsize=11, color="#C0CADC", family="sans-serif")

    # Two image axes
    ax1 = fig.add_axes([0.05, 0.18, 0.40, 0.65])
    ax2 = fig.add_axes([0.55, 0.18, 0.40, 0.65])

    ax1.imshow(raw, cmap="gray")
    ax1.set_title("Input CT patch (64 × 64)", fontsize=14, color=NAVY,
                  fontweight="bold", family="serif", pad=10)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_edgecolor(GRAY)

    ax2.imshow(raw, cmap="gray")
    ax2.imshow(sa_up, cmap="jet", alpha=0.50)
    ax2.set_title("CBAM spatial attention", fontsize=14, color=NAVY,
                  fontweight="bold", family="serif", pad=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_edgecolor(GRAY)

    # Caption
    color_box = GREEN if is_correct else RED
    mark      = "✓ correct" if is_correct else "✗ wrong"
    ax_cap = fig.add_axes([0.05, 0.02, 0.90, 0.13]); ax_cap.set_axis_off()
    ax_cap.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax_cap.transAxes,
                                        facecolor="white", edgecolor=GRAY,
                                        linewidth=0.5))
    ax_cap.text(0.02, 0.5,
                f"Prediction:  {pred_str}     Confidence:  {conf*100:.1f}%",
                transform=ax_cap.transAxes, ha="left", va="center",
                fontsize=18, color=NAVY, fontweight="bold", family="sans-serif")
    ax_cap.text(0.98, 0.5,
                f"Ground truth: {truth_str}    {mark}",
                transform=ax_cap.transAxes, ha="right", va="center",
                fontsize=15, color=color_box, fontweight="bold",
                family="sans-serif", style="italic")

    return fig_to_array(fig)

def render_summary():
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_patch(mpatches.Rectangle((0, 0), 0.04, 1, transform=ax.transAxes,
                                    facecolor=RED, edgecolor='none'))
    ax.text(0.5, 0.78, "How it does overall",
            ha="center", va="center", fontsize=20, color="#C0CADC",
            family="sans-serif", style="italic", transform=ax.transAxes)
    ax.text(0.5, 0.62, "1 528 unseen test patches  ·  176 patients",
            ha="center", va="center", fontsize=18, color="#C0CADC",
            transform=ax.transAxes)
    metrics = [
        ("F1-Score",   "0.9829"),
        ("ROC-AUC",    "0.9917"),
        ("Recall",     "0.9874"),
        ("Specificity","0.9228"),
    ]
    for i, (name, val) in enumerate(metrics):
        x = 0.18 + i * 0.21
        ax.text(x, 0.45, val, ha="center", va="center", fontsize=42,
                color="white", fontweight="bold", family="serif",
                transform=ax.transAxes)
        ax.text(x, 0.34, name, ha="center", va="center", fontsize=14,
                color="#C0CADC", family="sans-serif", transform=ax.transAxes)

    ax.add_patch(mpatches.Rectangle((0.45, 0.20), 0.10, 0.005,
                                    transform=ax.transAxes,
                                    facecolor=RED, edgecolor='none'))
    ax.text(0.5, 0.12, "Patient-level split  ·  zero leakage",
            ha="center", va="center", fontsize=14, color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.05, "Lung Cancer Classification  ·  Seminar 2  ·  12 May 2026",
            ha="center", va="center", fontsize=11, color="#C0CADC",
            transform=ax.transAxes)
    return fig_to_array(fig)

# ── Build frames ──
print("\nRendering frames...", flush=True)
frames = []
title = render_title()
frames.extend([title] * (5 * FPS))                      # 5 sec title
print(f"  title: {5*FPS} frames", flush=True)
for i, (idx, ok) in enumerate(selected, 1):
    f = render_patch_frame(idx, ok, frame_num=i, total_frames=len(selected))
    frames.extend([f] * (8 * FPS))                      # 8 sec each
    print(f"  patch {i}: {8*FPS} frames", flush=True)
summary = render_summary()
frames.extend([summary] * (7 * FPS))                    # 7 sec summary
print(f"  summary: {7*FPS} frames", flush=True)
print(f"Total: {len(frames)} frames @ {FPS} fps = {len(frames)/FPS:.1f} sec", flush=True)

# ── Write MP4 ──
print(f"\nWriting {OUT_PATH}...", flush=True)
writer = iio.get_writer(str(OUT_PATH), fps=FPS, codec="libx264",
                         quality=8, macro_block_size=1)
for f in frames:
    writer.append_data(f)
writer.close()
size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
print(f"Done. {OUT_PATH}  ({size_mb:.1f} MB)", flush=True)
