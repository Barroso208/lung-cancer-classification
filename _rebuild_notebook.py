"""
Rebuild lung_cancer_cnn.ipynb from the validated _run_notebook.py code.
Run with: py _rebuild_notebook.py
"""
import json
from pathlib import Path

nb_path = Path("lung_cancer_cnn.ipynb")

def code_cell(lines):
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

def md_cell(lines):
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
    }

cells = []

# ── Cell 0 : Title markdown ──────────────────────────────────────────
cells.append(md_cell([
    "# Lung Cancer Detection — ResNet-50 CNN (High-Distinction Quality)",
    "",
    "**Dataset**: IQ-OTH/NCCD CT scan images (Benign / Malignant / Normal)  ",
    "**Model**: ResNet-50 pretrained on ImageNet1K V2, single-phase end-to-end fine-tuning  ",
    "**Key techniques**: WeightedRandomSampler · OneCycleLR · Mixed-precision AMP · TTA  ",
    "**Metrics**: Accuracy · Precision · Recall · F1-Score (macro + per-class)",
]))

# ── Cell 1 : Imports ─────────────────────────────────────────────────
cells.append(code_cell([
    '"""',
    "Lung Cancer Detection CNN - ResNet-50, Single-Phase Fine-Tuning",
    "Benchmark: Accuracy, Precision, Recall, F1 (macro + per-class) + TTA",
    '"""',
    "import os, sys, time, json",
    "import numpy as np",
    "import matplotlib",
    "matplotlib.use('Agg')",
    "import matplotlib.pyplot as plt",
    "from pathlib import Path",
    "",
    "import torch",
    "import torch.nn as nn",
    "import torch.optim as optim",
    "from torch.amp import GradScaler, autocast",
    "from torch.utils.data import DataLoader, WeightedRandomSampler, Subset",
    "from torchvision import datasets, transforms, models",
    "",
    "from sklearn.metrics import (",
    "    accuracy_score, precision_score, recall_score, f1_score,",
    "    confusion_matrix, classification_report",
    ")",
    "from sklearn.model_selection import StratifiedShuffleSplit",
    "",
    "print(f\"PyTorch : {torch.__version__}  CUDA: {torch.version.cuda}\")",
    "sys.stdout.flush()",
]))

# ── Cell 2 : Config ───────────────────────────────────────────────────
cells.append(md_cell(["## Configuration"]))
cells.append(code_cell([
    "# ── Config ────────────────────────────────────────────────────────",
    'DATA_DIR       = Path("Data")',
    'MODEL_DIR      = Path("model")',
    'OUT_DIR        = Path("cnn_output")',
    "MODEL_DIR.mkdir(exist_ok=True)",
    "OUT_DIR.mkdir(exist_ok=True)",
    "",
    'CLASS_NAMES    = ["Benign", "Malignant", "Normal"]',
    "NUM_CLASSES    = 3",
    'BACKBONE       = "resnet50"',
    "IMG_SIZE       = 224",
    "BATCH_SIZE     = 32",
    "NUM_EPOCHS     = 40",
    "LR_MAX         = 5e-4      # OneCycleLR peak",
    "WEIGHT_DECAY   = 1e-4",
    "GRAD_CLIP      = 1.0",
    "EARLY_STOP_PAT = 10        # patience on val F1",
    "VAL_SPLIT      = 0.15",
    "TEST_SPLIT     = 0.10",
    "SEED           = 42",
    "TTA_ROUNDS     = 5",
]))

# ── Cell 3 : Device ───────────────────────────────────────────────────
cells.append(md_cell(["## Device Setup"]))
cells.append(code_cell([
    "# ── Device ────────────────────────────────────────────────────────",
    "torch.manual_seed(SEED)",
    "np.random.seed(SEED)",
    'device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    "USE_AMP = device.type == \"cuda\"",
    "print(f\"Device: {device}\", flush=True)",
    "if device.type == \"cuda\":",
    "    p = torch.cuda.get_device_properties(0)",
    '    print(f"  GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB  AMP: on", flush=True)',
]))

# ── Cell 4 : Transforms ───────────────────────────────────────────────
cells.append(md_cell(["## Data Transforms"]))
cells.append(code_cell([
    "# ── Transforms ────────────────────────────────────────────────────",
    "MEAN = [0.485, 0.456, 0.406]",
    "STD  = [0.229, 0.224, 0.225]",
    "",
    "train_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),",
    "    transforms.RandomCrop(IMG_SIZE),",
    "    transforms.RandomHorizontalFlip(p=0.5),",
    "    transforms.RandomVerticalFlip(p=0.3),",
    "    transforms.RandomRotation(degrees=15),",
    "    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(MEAN, STD),",
    "    transforms.RandomErasing(p=0.3, scale=(0.02, 0.12)),",
    "])",
    "val_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(MEAN, STD),",
    "])",
    "tta_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),",
    "    transforms.RandomCrop(IMG_SIZE),",
    "    transforms.RandomHorizontalFlip(p=0.5),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(MEAN, STD),",
    "])",
]))

# ── Cell 5 : Stratified split + DataLoaders ───────────────────────────
cells.append(md_cell(["## Stratified Dataset Split & DataLoaders"]))
cells.append(code_cell([
    "# ── Stratified split ──────────────────────────────────────────────",
    "full_ds  = datasets.ImageFolder(root=str(DATA_DIR))",
    "all_tgts = np.array(full_ds.targets)",
    "indices  = np.arange(len(all_tgts))",
    "",
    "sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SPLIT, random_state=SEED)",
    "trainval_idx, test_idx = next(sss1.split(indices, all_tgts))",
    "",
    "val_frac = VAL_SPLIT / (1 - TEST_SPLIT)",
    "sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)",
    "tr_rel, va_rel = next(sss2.split(trainval_idx, all_tgts[trainval_idx]))",
    "train_idx = trainval_idx[tr_rel]",
    "val_idx   = trainval_idx[va_rel]",
    "",
    "train_ds = Subset(datasets.ImageFolder(str(DATA_DIR), transform=train_tf), train_idx)",
    "val_ds   = Subset(datasets.ImageFolder(str(DATA_DIR), transform=val_tf),   val_idx)",
    "test_ds  = Subset(datasets.ImageFolder(str(DATA_DIR), transform=val_tf),   test_idx)",
    "",
    'print(f"Split: Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}", flush=True)',
    'for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:',
    "    counts = np.bincount(all_tgts[idx], minlength=NUM_CLASSES)",
    '    print(f"  {name}: {dict(zip(CLASS_NAMES, counts))}", flush=True)',
    "",
    "# Weighted sampler for class imbalance",
    "train_labels = [all_tgts[i] for i in train_idx]",
    "counts_tr    = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)",
    "w_cls        = 1.0 / (counts_tr + 1e-8)",
    "sampler      = WeightedRandomSampler([w_cls[l] for l in train_labels],",
    "                                      num_samples=len(train_labels), replacement=True)",
    "",
    "train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,",
    "                           num_workers=0, pin_memory=USE_AMP)",
    "val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,",
    "                           num_workers=0, pin_memory=USE_AMP)",
    "test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,",
    "                           num_workers=0, pin_memory=USE_AMP)",
]))

# ── Cell 6 : Model ────────────────────────────────────────────────────
cells.append(md_cell(["## Model Architecture (ResNet-50)"]))
cells.append(code_cell([
    "# ── Model ─────────────────────────────────────────────────────────",
    "model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)",
    "in_f  = model.fc.in_features  # 2048",
    "model.fc = nn.Sequential(",
    "    nn.BatchNorm1d(in_f),",
    "    nn.Dropout(p=0.4),",
    "    nn.Linear(in_f, NUM_CLASSES),",
    ")",
    "model = model.to(device)",
    "n_params = sum(p.numel() for p in model.parameters())",
    'print(f"Model: {BACKBONE}  Params: {n_params:,}", flush=True)',
]))

# ── Cell 7 : Loss / Optimizer / Scheduler ────────────────────────────
cells.append(md_cell(["## Loss, Optimizer & Scheduler"]))
cells.append(code_cell([
    "# ── Loss / Optimizer / Scheduler ──────────────────────────────────",
    "cls_weights = torch.tensor(1.0 / (counts_tr / counts_tr.sum()),",
    "                            dtype=torch.float32).to(device)",
    "criterion   = nn.CrossEntropyLoss(weight=cls_weights)",
    "optimizer   = optim.AdamW(model.parameters(), lr=LR_MAX / 25,",
    "                          weight_decay=WEIGHT_DECAY)",
    "scheduler   = optim.lr_scheduler.OneCycleLR(",
    "    optimizer, max_lr=LR_MAX,",
    "    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,",
    '    pct_start=0.3, anneal_strategy="cos",',
    "    div_factor=25, final_div_factor=1e4,",
    ")",
    "scaler = GradScaler('cuda', enabled=USE_AMP)",
    'print(f"Cls weights: {dict(zip(CLASS_NAMES, cls_weights.cpu().numpy().round(3)))}", flush=True)',
]))

# ── Cell 8 : Helpers ─────────────────────────────────────────────────
cells.append(md_cell(["## Metric Helpers & Training Functions"]))
cells.append(code_cell([
    "# ── Metric helpers ────────────────────────────────────────────────",
    "def compute_metrics(y_true, y_pred):",
    '    acc  = accuracy_score(y_true, y_pred)',
    '    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)',
    '    rec  = recall_score(y_true,    y_pred, average="macro", zero_division=0)',
    '    f1   = f1_score(y_true,        y_pred, average="macro", zero_division=0)',
    "    per  = {",
    "        c: {",
    '            "precision": float(precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)),',
    '            "recall":    float(recall_score(y_true,    y_pred, labels=[i], average="micro", zero_division=0)),',
    '            "f1":        float(f1_score(y_true,        y_pred, labels=[i], average="micro", zero_division=0)),',
    "        }",
    "        for i, c in enumerate(CLASS_NAMES)",
    "    }",
    "    return acc, prec, rec, f1, per",
    "",
    "",
    "def train_one_epoch(model, loader):",
    "    model.train()",
    "    run_loss, preds_all, labels_all = 0.0, [], []",
    "    for imgs, lbs in loader:",
    "        imgs, lbs = imgs.to(device), lbs.to(device)",
    "        optimizer.zero_grad()",
    "        with autocast('cuda', enabled=USE_AMP):",
    "            out  = model(imgs)",
    "            loss = criterion(out, lbs)",
    "        scaler.scale(loss).backward()",
    "        scaler.unscale_(optimizer)",
    "        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)",
    "        scaler.step(optimizer)",
    "        scaler.update()",
    "        scheduler.step()",
    "        run_loss    += loss.item() * imgs.size(0)",
    "        preds_all.extend(out.argmax(1).cpu().numpy())",
    "        labels_all.extend(lbs.cpu().numpy())",
    "    return run_loss / len(loader.dataset), *compute_metrics(labels_all, preds_all)[:4]",
    "",
    "",
    "def evaluate(model, loader):",
    "    model.eval()",
    "    run_loss, preds_all, labels_all = 0.0, [], []",
    "    with torch.no_grad():",
    "        for imgs, lbs in loader:",
    "            imgs, lbs = imgs.to(device), lbs.to(device)",
    "            with autocast('cuda', enabled=USE_AMP):",
    "                out  = model(imgs)",
    "                loss = criterion(out, lbs)",
    "            run_loss    += loss.item() * imgs.size(0)",
    "            preds_all.extend(out.argmax(1).cpu().numpy())",
    "            labels_all.extend(lbs.cpu().numpy())",
    "    avg_loss = run_loss / len(loader.dataset)",
    "    acc, prec, rec, f1, pc = compute_metrics(labels_all, preds_all)",
    "    return avg_loss, acc, prec, rec, f1, pc, np.array(preds_all), np.array(labels_all)",
]))

# ── Cell 9 : Training loop ────────────────────────────────────────────
cells.append(md_cell(["## Training Loop"]))
cells.append(code_cell([
    "# ── Training loop ─────────────────────────────────────────────────",
    'history    = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc",',
    '                                "train_f1","val_f1","train_prec","val_prec",',
    '                                "train_rec","val_rec"]}',
    "best_val_f1 = 0.0",
    "early_ctr   = 0",
    'best_ckpt   = MODEL_DIR / f"best_{BACKBONE}.pth"',
    "",
    'HDR = (f"{\'Ep\':>3}  {\'TrLoss\':>7}  {\'TrAcc\':>6}  {\'TrF1\':>6}  "',
    '       f"{\'VaLoss\':>7}  {\'VaAcc\':>6}  {\'VaF1\':>6}  {\'VaP\':>6}  {\'VaR\':>6}  {\'LR\':>8}  {\'s\':>5}")',
    'print("\\nStarting training...\\n")',
    "print(HDR)",
    'print("-" * len(HDR))',
    "sys.stdout.flush()",
    "",
    "for epoch in range(1, NUM_EPOCHS + 1):",
    "    t0 = time.time()",
    "    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = train_one_epoch(model, train_loader)",
    "    va_loss, va_acc, va_prec, va_rec, va_f1, _, _, _ = evaluate(model, val_loader)",
    "    elapsed = time.time() - t0",
    '    lr_now  = optimizer.param_groups[0]["lr"]',
    "",
    '    for k, v in [("train_loss",tr_loss),("val_loss",va_loss),',
    '                  ("train_acc",tr_acc),  ("val_acc",va_acc),',
    '                  ("train_f1",tr_f1),    ("val_f1",va_f1),',
    '                  ("train_prec",tr_prec),("val_prec",va_prec),',
    '                  ("train_rec",tr_rec),  ("val_rec",va_rec)]:',
    "        history[k].append(v)",
    "",
    '    print(f"{epoch:>3}  {tr_loss:>7.4f}  {tr_acc*100:>5.1f}%  {tr_f1:.4f}  "',
    '          f"{va_loss:>7.4f}  {va_acc*100:>5.1f}%  {va_f1:.4f}  "',
    '          f"{va_prec:.4f}  {va_rec:.4f}  {lr_now:.2e}  {elapsed:>4.0f}s", flush=True)',
    "",
    "    if va_f1 > best_val_f1:",
    "        best_val_f1 = va_f1",
    "        early_ctr   = 0",
    "        torch.save({",
    '            "epoch": epoch, "backbone": BACKBONE, "class_names": CLASS_NAMES,',
    '            "model_state_dict": model.state_dict(),',
    '            "optim_state_dict": optimizer.state_dict(),',
    '            "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1,',
    "        }, best_ckpt)",
    '        print(f"     [*] Best saved  val_acc={va_acc*100:.2f}%  val_f1={va_f1:.4f}  "',
    '              f"val_prec={va_prec:.4f}  val_rec={va_rec:.4f}", flush=True)',
    "    else:",
    "        early_ctr += 1",
    "        if early_ctr >= EARLY_STOP_PAT:",
    '            print(f"\\nEarly stop at epoch {epoch} (patience {EARLY_STOP_PAT})", flush=True)',
    "            break",
    "",
    'print("\\nTraining complete.", flush=True)',
]))

# ── Cell 10 : Training curves ─────────────────────────────────────────
cells.append(md_cell(["## Training Curves"]))
cells.append(code_cell([
    "# ── Training curves ────────────────────────────────────────────────",
    'ep_range = range(1, len(history["train_loss"]) + 1)',
    "fig, axes = plt.subplots(2, 2, figsize=(14, 9))",
    'fig.suptitle(f"Training History - {BACKBONE}", fontsize=14, fontweight="bold")',
    "for ax, tr_k, va_k, pct, title, ylabel in [",
    '    (axes[0,0],"train_loss","val_loss",False,"Cross-Entropy Loss","Loss"),',
    '    (axes[0,1],"train_acc", "val_acc", True, "Accuracy (%)","Accuracy (%)"),',
    '    (axes[1,0],"train_f1",  "val_f1",  False,"Macro F1-Score","F1"),',
    "]:",
    "    tr = [v*100 for v in history[tr_k]] if pct else history[tr_k]",
    "    va = [v*100 for v in history[va_k]] if pct else history[va_k]",
    '    ax.plot(ep_range, tr, "b-o", ms=3, label="Train")',
    '    ax.plot(ep_range, va, "r-o", ms=3, label="Val")',
    '    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)',
    "    ax.legend(fontsize=9); ax.grid(alpha=0.3)",
    'axes[1,1].plot(ep_range, history["val_prec"], "g-o", ms=3, label="Precision")',
    'axes[1,1].plot(ep_range, history["val_rec"],  "m-o", ms=3, label="Recall")',
    'axes[1,1].plot(ep_range, history["val_f1"],   "r-o", ms=3, label="F1")',
    'axes[1,1].set(title="Val Precision / Recall / F1 (macro)", xlabel="Epoch", ylabel="Score")',
    "axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)",
    "plt.tight_layout()",
    'plt.savefig(OUT_DIR / "training_curves.png", dpi=150)',
    "plt.close()",
    'print("Training curves saved.", flush=True)',
    "plt.show()",
]))

# ── Cell 11 : Load best & evaluate + TTA ─────────────────────────────
cells.append(md_cell(["## Load Best Checkpoint & Evaluate with TTA"]))
cells.append(code_cell([
    "# ── Load best & evaluate ───────────────────────────────────────────",
    "ckpt = torch.load(best_ckpt, map_location=device)",
    "model.load_state_dict(ckpt[\"model_state_dict\"])",
    'print(f"Best checkpoint: epoch {ckpt[\'epoch\']}  "',
    '      f"val_acc={ckpt[\'val_acc\']*100:.2f}%  val_f1={ckpt[\'val_f1\']:.4f}", flush=True)',
    "",
    "te_loss, te_acc, te_prec, te_rec, te_f1, per_class, te_preds, te_labels = evaluate(model, test_loader)",
    "",
    "# TTA",
    'print(f"Running TTA ({TTA_ROUNDS} rounds)...", flush=True)',
    "model.eval()",
    "tta_ds  = Subset(datasets.ImageFolder(str(DATA_DIR), transform=tta_tf), test_ds.indices)",
    "base_ds = Subset(datasets.ImageFolder(str(DATA_DIR), transform=val_tf),  test_ds.indices)",
    "b_ldr   = DataLoader(base_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)",
    "t_ldr   = DataLoader(tta_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)",
    "",
    "logits_sum, labels_all = [], []",
    "with torch.no_grad():",
    "    for imgs, lbs in b_ldr:",
    "        with autocast('cuda', enabled=USE_AMP):",
    "            lgt = model(imgs.to(device)).cpu()",
    "        logits_sum.append(lgt)",
    "        labels_all.extend(lbs.numpy())",
    "    logits_sum = torch.cat(logits_sum)",
    "    for _ in range(TTA_ROUNDS):",
    "        aug_l = []",
    "        for imgs, _ in t_ldr:",
    "            with autocast('cuda', enabled=USE_AMP):",
    "                lgt = model(imgs.to(device)).cpu()",
    "            aug_l.append(lgt)",
    "        logits_sum += torch.cat(aug_l)",
    "",
    "tta_preds = (logits_sum / (TTA_ROUNDS + 1)).argmax(1).numpy()",
    "labels_np = np.array(labels_all)",
    "tta_acc, tta_prec, tta_rec, tta_f1, tta_pc = compute_metrics(labels_np, tta_preds)",
]))

# ── Cell 12 : Benchmark table ─────────────────────────────────────────
cells.append(md_cell(["## Benchmark Results"]))
cells.append(code_cell([
    "# ── Benchmark table ────────────────────────────────────────────────",
    'print("\\n" + "="*64)',
    'print("  BENCHMARK RESULTS - TEST SET")',
    'print("="*64)',
    'print(f"  {\'Metric\':<20}  {\'Standard\':>10}  {\'With TTA\':>10}")',
    'print("-"*64)',
    'print(f"  {\'Accuracy\':<20}  {te_acc*100:>9.2f}%  {tta_acc*100:>9.2f}%")',
    'print(f"  {\'Precision (macro)\':<20}  {te_prec:>10.4f}  {tta_prec:>10.4f}")',
    'print(f"  {\'Recall (macro)\':<20}  {te_rec:>10.4f}  {tta_rec:>10.4f}")',
    'print(f"  {\'F1-Score (macro)\':<20}  {te_f1:>10.4f}  {tta_f1:>10.4f}")',
    'print("-"*64)',
    'print(f"  {\'Class\':<20}  {\'Precision\':>10}  {\'Recall\':>8}  {\'F1\':>8}")',
    'print("-"*64)',
    "for c, m in tta_pc.items():",
    "    print(f\"  {c:<20}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  {m['f1']:>8.4f}\")",
    'print("="*64)',
    'print("\\nFull Classification Report (TTA):")',
    "print(classification_report(labels_np, tta_preds, target_names=CLASS_NAMES, digits=4))",
    "sys.stdout.flush()",
]))

# ── Cell 13 : Confusion matrix ────────────────────────────────────────
cells.append(md_cell(["## Confusion Matrix"]))
cells.append(code_cell([
    "# ── Confusion matrix ───────────────────────────────────────────────",
    "cm      = confusion_matrix(labels_np, tta_preds)",
    "cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
    'fig.suptitle(f"Confusion Matrix - {BACKBONE} + TTA", fontsize=13, fontweight="bold")',
    'for ax, data, sub, fmt in [(axes[0],cm,"Counts","%d"),(axes[1],cm_norm,"Row-Normalised",".2f")]:',
    '    im = ax.imshow(data, interpolation="nearest", cmap="Blues")',
    "    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)",
    "    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),",
    "           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,",
    '           xlabel="Predicted", ylabel="True", title=sub)',
    '    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")',
    "    thresh = data.max() / 2.0",
    "    for i in range(data.shape[0]):",
    "        for j in range(data.shape[1]):",
    "            ax.text(j, i, fmt % data[i, j], ha=\"center\", va=\"center\",",
    "                    color=\"white\" if data[i, j] > thresh else \"black\", fontsize=11)",
    "plt.tight_layout()",
    'plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)',
    "plt.close()",
    'print("Confusion matrix saved.", flush=True)',
    "plt.show()",
]))

# ── Cell 14 : Benchmark bar chart ─────────────────────────────────────
cells.append(md_cell(["## Per-Class Benchmark Chart"]))
cells.append(code_cell([
    "# ── Benchmark bar chart ────────────────────────────────────────────",
    'metrics_list = ["Precision", "Recall", "F1"]',
    "x = np.arange(NUM_CLASSES); width = 0.25",
    'colors = ["#4C72B0", "#DD8452", "#55A868"]',
    "fig, ax = plt.subplots(figsize=(10, 6))",
    "for i, (metric, color) in enumerate(zip(metrics_list, colors)):",
    "    vals = [tta_pc[c][metric.lower()] for c in CLASS_NAMES]",
    "    bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)",
    "    for bar, v in zip(bars, vals):",
    "        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,",
    '                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")',
    'ax.axhline(tta_acc, color="black",   ls="--", lw=1.5, label=f"Accuracy={tta_acc:.3f}")',
    'ax.axhline(tta_f1,  color="crimson", ls="--", lw=1.5, label=f"Macro F1={tta_f1:.3f}")',
    "ax.set(xticks=x + width, xticklabels=CLASS_NAMES, ylim=(0, 1.18),",
    '       ylabel="Score", title=f"Per-Class Benchmark - {BACKBONE} + TTA")',
    'ax.legend(loc="lower right", fontsize=9)',
    'ax.grid(axis="y", alpha=0.3)',
    "plt.tight_layout()",
    'plt.savefig(OUT_DIR / "benchmark.png", dpi=150)',
    "plt.close()",
    'print("Benchmark chart saved.", flush=True)',
    "plt.show()",
]))

# ── Cell 15 : Save + final summary ────────────────────────────────────
cells.append(md_cell(["## Save Models & Final Summary"]))
cells.append(code_cell([
    "# ── Save models & benchmark JSON ──────────────────────────────────",
    'final_path = MODEL_DIR / f"final_{BACKBONE}.pth"',
    "torch.save({",
    '    "epoch": len(history["train_loss"]), "backbone": BACKBONE,',
    '    "class_names": CLASS_NAMES, "model_state_dict": model.state_dict(),',
    '    "val_loss": history["val_loss"][-1],',
    '    "val_acc":  history["val_acc"][-1],',
    '    "val_f1":   history["val_f1"][-1],',
    "}, final_path)",
    "",
    "benchmark = {",
    '    "backbone": BACKBONE, "best_epoch": int(ckpt["epoch"]),',
    '    "test_standard": {',
    '        "accuracy":  round(te_acc,  4), "precision": round(te_prec, 4),',
    '        "recall":    round(te_rec,  4), "f1":        round(te_f1,   4),',
    '        "loss":      round(float(te_loss), 4),',
    "    },",
    '    "test_tta": {',
    '        "accuracy":  round(tta_acc,  4), "precision": round(tta_prec, 4),',
    '        "recall":    round(tta_rec,  4), "f1":        round(tta_f1,   4),',
    "    },",
    '    "per_class_tta": {',
    "        c: {k: round(v, 4) for k, v in m.items()} for c, m in tta_pc.items()",
    "    },",
    '    "hyperparameters": {',
    '        "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,',
    '        "epochs_run": len(history["train_loss"]), "lr_max": LR_MAX,',
    '        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,',
    '        "early_stop_patience": EARLY_STOP_PAT, "tta_rounds": TTA_ROUNDS,',
    '        "scheduler": "OneCycleLR", "optimizer": "AdamW",',
    "    },",
    "}",
    '(MODEL_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2))',
    "",
    'print("\\nFiles in model/:")',
    "for f in sorted(MODEL_DIR.iterdir()):",
    '    print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")',
    "",
    'print("\\n" + "="*52)',
    'print("  FINAL RESULTS (with TTA)")',
    'print("="*52)',
    'print(f"  Accuracy  : {tta_acc*100:.2f}%")',
    'print(f"  Precision : {tta_prec:.4f}  (macro)")',
    'print(f"  Recall    : {tta_rec:.4f}  (macro)")',
    'print(f"  F1-Score  : {tta_f1:.4f}  (macro)")',
    'print("="*52)',
    "sys.stdout.flush()",
]))

# ── Build notebook JSON ───────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (lung_cancer_env)",
            "language": "python",
            "name": "lung_cancer_env",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written {nb_path}  ({nb_path.stat().st_size / 1024:.1f} KB)  cells={len(cells)}")
