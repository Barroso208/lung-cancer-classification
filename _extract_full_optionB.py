"""LUNA16 full extraction — Option B split.

Subset mapping:
  subsets 0-6 → train
  subset 7    → val
  subsets 8-9 → test

Output: luna16_patches_full/{split}/{class}/*.png
Idempotent: skips files that already exist.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

LUNA_DIR = Path("luna16")
OUT_DIR  = Path("luna16_patches_full")
OUT_DIR.mkdir(exist_ok=True)

SPLITS = {
    "subset0": "train", "subset1": "train", "subset2": "train",
    "subset3": "train", "subset4": "train", "subset5": "train",
    "subset6": "train",
    "subset7": "val",
    "subset8": "test", "subset9": "test",
}

PATCH_MM, PATCH_PX, NEG_RATIO = 50, 64, 3
HU_WIN = (-1000, 400)

def world_to_voxel(w, origin, spacing, direction):
    D = np.array(direction).reshape(3, 3)
    return np.round(np.linalg.inv(D) @ ((np.array(w) - np.array(origin)) / np.array(spacing))).astype(int)

def window_hu(arr):
    arr = np.clip(arr, *HU_WIN)
    return ((arr - HU_WIN[0]) / (HU_WIN[1] - HU_WIN[0]) * 255.0).astype(np.uint8)

def extract_patch(vol, spacing, vx):
    z = vx[2]
    if z < 0 or z >= vol.shape[0]: return None
    s = vol[z]
    hx = int(round((PATCH_MM/2) / spacing[0]))
    hy = int(round((PATCH_MM/2) / spacing[1]))
    cx, cy = vx[0], vx[1]
    y0, y1, x0, x1 = cy-hy, cy+hy, cx-hx, cx+hx
    if y0 < 0 or x0 < 0 or y1 > s.shape[0] or x1 > s.shape[1]: return None
    return Image.fromarray(window_hu(s[y0:y1, x0:x1])).resize((PATCH_PX, PATCH_PX), Image.BILINEAR)

cand = pd.read_csv(LUNA_DIR / "candidates_V2.csv")
print(f"Loaded {len(cand):,} candidates", flush=True)

# Make split dirs
for split in set(SPLITS.values()):
    (OUT_DIR / split / "nodule").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / "non_nodule").mkdir(parents=True, exist_ok=True)

stats = {s: {"nodule": 0, "non_nodule": 0, "patients": 0, "skipped": 0}
         for s in set(SPLITS.values())}

t_start = time.time()
for subset, split in SPLITS.items():
    inner = LUNA_DIR / subset / subset
    if not inner.exists(): inner = LUNA_DIR / subset
    mhd_files = sorted(inner.glob("*.mhd"))
    if not mhd_files:
        print(f"  [!] {subset}: no .mhd files at {inner}", flush=True)
        continue
    print(f"\n=== {subset} → {split}  ({len(mhd_files)} CTs) ===", flush=True)
    t0 = time.time()
    for i, mhd in enumerate(mhd_files, 1):
        uid = mhd.stem
        # Skip if already done (any patch with this uid exists)
        existing = list((OUT_DIR/split/"nodule").glob(f"{uid}_*.png")) + \
                   list((OUT_DIR/split/"non_nodule").glob(f"{uid}_*.png"))
        if existing:
            stats[split]["patients"] += 1
            stats[split]["skipped"] += 1
            continue

        pat = cand[cand.seriesuid == uid]
        if len(pat) == 0:
            stats[split]["skipped"] += 1
            continue
        try:
            vol = sitk.ReadImage(str(mhd))
            arr = sitk.GetArrayFromImage(vol)
            origin, spacing, direction = vol.GetOrigin(), vol.GetSpacing(), vol.GetDirection()
        except Exception as e:
            print(f"  [!] {uid}: {e}", flush=True)
            stats[split]["skipped"] += 1
            continue

        pos = pat[pat["class"] == 1]
        neg = pat[pat["class"] == 0]
        n_neg_target = max(NEG_RATIO * max(len(pos), 1), 3)
        if len(neg) > n_neg_target:
            neg = neg.sample(n=n_neg_target, random_state=42)

        for _, r in pos.iterrows():
            vx = world_to_voxel((r.coordX, r.coordY, r.coordZ), origin, spacing, direction)
            img = extract_patch(arr, spacing, vx)
            if img is not None:
                img.save(OUT_DIR/split/"nodule"/f"{uid}_{int(r.coordX)}_{int(r.coordY)}_{int(r.coordZ)}.png")
                stats[split]["nodule"] += 1
        for _, r in neg.iterrows():
            vx = world_to_voxel((r.coordX, r.coordY, r.coordZ), origin, spacing, direction)
            img = extract_patch(arr, spacing, vx)
            if img is not None:
                img.save(OUT_DIR/split/"non_nodule"/f"{uid}_{int(r.coordX)}_{int(r.coordY)}_{int(r.coordZ)}.png")
                stats[split]["non_nodule"] += 1
        stats[split]["patients"] += 1

        if i % 20 == 0 or i == len(mhd_files):
            dt = time.time() - t0
            print(f"  [{i:>3}/{len(mhd_files)}] +nod={stats[split]['nodule']} "
                  f"+non={stats[split]['non_nodule']} skip={stats[split]['skipped']} "
                  f"({dt/i:.1f}s/vol)", flush=True)

print("\n" + "="*70)
print("  EXTRACTION COMPLETE — Option B split (full LUNA16)")
print("="*70)
total_time = time.time() - t_start
for split in ["train", "val", "test"]:
    s = stats[split]
    tot = s["nodule"] + s["non_nodule"]
    print(f"  {split:<5}  patients={s['patients']:>3}  nodule={s['nodule']:>4}  "
          f"non_nodule={s['non_nodule']:>4}  total={tot:>5}")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Total runtime: {total_time/60:.1f} min")
print("="*70)

import json
(OUT_DIR / "extract_summary.json").write_text(json.dumps({
    "split_scheme": SPLITS, "stats": stats,
    "patch_mm": PATCH_MM, "patch_px": PATCH_PX, "neg_ratio": NEG_RATIO,
    "hu_window": list(HU_WIN),
}, indent=2))
print(f"Summary saved to {OUT_DIR/'extract_summary.json'}")
