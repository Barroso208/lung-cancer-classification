"""
LUNA16 Patch Extraction
- Reads .mhd/.raw CT volumes from subsets 7, 8, 9
- Extracts 2D axial patches around candidates (nodule / non-nodule)
- Patient-level split preserved: one subset = one split (train/val/test)
- Saves PNG patches to luna16_patches/{split}/{class}/*.png
"""
import os, sys, time, random, json
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

random.seed(42); np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────
LUNA_DIR  = Path("luna16")
OUT_DIR   = Path("luna16_patches")
OUT_DIR.mkdir(exist_ok=True)

PATCH_MM  = 50          # physical size of patch: 50mm × 50mm
PATCH_PX  = 64          # output resolution in pixels
NEG_RATIO = 3           # negatives per positive per patient (≤ this ratio)
HU_WIN    = (-1000, 400)  # Hounsfield window: lung tissue

SPLITS = {
    "subset7": "train",
    "subset8": "val",
    "subset9": "test",
}

# ── Helpers ─────────────────────────────────────────────────────────
def world_to_voxel(world_xyz, origin, spacing, direction):
    """Convert world (x,y,z) mm → voxel (z,y,x) index."""
    # SimpleITK direction is a 9-length flat matrix
    D = np.array(direction).reshape(3, 3)
    voxel = np.linalg.inv(D) @ ((np.array(world_xyz) - np.array(origin)) / np.array(spacing))
    return np.round(voxel).astype(int)  # (x, y, z)

def window_hu(arr, lo=HU_WIN[0], hi=HU_WIN[1]):
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)

def extract_patch(vol_np, spacing, vx_xyz, patch_mm=PATCH_MM, patch_px=PATCH_PX):
    """Extract 2D axial patch around voxel (x,y,z). Returns uint8 image."""
    z = vx_xyz[2]
    if z < 0 or z >= vol_np.shape[0]: return None
    slice2d = vol_np[z]                  # (H, W) = (y, x)
    # physical half-width in voxels (use x,y spacing)
    half_x = int(round((patch_mm / 2) / spacing[0]))
    half_y = int(round((patch_mm / 2) / spacing[1]))
    cx, cy = vx_xyz[0], vx_xyz[1]
    y0, y1 = cy - half_y, cy + half_y
    x0, x1 = cx - half_x, cx + half_x
    if y0 < 0 or x0 < 0 or y1 > slice2d.shape[0] or x1 > slice2d.shape[1]:
        return None
    crop = slice2d[y0:y1, x0:x1]
    crop = window_hu(crop)
    img = Image.fromarray(crop).resize((patch_px, patch_px), Image.BILINEAR)
    return img

# ── Load CSVs ───────────────────────────────────────────────────────
cand = pd.read_csv(LUNA_DIR / "candidates_V2.csv")
print(f"Loaded candidates_V2.csv: {len(cand):,} rows", flush=True)

# ── Process each subset ─────────────────────────────────────────────
stats = {split: {"nodule": 0, "non_nodule": 0, "patients": 0, "skipped": 0}
         for split in SPLITS.values()}

for subset, split in SPLITS.items():
    print(f"\n{'='*60}\n  Processing {subset} → {split}\n{'='*60}", flush=True)
    # Find mhd files
    subset_dir = LUNA_DIR / subset
    inner = subset_dir / subset
    if not inner.exists(): inner = subset_dir
    mhd_files = sorted(inner.glob("*.mhd"))
    print(f"  {len(mhd_files)} CT volumes found", flush=True)

    (OUT_DIR / split / "nodule").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split / "non_nodule").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, mhd in enumerate(mhd_files, 1):
        uid = mhd.stem
        pat_cand = cand[cand.seriesuid == uid]
        if len(pat_cand) == 0:
            stats[split]["skipped"] += 1
            continue

        try:
            vol = sitk.ReadImage(str(mhd))
            arr = sitk.GetArrayFromImage(vol)   # shape (z, y, x)
            origin    = vol.GetOrigin()
            spacing   = vol.GetSpacing()
            direction = vol.GetDirection()
        except Exception as e:
            print(f"    [!] {uid}: read error — {e}", flush=True)
            stats[split]["skipped"] += 1
            continue

        pos_rows = pat_cand[pat_cand["class"] == 1]
        neg_rows = pat_cand[pat_cand["class"] == 0]
        # Sample negatives at NEG_RATIO per positive (min 3 if patient has 0 positives)
        n_neg_target = max(NEG_RATIO * max(len(pos_rows), 1), 3)
        if len(neg_rows) > n_neg_target:
            neg_rows = neg_rows.sample(n=n_neg_target, random_state=42)

        # Extract positive patches
        for _, r in pos_rows.iterrows():
            vxyz = world_to_voxel((r.coordX, r.coordY, r.coordZ),
                                    origin, spacing, direction)
            img = extract_patch(arr, spacing, vxyz)
            if img is None: continue
            out = OUT_DIR / split / "nodule" / f"{uid}_{int(r.coordX)}_{int(r.coordY)}_{int(r.coordZ)}.png"
            img.save(out)
            stats[split]["nodule"] += 1

        # Extract negative patches
        for _, r in neg_rows.iterrows():
            vxyz = world_to_voxel((r.coordX, r.coordY, r.coordZ),
                                    origin, spacing, direction)
            img = extract_patch(arr, spacing, vxyz)
            if img is None: continue
            out = OUT_DIR / split / "non_nodule" / f"{uid}_{int(r.coordX)}_{int(r.coordY)}_{int(r.coordZ)}.png"
            img.save(out)
            stats[split]["non_nodule"] += 1

        stats[split]["patients"] += 1
        if i % 10 == 0 or i == len(mhd_files):
            dt = time.time() - t0
            print(f"  [{i:>3}/{len(mhd_files)}] "
                  f"nodule={stats[split]['nodule']} non={stats[split]['non_nodule']} "
                  f"| {dt/i:.1f}s/vol", flush=True)

# ── Save stats + summary ────────────────────────────────────────────
print("\n" + "="*60)
print("  EXTRACTION COMPLETE")
print("="*60)
for split, s in stats.items():
    total = s["nodule"] + s["non_nodule"]
    print(f"  {split:<5}  patients={s['patients']:>3}  nodule={s['nodule']:>4}  "
          f"non-nodule={s['non_nodule']:>4}  total={total:>4}")
print("="*60)

summary = {
    "source": "LUNA16 subsets 7-9",
    "patch_mm": PATCH_MM,
    "patch_px": PATCH_PX,
    "neg_ratio": NEG_RATIO,
    "hu_window": list(HU_WIN),
    "splits": stats,
    "split_mapping": SPLITS,
}
(OUT_DIR / "extract_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nSummary: {OUT_DIR / 'extract_summary.json'}")
