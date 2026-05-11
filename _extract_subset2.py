"""Extract patches from LUNA16 subset2 → add to luna16_patches/train.
Idempotent: skips if subset2 patches are already present."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

LUNA_DIR  = Path("luna16")
PATCH_DIR = Path("luna16_patches")
SUBSET    = "subset2"
SPLIT     = "train"

PATCH_MM = 50; PATCH_PX = 64; NEG_RATIO = 3
HU_WIN   = (-1000, 400)

def world_to_voxel(world_xyz, origin, spacing, direction):
    D = np.array(direction).reshape(3, 3)
    voxel = np.linalg.inv(D) @ ((np.array(world_xyz) - np.array(origin)) / np.array(spacing))
    return np.round(voxel).astype(int)

def window_hu(arr, lo=HU_WIN[0], hi=HU_WIN[1]):
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)

def extract_patch(vol_np, spacing, vx_xyz):
    z = vx_xyz[2]
    if z < 0 or z >= vol_np.shape[0]: return None
    s = vol_np[z]
    half_x = int(round((PATCH_MM / 2) / spacing[0]))
    half_y = int(round((PATCH_MM / 2) / spacing[1]))
    cx, cy = vx_xyz[0], vx_xyz[1]
    y0, y1 = cy - half_y, cy + half_y
    x0, x1 = cx - half_x, cx + half_x
    if y0 < 0 or x0 < 0 or y1 > s.shape[0] or x1 > s.shape[1]:
        return None
    crop = window_hu(s[y0:y1, x0:x1])
    return Image.fromarray(crop).resize((PATCH_PX, PATCH_PX), Image.BILINEAR)

cand = pd.read_csv(LUNA_DIR / "candidates_V2.csv")
print(f"Loaded {len(cand):,} candidates", flush=True)

(PATCH_DIR / SPLIT / "nodule").mkdir(parents=True, exist_ok=True)
(PATCH_DIR / SPLIT / "non_nodule").mkdir(parents=True, exist_ok=True)

mhd_files = sorted((LUNA_DIR / SUBSET).glob("*.mhd"))
print(f"Found {len(mhd_files)} CTs in {SUBSET}", flush=True)

# Track existing files to avoid double-counting
existing = {f.stem.split('_')[0] for f in (PATCH_DIR/SPLIT/"nodule").glob("*.png")}
existing |= {f.stem.split('_')[0] for f in (PATCH_DIR/SPLIT/"non_nodule").glob("*.png")}

n_pos, n_neg, n_skip, t0 = 0, 0, 0, time.time()
for i, mhd in enumerate(mhd_files, 1):
    uid = mhd.stem
    if uid in existing:
        n_skip += 1
        continue
    pat = cand[cand.seriesuid == uid]
    if len(pat) == 0: continue
    try:
        vol = sitk.ReadImage(str(mhd))
        arr = sitk.GetArrayFromImage(vol)
        origin, spacing, direction = vol.GetOrigin(), vol.GetSpacing(), vol.GetDirection()
    except Exception as e:
        print(f"  [!] {uid}: read error — {e}", flush=True); continue
    pos = pat[pat["class"] == 1]
    neg = pat[pat["class"] == 0]
    n_neg_target = max(NEG_RATIO * max(len(pos), 1), 3)
    if len(neg) > n_neg_target: neg = neg.sample(n=n_neg_target, random_state=42)
    for _, r in pos.iterrows():
        vx = world_to_voxel((r.coordX, r.coordY, r.coordZ), origin, spacing, direction)
        img = extract_patch(arr, spacing, vx)
        if img is not None:
            img.save(PATCH_DIR/SPLIT/"nodule"/f"{uid}_{int(r.coordX)}_{int(r.coordY)}_{int(r.coordZ)}.png")
            n_pos += 1
    for _, r in neg.iterrows():
        vx = world_to_voxel((r.coordX, r.coordY, r.coordZ), origin, spacing, direction)
        img = extract_patch(arr, spacing, vx)
        if img is not None:
            img.save(PATCH_DIR/SPLIT/"non_nodule"/f"{uid}_{int(r.coordX)}_{int(r.coordY)}_{int(r.coordZ)}.png")
            n_neg += 1
    if i % 10 == 0:
        print(f"  [{i:>3}/{len(mhd_files)}]  +nod={n_pos}  +non={n_neg}  "
              f"skip={n_skip}  {(time.time()-t0)/i:.1f}s/vol", flush=True)

print(f"\nDone. Added: nodule={n_pos}  non_nodule={n_neg}  (skipped {n_skip} already-present)")
print(f"Train split now has:")
n_nod = len(list((PATCH_DIR/SPLIT/'nodule').glob('*.png')))
n_non = len(list((PATCH_DIR/SPLIT/'non_nodule').glob('*.png')))
print(f"  nodule={n_nod}  non_nodule={n_non}  total={n_nod+n_non}")
