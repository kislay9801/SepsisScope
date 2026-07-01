"""Unified loader for the 8-benchmark retinal-vessel-fundus-dataset-collection/.

Binary vessel-segmentation ground truth only. Artery/vein labels exist for
LES-AV alone (handled separately in train.py); every other benchmark here only
ships a single vessel mask. Each dataset uses its own images<->mask naming
convention between `images/` and its mask folder — the STEM_FN below encodes
each one, verified against the actual folder contents.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COLLECTION = os.path.join(ROOT, "retinal-vessel-fundus-dataset-collection")

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# (dataset folder, mask subfolder, image-stem -> mask-stem)
DATASET_SPECS = [
    ("DRIVE",    "masks",   lambda s: s.replace("_training", "_manual1")),
    ("STARE",    "masks_1", lambda s: s + "_1stHO"),
    ("CHASEDB1", "masks_1", lambda s: s + "_1stHO"),
    ("HRF",      "masks",   lambda s: s),
    ("FIVES",    "masks",   lambda s: s),
    ("LES-AV",   "masks",   lambda s: s),
    ("RETA",     "masks",   lambda s: s + "_vessel"),
    ("TRENDS",   "masks",   lambda s: s + "_SEG"),
]


def _find_ci(dir_path, stem, ext_hint=None):
    """Case-insensitive filename lookup by stem (tries ext_hint, then IMG_EXTS)."""
    if not os.path.isdir(dir_path):
        return None
    exts = ([ext_hint] if ext_hint else []) + list(IMG_EXTS)
    lower_map = {f.lower(): f for f in os.listdir(dir_path)}
    for ext in exts:
        cand = (stem + ext).lower()
        if cand in lower_map:
            return os.path.join(dir_path, lower_map[cand])
    return None


def list_vessel_pairs(verbose=True):
    """Return [(image_path, mask_path, dataset_name), ...] across all 8 benchmarks."""
    pairs = []
    skipped = 0
    for name, mask_sub, stem_fn in DATASET_SPECS:
        img_dir = os.path.join(COLLECTION, name, "images")
        mask_dir = os.path.join(COLLECTION, name, mask_sub)
        if not os.path.isdir(img_dir):
            if verbose:
                print(f"  [datasets] {name}: images/ not found, skipping")
            continue
        n_before = len(pairs)
        for fname in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMG_EXTS:
                continue
            mask_path = _find_ci(mask_dir, stem_fn(stem))
            if mask_path is None:
                skipped += 1
                continue
            pairs.append((os.path.join(img_dir, fname), mask_path, name))
        if verbose:
            print(f"  [datasets] {name}: {len(pairs) - n_before} pairs")
    if verbose and skipped:
        print(f"  [datasets] {skipped} images skipped (no matching mask found)")
    return pairs


if __name__ == "__main__":
    p = list_vessel_pairs()
    print(f"total: {len(p)} pairs")
