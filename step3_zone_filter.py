"""
SepsisScope — Step 3: Zone Filtering (1r to 2r)
================================================
Filters vessel segments from Step 1 keeping only those that fall
within the standard CRAE/CRVE measurement zone: 1r to 2r from
the optic disc centre, where r = disc radius from Step 2.

Filtering method: MAJORITY PIXELS
    A segment is included if more than 50% of its skeleton pixels
    fall within the 1r-2r annular zone. This is the most accurate
    method as it represents where the vessel actually lies.

Usage
-----
    python step3_zone_filter.py --step1_out <folder> --step2_out <folder> --output <folder>

Outputs
-------
    <out>/<stem>_zone.png          Overlay showing 1r-2r zone + kept segments
    <out>/<stem>_filtered.csv      Filtered segments for this image
    <out>/step3_summary.csv        One row per image: n_kept, n_rejected, coverage
"""

import os
import csv
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
from skimage import measure

warnings.filterwarnings("ignore")

SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".bmp"}


# ──────────────────────────────────────────────
# LOAD DISC RESULTS
# ──────────────────────────────────────────────
def load_disc_results(step2_csv):
    """Load disc detection results into a dict keyed by filename stem."""
    disc_map = {}
    with open(step2_csv, newline="") as f:
        for row in csv.DictReader(f):
            disc_map[row["file"]] = {
                "cx":         int(row["disc_cx"]),
                "cy":         int(row["disc_cy"]),
                "r":          int(row["disc_r"]),
                "confidence": float(row["confidence"]),
                "method":     row["method"],
                "flag":       row["flag"],
            }
    return disc_map


# ──────────────────────────────────────────────
# ZONE FILTER — MAJORITY PIXELS METHOD
# ──────────────────────────────────────────────
def filter_segments_by_zone(skeleton, disc, h, w, majority_threshold=0.3):
    """
    Filter using Step 1 segment centroids against the 1r-2r zone.
    Also does a pixel-majority check as secondary confirmation.
    """
    cx, cy, r = disc["cx"], disc["cy"], disc["r"]
    inner_r = r
    outer_r = 2 * r

    Y, X = np.ogrid[:h, :w]
    dist_map = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    zone_mask = (dist_map >= inner_r) & (dist_map <= outer_r)

    labeled = measure.label(skeleton, connectivity=2)
    regions = measure.regionprops(labeled)

    kept     = []
    rejected = []

    for reg in regions:
        coords = reg.coords
        n = len(coords)
        if n < 5:
            continue

        cy_seg, cx_seg = reg.centroid
        centroid_dist = float(np.sqrt((cx_seg - cx) ** 2 + (cy_seg - cy) ** 2))

        # Primary check: centroid in zone
        centroid_in_zone = inner_r <= centroid_dist <= outer_r

        # Secondary check: pixel majority in zone
        in_zone = zone_mask[coords[:, 0], coords[:, 1]].sum()
        zone_fraction = float(in_zone) / n
        pixels_in_zone = zone_fraction > majority_threshold

        # Keep if EITHER condition is met
        include = centroid_in_zone or pixels_in_zone

        seg_data = {
            "segment_id":    reg.label,
            "length_px":     n,
            "centroid_x":    round(cx_seg, 1),
            "centroid_y":    round(cy_seg, 1),
            "centroid_dist": round(centroid_dist, 1),
            "zone_fraction": round(zone_fraction, 3),
            "in_zone":       include,
        }

        if include:
            kept.append(seg_data)
        else:
            rejected.append(seg_data)

    return kept, rejected
    """
    For each connected segment in the skeleton, compute what fraction
    of its pixels fall within the 1r-2r annular zone.
    Keep segments where fraction > majority_threshold.

    Returns:
        kept    — list of segment dicts with zone_fraction added
        rejected — list of rejected segment dicts
    """
    cx, cy, r = disc["cx"], disc["cy"], disc["r"]
    inner_r = r         # 1r
    outer_r = 2 * r     # 2r

    # Pre-compute distance from disc centre for every pixel
    Y, X = np.ogrid[:h, :w]
    dist_map = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Zone mask: True where pixel is in 1r-2r ring
    zone_mask = (dist_map >= inner_r) & (dist_map <= outer_r)

    # Label skeleton into individual segments
    labeled = measure.label(skeleton, connectivity=2)
    regions = measure.regionprops(labeled)

    kept     = []
    rejected = []

    for reg in regions:
        coords = reg.coords  # (N, 2) — (row, col)
        n = len(coords)

        if n < 5:
            continue  # skip tiny fragments

        # Count how many pixels fall in the zone
        in_zone = zone_mask[coords[:, 0], coords[:, 1]].sum()
        zone_fraction = float(in_zone) / n

        # Distance of centroid from disc centre
        cy_seg, cx_seg = reg.centroid
        centroid_dist = float(np.sqrt((cx_seg - cx) ** 2 + (cy_seg - cy) ** 2))

        seg_data = {
            "segment_id":    reg.label,
            "length_px":     n,
            "centroid_x":    round(cx_seg, 1),
            "centroid_y":    round(cy_seg, 1),
            "centroid_dist": round(centroid_dist, 1),
            "zone_fraction": round(zone_fraction, 3),
            "in_zone":       zone_fraction > majority_threshold,
        }

        if zone_fraction > majority_threshold:
            kept.append(seg_data)
        else:
            rejected.append(seg_data)

    return kept, rejected


# ──────────────────────────────────────────────
# MERGE WITH STEP 1 MEASUREMENTS
# ──────────────────────────────────────────────
def merge_with_step1(kept_segments, step1_csv_path):
    """
    Merge zone-filtered segments with Step 1 measurements
    by matching centroids (within 5px tolerance).
    Segment IDs differ between Step 1 and Step 3 re-labeling,
    so we match spatially instead.
    """
    if not Path(step1_csv_path).exists():
        return kept_segments

    # Load step1 segments
    step1_rows = []
    with open(step1_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            step1_rows.append(row)

    if not step1_rows:
        return kept_segments

    # Build array of step1 centroids for fast matching
    s1_cx = np.array([float(r['centroid_x']) for r in step1_rows])
    s1_cy = np.array([float(r['centroid_y']) for r in step1_rows])

    merged = []
    tolerance = 55.0  # px — centroids within 8px = same vessel

    for seg in kept_segments:
        cx_seg = float(seg['centroid_x'])
        cy_seg = float(seg['centroid_y'])

        # Find closest Step 1 segment by centroid distance
        dists = np.sqrt((s1_cx - cx_seg)**2 + (s1_cy - cy_seg)**2)
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist <= tolerance:
            # Match found — merge Step 1 measurements in
            s1 = step1_rows[min_idx]
            combined = {**seg}
            combined['width_px']    = float(s1['width_px'])
            combined['r_mean']      = float(s1['r_mean'])
            combined['g_mean']      = float(s1['g_mean'])
            combined['b_mean']      = float(s1['b_mean'])
            combined['color_score'] = float(s1['color_score'])
            combined['contrast']    = float(s1['contrast'])
            merged.append(combined)
        else:
            # No match — keep segment but flag missing measurements
            combined = {**seg}
            combined['width_px']    = None
            combined['r_mean']      = None
            combined['g_mean']      = None
            combined['b_mean']      = None
            combined['color_score'] = None
            combined['contrast']    = None
            merged.append(combined)

    return merged


# ──────────────────────────────────────────────
# PROCESS ONE IMAGE
# ──────────────────────────────────────────────
def process_image(img_path, step1_out, disc_info, out_dir, img_rgb=None):
    stem = Path(img_path).stem

    # Load skeleton
    skel_path = Path(step1_out) / f"{stem}_skeleton.png"
    if not skel_path.exists():
        return {"file": stem, "status": "ERROR: skeleton not found"}

    skeleton_img = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
    skeleton = (skeleton_img > 127).astype(np.uint8)
    h, w = skeleton.shape

    # Get disc info
    disc = disc_info.get(stem)
    if disc is None:
        return {"file": stem, "status": "ERROR: no disc info — run Step 2 first"}

    # Skip low confidence disc detections
    if disc["flag"] == "LOW_CONFIDENCE_FALLBACK":
        return {"file": stem, "status": "SKIPPED: low confidence disc detection"}

    # Zone filter
    kept, rejected = filter_segments_by_zone(skeleton, disc, h, w,  majority_threshold=0.3)

    # Merge with Step 1 measurements
    step1_csv = Path(step1_out) / f"{stem}_segments.csv"
    kept = merge_with_step1(kept, step1_csv)

    # Save filtered segments CSV
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if kept:
            all_keys = list(kept[0].keys())
            for seg in kept[1:]:
                for k in seg.keys():
                    if k not in all_keys:
                        all_keys.append(k)
            with open(out / f"{stem}_filtered.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(kept)

    # Save zone overlay
    if img_rgb is not None:
        overlay = img_rgb.copy()
        cx, cy, r = disc["cx"], disc["cy"], disc["r"]

        # Draw measurement zone rings
        cv2.circle(overlay, (cx, cy), r,     [100, 100, 255], 1)  # 1r inner
        cv2.circle(overlay, (cx, cy), 2 * r, [100, 100, 255], 1)  # 2r outer
        cv2.circle(overlay, (cx, cy), 3,     [255,   0,   0], -1) # disc centre

        # Colour kept segments GREEN, rejected segments RED
        labeled = measure.label(skeleton, connectivity=2)
        kept_ids = {seg["segment_id"] for seg in kept}

        for reg in measure.regionprops(labeled):
            coords = reg.coords
            if reg.label in kept_ids:
                overlay[coords[:, 0], coords[:, 1]] = [0, 220, 80]   # green = kept
            else:
                overlay[coords[:, 0], coords[:, 1]] = [180, 60, 60]  # red = rejected

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Label
        label = f"kept={len(kept)}  rejected={len(rejected)}  disc r={r}px"
        cv2.putText(overlay_bgr, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(out / f"{stem}_zone.png"), overlay_bgr)

    return {
        "file":          stem,
        "status":        "OK",
        "n_kept":        len(kept),
        "n_rejected":    len(rejected),
        "disc_r":        disc["r"],
        "disc_cx":       disc["cx"],
        "disc_cy":       disc["cy"],
        "disc_conf":     disc["confidence"],
    }


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────
def collect_images(input_dir):
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED:
                paths.append(Path(root) / f)
    return sorted(paths)


def run_batch(images_dir, step1_out, step2_csv, out_dir):
    images = collect_images(images_dir)
    if not images:
        print(f"[ERROR] No images found in: {images_dir}")
        return

    disc_info = load_disc_results(step2_csv)

    print(f"\n{'='*60}")
    print(f"  SepsisScope — Step 3: Zone Filtering (1r to 2r)")
    print(f"{'='*60}")
    print(f"  Images folder : {images_dir}")
    print(f"  Step 1 output : {step1_out}")
    print(f"  Step 2 CSV    : {step2_csv}")
    print(f"  Output folder : {out_dir}")
    print(f"  Images found  : {len(images)}")
    print(f"  Method        : Majority pixels (>50% in zone)")
    print(f"{'='*60}\n")

    results = []
    for i, img_path in enumerate(images, 1):
        stem = img_path.stem

        # Load image for overlay
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr is not None else None

        res = process_image(img_path, step1_out, disc_info, out_dir, img_rgb)
        results.append(res)

        status = res["status"]
        if status == "OK":
            print(f"  [{i:4d}/{len(images)}] {stem:<40} "
                  f"kept={res['n_kept']:3d}  rejected={res['n_rejected']:3d}")
        else:
            print(f"  [{i:4d}/{len(images)}] {stem:<40} {status}")

    # Summary CSV
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "step3_summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    ok      = sum(1 for r in results if r["status"] == "OK")
    skipped = sum(1 for r in results if "SKIPPED" in r["status"])
    errors  = sum(1 for r in results if "ERROR" in r["status"])
    kept_counts = [r["n_kept"] for r in results if r["status"] == "OK"]

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Succeeded : {ok}")
    print(f"  Skipped   : {skipped}  (low confidence disc)")
    print(f"  Errors    : {errors}")
    if kept_counts:
        print(f"  Segments kept — "
              f"min:{min(kept_counts)}  "
              f"max:{max(kept_counts)}  "
              f"avg:{sum(kept_counts)//len(kept_counts)}")
    print(f"  Summary   → {summary_path}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="SepsisScope Step 3 — Zone filtering 1r to 2r")
    p.add_argument("--images",    required=True,
                   help="Folder containing original fundus images")
    p.add_argument("--step1_out", required=True,
                   help="Folder containing Step 1 skeleton + segment CSV outputs")
    p.add_argument("--step2_csv", required=True,
                   help="Path to step2_disc_results.csv from Step 2")
    p.add_argument("--output",    required=True,
                   help="Folder to write filtered CSVs and overlays into")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(args.images, args.step1_out, args.step2_csv, args.output)
