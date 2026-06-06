"""
SepsisScope — Step 1: Vessel Segmentation + Skeletonisation
============================================================
Processes a folder of fundus images (any mix of .tif/.tiff/.png/.jpg/.ppm).

For every image it produces:
  <out_dir>/<stem>_mask.png        — binary vessel mask
  <out_dir>/<stem>_skeleton.png    — single-pixel centreline skeleton
  <out_dir>/<stem>_overlay.png     — colour overlay (vessels in green on original)
  <out_dir>/<stem>_segments.csv    — one row per vessel segment between branch points

Usage
-----
  python step1_segment_skeleton.py --input <folder> --output <folder> [options]

Key options
-----------
  --sigmas      Frangi sigma range  (default: 1 5)   e.g. --sigmas 1 8
  --threshold   Vesselness cutoff   (default: 0.02)
  --min_area    Min segment pixels  (default: 30)
  --workers     Parallel processes  (default: 4)
  --no_save_overlay   Skip overlay images (faster for large batches)

Examples
--------
  # DRIVE dataset
  python step1_segment_skeleton.py --input C:/SepsisScope/data/DRIVE --output C:/SepsisScope/out/step1

  # STARE dataset, full sigma range, 8 parallel workers
  python step1_segment_skeleton.py --input C:/SepsisScope/data/STARE --output C:/SepsisScope/out/step1 --sigmas 1 8 --workers 8
"""

import os
import sys
import csv
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from skimage.filters import frangi
from skimage import exposure, morphology, measure

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# SUPPORTED IMAGE EXTENSIONS
# ──────────────────────────────────────────────
SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".bmp"}


# ──────────────────────────────────────────────
# RETINAL CIRCULAR MASK
# ──────────────────────────────────────────────
def make_retinal_mask(h, w, img_gray=None, shrink_px=15):
    """
    Auto-detects the true retinal disc centre and radius
    by finding the largest bright circular region in the image.
    Falls back to image centre if detection fails.
    Erodes boundary by 7px to prevent hard edge artefacts.
    """
    cx, cy, radius = w // 2, h // 2, min(h, w) // 2  # fallback defaults

    if img_gray is not None:
        try:
            # Threshold to isolate the lit retinal area
            _, bright = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)

            # Fill holes so the disc is one solid blob
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

            # Find largest contour = retinal boundary
            contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                (fx, fy), fr = cv2.minEnclosingCircle(largest)
                cx, cy, radius = int(fx), int(fy), int(fr)
        except Exception:
            pass  # silently fall back to defaults

    # Shrink inward and erode to kill edge transition artefacts
    radius = max(10, radius - shrink_px)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = (dist <= radius).astype(np.uint8)

    # Erode boundary to soften hard edge before Frangi sees it
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# ──────────────────────────────────────────────
# CORE PROCESSING FUNCTION (one image)
# ──────────────────────────────────────────────
def process_image(args):
    """
    Full Step 1 pipeline for a single fundus image.
    Returns a dict summarising the result (or an error string).
    """
    img_path, out_dir, cfg = args
    stem = Path(img_path).stem

    try:
        # ── Load ──────────────────────────────────────────────────
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"file": stem, "status": "ERROR: could not read image"}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # ── Green channel ─────────────────────────────────────────
        green = img_rgb[:, :, 1].astype(np.float32) / 255.0

        # ── Circular retinal mask ─────────────────────────────────
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ret_mask = make_retinal_mask(h, w, img_gray=gray, shrink_px=15)
        green_masked = green * ret_mask

        # ── CLAHE contrast enhancement ────────────────────────────
        green_uint8 = (green_masked * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        green_clahe = clahe.apply(green_uint8).astype(np.float32) / 255.0
        green_clahe = green_clahe * ret_mask          # re-apply mask after CLAHE

        # ── Frangi vesselness filter ──────────────────────────────
        sigmas = range(cfg["sigma_low"], cfg["sigma_high"] + 1)
        vessel_map = frangi(
            green_clahe,
            sigmas=sigmas,
            black_ridges=True,   # vessels are darker than background in green ch
            alpha=0.5,
            beta=0.5,
            gamma=15,
        )

        # Rescale to 0–1 and re-mask
        vessel_map = exposure.rescale_intensity(vessel_map, out_range=(0, 1))
        vessel_map = vessel_map * ret_mask

        # ── Threshold → binary mask ───────────────────────────────
        vessel_binary = (vessel_map > cfg["threshold"]).astype(np.uint8)

        # Morphological clean-up: remove tiny speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vessel_binary = cv2.morphologyEx(vessel_binary, cv2.MORPH_OPEN, kernel)
        vessel_binary = vessel_binary * ret_mask      # final mask enforcement

        # ── Skeletonise → single-pixel centrelines ────────────────
        skeleton = morphology.skeletonize(vessel_binary.astype(bool)).astype(np.uint8)

        # ── Detect branch points ──────────────────────────────────
        # A branch point is a skeleton pixel with 3+ neighbours
        branch_mask = _find_branch_points(skeleton)

        # Remove branch points from skeleton, leaving individual segments
        skeleton_no_branch = skeleton.copy()
        skeleton_no_branch[branch_mask == 1] = 0

        # ── Label individual segments ─────────────────────────────
        labeled = measure.label(skeleton_no_branch, connectivity=2)
        regions = measure.regionprops(labeled, intensity_image=img_rgb[:, :, 0])

        # ── Extract per-segment properties ───────────────────────
        # (Width is estimated from the original thick mask — not the skeleton)
        segments = []
        for reg in regions:
            if reg.area < cfg["min_area"]:
                continue

            coords = reg.coords  # (N, 2) array of (row, col)

            # Average width: cross-check against the original binary mask
            # For each skeleton pixel, find the local vessel thickness via
            # distance transform
            seg_mask = np.zeros((h, w), dtype=np.uint8)
            seg_mask[coords[:, 0], coords[:, 1]] = 1

            # Width = 2 × mean distance-transform value at skeleton pixels
            dist_transform = cv2.distanceTransform(vessel_binary, cv2.DIST_L2, 5)
            widths_at_skel = dist_transform[coords[:, 0], coords[:, 1]]
            avg_width_px = float(2.0 * widths_at_skel.mean())

            # Colour intensities from the original RGB image
            r_vals = img_rgb[:, :, 0][coords[:, 0], coords[:, 1]]
            g_vals = img_rgb[:, :, 1][coords[:, 0], coords[:, 1]]
            b_vals = img_rgb[:, :, 2][coords[:, 0], coords[:, 1]]

            r_mean = float(r_vals.mean())
            g_mean = float(g_vals.mean())
            b_mean = float(b_vals.mean())

            # Colour score: arterioles redder, venules darker/greener
            color_score = r_mean / (g_mean + b_mean + 1e-6)

            # Vessel contrast: how different is the vessel from its local background
            local_bg = _local_background_mean(img_rgb[:, :, 1], coords)
            contrast = float(abs(g_mean - local_bg))

            # Segment length (px): number of skeleton pixels
            length_px = int(reg.area)

            # Centroid
            cy_seg, cx_seg = reg.centroid

            segments.append({
                "segment_id":  reg.label,
                "length_px":   length_px,
                "width_px":    round(avg_width_px, 3),
                "r_mean":      round(r_mean, 3),
                "g_mean":      round(g_mean, 3),
                "b_mean":      round(b_mean, 3),
                "color_score": round(color_score, 5),
                "contrast":    round(contrast, 3),
                "centroid_x":  round(cx_seg, 1),
                "centroid_y":  round(cy_seg, 1),
            })

        n_segments = len(segments)

        # ── Save outputs ──────────────────────────────────────────
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Binary mask
        cv2.imwrite(str(out / f"{stem}_mask.png"), vessel_binary * 255)

        # Skeleton
        cv2.imwrite(str(out / f"{stem}_skeleton.png"), skeleton * 255)

        # Colour overlay (vessels highlighted in bright green on original)
        if cfg["save_overlay"]:
            overlay = img_rgb.copy()
            overlay[vessel_binary == 1] = [0, 220, 80]      # green = vessels
            overlay[branch_mask == 1]  = [255, 80,  80]     # red   = branch points
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay_bgr)

        # Segments CSV
        csv_path = out / f"{stem}_segments.csv"
        if segments:
            fieldnames = list(segments[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(segments)

        # Coverage stat
        coverage = 100.0 * vessel_binary.sum() / max(ret_mask.sum(), 1)

        return {
            "file":        stem,
            "status":      "OK",
            "n_segments":  n_segments,
            "coverage_%":  round(coverage, 2),
            "h":           h,
            "w":           w,
        }

    except Exception as e:
        return {"file": stem, "status": f"ERROR: {e}"}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _find_branch_points(skeleton):
    """
    Returns a binary mask where 1 = branch point (3+ neighbours).
    Uses 8-connectivity neighbour count via convolution.
    """
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbour_count = convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0)
    # Branch point: skeleton pixel with 3 or more neighbours
    branch = (skeleton == 1) & (neighbour_count >= 3)
    return branch.astype(np.uint8)


def _local_background_mean(channel, coords, radius=5):
    """
    Estimate mean intensity of pixels near the segment but not on it.
    Used to compute vessel contrast against local background.
    """
    h, w = channel.shape
    seg_set = set(map(tuple, coords))
    bg_vals = []
    step = max(1, len(coords) // 20)   # sample at most 20 points for speed
    for r, c in coords[::step]:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seg_set:
                    bg_vals.append(channel[nr, nc])
    return float(np.mean(bg_vals)) if bg_vals else 128.0


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────
def collect_images(input_dir):
    """Recursively collect all supported image files."""
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED:
                paths.append(Path(root) / f)
    return sorted(paths)


def run_batch(input_dir, out_dir, cfg):
    images = collect_images(input_dir)
    if not images:
        print(f"[ERROR] No supported images found in: {input_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  SepsisScope — Step 1: Segmentation + Skeletonisation")
    print(f"{'='*60}")
    print(f"  Input folder : {input_dir}")
    print(f"  Output folder: {out_dir}")
    print(f"  Images found : {len(images)}")
    print(f"  Frangi sigmas: {cfg['sigma_low']}–{cfg['sigma_high']}")
    print(f"  Threshold    : {cfg['threshold']}")
    print(f"  Min segment  : {cfg['min_area']} px")
    print(f"  Workers      : {cfg['workers']}")
    print(f"{'='*60}\n")

    job_args = [(img, out_dir, cfg) for img in images]

    results = []
    if cfg["workers"] == 1:
        # Single-threaded: easier to debug
        for i, arg in enumerate(job_args, 1):
            res = process_image(arg)
            results.append(res)
            status = res["status"]
            if status == "OK":
                print(f"  [{i:4d}/{len(images)}] {res['file']:<40} "
                      f"segs={res['n_segments']:4d}  cov={res['coverage_%']:5.1f}%")
            else:
                print(f"  [{i:4d}/{len(images)}] {res['file']:<40} {status}")
    else:
        with Pool(processes=cfg["workers"]) as pool:
            for i, res in enumerate(pool.imap(process_image, job_args), 1):
                results.append(res)
                status = res["status"]
                if status == "OK":
                    print(f"  [{i:4d}/{len(images)}] {res['file']:<40} "
                          f"segs={res['n_segments']:4d}  cov={res['coverage_%']:5.1f}%")
                else:
                    print(f"  [{i:4d}/{len(images)}] {res['file']:<40} {status}")

    # ── Summary CSV ───────────────────────────────────────────────
    summary_path = Path(out_dir) / "step1_summary.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    ok      = sum(1 for r in results if r["status"] == "OK")
    errors  = len(results) - ok
    print(f"\n{'='*60}")
    print(f"  Done.  {ok} succeeded,  {errors} failed.")
    print(f"  Summary saved → {summary_path}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SepsisScope Step 1 — Vessel segmentation & skeletonisation")
    p.add_argument("--input",    required=True,  help="Folder containing fundus images")
    p.add_argument("--output",   required=True,  help="Folder to write results into")
    p.add_argument("--sigmas",   nargs=2, type=int, default=[1, 5],
                   metavar=("LOW", "HIGH"),
                   help="Frangi sigma range (default: 1 5). Use 1 8 for final runs.")
    p.add_argument("--threshold",type=float, default=0.02,
                   help="Vesselness threshold (default: 0.02)")
    p.add_argument("--min_area", type=int, default=30,
                   help="Min skeleton pixels per segment (default: 30)")
    p.add_argument("--workers",  type=int, default=min(4, cpu_count()),
                   help="Parallel worker processes (default: 4 or cpu count)")
    p.add_argument("--no_save_overlay", action="store_true",
                   help="Skip saving colour overlay images (faster)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {
        "sigma_low":    args.sigmas[0],
        "sigma_high":   args.sigmas[1],
        "threshold":    args.threshold,
        "min_area":     args.min_area,
        "workers":      args.workers,
        "save_overlay": not args.no_save_overlay,
    }
    run_batch(args.input, args.output, cfg)
