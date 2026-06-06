"""
SepsisScope — Step 2: Optic Disc Detection
===========================================
Detects the optic disc centre (x, y) and radius (r) for every fundus image.

Method 1 — Vessel convergence (primary)
    Reads the skeleton + segments CSV from Step 1.
    For each vessel segment, fits a line through its proximal 20% of pixels
    and extrapolates inward. Votes accumulate in a 2D grid.
    The peak of the vote map = disc centre.
    Confidence = peak sharpness (how focused the convergence is).

Method 2 — Positional prior (fallback, used when confidence < threshold)
    Optic disc is almost always in the left 1/3 or right 1/3 of the image.
    Picks the side with higher mean brightness in the green channel.
    Returns a large uncertainty radius and flags the image in the CSV.

Usage
-----
    python step2_disc_detect.py --step1_out <folder> --images <folder> --output <folder>

Outputs
-------
    <out>/<stem>_disc.png          Overlay showing detected disc circle + centre
    <out>/step2_disc_results.csv   One row per image:
                                   disc_cx, disc_cy, disc_radius,
                                   method, confidence, flag
"""

import os
import csv
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from skimage import measure

warnings.filterwarnings("ignore")

SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".bmp"}

# Confidence threshold below which we fall back to Method 3
CONFIDENCE_THRESHOLD = 0.25

# Typical disc radius as fraction of image min dimension
DISC_RADIUS_FRACTION = 0.08   # ~8% of image width — standard across datasets


# ──────────────────────────────────────────────────────────────
# METHOD 1 — VESSEL CONVERGENCE
# ──────────────────────────────────────────────────────────────
def detect_disc_convergence(skeleton, segments_csv, h, w):
    """
    Vessel density approach:
    The optic disc is where vessel density is highest.
    Blur the skeleton heavily to create a density map,
    find the peak — that's the disc centre.
    """
    if skeleton is None or skeleton.sum() < 10:
        return None, None, 0.0

    # Cast skeleton to float
    skel_float = skeleton.astype(np.float32)

    # Heavy Gaussian blur creates a vessel density heatmap
    # Large kernel = looks for regional concentration not local
    density = cv2.GaussianBlur(skel_float, (151, 151), 40)

    # Find peak density location
    _, max_val, _, max_loc = cv2.minMaxLoc(density)
    cx, cy = max_loc  # (col, row) = (x, y)

    # Confidence: how sharp is the peak relative to mean
    mean_val = density.mean()
    confidence = float(max_val / (mean_val + 1e-6)) / 5.0
    confidence = min(1.0, max(0.0, confidence))

    return int(cx), int(cy), confidence

# ──────────────────────────────────────────────────────────────
# METHOD 3 — POSITIONAL PRIOR FALLBACK
# ──────────────────────────────────────────────────────────────
def detect_disc_brightness(img_rgb, h, w):
    """
    Brightness-guided disc detection.
    The optic disc is the brightest, most compact circular region.
    Works when vessel density fails (disc near edge, sparse vessels).
    """
    # Use green channel — best contrast for disc vs background
    green = img_rgb[:, :, 1].astype(np.float32)

    # Blur to suppress noise and small bright lesions
    bright_map = cv2.GaussianBlur(green, (61, 61), 20)

    # Suppress the image centre — if vessel density already failed,
    # the disc is likely NOT in the centre
    # Apply a gentle centre-suppression mask
    cy_img, cx_img = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_centre = np.sqrt((X - cx_img)**2 + (Y - cy_img)**2)
    # Suppress pixels within 20% of image centre
    centre_suppress = np.clip(dist_from_centre / (0.2 * min(h, w)), 0, 1)
    bright_map = bright_map * centre_suppress

    # Find peak brightness location
    _, max_val, _, max_loc = cv2.minMaxLoc(bright_map)
    cx, cy = max_loc

    # Confidence based on how bright the peak is vs surroundings
    mean_val = bright_map.mean()
    confidence = float(max_val / (mean_val + 1e-6)) / 8.0
    confidence = min(1.0, max(0.0, confidence))

    return int(cx), int(cy), confidence

def detect_disc_positional(img_rgb, h, w):
    """
    Disc is in the left 1/3 or right 1/3 of the image.
    Pick the side with higher mean brightness in green channel.
    Returns (cx, cy, confidence=0.0) — always flagged as fallback.
    """
    green = img_rgb[:, :, 1].astype(np.float32)
    third = w // 3

    left_mean  = green[:, :third].mean()
    right_mean = green[:, -third:].mean()

    if left_mean >= right_mean:
        # Disc likely on left
        cx = third // 2
    else:
        # Disc likely on right
        cx = w - third // 2

    cy = h // 2  # vertically centred as prior

    return int(cx), int(cy), 0.0


# ──────────────────────────────────────────────────────────────
# ESTIMATE DISC RADIUS
# ──────────────────────────────────────────────────────────────
def estimate_disc_radius(img_rgb, cx, cy, h, w):
    """
    Once we have the centre, estimate the disc radius by looking at
    how far brightness stays high around that centre point.
    Falls back to a fraction-based estimate if this fails.
    """
    default_r = int(DISC_RADIUS_FRACTION * min(h, w))

    try:
        green = img_rgb[:, :, 1].astype(np.float32)

        # Sample brightness at increasing radii from centre
        centre_brightness = green[cy, cx]
        threshold = centre_brightness * 0.45  # disc edge is ~65% of centre brightness

        radii_tested = range(5, int(0.25 * min(h, w)), 2)
        disc_r = default_r

        for r in radii_tested:
            # Sample 16 points on a circle of this radius
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring_vals = []
            for angle in angles:
                sr = int(cy + r * np.sin(angle))
                sc = int(cx + r * np.cos(angle))
                if 0 <= sr < h and 0 <= sc < w:
                    ring_vals.append(green[sr, sc])

            if not ring_vals:
                continue

            if np.mean(ring_vals) < threshold:
                disc_r = r
                break

        # Cap: disc radius can't exceed 15% of image min dimension
        max_allowed = int(0.15 * min(h, w))
        return max(15, min(disc_r, max_allowed))

    except Exception:
        return default_r


# ──────────────────────────────────────────────────────────────
# CORE PROCESSING FUNCTION (one image)
# ──────────────────────────────────────────────────────────────
def process_image(args):
    img_path, step1_out, out_dir, cfg = args
    stem = Path(img_path).stem

    try:
        # ── Load original image ───────────────────────────────
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"file": stem, "status": "ERROR: could not read image"}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # ── Load skeleton from Step 1 ─────────────────────────
        skel_path = Path(step1_out) / f"{stem}_skeleton.png"
        if not skel_path.exists():
            return {"file": stem, "status": "ERROR: skeleton not found — run Step 1 first"}

        skeleton = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
        skeleton = (skeleton > 127).astype(np.uint8)

        # ── Method 1: Vessel convergence ──────────────────────
        cx, cy, confidence = detect_disc_convergence(skeleton, None, h, w)

        method = "convergence"
        flag   = ""

        if cx is None or confidence < CONFIDENCE_THRESHOLD:
        # ── Method 2b: Brightness-guided refinement ───────
            cx, cy, confidence = detect_disc_brightness(img_rgb, h, w)
            method = "brightness"
            flag   = ""

        if confidence < CONFIDENCE_THRESHOLD:
    # ── Method 3: Positional prior (last resort) ──────
            cx, cy, confidence = detect_disc_positional(img_rgb, h, w)
            method = "positional_prior"
            flag   = "LOW_CONFIDENCE_FALLBACK"

        # ── Estimate disc radius ──────────────────────────────
        disc_r = estimate_disc_radius(img_rgb, cx, cy, h, w)

        # ── Save overlay ─────────────────────────────────────
        # ── Save overlay ──────────────────────────────────────────────
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        overlay = img_rgb.copy()

        cv2.circle(overlay, (cx, cy), disc_r,     [255, 255,   0], 2)
        cv2.circle(overlay, (cx, cy), 4,           [255,   0,   0], -1)
        cv2.circle(overlay, (cx, cy), disc_r,     [100, 100, 255], 1)
        cv2.circle(overlay, (cx, cy), 2 * disc_r, [100, 100, 255], 1)

        label = f"disc r={disc_r}px  conf={confidence:.2f}  [{method}]"

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(overlay_bgr, label, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_disc.png"), overlay_bgr)

        return {
            "file":       stem,
            "status":     "OK",
            "disc_cx":    cx,
            "disc_cy":    cy,
            "disc_r":     disc_r,
            "confidence": round(confidence, 4),
            "method":     method,
            "flag":       flag,
            "img_w":      w,
            "img_h":      h,
        }

    except Exception as e:
        return {"file": stem, "status": f"ERROR: {e}"}


# ──────────────────────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────────────────────
def collect_images(input_dir):
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED:
                paths.append(Path(root) / f)
    return sorted(paths)


def run_batch(images_dir, step1_out, out_dir, cfg):
    images = collect_images(images_dir)
    if not images:
        print(f"[ERROR] No images found in: {images_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  SepsisScope — Step 2: Optic Disc Detection")
    print(f"{'='*60}")
    print(f"  Images folder : {images_dir}")
    print(f"  Step 1 output : {step1_out}")
    print(f"  Output folder : {out_dir}")
    print(f"  Images found  : {len(images)}")
    print(f"  Conf threshold: {CONFIDENCE_THRESHOLD}")
    print(f"{'='*60}\n")

    job_args = [(img, step1_out, out_dir, cfg) for img in images]

    results = []
    for i, arg in enumerate(job_args, 1):
        res = process_image(arg)
        results.append(res)
        if res["status"] == "OK":
            flag_str = f"  *** {res['flag']}" if res["flag"] else ""
            print(f"  [{i:4d}/{len(images)}] {res['file']:<35} "
                  f"centre=({res['disc_cx']:3d},{res['disc_cy']:3d})  "
                  f"r={res['disc_r']:3d}  "
                  f"conf={res['confidence']:.2f}  "
                  f"[{res['method'][:11]}]{flag_str}")
        else:
            print(f"  [{i:4d}/{len(images)}] {res['file']:<35} {res['status']}")

    # ── Summary CSV ───────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "step2_disc_results.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    ok        = sum(1 for r in results if r["status"] == "OK")
    fallbacks = sum(1 for r in results if r.get("flag") == "LOW_CONFIDENCE_FALLBACK")
    errors    = len(results) - ok

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Succeeded      : {ok}")
    print(f"  Fallbacks      : {fallbacks}  (positional prior used)")
    print(f"  Errors         : {errors}")
    print(f"  Results saved  → {summary_path}")
    print(f"{'='*60}\n")

    # ── Print high / low confidence examples for manual review ─
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        sorted_by_conf = sorted(ok_results, key=lambda r: r["confidence"], reverse=True)
        print("  Top 5 HIGH confidence detections:")
        for r in sorted_by_conf[:5]:
            print(f"    {r['file']:<35} conf={r['confidence']:.3f}")
        print()
        print("  Top 5 LOW confidence detections (check these):")
        for r in sorted_by_conf[-5:]:
            print(f"    {r['file']:<35} conf={r['confidence']:.3f}  [{r['flag']}]")
        print()


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="SepsisScope Step 2 — Optic disc detection")
    p.add_argument("--images",    required=True,
                   help="Folder containing original fundus images")
    p.add_argument("--step1_out", required=True,
                   help="Folder containing Step 1 skeleton outputs")
    p.add_argument("--output",    required=True,
                   help="Folder to write disc overlays and CSV into")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = {}
    run_batch(args.images, args.step1_out, args.output, cfg)
