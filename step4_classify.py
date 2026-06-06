"""
SepsisScope — Step 4: Arteriole/Venule Classification
======================================================
Classifies each vessel segment as arteriole (A) or venule (V)
using a combined score of colour (r_mean / (g_mean + b_mean))
and width, computed per image so thresholds adapt to each image.

Method
------
For each image:
  1. Normalise color_score to 0-1 range (higher = more arteriole)
  2. Normalise width_px   to 0-1 range (lower  = more arteriole)
  3. Combined score = norm_color - norm_width
     - Positive score -> arteriole (redder AND narrower)
     - Negative score -> venule   (darker  AND wider)
  4. Confidence = |combined_score| — distance from the zero decision boundary
  5. Drop segments below confidence threshold (uncertain)
  6. Rank remaining segments by confidence (descending)

Outputs
-------
  <out>/<stem>_classified.csv   — segments with label, score, confidence
  <out>/<stem>_classified.png   — overlay: arterioles=red, venules=blue
  <out>/step4_summary.csv       — per-image counts and stats

Usage
-----
  python step4_classify.py
      --step3_drive  <folder>
      --step3_stare  <folder>
      --step1_drive  <folder>
      --step1_stare  <folder>
      --output_drive <folder>
      --output_stare <folder>
      [--min_confidence 0.05]
      [--min_segments   6]
"""

import os
import csv
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# CLASSIFY ONE IMAGE
# ──────────────────────────────────────────────
def classify_image(filtered_csv, step1_out, out_dir, stem,
                   min_confidence=0.05, min_segments=6):

    # Load filtered segments
    with open(filtered_csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"file": stem, "status": "ERROR: empty filtered CSV",
                "n_arteriole": 0, "n_venule": 0, "n_uncertain": 0, "n_total": 0}

    # Parse values once — skip rows with missing/invalid measurements.
    # Keep the parsed numbers alongside the row so we don't re-parse later.
    valid = []
    for r in rows:
        try:
            color = float(r['color_score'])
            width = float(r['width_px'])
            if color <= 0 or width <= 0:
                continue
            valid.append((r, color, width))
        except (ValueError, TypeError):
            continue

    if len(valid) < min_segments:
        return {"file": stem, "status": f"SKIPPED: only {len(valid)} valid segments",
                "n_arteriole": 0, "n_venule": 0, "n_uncertain": 0, "n_total": len(valid)}

    # Build arrays directly from the already-parsed values
    colors = np.array([c for (_, c, _) in valid])
    widths = np.array([w for (_, _, w) in valid])

    # Normalise per image (0-1 range)
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)

    norm_color = norm(colors)   # high = arteriole
    norm_width = norm(widths)   # high = venule (so subtract)

    # Combined score: positive = arteriole, negative = venule.
    # Decision boundary is zero, which is what `confidence` measures
    # distance from — so label and confidence are consistent.
    combined = norm_color - norm_width
    confidence = np.abs(combined)

    # Assign labels
    results = []
    n_arteriole = n_venule = n_uncertain = 0

    for i, (r, _, _) in enumerate(valid):
        score = float(combined[i])
        conf  = float(confidence[i])

        if conf < min_confidence:
            label = 'uncertain'
            n_uncertain += 1
        elif score > 0:
            label = 'arteriole'
            n_arteriole += 1
        else:
            label = 'venule'
            n_venule += 1

        results.append({
            **r,
            'label':          label,
            'combined_score': round(score, 6),
            'confidence':     round(conf,  6),
        })

    # Sort by confidence descending
    results.sort(key=lambda x: float(x['confidence']), reverse=True)

    # Save classified CSV
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    with open(out / f"{stem}_classified.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save overlay image
    skel_path = Path(step1_out) / f"{stem}_skeleton.png"
    overlay_path = Path(step1_out) / f"{stem}_overlay.png"

    # Load the skeleton once; reuse it for both the fallback base image
    # and the per-segment colouring below.
    skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE) if skel_path.exists() else None

    base_img = None
    if overlay_path.exists():
        base_img = cv2.imread(str(overlay_path))
    elif skel is not None:
        base_img = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)

    if base_img is not None:
        overlay = base_img.copy()

        if skel is not None:
            from skimage import measure
            skeleton_bin = (skel > 127).astype(np.uint8)
            labeled = measure.label(skeleton_bin, connectivity=2)

            # Build segment_id → label map
            id_label = {}
            for r in results:
                try:
                    id_label[int(r['segment_id'])] = r['label']
                except (ValueError, KeyError):
                    pass

            for reg in measure.regionprops(labeled):
                coords = reg.coords
                lbl = id_label.get(reg.label, 'unknown')
                if lbl == 'arteriole':
                    color = [0, 0, 255]    # red (BGR)
                elif lbl == 'venule':
                    color = [255, 0, 0]    # blue (BGR)
                elif lbl == 'uncertain':
                    color = [0, 255, 255]  # yellow
                else:
                    continue
                overlay[coords[:, 0], coords[:, 1]] = color

        # Add label
        label_str = f"A={n_arteriole}  V={n_venule}  ?={n_uncertain}"
        cv2.putText(overlay, label_str, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{stem}_classified.png"), overlay)

    return {
        "file":        stem,
        "status":      "OK",
        "n_arteriole": n_arteriole,
        "n_venule":    n_venule,
        "n_uncertain": n_uncertain,
        "n_total":     len(valid),
    }


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────
def run_batch(step3_folder, step1_folder, out_folder,
              min_confidence=0.05, min_segments=6):

    csvs = sorted([f for f in os.listdir(step3_folder)
                   if f.endswith('_filtered.csv')])

    if not csvs:
        print(f"[ERROR] No filtered CSVs found in: {step3_folder}")
        return []

    results = []
    for i, c in enumerate(csvs, 1):
        stem = c.replace('_filtered.csv', '')
        filtered_csv = os.path.join(step3_folder, c)

        res = classify_image(
            filtered_csv, step1_folder, out_folder, stem,
            min_confidence=min_confidence,
            min_segments=min_segments
        )
        results.append(res)

        status = res['status']
        if status == 'OK':
            print(f"  [{i:4d}/{len(csvs)}] {stem:<40} "
                  f"A={res['n_arteriole']:3d}  V={res['n_venule']:3d}  "
                  f"?={res['n_uncertain']:3d}  total={res['n_total']:3d}")
        else:
            print(f"  [{i:4d}/{len(csvs)}] {stem:<40} {status}")

    return results


def save_summary(results, out_folder, dataset_name):
    if not results:
        return
    summary_path = Path(out_folder) / "step4_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if r['status'] == 'OK']
    print(f"\n  {dataset_name} Summary:")
    print(f"  OK       : {len(ok)}")
    print(f"  Skipped  : {len(results) - len(ok)}")
    if ok:
        arts = [r['n_arteriole'] for r in ok]
        vens = [r['n_venule']    for r in ok]
        print(f"  Avg arterioles/image : {sum(arts)/len(arts):.1f}")
        print(f"  Avg venules/image    : {sum(vens)/len(vens):.1f}")
    print(f"  Summary  : {summary_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SepsisScope Step 4: A/V Classification")
    parser.add_argument('--step3_drive',  default=r'C:\SepsisScope\out\step3')
    parser.add_argument('--step3_stare',  default=r'C:\SepsisScope\out\step3_stare')
    parser.add_argument('--step1_drive',  default=r'C:\SepsisScope\out\step1_v5')
    parser.add_argument('--step1_stare',  default=r'C:\SepsisScope\out\step1_stare')
    parser.add_argument('--output_drive', default=r'C:\SepsisScope\out\step4_drive')
    parser.add_argument('--output_stare', default=r'C:\SepsisScope\out\step4_stare')
    parser.add_argument('--min_confidence', type=float, default=0.05)
    parser.add_argument('--min_segments',   type=int,   default=6)
    parser.add_argument('--dataset',
                        choices=['drive', 'stare', 'both'],
                        default='both',
                        help='Which dataset to process')
    args = parser.parse_args()

    SEP = '=' * 60

    if args.dataset in ('drive', 'both'):
        print(f"\n{SEP}")
        print(f"  SepsisScope — Step 4: Classification (DRIVE)")
        print(SEP)
        drive_results = run_batch(
            args.step3_drive, args.step1_drive, args.output_drive,
            args.min_confidence, args.min_segments
        )
        save_summary(drive_results, args.output_drive, 'DRIVE')

    if args.dataset in ('stare', 'both'):
        print(f"\n{SEP}")
        print(f"  SepsisScope — Step 4: Classification (STARE)")
        print(SEP)
        stare_results = run_batch(
            args.step3_stare, args.step1_stare, args.output_stare,
            args.min_confidence, args.min_segments
        )
        save_summary(stare_results, args.output_stare, 'STARE')
