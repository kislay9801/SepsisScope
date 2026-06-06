"""
SepsisScope — Step 5: CRAE / CRVE / AVR Calculation
=====================================================
Applies the Knudtson-modified Hubbard formula to compute:
  CRAE — Central Retinal Arteriole Equivalent
  CRVE — Central Retinal Venule Equivalent
  AVR  — Arteriovenous Ratio = CRAE / CRVE

Protocol
--------
  1. From Step 4 classified segments, take top 6 widest arterioles
     and top 6 widest venules (by width_px, descending)
  2. Apply Knudtson formula iteratively to pairs:
       W_crae = 0.88 * sqrt(a1^2 + a2^2)   (arteriole pairs)
       W_crve = 0.95 * sqrt(v1^2 + v2^2)   (venule pairs)
     until one value remains per type
  3. AVR = CRAE / CRVE
  4. Flag if AVR outside normal range (0.6 - 0.8)

Knudtson formula reference:
  Knudtson et al. (2003) Ophthalmology 110(8):1491-1496

Usage
-----
  python step5_avr.py
      [--step4_drive <folder>]
      [--step4_stare <folder>]
      [--output_drive <folder>]
      [--output_stare <folder>]
      [--dataset drive|stare|both]
      [--min_vessels 6]
"""

import os
import csv
import math
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

AVR_NORMAL_LOW  = 0.6
AVR_NORMAL_HIGH = 0.8


# ──────────────────────────────────────────────
# KNUDTSON FORMULA
# ──────────────────────────────────────────────
def knudtson_combine(widths, vessel_type='arteriole'):
    """
    Apply Knudtson-modified Hubbard formula iteratively.
    Takes a list of vessel widths, returns the combined equivalent.

    Arteriole: W = 0.88 * sqrt(w1^2 + w2^2)
    Venule:    W = 0.95 * sqrt(v1^2 + v2^2)
    """
    factor = 0.88 if vessel_type == 'arteriole' else 0.95

    # Work with a mutable list, sorted descending (widest first)
    w = sorted([float(x) for x in widths], reverse=True)

    # Iteratively combine pairs from widest down
    while len(w) > 1:
        # Take the two widest remaining
        w1, w2 = w[0], w[1]
        combined = factor * math.sqrt(w1**2 + w2**2)
        # Remove the two used, insert the combined value
        w = [combined] + w[2:]

    return round(w[0], 4)


# ──────────────────────────────────────────────
# PROCESS ONE IMAGE
# ──────────────────────────────────────────────
def compute_avr(classified_csv, out_dir, stem, min_vessels=6):

    with open(classified_csv, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Separate arterioles and venules, skip uncertain
    arterioles = []
    venules    = []
    for r in rows:
        if r['label'] == 'arteriole':
            try:
                arterioles.append((float(r['width_px']), float(r['confidence']), r))
            except (ValueError, TypeError):
                pass
        elif r['label'] == 'venule':
            try:
                venules.append((float(r['width_px']), float(r['confidence']), r))
            except (ValueError, TypeError):
                pass

    if len(arterioles) < min_vessels:
        return {
            "file": stem, "status": f"SKIPPED: only {len(arterioles)} arterioles",
            "n_arterioles": len(arterioles), "n_venules": len(venules),
            "CRAE": None, "CRVE": None, "AVR": None, "flag": "insufficient_arterioles"
        }

    if len(venules) < min_vessels:
        return {
            "file": stem, "status": f"SKIPPED: only {len(venules)} venules",
            "n_arterioles": len(arterioles), "n_venules": len(venules),
            "CRAE": None, "CRVE": None, "AVR": None, "flag": "insufficient_venules"
        }

    # Sort by width descending, take top 6
    arterioles.sort(key=lambda x: x[0], reverse=True)
    venules.sort(key=lambda x: x[0], reverse=True)

    top6_art = arterioles[:6]
    top6_ven = venules[:6]

    art_widths = [w for w, c, r in top6_art]
    ven_widths = [w for w, c, r in top6_ven]

    # Apply Knudtson formula
    CRAE = knudtson_combine(art_widths, vessel_type='arteriole')
    CRVE = knudtson_combine(ven_widths, vessel_type='venule')

    if CRVE == 0:
        return {
            "file": stem, "status": "ERROR: CRVE is zero",
            "n_arterioles": len(arterioles), "n_venules": len(venules),
            "CRAE": CRAE, "CRVE": CRVE, "AVR": None, "flag": "zero_crve"
        }

    AVR = round(CRAE / CRVE, 4)

    # Flag if outside normal range
    if AVR_NORMAL_LOW <= AVR <= AVR_NORMAL_HIGH:
        flag = "normal"
    elif AVR < AVR_NORMAL_LOW:
        flag = "low_avr"
    else:
        flag = "high_avr"

    # Save per-image AVR CSV
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    avr_rows = []
    for w, c, r in top6_art:
        avr_rows.append({**r, 'selected_for': 'CRAE', 'rank_width': round(w, 4)})
    for w, c, r in top6_ven:
        avr_rows.append({**r, 'selected_for': 'CRVE', 'rank_width': round(w, 4)})

    fieldnames = list(avr_rows[0].keys())
    with open(out / f"{stem}_avr.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(avr_rows)

    return {
        "file":         stem,
        "status":       "OK",
        "n_arterioles": len(arterioles),
        "n_venules":    len(venules),
        "art_widths":   [round(w, 3) for w in art_widths],
        "ven_widths":   [round(w, 3) for w in ven_widths],
        "CRAE":         CRAE,
        "CRVE":         CRVE,
        "AVR":          AVR,
        "flag":         flag,
    }


# ──────────────────────────────────────────────
# BATCH RUNNER
# ──────────────────────────────────────────────
def run_batch(step4_folder, out_folder, min_vessels=6):

    csvs = sorted([f for f in os.listdir(step4_folder)
                   if f.endswith('_classified.csv')])

    if not csvs:
        print(f"[ERROR] No classified CSVs found in: {step4_folder}")
        return []

    results = []
    for i, c in enumerate(csvs, 1):
        stem = c.replace('_classified.csv', '')
        classified_csv = os.path.join(step4_folder, c)

        res = compute_avr(classified_csv, out_folder, stem, min_vessels)
        results.append(res)

        status = res['status']
        if status == 'OK':
            print(f"  [{i:4d}/{len(csvs)}] {stem:<40} "
                  f"CRAE={res['CRAE']:6.3f}  CRVE={res['CRVE']:6.3f}  "
                  f"AVR={res['AVR']:.4f}  [{res['flag']}]")
        else:
            print(f"  [{i:4d}/{len(csvs)}] {stem:<40} {status}")

    return results


def save_summary(results, out_folder, dataset_name):
    if not results:
        return

    summary_path = Path(out_folder) / "step5_summary.csv"
    fieldnames = ['file', 'status', 'n_arterioles', 'n_venules',
                  'CRAE', 'CRVE', 'AVR', 'flag']

    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    ok  = [r for r in results if r['status'] == 'OK']
    avrs = [r['AVR'] for r in ok if r['AVR'] is not None]

    print(f"\n  {dataset_name} AVR Summary:")
    print(f"  Images OK        : {len(ok)} / {len(results)}")
    if avrs:
        print(f"  AVR — min  : {min(avrs):.4f}")
        print(f"  AVR — max  : {max(avrs):.4f}")
        print(f"  AVR — mean : {np.mean(avrs):.4f}")
        print(f"  AVR — std  : {np.std(avrs):.4f}")
        normal = sum(1 for a in avrs if AVR_NORMAL_LOW <= a <= AVR_NORMAL_HIGH)
        low    = sum(1 for a in avrs if a < AVR_NORMAL_LOW)
        high   = sum(1 for a in avrs if a > AVR_NORMAL_HIGH)
        print(f"  Normal (0.6-0.8) : {normal} ({100*normal/len(avrs):.1f}%)")
        print(f"  Low    (<0.6)    : {low}    ({100*low/len(avrs):.1f}%)")
        print(f"  High   (>0.8)    : {high}   ({100*high/len(avrs):.1f}%)")
    print(f"  Summary saved : {summary_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SepsisScope Step 5: AVR Calculation")
    parser.add_argument('--step4_drive',  default=r'C:\SepsisScope\out\step4_drive')
    parser.add_argument('--step4_stare',  default=r'C:\SepsisScope\out\step4_stare')
    parser.add_argument('--output_drive', default=r'C:\SepsisScope\out\step5_drive')
    parser.add_argument('--output_stare', default=r'C:\SepsisScope\out\step5_stare')
    parser.add_argument('--min_vessels',  type=int, default=6)
    parser.add_argument('--dataset',
                        choices=['drive', 'stare', 'both'],
                        default='both')
    args = parser.parse_args()

    SEP = '=' * 60

    if args.dataset in ('drive', 'both'):
        print(f"\n{SEP}")
        print(f"  SepsisScope — Step 5: AVR Calculation (DRIVE)")
        print(SEP)
        drive_results = run_batch(args.step4_drive, args.output_drive, args.min_vessels)
        save_summary(drive_results, args.output_drive, 'DRIVE')

    if args.dataset in ('stare', 'both'):
        print(f"\n{SEP}")
        print(f"  SepsisScope — Step 5: AVR Calculation (STARE)")
        print(SEP)
        stare_results = run_batch(args.step4_stare, args.output_stare, args.min_vessels)
        save_summary(stare_results, args.output_stare, 'STARE')
