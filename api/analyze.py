"""
SepsisScope API — Flask serverless function for Vercel
Accepts a fundus image upload and runs the 5-step AVR pipeline.
"""

import os
import sys
import csv
import base64
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to Python path so pipeline step modules are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── scikit-image shim ─────────────────────────────────────────────────────────
# Vercel's Python runtime crashes when scikit-image is installed because it
# tries to load .pyi stub files that are absent in the vendored bundle:
#   "Cannot load imports from non-existent stub /var/task/_vendor/skimage/__init__.pyi"
#
# Fix: load the shim that lives next to this file (api/skimage_shim.py) and
# register it in sys.modules under 'skimage' *before* any pipeline step module
# is imported, so their `from skimage import ...` calls resolve to the shim.
_API_DIR = os.path.dirname(os.path.abspath(__file__))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from skimage_shim import filters as _sk_filters, exposure as _sk_exposure  # noqa: E402
from skimage_shim import morphology as _sk_morphology, measure as _sk_measure  # noqa: E402
import types as _types

def _make_skimage_pkg():
    pkg = _types.ModuleType("skimage")
    pkg.filters   = _sk_filters
    pkg.exposure  = _sk_exposure
    pkg.morphology = _sk_morphology
    pkg.measure   = _sk_measure
    return pkg

_skimage_pkg = _make_skimage_pkg()
sys.modules.setdefault("skimage",            _skimage_pkg)
sys.modules.setdefault("skimage.filters",    _sk_filters)
sys.modules.setdefault("skimage.exposure",   _sk_exposure)
sys.modules.setdefault("skimage.morphology", _sk_morphology)
sys.modules.setdefault("skimage.measure",    _sk_measure)
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow cross-origin requests from any origin (frontend on a different Render
# subdomain).  Be explicit so newer flask-cors versions don't silently skip
# preflight responses.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=False,
)

@app.after_request
def _cors_headers(response):
    """Belt-and-suspenders CORS headers on every response including 4xx/5xx."""
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ppm", ".bmp"}

# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def img_to_base64(path: str):
    """Read an image file and return a base64 data URI, or None if missing."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(path).suffix.lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        return f"data:image/{mime};base64,{data}"
    except Exception:
        return None


def filter_step1_by_zone(step1_csv_path: str, disc: dict, h: int, w: int) -> list[dict]:
    """
    Filter Step 1 segment measurements directly by the 1r–2r annular zone
    using each segment's centroid.

    This is more robust than Step 3's skeleton re-labelling + centroid merge
    because Step 1 centroids are computed on branch-point-free skeleton while
    Step 3 re-labels the full skeleton — the two labellings diverge, causing
    the merge to miss most segments.
    """
    import numpy as np

    cx, cy, r = disc["cx"], disc["cy"], disc["r"]
    inner_r, outer_r = float(r), float(2 * r)

    kept = []
    try:
        with open(step1_csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    seg_cx = float(row["centroid_x"])
                    seg_cy = float(row["centroid_y"])
                    dist = float(np.sqrt((seg_cx - cx) ** 2 + (seg_cy - cy) ** 2))

                    # Keep segments whose centroid falls in 1r–2r zone
                    if inner_r <= dist <= outer_r:
                        row["centroid_dist"] = round(dist, 1)
                        row["zone_fraction"] = 1.0
                        row["in_zone"] = True
                        kept.append(row)
                except (ValueError, TypeError):
                    continue
    except FileNotFoundError:
        pass

    return kept


def write_filtered_csv(segments: list[dict], path: str):
    """Write a list of segment dicts to a CSV file."""
    if not segments:
        return
    fieldnames = list(segments[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(segments)


def draw_zone_overlay(img_rgb, disc: dict, kept_segments: list[dict],
                      h: int, w: int, out_path: str):
    """Draw 1r/2r rings and colour-code kept/rejected segments."""
    import cv2
    import numpy as np
    from skimage import measure

    cx, cy, r = disc["cx"], disc["cy"], disc["r"]
    overlay = img_rgb.copy()

    # 1r and 2r rings
    cv2.circle(overlay, (cx, cy), r,     [100, 100, 255], 1)
    cv2.circle(overlay, (cx, cy), 2 * r, [100, 100, 255], 1)
    cv2.circle(overlay, (cx, cy), 3,     [255,   0,   0], -1)

    # Build set of kept centroid coords for colouring
    kept_cx = {round(float(s["centroid_x"]), 1) for s in kept_segments}

    # Load skeleton to colour segments
    skel_path = out_path.replace("_zone.png", "").replace(
        os.path.basename(out_path), ""
    )

    label_str = f"zone kept={len(kept_segments)}"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.putText(overlay_bgr, label_str, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(img_path: str, work_dir: str) -> dict:
    """
    Run all 5 SepsisScope pipeline steps for a single fundus image.

    Key improvement over the original CLI pipeline:
    • Step 3 zone filtering is done directly on Step 1 segment centroids
      (avoids the fragile skeleton re-labelling + centroid merge that drops
      most segments and causes Step 4 to see too few valid measurements).
    • Steps 4 and 5 are not fatal if they find fewer segments than ideal —
      the result is returned with an appropriate flag instead of an error.
    """
    import cv2
    import numpy as np

    stem = Path(img_path).stem
    s1_dir = os.path.join(work_dir, "step1")
    s2_dir = os.path.join(work_dir, "step2")
    s3_dir = os.path.join(work_dir, "step3")
    s4_dir = os.path.join(work_dir, "step4")
    s5_dir = os.path.join(work_dir, "step5")

    for d in [s1_dir, s2_dir, s3_dir, s4_dir, s5_dir]:
        os.makedirs(d, exist_ok=True)

    result = {"status": "error", "steps": {}, "result": {}, "images": {}}

    # Encode original image
    result["images"]["original"] = img_to_base64(img_path)

    # Load image once for reuse
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        result["error"] = "Could not read the uploaded image file."
        return result
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # ── STEP 1: Vessel Segmentation + Skeletonisation ─────────────────
    try:
        from step1_segment_skeleton import process_image as step1_fn

        # Try primary sigma range first; if too few segments found, widen range
        for sigma_high, threshold in [(5, 0.02), (8, 0.01), (10, 0.005)]:
            cfg1 = {
                "sigma_low":    1,
                "sigma_high":   sigma_high,
                "threshold":    threshold,
                "min_area":     20,      # lowered from 30 for better coverage
                "workers":      1,
                "save_overlay": True,
            }
            s1 = step1_fn((img_path, s1_dir, cfg1))
            if s1.get("status") == "OK" and s1.get("n_segments", 0) >= 4:
                break

        result["steps"]["step1"] = s1

        if s1.get("status") != "OK":
            result["error"] = f"Vessel segmentation failed: {s1.get('status')}"
            return result

        overlay_path = os.path.join(s1_dir, f"{stem}_overlay.png")
        result["images"]["overlay"] = img_to_base64(overlay_path)

    except Exception as e:
        result["steps"]["step1"] = {"status": f"ERROR: {e}"}
        result["error"] = f"Step 1 exception: {e}"
        return result

    # ── STEP 2: Optic Disc Detection ──────────────────────────────────
    try:
        from step2_disc_detect import process_image as step2_fn

        s2 = step2_fn((img_path, s1_dir, s2_dir, {}))
        result["steps"]["step2"] = s2

        if s2.get("status") != "OK":
            result["error"] = f"Optic disc detection failed: {s2.get('status')}"
            return result

        disc_path = os.path.join(s2_dir, f"{stem}_disc.png")
        result["images"]["disc"] = img_to_base64(disc_path)

    except Exception as e:
        result["steps"]["step2"] = {"status": f"ERROR: {e}"}
        result["error"] = f"Step 2 exception: {e}"
        return result

    # ── STEP 3: Zone Filtering — direct centroid approach ─────────────
    #
    # Instead of Step 3's skeleton re-labelling + centroid merge (which
    # frequently drops measurements), filter Step 1 segments directly by
    # checking each segment's centroid against the 1r–2r annulus.
    #
    try:
        disc = {
            "cx": s2["disc_cx"],
            "cy": s2["disc_cy"],
            "r":  s2["disc_r"],
            "confidence": s2["confidence"],
            "method":     s2["method"],
            "flag":       s2.get("flag", ""),
        }

        step1_csv = os.path.join(s1_dir, f"{stem}_segments.csv")
        kept_segments = filter_step1_by_zone(step1_csv, disc, h, w)

        # Write the filtered CSV for Step 4
        filtered_csv = os.path.join(s3_dir, f"{stem}_filtered.csv")
        write_filtered_csv(kept_segments, filtered_csv)

        # Draw zone overlay
        zone_path = os.path.join(s3_dir, f"{stem}_zone.png")
        _draw_zone_overlay_full(img_rgb, disc, kept_segments, step1_csv, zone_path)
        result["images"]["zone"] = img_to_base64(zone_path)

        s3 = {
            "file":       stem,
            "status":     "OK",
            "n_kept":     len(kept_segments),
            "n_rejected": s1.get("n_segments", 0) - len(kept_segments),
            "disc_r":     disc["r"],
            "disc_cx":    disc["cx"],
            "disc_cy":    disc["cy"],
            "disc_conf":  disc["confidence"],
        }
        result["steps"]["step3"] = s3

    except Exception as e:
        result["steps"]["step3"] = {"status": f"ERROR: {e}"}
        result["error"] = f"Step 3 exception: {e}"
        return result

    # ── STEP 4: Arteriole / Venule Classification ─────────────────────
    #
    # min_segments=1 so we attempt classification even with sparse images.
    # A SKIPPED result is NOT fatal — we still return the zone overlay and
    # disc info with an appropriate flag.
    #
    s4 = {"status": "SKIPPED: no filtered CSV", "n_arteriole": 0,
          "n_venule": 0, "n_uncertain": 0, "n_total": 0}

    if os.path.exists(filtered_csv) and len(kept_segments) > 0:
        try:
            from step4_classify import classify_image

            s4 = classify_image(filtered_csv, s1_dir, s4_dir, stem,
                                min_confidence=0.05,
                                min_segments=1)      # was 3/6 — lowered

            classified_path = os.path.join(s4_dir, f"{stem}_classified.png")
            if os.path.exists(classified_path):
                result["images"]["classified"] = img_to_base64(classified_path)

        except Exception as e:
            s4 = {"status": f"ERROR: {e}", "n_arteriole": 0,
                  "n_venule": 0, "n_uncertain": 0, "n_total": 0}

    result["steps"]["step4"] = s4

    # ── STEP 5: CRAE / CRVE / AVR Calculation ────────────────────────
    #
    # min_vessels=1 so we attempt AVR even with very few vessels.
    # All SKIPPED / insufficient flags are returned as valid results, not errors.
    #
    s5 = {"status": "SKIPPED: Step 4 produced no classified CSV",
          "CRAE": None, "CRVE": None, "AVR": None,
          "flag": "insufficient_arterioles",
          "n_arterioles": 0, "n_venules": 0}

    classified_csv = os.path.join(s4_dir, f"{stem}_classified.csv")
    if s4.get("status") == "OK" and os.path.exists(classified_csv):
        try:
            from step5_avr import compute_avr

            s5 = compute_avr(classified_csv, s5_dir, stem, min_vessels=1)

        except Exception as e:
            s5 = {"status": f"ERROR: {e}", "CRAE": None, "CRVE": None,
                  "AVR": None, "flag": "unknown",
                  "n_arterioles": 0, "n_venules": 0}

    result["steps"]["step5"] = s5

    # ── Build final result ────────────────────────────────────────────
    avr  = s5.get("AVR")
    crae = s5.get("CRAE")
    crve = s5.get("CRVE")
    flag = s5.get("flag", "unknown")

    # Determine reason when AVR is unavailable
    if avr is None and flag == "unknown":
        if s4.get("status", "").startswith("SKIPPED"):
            flag = "insufficient_segments"
        elif s5.get("status", "").startswith("SKIPPED"):
            flag = s5.get("flag", "insufficient_arterioles")

    INTERPRETATIONS = {
        "normal":
            "AVR within normal range (0.6–0.8) — microvascular calibre appears healthy.",
        "low_avr":
            "AVR below 0.6 — possible arteriolar narrowing associated with hypertension, "
            "cardiovascular risk, or sepsis-related microvascular injury.",
        "high_avr":
            "AVR above 0.8 — possible venular dilation associated with inflammation "
            "or metabolic syndrome.",
        "insufficient_arterioles":
            "Too few arterioles identified in the measurement zone for a reliable AVR. "
            "Try a higher-quality fundus image with clear vessel contrast.",
        "insufficient_venules":
            "Too few venules identified in the measurement zone for a reliable AVR. "
            "Try a higher-quality fundus image with clear vessel contrast.",
        "insufficient_segments":
            "Too few vessel segments passed quality checks in the measurement zone. "
            "The image may have low contrast, heavy noise, or poor vessel visibility.",
        "zero_crve":
            "CRVE computed as zero — cannot divide to produce AVR.",
        "unknown":
            "Analysis could not be completed for this image.",
    }

    result["result"] = {
        "AVR":   avr,
        "CRAE":  crae,
        "CRVE":  crve,
        "flag":  flag,
        "interpretation": INTERPRETATIONS.get(flag, "Analysis result available."),
        "segments": {
            "total":      s1.get("n_segments", 0),
            "in_zone":    s3.get("n_kept", 0),
            "rejected":   s3.get("n_rejected", 0),
            "arterioles": s4.get("n_arteriole", 0),
            "venules":    s4.get("n_venule", 0),
            "uncertain":  s4.get("n_uncertain", 0),
        },
        "disc": {
            "cx":         s2.get("disc_cx"),
            "cy":         s2.get("disc_cy"),
            "r":          s2.get("disc_r"),
            "confidence": s2.get("confidence"),
            "method":     s2.get("method"),
            "flag":       s2.get("flag", ""),
        },
    }

    # Always return success — let the flag communicate the quality of the result
    result["status"] = "success"
    result.pop("error", None)
    return result


def _draw_zone_overlay_full(img_rgb, disc: dict, kept_segments: list[dict],
                             step1_csv: str, out_path: str):
    """Draw 1r/2r annulus rings + colour all Step 1 segments kept/rejected."""
    import cv2
    import numpy as np

    overlay = img_rgb.copy()
    cx, cy, r = disc["cx"], disc["cy"], disc["r"]

    # Draw rings
    cv2.circle(overlay, (cx, cy), r,     [100, 100, 255], 1)
    cv2.circle(overlay, (cx, cy), 2 * r, [100, 100, 255], 1)
    cv2.circle(overlay, (cx, cy), 3,     [255,   0,   0], -1)

    # Load skeleton and colour segments
    skel_dir = os.path.dirname(step1_csv)
    stem = Path(step1_csv).stem.replace("_segments", "")
    skel_path = os.path.join(skel_dir, f"{stem}_skeleton.png")

    if os.path.exists(skel_path):
        from skimage import measure

        skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
        skel_bin = (skel > 127).astype("uint8") if skel is not None else None

        if skel_bin is not None:
            kept_cx_set = {round(float(s["centroid_x"]), 1) for s in kept_segments}
            labeled = measure.label(skel_bin, connectivity=2)

            for reg in measure.regionprops(labeled):
                coords = reg.coords
                cy_seg, cx_seg = reg.centroid
                if round(cx_seg, 1) in kept_cx_set:
                    overlay[coords[:, 0], coords[:, 1]] = [0, 220, 80]   # green = kept
                else:
                    overlay[coords[:, 0], coords[:, 1]] = [180, 60, 60]  # red = rejected

    label_str = f"zone kept={len(kept_segments)}"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.putText(overlay_bgr, label_str, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay_bgr)


# ──────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ──────────────────────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    # Browser sends OPTIONS preflight before the actual POST
    if request.method == "OPTIONS":
        return "", 204

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED:
        return jsonify({
            "error": f"Unsupported format '{suffix}'. Accepted: {', '.join(sorted(SUPPORTED))}"
        }), 400

    with tempfile.TemporaryDirectory() as work_dir:
        img_path = os.path.join(work_dir, f"input{suffix}")
        file.save(img_path)
        data = run_pipeline(img_path, work_dir)

    return jsonify(data)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "SepsisScope API"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
