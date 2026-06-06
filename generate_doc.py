"""Generate SepsisScope Technical Documentation PDF using reportlab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus.tableofcontents import TableOfContents
import os

OUTPUT = r"C:\SepsisScope\SepsisScope_Technical_Documentation.pdf"

# ── Colour palette ─────────────────────────────────────────────
DARK_BLUE  = colors.HexColor("#1a2e4a")
MID_BLUE   = colors.HexColor("#2c5282")
LIGHT_BLUE = colors.HexColor("#ebf4ff")
ACCENT     = colors.HexColor("#3182ce")
GREY_BG    = colors.HexColor("#f7f8fa")
GREY_LINE  = colors.HexColor("#cbd5e0")
CODE_BG    = colors.HexColor("#1e2030")
CODE_FG    = colors.HexColor("#cdd6f4")
RED        = colors.HexColor("#c53030")
GREEN      = colors.HexColor("#276749")
AMBER      = colors.HexColor("#b7791f")

# ── Styles ─────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

Title = S("DocTitle",
    fontSize=28, leading=34, textColor=DARK_BLUE,
    fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)

Subtitle = S("DocSubtitle",
    fontSize=13, leading=18, textColor=MID_BLUE,
    fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)

H1 = S("H1",
    fontSize=16, leading=22, textColor=colors.white,
    fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=8,
    backColor=DARK_BLUE, leftIndent=-12, rightIndent=-12,
    borderPad=6)

H2 = S("H2",
    fontSize=13, leading=18, textColor=DARK_BLUE,
    fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=5,
    borderPad=0)

H3 = S("H3",
    fontSize=11, leading=15, textColor=MID_BLUE,
    fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)

Body = S("Body",
    fontSize=9.5, leading=15, textColor=colors.HexColor("#2d3748"),
    fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY)

Bullet = S("Bullet",
    fontSize=9.5, leading=14, textColor=colors.HexColor("#2d3748"),
    fontName="Helvetica", spaceAfter=3, leftIndent=18, bulletIndent=6)

Code = S("Code",
    fontSize=8.5, leading=13, textColor=CODE_FG,
    fontName="Courier", backColor=CODE_BG,
    leftIndent=12, rightIndent=12, spaceAfter=8, spaceBefore=4,
    borderPad=8)

Caption = S("Caption",
    fontSize=8, leading=12, textColor=colors.HexColor("#718096"),
    fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=6)

Label = S("Label",
    fontSize=8, leading=11, textColor=colors.HexColor("#4a5568"),
    fontName="Helvetica-Bold")

Note = S("Note",
    fontSize=9, leading=13, textColor=colors.HexColor("#744210"),
    fontName="Helvetica-Oblique", backColor=colors.HexColor("#fffff0"),
    leftIndent=10, borderPad=6, spaceAfter=8)


def hr(): return HRFlowable(width="100%", thickness=0.5, color=GREY_LINE, spaceAfter=8, spaceBefore=4)
def sp(n=6): return Spacer(1, n)
def p(text, style=None): return Paragraph(text, style or Body)
def h1(text): return Paragraph(text, H1)
def h2(text): return Paragraph(text, H2)
def h3(text): return Paragraph(text, H3)
def bullet(text): return Paragraph(f"• {text}", Bullet)
def code(text): return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), Code)
def note(text): return Paragraph(f"<i>Note: {text}</i>", Note)

def table(data, col_widths, header_row=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUND",(0,0),(-1,0),  LIGHT_BLUE),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0), (-1,0),  DARK_BLUE),
        ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("GRID",        (0,0), (-1,-1), 0.4, GREY_LINE),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0),(-1,-1), 6),
    ]
    for i in range(1, len(data)):
        bg = GREY_BG if i % 2 == 0 else colors.white
        style.append(("ROWBACKGROUND", (0,i), (-1,i), bg))
    t.setStyle(TableStyle(style))
    return t


def badge(text, bg):
    style = S("badge", fontSize=7.5, leading=10, textColor=colors.white,
              fontName="Helvetica-Bold", backColor=bg, borderPad=3)
    return Paragraph(text, style)


# ── Document builder ───────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.4*cm, bottomMargin=2.4*cm,
        title="SepsisScope Technical Documentation",
        author="SepsisScope",
    )

    W = A4[0] - 4.4*cm  # usable text width

    story = []

    # ── Cover ──────────────────────────────────────────────────
    story += [
        sp(40),
        Paragraph("SepsisScope", Title),
        Paragraph("Retinal Fundus AVR Pipeline", Subtitle),
        Paragraph("Technical Documentation — v1.0", Subtitle),
        sp(6),
        HRFlowable(width="60%", thickness=2, color=ACCENT, hAlign="CENTER"),
        sp(30),
        p("This document describes the complete SepsisScope image-processing pipeline: its "
          "architecture, algorithms, assumptions, edge-case handling, known corrections, "
          "and final outputs.", Body),
        sp(6),
        p(f"<font color='#718096' size='8'>Generated: 2026-05-18&nbsp;&nbsp;|&nbsp;&nbsp;"
          f"Datasets: DRIVE · STARE · ARIA&nbsp;&nbsp;|&nbsp;&nbsp;"
          f"Language: Python 3</font>", Caption),
        PageBreak(),
    ]

    # ── 1. Overview ────────────────────────────────────────────
    story += [
        h1("1.  Overview"),
        sp(4),
        p("<b>SepsisScope</b> is a fully automated retinal fundus image analysis pipeline "
          "that computes the <b>Arteriovenous Ratio (AVR)</b> — a non-invasive biomarker "
          "for microvascular health. Low AVR values are associated with arteriolar narrowing "
          "seen in hypertension, cardiovascular disease, and sepsis-related microvascular "
          "injury. High AVR values indicate venular dilation linked to inflammation and "
          "metabolic syndrome."),
        sp(4),
        p("The pipeline accepts raw colour fundus photographs from standard retinal cameras "
          "and produces a numeric AVR per image through five sequential, independently "
          "runnable processing steps."),
        sp(6),
        h2("Supported Datasets"),
        table(
            [["Dataset", "Format", "Typical resolution", "Notes"],
             ["DRIVE",  ".tif",  "565 × 584 px",  "20 training + 20 test images"],
             ["STARE",  ".ppm",  "700 × 605 px",  "Mixed pathology"],
             ["ARIA",   ".tif",  "768 × 576 px",  "Diabetic / AMD / control groups"]],
            [2.5*cm, 2*cm, 3.5*cm, W-8*cm]
        ),
        sp(10),
        h2("Technology Stack"),
        table(
            [["Library",    "Purpose"],
             ["OpenCV",     "Image I/O, CLAHE, morphology, distance transform, drawing"],
             ["scikit-image","Frangi filter, skeletonisation, connected-component labelling"],
             ["NumPy / SciPy","Array operations, convolution for branch-point detection"],
             ["Python csv", "All intermediate data stored as plain CSV — no database"]],
            [3.5*cm, W-3.5*cm]
        ),
    ]

    # ── 2. Architecture ────────────────────────────────────────
    story += [
        sp(12),
        h1("2.  Pipeline Architecture"),
        sp(4),
        p("Each step is a self-contained script that reads its inputs from disk and writes "
          "outputs to a specified folder. Steps run sequentially; each step depends only on "
          "the outputs of the preceding one."),
        sp(8),
        code(
            "Fundus Image\n"
            "     │\n"
            "     ▼\n"
            "[step_green_channel.py]  ← Exploratory channel analysis\n"
            "     │\n"
            "     ▼\n"
            "[Step 1]  Vessel Segmentation + Skeletonisation\n"
            "     │    → binary mask · skeleton · per-segment CSV\n"
            "     ▼\n"
            "[Step 2]  Optic Disc Detection\n"
            "     │    → disc centre (cx, cy) and radius r\n"
            "     ▼\n"
            "[Step 3]  Zone Filtering  (1r – 2r annular zone)\n"
            "     │    → filtered segments in measurement zone\n"
            "     ▼\n"
            "[Step 4]  Arteriole / Venule Classification\n"
            "     │    → each segment labelled A or V\n"
            "     ▼\n"
            "[Step 5]  CRAE / CRVE / AVR Calculation\n"
            "          → Arteriovenous Ratio per image"
        ),
    ]

    # ── 3. Step 0 ─────────────────────────────────────────────
    story += [
        sp(12),
        h1("3.  Step 0 — Green Channel Exploration"),
        p("<i>File: step_green_channel.py</i>", Caption),
        sp(4),
        h2("Purpose"),
        p("An exploratory/diagnostic script that splits a single DRIVE training image into "
          "its RGB channels and displays them side by side. This step is not part of the "
          "automated batch pipeline — it exists to visually justify a foundational design choice."),
        sp(6),
        h2("Key Finding"),
        p("The <b>green channel</b> provides the highest vessel-to-background contrast in "
          "fundus photography:"),
        bullet("Red channel: saturates near the bright optic disc; vessels poorly separated."),
        bullet("Blue channel: high noise floor due to short-wavelength scattering."),
        bullet("Green channel: vessels appear as dark ridges on a bright retinal background — "
               "ideal for ridge-detection filters."),
        sp(4),
        note("This channel choice propagates to every downstream step. All vesselness, "
             "brightness, and disc detection computations use the green channel."),
    ]

    # ── 4. Step 1 ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("4.  Step 1 — Vessel Segmentation + Skeletonisation"),
        p("<i>File: step1_segment_skeleton.py</i>", Caption),
        sp(4),
        h2("Goal"),
        p("Detect and map every blood vessel in a fundus image, reduce each vessel to its "
          "single-pixel centreline (skeleton), and extract per-segment geometric and "
          "photometric measurements for downstream analysis."),
        sp(6),
        h2("Processing Pipeline (per image)"),
        table(
            [["Sub-step",          "Method / Detail"],
             ["Load",              "cv2.imread → BGR → RGB conversion"],
             ["Green channel",     "img_rgb[:,:,1], normalised to [0, 1]"],
             ["Retinal mask",      "Largest bright contour → min-enclosing circle → erode 15 px inward"],
             ["Contrast enhance",  "CLAHE (clipLimit=2.0, tileGridSize=8×8) → re-apply mask"],
             ["Vesselness",        "Frangi filter: sigmas 1–5, black_ridges=True, α=β=0.5, γ=15"],
             ["Binarisation",      "Threshold at 0.02 on rescaled vesselness map"],
             ["Morphological clean","MORPH_OPEN with 3×3 elliptical kernel — removes speckle noise"],
             ["Skeletonisation",   "skimage.morphology.skeletonize → single-pixel centrelines"],
             ["Branch detection",  "3×3 convolution counts 8-connected neighbours; ≥3 = branch point"],
             ["Segment labelling", "measure.label (connectivity=2) — each disconnected piece = one segment"]],
            [3.5*cm, W-3.5*cm]
        ),
        sp(8),
        h2("Per-Segment Measurements"),
        table(
            [["Feature",        "How computed"],
             ["length_px",      "Number of skeleton pixels in segment"],
             ["width_px",       "2 × mean(distance_transform at skeleton pixels)"],
             ["r_mean / g_mean / b_mean", "Mean RGB at skeleton pixel locations in original image"],
             ["color_score",    "r_mean / (g_mean + b_mean + ε)  — arterioles score higher"],
             ["contrast",       "|g_mean − local_background_mean|  sampled in 5 px radius"],
             ["centroid_x / y", "From skimage regionprops"]],
            [3.5*cm, W-3.5*cm]
        ),
        sp(8),
        h2("Assumptions"),
        bullet("Vessels are <b>dark ridges</b> on a bright background in the green channel "
               "(black_ridges=True). Valid for standard fundus cameras; may fail for "
               "non-mydriatic or infrared images."),
        bullet("Frangi sigma range [1, 5] captures capillaries through medium vessels. "
               "Use [1, 8] for high-quality final runs."),
        bullet("Width via distance transform is a proxy for true vessel calibre (no "
               "calibration to physical microns)."),
        bullet("Segments shorter than min_area=30 px are noise and discarded."),
        bullet("Local background is sampled at ≤20 evenly-spaced points per segment for speed."),
        sp(8),
        h2("Corrections / Edge Cases"),
        p("The hard circular boundary of the retinal disc can appear as a strong ridge to "
          "the Frangi filter, producing false vessel detections at the image border. "
          "This is corrected by:"),
        bullet("Eroding the retinal mask 15 px inward <i>before</i> Frangi processes the image."),
        bullet("Re-applying the mask after CLAHE to suppress CLAHE artefacts at the edge."),
        bullet("Falling back to image centre if contour detection fails (very dark images)."),
        sp(8),
        h2("Outputs"),
        table(
            [["File",                  "Description"],
             ["<stem>_mask.png",       "Binary vessel mask"],
             ["<stem>_skeleton.png",   "Single-pixel centreline image"],
             ["<stem>_overlay.png",    "Colour overlay: vessels=green, branch points=red"],
             ["<stem>_segments.csv",   "One row per segment with all measurements"],
             ["step1_summary.csv",     "Batch summary: n_segments, coverage% per image"]],
            [4*cm, W-4*cm]
        ),
    ]

    # ── 5. Step 2 ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("5.  Step 2 — Optic Disc Detection"),
        p("<i>File: step2_disc_detect.py</i>", Caption),
        sp(4),
        h2("Goal"),
        p("Locate the optic disc centre (cx, cy) and estimate its radius r. This is the "
          "anatomical reference point for the 1r–2r measurement annulus required by the "
          "Knudtson AVR protocol."),
        sp(6),
        h2("Three-Tier Detection Cascade"),
        p("Three methods are tried in sequence; the first to exceed the confidence "
          "threshold (0.25) is used:"),
        sp(4),
        h3("Method 1 — Vessel Convergence (Primary)"),
        p("All major retinal vessels radiate from the optic disc. The skeleton from Step 1 "
          "is blurred with a large Gaussian (kernel 151×151, σ=40) to create a vessel "
          "density heatmap. The peak of this map is taken as the disc centre."),
        p("<b>Confidence</b> = peak_density / (5 × mean_density), clamped to [0, 1]."),
        sp(4),
        h3("Method 2 — Brightness-Guided (Fallback, conf &lt; 0.25)"),
        p("The optic disc is the brightest compact region in the fundus image. "
          "The green channel is blurred (61×61, σ=20) and the image centre is suppressed "
          "(within 20% of min-dimension radius) to avoid mistaking it for the disc. "
          "The brightest remaining point is the disc centre."),
        sp(4),
        h3("Method 3 — Positional Prior (Last Resort, always flagged)"),
        p("The disc lies in the left or right third of the image. The side with higher "
          "mean green intensity is chosen; the centre is placed at its horizontal midpoint "
          "and the image vertical midpoint. Confidence = 0.0. "
          "Flag = LOW_CONFIDENCE_FALLBACK."),
        sp(6),
        h2("Radius Estimation"),
        p("Radial brightness is sampled at 16 points on circles of increasing radius. "
          "The disc edge is defined where mean ring brightness drops below 45% of centre "
          "brightness. Radius is capped at 15% of image min-dimension. "
          "Default fallback = 8% of image min-dimension."),
        sp(6),
        h2("Assumptions"),
        bullet("Disc = highest local vessel density in the image."),
        bullet("Disc radius ≈ 8% of image size — consistent across DRIVE/STARE/ARIA."),
        bullet("Images are roughly horizontally oriented (disc in left or right third). "
               "Fails for rotated or vertically-framed images."),
        sp(6),
        h2("Outputs"),
        table(
            [["File",                      "Description"],
             ["<stem>_disc.png",           "Overlay: yellow circle=disc, red dot=centre, blue rings=1r/2r"],
             ["step2_disc_results.csv",    "disc_cx, disc_cy, disc_r, confidence, method, flag per image"]],
            [4.5*cm, W-4.5*cm]
        ),
    ]

    # ── 6. Step 3 ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("6.  Step 3 — Zone Filtering (1r – 2r)"),
        p("<i>File: step3_zone_filter.py</i>", Caption),
        sp(4),
        h2("Goal"),
        p("Retain only vessel segments that lie in the <b>1r to 2r annular zone</b> — "
          "the standard clinical measurement annulus used by ophthalmologists to assess "
          "CRAE, CRVE, and AVR per the Knudtson 2003 protocol."),
        sp(6),
        h2("Dual Inclusion Criterion"),
        p("A segment is <b>kept</b> if <b>either</b> condition is met:"),
        bullet("<b>Centroid in zone</b> — centroid distance from disc centre lies in [r, 2r]."),
        bullet("<b>Pixel majority</b> — ≥ 30% of skeleton pixels fall within the annular zone."),
        sp(4),
        p("Using OR instead of AND increases recall, correctly accepting segments that "
          "straddle the zone boundary rather than discarding them arbitrarily."),
        sp(6),
        h2("Merge with Step 1 Measurements"),
        p("Step 3 re-labels segments independently from Step 1, so IDs don't match. "
          "Centroids are matched spatially with a <b>55 px tolerance</b> to retrieve the "
          "original width_px, color_score, contrast, and r/g/b_mean from the Step 1 CSV."),
        sp(6),
        h2("Assumptions"),
        bullet("Low-confidence disc detections (flag=LOW_CONFIDENCE_FALLBACK) are skipped "
               "entirely — an unreliable disc centre would incorrectly filter all segments."),
        bullet("55 px centroid tolerance is intentionally generous because re-labelling can "
               "shift segment centroids by several pixels."),
        bullet("Skeleton fragments shorter than 5 px are noise and discarded."),
        sp(6),
        h2("Known Issue — Dead Code"),
        note("filter_segments_by_zone() contains an unreachable duplicate implementation "
             "after its return statement (lines 116–173 of step3_zone_filter.py). "
             "This is a remnant of an earlier refactor and has no runtime impact, "
             "but should be removed to avoid confusion."),
        sp(6),
        h2("Outputs"),
        table(
            [["File",                  "Description"],
             ["<stem>_zone.png",       "Overlay: kept=green, rejected=red, 1r/2r rings=blue"],
             ["<stem>_filtered.csv",   "Zone-passing segments with Step 1 measurements merged"],
             ["step3_summary.csv",     "Per-image: n_kept, n_rejected, disc_r, disc_conf"]],
            [4*cm, W-4*cm]
        ),
    ]

    # ── 7. Step 4 ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("7.  Step 4 — Arteriole / Venule Classification"),
        p("<i>File: step4_classify.py</i>", Caption),
        sp(4),
        h2("Goal"),
        p("Label each zone-filtered vessel segment as <b>arteriole (A)</b>, "
          "<b>venule (V)</b>, or <b>uncertain</b>. This classification feeds the "
          "Knudtson width-combination formula in Step 5."),
        sp(6),
        h2("Classification Algorithm (per image)"),
        p("1. <b>Normalise</b> color_score and width_px to [0, 1] range within that image."),
        p("2. <b>Combined score</b> = norm_color − norm_width"),
        table(
            [["Sign of combined score", "Interpretation"],
             ["Positive (redder AND narrower)", "Arteriole"],
             ["Negative (darker AND wider)",    "Venule"]],
            [7*cm, W-7*cm]
        ),
        sp(6),
        p("3. <b>Label assignment</b>:"),
        bullet("|combined_score| &lt; 0.05  →  uncertain"),
        bullet("combined_score &gt; median(combined)  →  arteriole"),
        bullet("else  →  venule"),
        sp(4),
        p("4. Sort all results by confidence (= |combined_score|) descending."),
        sp(6),
        h2("Assumptions"),
        bullet("Arterioles are <b>narrower and redder</b> than venules in the same image. "
               "This is a well-established clinical fact in retinal imaging."),
        bullet("Per-image normalisation makes the classifier dataset-agnostic — no "
               "cross-image calibration or training data required."),
        bullet("Median split as the decision boundary assumes a roughly equal mix of A and V "
               "per image, which holds for normal retinae."),
        bullet("Segments with color_score ≤ 0 or width_px ≤ 0 are invalid and discarded."),
        sp(6),
        h2("Known Limitations"),
        bullet("Entirely rule-based — no ground-truth training. On pathological images "
               "(silver-wiring, AV nicking), the A/V colour difference may be reduced, "
               "lowering classification accuracy."),
        bullet("Segment IDs in the overlay image may not perfectly align with classified IDs "
               "because the skeleton used for overlay (from Step 1) differs from the "
               "re-labelled skeleton in Step 3."),
        sp(6),
        h2("Outputs"),
        table(
            [["File",                      "Description"],
             ["<stem>_classified.csv",     "Segments with label, combined_score, confidence"],
             ["<stem>_classified.png",     "Overlay: arterioles=red, venules=blue, uncertain=yellow"],
             ["step4_summary.csv",         "Per-image: n_arteriole, n_venule, n_uncertain, n_total"]],
            [4.5*cm, W-4.5*cm]
        ),
    ]

    # ── 8. Step 5 ─────────────────────────────────────────────
    story += [
        PageBreak(),
        h1("8.  Step 5 — CRAE / CRVE / AVR Calculation"),
        p("<i>File: step5_avr.py</i>", Caption),
        sp(4),
        h2("Goal"),
        p("Compute the <b>Central Retinal Arteriole Equivalent (CRAE)</b>, "
          "<b>Central Retinal Venule Equivalent (CRVE)</b>, and their ratio "
          "<b>AVR = CRAE / CRVE</b> — the primary clinical output of the pipeline."),
        sp(6),
        h2("Knudtson-Modified Hubbard Formula"),
        p("Reference: Knudtson et al. (2003) <i>Ophthalmology</i> 110(8):1491–1496"),
        sp(4),
        h3("Protocol"),
        bullet("Take the <b>6 widest arterioles</b> and <b>6 widest venules</b> by width_px."),
        bullet("Apply iteratively, combining pairs widest-first:"),
        sp(4),
        code(
            "Arteriole:  W = 0.88 × √(w₁² + w₂²)\n"
            "Venule:     W = 0.95 × √(v₁² + v₂²)\n\n"
            "Repeat until one value remains per type.\n"
            "AVR = CRAE / CRVE"
        ),
        sp(6),
        h2("AVR Interpretation"),
        table(
            [["AVR Range",  "Flag",      "Clinical Interpretation"],
             ["0.6 – 0.8",  "normal",    "Normal microvascular calibre ratio"],
             ["< 0.6",      "low_avr",   "Arteriolar narrowing — hypertension, cardiovascular risk"],
             ["> 0.8",      "high_avr",  "Venular dilation — inflammation, metabolic syndrome"]],
            [2.5*cm, 2.5*cm, W-5*cm]
        ),
        sp(6),
        h2("Assumptions"),
        bullet("Vessel width in <b>pixels</b> is used directly (not converted to microns). "
               "CRAE/CRVE values are therefore image-resolution-dependent and not directly "
               "comparable across datasets with different camera fields of view."),
        bullet("Minimum 6 vessels of each type required. Images with fewer are skipped."),
        bullet("Top-6 selection by width matches the Knudtson protocol's intent to "
               "use the largest, most clinically significant vessels."),
        bullet("Width from Step 1 distance transform is a reasonable proxy for true "
               "vessel diameter under the symmetric cross-section assumption."),
        sp(6),
        h2("Outputs"),
        table(
            [["File",                "Description"],
             ["<stem>_avr.csv",     "12 selected vessels (6A + 6V) with selected_for and rank_width"],
             ["step5_summary.csv",  "CRAE, CRVE, AVR, flag per image + aggregate stats (min/max/mean/std, % in each range)"]],
            [3.5*cm, W-3.5*cm]
        ),
    ]

    # ── 9. End-to-end data flow ────────────────────────────────
    story += [
        PageBreak(),
        h1("9.  End-to-End Data Flow"),
        sp(4),
        code(
            "Input image\n"
            "    │\n"
            "    ├──[Step 1]──► <stem>_segments.csv      (all vessel segments + measurements)\n"
            "    │               <stem>_skeleton.png\n"
            "    │\n"
            "    ├──[Step 2]──► step2_disc_results.csv   (disc cx, cy, r, confidence)\n"
            "    │\n"
            "    ├──[Step 3]──► <stem>_filtered.csv      (zone-passing segments, merged measurements)\n"
            "    │\n"
            "    ├──[Step 4]──► <stem>_classified.csv    (A/V labels + scores)\n"
            "    │\n"
            "    └──[Step 5]──► <stem>_avr.csv           (top-6 A and V used)\n"
            "                   step5_summary.csv        (CRAE, CRVE, AVR, flag)"
        ),
    ]

    # ── 10. Global assumptions ────────────────────────────────
    story += [
        sp(12),
        h1("10.  Global Assumptions"),
        sp(4),
        table(
            [["#", "Assumption",                                             "Scope"],
             ["1",  "Green channel gives best vessel contrast",              "All steps"],
             ["2",  "Pixel width ≈ vessel calibre (no micron conversion)",   "Steps 1, 5"],
             ["3",  "Vessels are dark ridges on bright background (green ch)","Step 1"],
             ["4",  "Disc is in left or right third of image",               "Step 2 fallback"],
             ["5",  "Roughly equal mix of A and V per image",                "Step 4"],
             ["6",  "Top-6 widest vessels best represent CRAE/CRVE",         "Step 5"]],
            [0.7*cm, 9*cm, W-9.7*cm]
        ),
    ]

    # ── 11. Known issues ──────────────────────────────────────
    story += [
        sp(12),
        h1("11.  Known Issues & Corrections Needed"),
        sp(4),
        table(
            [["Location",                   "Issue",                                                              "Severity"],
             ["step3_zone_filter.py:116–173","Dead code after return — unreachable duplicate implementation",     "Low"],
             ["Step 4 overlay",             "Segment IDs from re-labelling may not match skeleton — wrong overlay colours", "Medium"],
             ["Step 5",                     "Width in pixels not microns — CRAE/CRVE not comparable cross-dataset","Medium"],
             ["Step 2 fallback",            "Positional prior assumes horizontal orientation — fails for rotated images", "Low"],
             ["Step 1 width",               "Distance transform × 2 assumes symmetric vessel cross-section",     "Low"]],
            [4*cm, 8.5*cm, W-12.5*cm]
        ),
    ]

    # ── 12. Usage reference ───────────────────────────────────
    story += [
        PageBreak(),
        h1("12.  Usage Quick Reference"),
        sp(4),
        h2("Step 1 — Segmentation"),
        code(
            "python step1_segment_skeleton.py \\\n"
            "    --input  C:/SepsisScope/data/DRIVE \\\n"
            "    --output C:/SepsisScope/out/step1 \\\n"
            "    --sigmas 1 8   --threshold 0.02   --workers 4"
        ),
        h2("Step 2 — Disc Detection"),
        code(
            "python step2_disc_detect.py \\\n"
            "    --images    C:/SepsisScope/data/DRIVE \\\n"
            "    --step1_out C:/SepsisScope/out/step1 \\\n"
            "    --output    C:/SepsisScope/out/step2"
        ),
        h2("Step 3 — Zone Filtering"),
        code(
            "python step3_zone_filter.py \\\n"
            "    --images    C:/SepsisScope/data/DRIVE \\\n"
            "    --step1_out C:/SepsisScope/out/step1 \\\n"
            "    --step2_csv C:/SepsisScope/out/step2/step2_disc_results.csv \\\n"
            "    --output    C:/SepsisScope/out/step3"
        ),
        h2("Step 4 — Classification"),
        code(
            "python step4_classify.py \\\n"
            "    --step3_drive  C:/SepsisScope/out/step3 \\\n"
            "    --step1_drive  C:/SepsisScope/out/step1 \\\n"
            "    --output_drive C:/SepsisScope/out/step4_drive \\\n"
            "    --dataset drive"
        ),
        h2("Step 5 — AVR Calculation"),
        code(
            "python step5_avr.py \\\n"
            "    --step4_drive  C:/SepsisScope/out/step4_drive \\\n"
            "    --output_drive C:/SepsisScope/out/step5_drive \\\n"
            "    --dataset drive"
        ),
        sp(12),
        HRFlowable(width="100%", thickness=1, color=GREY_LINE),
        sp(6),
        p("<i>End of SepsisScope Technical Documentation — v1.0 · 2026-05-18</i>", Caption),
    ]

    doc.build(story)
    print(f"PDF written -> {OUTPUT}")


if __name__ == "__main__":
    build()
