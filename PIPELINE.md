# SepsisScope — Project Pipeline

This document explains, end to end, how SepsisScope turns a **fundus (retina)
photo** into an **Arteriovenous Ratio (AVR)** — the ratio of average artery
width to average vein width near the optic disc, a marker used to screen for
hypertensive/vascular changes in the eye.

The project contains **two independent pipelines** that compute the same number
in two different ways:

| Pipeline | Where it lives | How it works | Status |
|----------|----------------|--------------|--------|
| **Classical** | `step*.py` + `api/analyze.py` + `src/` | Image-processing heuristics | **Deployed** (the web app) |
| **Deep-learning** | `dl_av/` | A U-Net segments arteries/veins, classical maths does the rest | **Local experiment** (validation / research) |

Both produce **CRAE**, **CRVE**, and **AVR = CRAE / CRVE** using the same
**Knudtson-modified Hubbard formula (2003)**. Normal AVR is ~**0.6–0.8**.

---

## 1. The big picture

```
                        ┌─────────────────────────────────────────────┐
   Fundus image  ─────► │  SEGMENT vessels  →  FIND optic disc  →      │
                        │  DEFINE 1r–2r zone  →  CLASSIFY artery/vein  │ ─────► AVR
                        │  →  MEASURE widths  →  KNUDTSON formula      │        + CRAE/CRVE
                        └─────────────────────────────────────────────┘        + reliability
```

Everything hinges on one clinical convention: vessel calibre is only measured in
a **ring around the optic disc**, between **1× and 2× the disc radius (1r–2r)**.
That is where the six largest arteries and six largest veins are combined into
CRAE and CRVE.

---

## 2. The deployed web app (classical pipeline)

### 2.1 Front end — `src/` (Next.js + React + TypeScript)

The browser app the user actually sees.

| File | Role |
|------|------|
| `src/app/page.tsx` | Main page; holds state and calls the API |
| `src/components/UploadZone.tsx` | Drag-and-drop image upload |
| `src/components/ProgressSteps.tsx` | Shows the 5 pipeline stages running |
| `src/components/ImageViewer.tsx` | Displays overlays (vessels, disc, zone, accept/reject audit) |
| `src/components/ResultsPanel.tsx` | Shows AVR, CRAE, CRVE, reliability banner, accepted/rejected vessel funnel |
| `src/components/AVRGauge.tsx` | The gauge/needle that visualises the AVR value |
| `src/types/index.ts` | Shared TypeScript types for the API response |

**Flow:** user uploads an image → the front end POSTs it to the API → the API
returns AVR + intermediate overlays as base64 images + a reliability assessment →
the components render them.

### 2.2 API — `api/analyze.py` (Flask)

The orchestrator. It receives the uploaded image and runs the five steps in
order, then packages the result.

Key responsibilities beyond just calling the steps:

- **`skimage_shim`** — a hand-written stand-in for scikit-image. The hosting
  runtime (Vercel/Render) crashes on the real scikit-image package, so
  `api/skimage_shim.py` provides just the functions we use (`frangi`,
  `skeletonize` via Zhang–Suen thinning, `remove_small_objects`, etc.). It is
  registered in `sys.modules` as `skimage` **before** the step modules import it.
- **Auto-downscale** — very large uploads are resized so the longest side is
  ≤1024 px (`MAX_DIM`), which keeps processing fast (seconds, not 30 s+).
- **`filter_step1_by_zone()`** — keeps only vessels whose location falls inside
  the measurement ring, with an adaptive fallback if too few survive.
- **`_build_vessel_audit()`** — records which vessels were **accepted** vs
  **rejected** (and why), so the UI can show it visually.
- **`_assess_reliability()`** — returns `high` / `moderate` / `low` based on how
  many vessels were usable and whether the AVR is physiologically plausible.
- **`_draw_audit_overlay()`** — renders the accept/reject overlay image.

### 2.3 The five steps

Each step is a separate Python module so it can be understood and tested alone.

```
step_green_channel.py   (preprocessing)
        │
        ▼
step1_segment_skeleton.py ── vessel map + skeleton + per-vessel features
        │
        ▼
step2_disc_detect.py ────── optic disc centre + radius
        │
        ▼
step3_zone_filter.py ────── keep only vessels in the 1r–2r ring
        │
        ▼
step4_classify.py ───────── label each vessel artery or vein
        │
        ▼
step5_avr.py ────────────── CRAE, CRVE, AVR (Knudtson)
```

**Step 0 — `step_green_channel.py` (preprocessing).**
The green channel of a fundus photo has the best vessel contrast. This extracts
it and applies **CLAHE** (contrast-limited adaptive histogram equalisation) and
shade correction so vessels stand out evenly across the image.

**Step 1 — `step1_segment_skeleton.py` (find the vessels).**
- Builds a **retinal FOV mask** (`retinal_fov_mask()`) using HSV saturation +
  image moments to find the circular retina region and its centre/radius — this
  stops the disc and zone from being placed *outside* the eye.
- **Frangi vesselness** filter highlights tube-like structures (vessels).
- **Percentile thresholding** turns that into a binary vessel map.
- **Zhang–Suen thinning** reduces each vessel to a 1-pixel-wide **skeleton**
  (centre-line). *(This replaced a morphological skeletoniser that was
  shattering vessels and causing the original "Insufficient arterioles" error.)*
- Computes a per-vessel **`reflex_ratio`** feature (centre-line green brightness
  ÷ whole-vessel green) used later for artery/vein classification.

**Step 2 — `step2_disc_detect.py` (find the optic disc).**
- `detect_disc_combined()` fuses **vessel density** (vessels converge on the
  disc) and **brightness** (the disc is the brightest region) to locate the
  disc centre.
- `estimate_disc_radius()` estimates its radius.
- Guardrails clamp the centre and radius to stay inside the FOV mask.

**Step 3 — `step3_zone_filter.py` (the measurement ring).**
Keeps only the vessel segments whose distance from the disc centre lies between
**1× and 2×** the disc radius — the standard AVR measurement zone.

**Step 4 — `step4_classify.py` (artery vs vein).**
- Arteries are brighter/have a stronger central light reflex; veins are darker
  and wider. The classifier scores each vessel on the `reflex_ratio` /
  green-brightness feature and does a **median split** into artery vs vein.
- Returns a `separation` score (Cohen's d) indicating how cleanly the two groups
  divided — feeds the reliability estimate.
- *Important caveat:* width is deliberately **not** used to classify (that would
  be circular — we then measure width to compute AVR).

**Step 5 — `step5_avr.py` (the number).**
- `_drop_width_outliers()` removes implausible widths using a Tukey fence.
- Takes the widths of the largest arteries and veins and combines them with
  `knudtson_combine()`:
  - **CRAE** = combined arteriolar equivalent (arteries)
  - **CRVE** = combined venular equivalent (veins)
- **AVR = CRAE / CRVE.**
- Also returns how many outliers were dropped and how many vessels were used.

### 2.4 The AVR formula (Knudtson-modified Hubbard, 2003)

Vessels are combined **pairwise** (widest with narrowest) rather than simply
averaged. For each pair of widths `w1 ≥ w2`:

- Arteries: `W = 0.88 × √(w1² + w2²)`
- Veins:   `W = 0.95 × √(w1² + w2²)`

This is applied iteratively until a single CRAE / CRVE value remains. The
constants make the result independent of how many vessels you started with —
which is why AVR is a *ratio* and pixel size cancels out.

---

## 3. The deep-learning pipeline — `dl_av/`

A separate, **local-only** experiment built to the same design as
[AutoMorph](https://github.com/rmaphoh/AutoMorph): **deep learning for
perception, classical maths for the number.** Trains on GPU (CUDA
auto-detected, falls back to CPU) in two stages, since only some of the
available benchmarks have artery/vein ground truth:

```
Stage 1 (train_vessel.py): 8 vessel-only benchmarks ──► binary vessel encoder
Stage 2 (train.py):        A/V-labelled images, init'd from stage 1 ──► A/V model

Fundus image
   │
   ├─►  U-Net  ──►  per-pixel {background, artery, vein, crossing}   ← deep learning
   │
   └─►  classical morphometry (reuses the app's disc + zone + Knudtson):
          • retinal FOV mask + optic-disc detection
          • 1r–2r measurement zone
          • per-vessel width (distance transform, median)
          • CRAE / CRVE / AVR
```

| File | Role |
|------|------|
| `unet.py` | Compact U-Net (configurable classes/base); trained here with base=64 |
| `datasets.py` | Pairs images↔masks across the 8-benchmark vessel-only collection |
| `download_data.py` | Fetches the DRIVE_AV labelled training set |
| `train_vessel.py` | Stage 1: pretrains a binary vessel encoder on all 8 vessel-only benchmarks |
| `train.py` | Stage 2: fine-tunes the 4-class A/V model on DRIVE_AV + LES-AV + Fundus-AVSeg |
| `infer.py` | Runs the model on an image/folder → overlay + CRAE/CRVE/AVR |
| `vessel_encoder.pth` / `av_unet.pth` | Stage 1 / stage 2 trained weights |

**Why a U-Net and not an "AVR network"?** There is **no** neural network that
outputs AVR directly — neither here nor in AutoMorph — because there is no
large-scale ground-truth AVR dataset to train one on. The network only does
**segmentation** (which pixels are artery vs vein); the AVR is then computed
from that map by the same classical geometry as the web app.

**Training data**

A/V-labelled (stage 2, `train.py`):
- **DRIVE_AV** — 20 train / 20 test, pixel-level A/V labels.
- **LES-AV** — 22 images (colour-coded A/V labels, converted to indices).
- **Fundus-AVSeg** — 100 images (85 train / 15 test), same colour-coded scheme.

Vessel-only, no A/V labels (stage 1 pretraining, `train_vessel.py`):
- **`retinal-vessel-fundus-dataset-collection/`** — DRIVE, STARE, CHASEDB1,
  HRF, FIVES, RETA, TRENDS (+ LES-AV again) — 1042 image/mask pairs total.

Label scheme everywhere: `0=background, 1=artery, 2=vein, 3=crossing`.

**Augmentation (`train.py`, `train_vessel.py`)** — beyond flips/rotations,
training applies **photometric** augmentation (brightness/contrast, gamma,
colour cast, blur, noise) and (stage 2 only) **synthetic lesions** (dark
haemorrhage / bright exudate blobs), so the network stays accurate on
diseased-looking images. Photometric transforms touch the image only, never
the label, and run *before* normalisation.

**Measured performance** (GPU, base=64; see `dl_av/README.md` for the full
per-class breakdown): A/V balanced accuracy on held-out test sets — DRIVE
0.878, LES-AV 0.890, Fundus-AVSeg 0.907 (up from a CPU/base=32 proof-of-concept
that scored DRIVE 0.736-0.787, LES-AV 0.851). Confusion-matrix analysis shows
recall is high (0.87-0.91) but **precision is lower** (0.39-0.53) — the model
over-predicts artery/vein onto background pixels — and the **crossing class is
barely learned** (recall 0.05-0.21), a side effect of the class weights used
to fight background dominance.

**Run it on a single image:**
```bash
python dl_av/infer.py "path/to/image.jpg" --save dl_av/out
```
Prints `AVR / CRAE / CRVE / artery & vein counts / reliability`, and (with
`--save`) writes an overlay: arteries red, veins blue, disc + 1r/2r rings.

---

## 4. Validation against AutoMorph

To check whether the DL pipeline is genuine, its AVR was compared image-by-image
against the real, published **AutoMorph** tool (run on Colab; its
`AVR_Knudtson_zone_b` column is the reference). See
`dl_av/AUTOMORPH_COLAB.md` and the `automorph_results*.csv` files.

> This comparison predates the GPU retrain in section 3 (it ran against the
> CPU/base=32 model). It hasn't been re-run against the current `av_unet.pth`
> — the findings below reflect the older, less accurate A/V model.

**Findings:**
- **Normal eyes:** both tools agree on the mean AVR (~0.67–0.75, both in the
  healthy band) → our **calibration is correct**.
- **Diseased (hypertensive) eyes:** AutoMorph's AVR correctly trends *lower*;
  ours does not, and per-image the two anti-correlate (r ≈ −0.53). AutoMorph is
  **better on diseased images** because it has a better-trained A/V network, a
  deep-learning disc detector, and image-quality gating.

---

## 5. Known limitations (both pipelines)

- **AVR cannot, by itself, reliably detect hypertension.** AVR is computed from
  the **outer** vessel calibre, but hypertensive narrowing happens in the
  **lumen** (inside), which the outer width barely reflects. Even AutoMorph
  shifts AVR only ~0.1 between normal and hypertensive. This is an *intrinsic*
  ceiling, not a bug.
- Values are a research **estimate**, not a clinically calibrated measurement
  (no ground-truth AVR labels were available to calibrate against).
- Measurements are in **pixels**, not microns (no per-image resolution
  calibration) — fine for a ratio like AVR.
- The classical disc detector and lack of quality gating make both pipelines
  degrade on low-quality or heavily diseased images.

**If the goal is detecting disease** (rather than the AVR number), an end-to-end
image classifier trained directly on labelled fundus images — which sees the
haemorrhages/exudates directly — will outperform any computed-AVR approach.
(`dl_classifier/` is reserved for this direction but is currently empty.)
