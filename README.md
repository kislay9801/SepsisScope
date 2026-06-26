# SepsisScope — Retinal AVR Analysis

A professionally-designed web application for automated retinal fundus image analysis. Computes the **Arteriovenous Ratio (AVR)** — a non-invasive microvascular biomarker associated with hypertension, cardiovascular risk, and sepsis-related microvascular injury.

## Features

- **Upload** any fundus image (PNG, JPG, TIFF, BMP, PPM)
- **5-step automated pipeline** with live progress indicators
- **AVR gauge** with clinical interpretation
- **Pipeline visualisations** — vessel overlay, disc detection, zone filter, A/V classification
- **Deployment-ready** for Vercel (Next.js 16 + Python serverless)

## Pipeline

| Step | Description          | Algorithm                                                            |
| ---- | -------------------- | ------------------------------------------------------------------- |
| 1    | Vessel Segmentation  | Shade correction + CLAHE + Frangi vesselness + Zhang–Suen thinning  |
| 2    | Optic Disc Detection | Fused vessel-density × brightness, confined to the retina (FOV)     |
| 3    | Zone Filtering       | Adaptive 1r–2r annular zone around the optic disc                   |
| 4    | A/V Classification   | Central light-reflex ratio, median split (width never used)         |
| 5    | AVR Calculation      | Knudtson-modified Hubbard formula (2003), width-outlier trimmed      |

### Robustness & preprocessing

The pipeline is built to handle real-world uploads, not just clean research-dataset images:

- **Colour-based field-of-view (FOV) detection** — the retina is isolated by HSV
  saturation (it is the only strongly coloured region), so black, white and
  transparent/checkerboard backgrounds are all handled. The FOV circle is fit
  from the contour's area centroid and area-equivalent radius, which stays
  inside the visible retina even when the image is zoomed in and clipped —
  reliably excluding the bright camera-aperture rim that would otherwise be
  mistaken for vessels or the optic disc.
- **Illumination correction** — a large-scale background is subtracted so faint
  arterioles survive uneven lighting.
- **Stable thresholding** — vessels are kept by a percentile of vesselness
  strength rather than a fixed cutoff, so coverage is consistent across images.
- **Disc guardrails** — the optic disc centre and the 1r–2r zone are always kept
  inside the field of view.
- **Adaptive zone** — if too few segments fall in the nominal 1r–2r annulus the
  band widens automatically, so the AVR is computed instead of failing with
  "insufficient arterioles".
- **Auto-downscaling** — uploads larger than 1024 px on the longest side are
  resized before processing, keeping analysis fast (a few seconds) regardless of
  source resolution.
- **A/V classification by the central light reflex** — arterioles have a bright
  reflective stripe down their centre, so each vessel's centre-line brightness
  is compared with its whole-vessel brightness. Width is **never** used to
  classify (that would bias the very widths Step 5 measures).
- **Width-outlier trimming** — vessels that merge near the disc read as one
  abnormally thick vessel; these Tukey-fence outliers are dropped before the
  Knudtson formula so a single artefact can't dominate CRVE.
- **Reliability indicator** — every result carries a confidence level
  (high / moderate / low) with reasons (e.g. implausible AVR > 1, too few
  vessels, weak A/V colour separation, low disc confidence), shown in the UI.
- **Vessel acceptance breakdown** — an overlay + funnel show exactly which
  detected vessels were accepted vs rejected (and why) at each stage.

## AVR Reference Ranges

| AVR       | Interpretation                                                                 |
| --------- | ------------------------------------------------------------------------------ |
| < 0.6     | **Low AVR** — arteriolar narrowing (hypertension, cardiovascular risk, sepsis) |
| 0.6 – 0.8 | **Normal** — healthy microvascular calibre                                     |
| > 0.8     | **High AVR** — venular dilation (inflammation, metabolic syndrome)             |

## Validation & Limitations

> **This tool produces an AVR _estimate_. It is not validated for disease detection.**

The pipeline was tested on a 38-image normal-eye dataset and a 15-image severe
hypertensive-retinopathy dataset:

- **On normal eyes** the AVR estimate is stable and centred near the lower end of
  the normal range (mean ≈ 0.56, low scatter, no label flips).
- **It does not separate hypertensive from normal eyes** (AUC ≈ 0.4 across five
  different classification features, including an oximetry-based optical-density
  ratio). Healthy and hypertensive distributions overlap almost completely.

This is a fundamental limit of fully-automated, single-image AVR, not a tuning
bug, for two reasons:

1. **Artery/vein colour difference is tiny** in an uncalibrated fundus photo and
   is swamped by illumination, camera, and exposure variation — so unsupervised
   classification is inherently noisy.
2. **We measure outer vessel calibre**, but hypertensive narrowing is of the
   blood-column (lumen), which the outer width barely reflects.

Clinical-grade AVR would require a **supervised deep-learning artery/vein
classifier** (e.g. trained on RITE / INSPIRE-AVR / VICAVR) plus lumen-aware,
full-resolution calibre measurement, and ground-truth AVRs for validation. That
is deliberately out of scope here — treat the output as a research estimate.

## Local Development

### Frontend

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Backend (Python API)

```bash
pip install -r requirements.txt
cd api
python analyze.py
# API runs at http://localhost:5000
```

## Vercel Deployment

1. **Fork / push** this repository to GitHub
2. **Import** the repo in [Vercel](https://vercel.com/new)
3. Vercel auto-detects Next.js + Python serverless function
4. Deploy — no extra configuration needed

> **Note:** The Python pipeline runs on `opencv-python-headless`, `scipy` and
> `numpy` only. The scikit-image routines it needs (Frangi filter, Zhang–Suen
> thinning, connected-component labelling) are reimplemented in
> [`api/skimage_shim.py`](api/skimage_shim.py) on top of those libraries, which
> avoids a packaging bug in scikit-image on serverless runtimes and keeps the
> bundle well within Vercel's 250 MB function limit. First cold start may take
> ~5–10 s.

### Environment Variables

No environment variables are required for the default deployment.

## Tech Stack

- **Frontend:** Next.js 16, React 18, TypeScript, Tailwind CSS
- **Backend:** Python 3.12, Flask, OpenCV (headless), SciPy, NumPy
- **Algorithm:** Frangi vesselness, Zhang–Suen thinning, Knudtson 2003 formula
- **Deployment:** Vercel or Render (Next.js + Python serverless)

## Disclaimer

> **Research use only.** This tool computes pixel-based vessel widths (not microns) and is not validated for clinical diagnosis. Results should be interpreted by a qualified clinician. SepsisScope is not a medical device.

## Reference

Knudtson MD et al. (2003). _Revised formulas for summarizing retinal vessel diameters._ Current Eye Research, **27**(3):143–149. PMID: 12917150
