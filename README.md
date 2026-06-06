# SepsisScope — Retinal AVR Analysis

A professionally-designed web application for automated retinal fundus image analysis. Computes the **Arteriovenous Ratio (AVR)** — a non-invasive microvascular biomarker associated with hypertension, cardiovascular risk, and sepsis-related microvascular injury.

## Features

- **Upload** any fundus image (PNG, JPG, TIFF, BMP, PPM)
- **5-step automated pipeline** with live progress indicators
- **AVR gauge** with clinical interpretation
- **Pipeline visualisations** — vessel overlay, disc detection, zone filter, A/V classification
- **Deployment-ready** for Vercel (Next.js 16 + Python serverless)

## Pipeline

| Step | Description | Algorithm |
|------|-------------|-----------|
| 1 | Vessel Segmentation | Frangi vesselness + CLAHE + skeletonisation |
| 2 | Optic Disc Detection | Vessel density heatmap + brightness fallback |
| 3 | Zone Filtering | 1r–2r annular zone around optic disc |
| 4 | A/V Classification | Colour score + width normalisation |
| 5 | AVR Calculation | Knudtson-modified Hubbard formula (2003) |

## AVR Reference Ranges

| AVR | Interpretation |
|-----|----------------|
| < 0.6 | **Low AVR** — arteriolar narrowing (hypertension, cardiovascular risk, sepsis) |
| 0.6 – 0.8 | **Normal** — healthy microvascular calibre |
| > 0.8 | **High AVR** — venular dilation (inflammation, metabolic syndrome) |

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

> **Note:** The Python pipeline uses `opencv-python-headless`, `scikit-image`, `scipy` and `numpy`. These fit within Vercel's 250 MB function limit. First cold start may take ~5–10 s.

### Environment Variables

No environment variables are required for the default deployment.

## Tech Stack

- **Frontend:** Next.js 16, React 18, TypeScript, Tailwind CSS
- **Backend:** Python 3.12, Flask, OpenCV (headless), scikit-image, scipy
- **Algorithm:** Frangi filter, Knudtson 2003 formula
- **Deployment:** Vercel (Next.js + Python serverless)

## Disclaimer

> **Research use only.** This tool computes pixel-based vessel widths (not microns) and is not validated for clinical diagnosis. Results should be interpreted by a qualified clinician. SepsisScope is not a medical device.

## Reference

Knudtson MD et al. (2003). *Revised formulas for summarizing retinal vessel diameters.* Current Eye Research, **27**(3):143–149. PMID: 12917150
