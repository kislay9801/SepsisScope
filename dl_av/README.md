# dl_av — Deep-Learning AVR pipeline (AutoMorph-style)

A self-contained, **local-only** experiment that computes the Arteriovenous
Ratio (AVR) using a deep-learning artery/vein segmenter — the same architecture
as [AutoMorph](https://github.com/rmaphoh/AutoMorph): **deep learning for
perception, classical math for the AVR number.** It is kept separate from the
deployed web app (which uses the lighter classical pipeline).

```
Fundus image
   │
   ├─►  U-Net  ──►  per-pixel {background, artery, vein, crossing}   ← deep learning
   │
   └─►  classical morphometry:
          • retinal FOV mask + optic-disc detection
          • 1r–2r measurement zone
          • per-vessel width (distance transform, median)
          • CRAE / CRVE / AVR  (Knudtson-modified Hubbard, 2003)
```

> There is **no neural network that outputs AVR directly** — neither here nor in
> AutoMorph. A per-image AVR regressor can't be trained (no large-scale AVR
> ground truth). The DL part is segmentation + A/V classification; the AVR is
> computed from those maps by classical geometry.

## Files
| File | Purpose |
|------|---------|
| `unet.py` | Compact 4-class U-Net (bg / artery / vein / crossing), base=32 |
| `download_data.py` | Fetch the DRIVE_AV training/test set |
| `train.py` | Train on DRIVE_AV + LES-AV, report A/V accuracy per dataset |
| `infer.py` | Run on an image/folder → A/V overlay + CRAE/CRVE/AVR |
| `av_unet.pth` | Trained weights (produced by `train.py`) |

## Usage
```bash
# 1. (once) download DRIVE_AV labelled data
python dl_av/download_data.py

# 2. (once) train — CPU, ~ a couple of hours; produces av_unet.pth
python dl_av/train.py

# 3. run on an image or a folder
python dl_av/infer.py sample/fundus.png --save sample/dl_results
python dl_av/infer.py "0.0.Normal" --save out_overlays
```
Output per image: `AVR`, `CRAE`, `CRVE`, artery/vein counts, and a reliability
flag (`high` / `moderate` / `low`). Overlays show arteries red, veins blue, with
the disc + 1r/2r zone rings.

## Training data
- **DRIVE_AV** — 20 train / 20 test, pixel-level A/V labels (downloaded).
- **LES-AV** — 22 images, A/V labels from `masks_multiclass` (added locally).
- *(HRF added locally has only binary vessel masks — no A/V labels — so it is
  not used for A/V training.)*

Labels use the scheme `0=background, 1=artery, 2=vein, 3=crossing`.

## Measured performance
A/V balanced accuracy on held-out test sets (CPU proof-of-concept, 60 epochs):

| Model | DRIVE test | LES test |
|-------|-----------|----------|
| DRIVE-only | 0.736 | – |
| **DRIVE + LES** | **0.787** | **0.851** |

Adding the second dataset improved DRIVE *and* generalised to LES's different
camera/resolution. Accuracy was still rising at 60 epochs — more epochs / a GPU
would close the gap toward the ~0.95 published SOTA.

On a 38-image normal set the AVR is **~0.78, std ~0.12** (believable, in the
normal 0.6–0.8 band) and generalises across image types — a clear improvement
over the classical heuristic (which classified A/V near-randomly).

## Honest limitations
- **Does not detect hypertension.** On a 15-image severe-hypertensive set the
  AVR does **not** separate from normal (AUC ≈ 0.46). This is **not** a
  classification problem — improving A/V accuracy (0.74→0.85) did not help. The
  cause is intrinsic: AVR is computed from **outer vessel caliber**, but
  hypertensive narrowing is in the **lumen**, which outer width barely reflects;
  pathology (hemorrhages/exudates) further corrupts measurement. AutoMorph has
  the **same** ceiling for the same reason.
- **Not validated** against expert AVRs (no ground-truth AVR labels available),
  so values are a research **estimate**, not a calibrated measurement.
- **Pixel units**, not microns (no per-image resolution calibration).
- Classical disc detection (no DL disc model) and no image-quality gating.

## If you need more
- **Better-validated AVR values:** run real AutoMorph (Colab/Docker) — better A/V
  weights, DL disc, quality gating, micron calibration.
- **Detecting disease** (the actual clinical goal): train an end-to-end image
  classifier on labelled images — it sees hemorrhages/exudates directly and will
  beat computed-AVR, which has a real ceiling for this task.
