# dl_av — Deep-Learning AVR pipeline (AutoMorph-style)

> **GPU update:** this now trains in two stages. `train_vessel.py` pretrains a
> binary vessel-segmentation encoder on the 8-benchmark
> `retinal-vessel-fundus-dataset-collection/` (project root) — only LES-AV in
> that collection has artery/vein labels, the other 7 are vessel-only, so they
> can't feed the 4-class A/V task directly. `train.py` then fine-tunes the
> real A/V model on DRIVE_AV + LES-AV, initialised from that pretrained
> encoder. Both scripts use CUDA automatically if available.

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
| `unet.py` | Compact U-Net (configurable classes/base); trained here with base=64 |
| `datasets.py` | Pairs images<->masks across the 8-benchmark vessel collection |
| `download_data.py` | Fetch the DRIVE_AV training/test set |
| `train_vessel.py` | Stage 1: pretrain a binary vessel encoder on the full collection (GPU) |
| `train.py` | Stage 2: fine-tune the 4-class A/V model on DRIVE_AV + LES-AV + Fundus-AVSeg, report A/V accuracy per dataset |
| `infer.py` | Run on an image/folder → A/V overlay + CRAE/CRVE/AVR |
| `vessel_encoder.pth` | Stage-1 weights (produced by `train_vessel.py`) |
| `av_unet.pth` | Stage-2 weights (produced by `train.py`) — used by `infer.py` |

## Usage
```bash
# 1. (once) download DRIVE_AV labelled data
python dl_av/download_data.py

# 2. (once) place retinal-vessel-fundus-dataset-collection/ at the project root
#    (8 public benchmarks; only LES-AV in it has artery/vein labels, the rest
#    are vessel-only masks)

# 3. stage 1 — pretrain vessel encoder on all 8 benchmarks (GPU)
python dl_av/train_vessel.py                # -> dl_av/vessel_encoder.pth

# 4. stage 2 — fine-tune the A/V model, initialised from stage 1
python dl_av/train.py                       # -> dl_av/av_unet.pth

# 5. run on an image or a folder
python dl_av/infer.py sample/fundus.png --save sample/dl_results
python dl_av/infer.py "0.0.Normal" --save out_overlays
```
Both training scripts use CUDA automatically when available
(`torch.cuda.is_available()`), falling back to CPU otherwise. `train.py
--init-vessel ""` skips stage-1 transfer and trains the A/V encoder from
scratch, as before.

Output per image: `AVR`, `CRAE`, `CRVE`, artery/vein counts, and a reliability
flag (`high` / `moderate` / `low`). Overlays show arteries red, veins blue, with
the disc + 1r/2r zone rings.

## Training data
A/V-labelled (feed `train.py` directly):
- **DRIVE_AV** — 20 train / 20 test, pixel-level A/V labels (downloaded via `download_data.py`).
- **LES-AV** — 22 images, A/V labels from `masks_multiclass` (inside `retinal-vessel-fundus-dataset-collection/`).
- **Fundus-AVSeg** — 100 images (85 train / 15 test), colour-coded A/V labels in `annotation/` (project root).

Vessel-only, no A/V labels (feed `train_vessel.py`'s stage-1 pretraining instead):
- **`retinal-vessel-fundus-dataset-collection/`** — DRIVE, STARE, CHASEDB1, HRF,
  FIVES, RETA, TRENDS (+ LES-AV again, used for both stages) — 1042 image/mask
  pairs total. Binary vessel masks only; can't feed the 4-class A/V task.

Labels use the scheme `0=background, 1=artery, 2=vein, 3=crossing`.

## Measured performance
Stage 1 (`train_vessel.py`, 30 epochs, base=64, GPU): binary vessel Dice ~0.74
overall on held-out data, per dataset from 0.50 (TRENDS) to 0.79 (FIVES).

Stage 2 (`train.py`, 80 epochs, base=64, GPU, initialised from stage 1) — A/V
balanced accuracy on held-out test sets:

| Test set | Balanced accuracy |
|----------|-------------------|
| DRIVE | 0.878 |
| LES-AV | 0.890 |
| Fundus-AVSeg | 0.907 |

(Previous CPU proof-of-concept, DRIVE+LES only, 60 epochs, base=32, no
pretraining: DRIVE=0.736-0.787, LES=0.851 — GPU + more data + encoder
pretraining closed most of the gap toward the ~0.95 published SOTA.)

**Fuller picture (confusion matrix + per-class precision/recall/F1, via
`evaluate(..., full=True)` / `print_full_report()`):** balanced accuracy above
only tracks *recall* (did we find the artery/vein pixels), and it hides two
real weaknesses:
- **Precision is much lower than recall** (artery/vein precision ~0.39-0.53
  vs. recall ~0.87-0.91 on all 3 test sets) — the model over-predicts vessel
  classes, i.e. a fair number of background pixels get called artery/vein.
  Likely cause: the `[0.05, 2.5, 2.0, 1.0]` class weighting that fixes the
  background-dominance problem also pushes precision down as a side effect.
- **The `crossing` class is barely learned** (recall 0.05-0.21) — crossings are
  rare and visually ambiguous, and are outweighed by the artery/vein loss
  terms. Not critical for AVR (crossings aren't measured for width) but means
  the 4th class is close to unused in practice.

Stage 1 (vessel) has the same pattern: sensitivity 0.88-0.98 but precision
only 0.35-0.63 — same over-prediction bias, consistent with its own
`[0.2, 2.5]` vessel up-weighting.

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
