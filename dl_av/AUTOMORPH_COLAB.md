# Validating our AVR against AutoMorph (Google Colab)

Goal: run the real, validated **AutoMorph** on a sample of your images using
Colab's free GPU, get its per-image AVR, and compare it to ours. It uses the
**Knudtson** formula in a disc-centred **Zone B** — same formula/zone concept as
our `dl_av`, so `AVR_Knudtson` is a fair comparison.

## How Colab works (the basics)
- Go to **https://colab.research.google.com** → **New notebook**.
- A notebook is a list of **cells** (grey boxes). You type code in a cell and
  **run it** by clicking the ▶ button on its left, or pressing **Shift+Enter**.
- Run the cells **top to bottom, one at a time**, waiting for each to finish (a
  green ✓ appears) before running the next.
- Add a new cell with **+ Code**.
- First, turn on the GPU: **Runtime → Change runtime type → T4 GPU → Save**.

Now make 7 cells, paste one block into each, and run them in order.

---

### Cell 1 — confirm the GPU is on
```python
!nvidia-smi
```

### Cell 2 — download AutoMorph
```python
%cd /content
!git clone https://github.com/rmaphoh/AutoMorph.git
%cd /content/AutoMorph
```

### Cell 3 — install dependencies (~3–5 min)
```python
!pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirement.txt
!pip install efficientnet_pytorch==0.7.1 --no-deps
```
> If Colab pops up **"Restart runtime"**, click it, then re-run **Cell 2**
> (`%cd /content/AutoMorph`) and carry on from Cell 4. (This installs the
> versions AutoMorph was tested with.)

### Cell 4 — upload your fundus images
```python
import os, shutil
from google.colab import files
os.makedirs('images', exist_ok=True)
for f in os.listdir('images'):                 # clear any demo images
    fp = os.path.join('images', f)
    if os.path.isfile(fp): os.remove(fp)
print("Pick 15–20 of your fundus images in the dialog…")
up = files.upload()                            # opens a file picker
for fn in list(up):
    shutil.move(fn, os.path.join('images', fn))
print(len(os.listdir('images')), "images ready in ./images")
```
Pick images where the **optic disc is clearly visible** (mix of your
`0.0.Normal` and `testx` sets is good). Keep original filenames.

### Cell 5 — resolution file (value doesn't matter for AVR)
```python
!python generate_resolution.py
```
AVR is a ratio, so the pixel-size value cancels out — the default is fine.

### Cell 6 — run the whole AutoMorph pipeline (~few min on GPU)
```python
!bash run.sh
```
This runs preprocess → quality → vessel/artery-vein + disc segmentation →
feature measurement.

### Cell 7 — show + download the AVR results
```python
import pandas as pd
df = pd.read_csv('Results/M3/Disc_centred/Disc_Zone_B_Measurement.csv')
print(df[['Name','AVR_Knudtson','CRAE_Knudtson','CRVE_Knudtson']].to_string())
from google.colab import files
files.download('Results/M3/Disc_centred/Disc_Zone_B_Measurement.csv')
```
This prints AutoMorph's AVR per image and downloads the CSV to your computer.

---

## Bring it back
Save the downloaded CSV into the project as **`dl_av/automorph_results.csv`** and
tell me. I'll join it by filename against `dl_av/our_avr_results.csv` and report
how closely our AVR tracks AutoMorph's (correlation, mean difference, ranking,
and where/why they diverge).

## If something breaks
- Disc not found in an image → that row is `NaN`; just ignore those.
- If Cell 3 fights versions, the **official AutoMorph Colab** (in their README)
  is an alternative — run its cells top-to-bottom the same way, upload images to
  its `images` folder, and grab the same `Disc_Zone_B_Measurement.csv`.
