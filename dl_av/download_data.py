"""Download the DRIVE_AV dataset (fundus images + colour-coded artery/vein
ground truth) from the public Learning-AVSegmentation repo into dl_av/data/.

A/V label colour code (1st_manual PNGs): red = artery, blue = vein,
green = artery/vein crossing, white = uncertain, black = background.
"""
import os, json, urllib.request, ssl

REPO = "rmaphoh/Learning-AVSegmentation"
BRANCH = "main"
HERE = os.path.dirname(os.path.abspath(__file__))
ctx = ssl.create_default_context()

def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx).read()

tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"))
wanted = [t["path"] for t in tree["tree"]
          if t["path"].startswith("data/DRIVE_AV/")
          and t["path"].split("/")[2] in ("training", "test")
          and ("/images/" in t["path"] or "/1st_manual/" in t["path"])
          and t["path"].rsplit(".", 1)[-1] in ("tif", "png")]

print(f"{len(wanted)} files to download")
for i, path in enumerate(wanted, 1):
    dest = os.path.join(HERE, path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        continue
    raw = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    try:
        open(dest, "wb").write(fetch(raw))
        if i % 10 == 0 or i == len(wanted):
            print(f"  {i}/{len(wanted)}")
    except Exception as e:
        print(f"  FAIL {path}: {e}")
print("done ->", os.path.join(HERE, "data/DRIVE_AV"))
