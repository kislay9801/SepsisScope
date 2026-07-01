"""Run the trained A/V U-Net on a fundus image (or a folder) and compute AVR.

The deep model replaces ONLY the artery/vein decision; disc detection, the
1r–2r measurement zone, and the Knudtson AVR formula are reused from the main
pipeline so results are comparable to the classical version.

Usage:
  python dl_av/infer.py <image_or_folder> [--save overlay_dir]
"""
import os, sys, glob, argparse, math
import numpy as np
import cv2
import torch
from skimage.morphology import skeletonize
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from unet import UNet
from step1_segment_skeleton import retinal_fov_mask, _find_branch_points
from step2_disc_detect import detect_disc_combined, estimate_disc_radius
from step5_avr import knudtson_combine

SIZE = 512
_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        net = UNet(3, 4, base=32)
        net.load_state_dict(torch.load(os.path.join(HERE, "av_unet.pth"),
                                       map_location="cpu"))
        net.eval()
        _MODEL = net
    return _MODEL


def _preprocess(bgr):
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(bgr[:, :, 1])
    x = g.astype(np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return np.stack([x, x, x], 0)


@torch.no_grad()
def predict_av(bgr):
    """Return artery & vein binary masks at the image's native resolution."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (SIZE, SIZE))
    x = torch.from_numpy(_preprocess(small)[None])
    pred = _model()(x).argmax(1)[0].cpu().numpy().astype(np.uint8)   # 0..3
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    artery = (pred == 1).astype(np.uint8)
    vein = (pred == 2).astype(np.uint8)
    return artery, vein


def _seg_widths_in_zone(class_mask, disc, h, w):
    """Width (px) of each vessel segment of one class whose centroid lies in the
    1r–2r zone. Width = 2·mean distance-transform along the segment skeleton."""
    cx, cy, r = disc["cx"], disc["cy"], disc["r"]
    dt = cv2.distanceTransform(class_mask, cv2.DIST_L2, 5)
    skel = skeletonize(class_mask.astype(bool)).astype(np.uint8)
    skel[_find_branch_points(skel) == 1] = 0
    widths = []
    for reg in measure.regionprops(measure.label(skel, connectivity=2)):
        if reg.area < 8:
            continue
        ys, xs = reg.coords[:, 0], reg.coords[:, 1]
        scy, scx = reg.centroid
        dist = math.hypot(scx - cx, scy - cy)
        if r <= dist <= 2 * r:
            # Median (not mean) of the per-pixel calibre → robust to the wide
            # readings at crossings/branch stubs along the segment.
            widths.append(2.0 * float(np.median(dt[ys, xs])))
    return widths


def analyze(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    # downscale very large images (same as the main pipeline)
    scale = 1024.0 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    artery, vein = predict_av(bgr)
    fov, fcx, fcy, fr = retinal_fov_mask(bgr, shrink_frac=0.08)
    artery &= (fov > 0); vein &= (fov > 0)

    vessel = ((artery | vein) > 0).astype(np.uint8)
    skel = skeletonize(vessel.astype(bool)).astype(np.uint8)
    cx, cy, conf = detect_disc_combined(skel, img_rgb, h, w, fov)
    if cx is None:
        cx, cy = fcx, fcy
    disc_r = estimate_disc_radius(img_rgb, cx, cy, h, w)
    disc = {"cx": cx, "cy": cy, "r": disc_r}

    aw = sorted(_seg_widths_in_zone(artery, disc, h, w), reverse=True)[:6]
    vw = sorted(_seg_widths_in_zone(vein, disc, h, w), reverse=True)[:6]
    if not aw or not vw:
        return {"avr": None, "n_art": len(aw), "n_ven": len(vw),
                "artery": artery, "vein": vein, "disc": disc, "img": img_rgb}
    crae = knudtson_combine(aw, "arteriole")
    crve = knudtson_combine(vw, "venule")
    avr = round(crae / crve, 4) if crve else None
    # Simple reliability flag: enough vessels + a physiologically plausible AVR.
    used = len(aw) + len(vw)
    if used < 8 or (avr is not None and avr > 1.0):
        reliab = "low"
    elif used < 10:
        reliab = "moderate"
    else:
        reliab = "high"
    return {"avr": avr, "crae": round(crae, 3), "crve": round(crve, 3),
            "n_art": len(aw), "n_ven": len(vw), "reliab": reliab,
            "artery": artery, "vein": vein, "disc": disc, "img": img_rgb}


def save_overlay(res, out_path):
    ov = res["img"].copy()
    ov[res["artery"] > 0] = [255, 40, 40]     # red = artery
    ov[res["vein"] > 0] = [40, 80, 255]       # blue = vein
    d = res["disc"]
    cv2.circle(ov, (d["cx"], d["cy"]), d["r"], [255, 255, 0], 2)
    cv2.circle(ov, (d["cx"], d["cy"]), 2 * d["r"], [255, 255, 0], 1)
    cv2.putText(ov, f"AVR={res['avr']}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--save", default=None, help="dir to write overlays")
    args = ap.parse_args()

    paths = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "*"))))
    paths = [p for p in paths if p.lower().endswith(
        (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm"))]
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    avrs = []
    for p in paths:
        res = analyze(p)
        if res is None:
            continue
        if res["avr"] is not None:
            avrs.append(res["avr"])
        print(f"{os.path.basename(p)[:24]:<26} AVR={res['avr']}  "
              f"CRAE={res.get('crae')} CRVE={res.get('crve')}  "
              f"A={res['n_art']} V={res['n_ven']}  [{res.get('reliab','-')}]")
        if args.save and res["avr"] is not None:
            save_overlay(res, os.path.join(args.save, os.path.basename(p) + "_av.png"))
    if len(avrs) > 1:
        a = np.array(avrs)
        print(f"\nn={len(a)}  mean={a.mean():.3f}  median={np.median(a):.3f}  std={a.std():.3f}")


if __name__ == "__main__":
    main()
