"""Train the A/V U-Net on DRIVE_AV + LES-AV and report artery/vein accuracy on
the held-out test split. Uses GPU automatically if available.

Label encoding (verified): 0=background, 1=artery, 2=vein, 3=crossing.
Run:  python dl_av/train.py
"""
import os, glob, time, argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from unet import UNet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data", "DRIVE_AV")
# LES-AV lives inside the merged benchmark collection, not a standalone folder.
LES = os.path.join(ROOT, "retinal-vessel-fundus-dataset-collection", "LES-AV")
FUNDUS_AVSEG = os.path.join(ROOT, "Fundus-AVSeg")
VESSEL_ENCODER = os.path.join(HERE, "vessel_encoder.pth")
SIZE = 512                      # train/infer resolution
# Class scheme everywhere: 0=background, 1=artery, 2=vein, 3=crossing/uncertain
torch.manual_seed(0)
np.random.seed(0)


def preprocess_img(bgr):
    """Green-channel CLAHE + per-image normalisation → 3-ch float tensor input.
    Using the contrast-enhanced green channel (replicated) makes vessels pop and
    is a standard retinal preprocessing choice."""
    g = bgr[:, :, 1]
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    x = g.astype(np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return np.stack([x, x, x], 0)          # (3, H, W)


def _drive_label(lp):
    """DRIVE_AV labels already use index values 0/1/2/3."""
    return np.array(Image.open(lp).resize((SIZE, SIZE), Image.NEAREST)).astype(np.int64)


def _rgb_av_label(lp):
    """Colour-coded A/V mask, RGB: red=artery, blue=vein, green=cross,
    white=uncertain. Convert to the 0/1/2/3 index scheme. Shared by LES-AV's
    masks_multiclass and Fundus-AVSeg's annotation/ — both use this scheme."""
    rgb = np.array(Image.open(lp).convert("RGB").resize((SIZE, SIZE), Image.NEAREST))
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lab = np.zeros(rgb.shape[:2], np.int64)
    lab[(r > 128) & (g < 128) & (b < 128)] = 1                 # artery
    lab[(b > 128) & (r < 128) & (g < 128)] = 2                 # vein
    lab[(g > 128) & (r < 128) & (b < 128)] = 3                 # crossing
    lab[(r > 128) & (g > 128) & (b > 128)] = 3                 # uncertain → cross
    return lab


def _load(pairs, label_fn):
    """Keep the RAW resized BGR (not the preprocessed tensor) so photometric
    augmentation can run on the real image each epoch, before normalisation."""
    items = []
    for ip, lp in pairs:
        bgr = cv2.resize(cv2.imread(ip), (SIZE, SIZE))
        items.append((bgr, label_fn(lp)))
    return items


def load_drive(split):
    imgs = sorted(glob.glob(os.path.join(DATA, split, "images", "*.tif")))
    pairs = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(DATA, split, "1st_manual", stem + ".png")
        if os.path.exists(lp):
            pairs.append((ip, lp))
    return _load(pairs, _drive_label)


def load_les():
    """All LES-AV images with multiclass A/V labels (last 4 held out for test)."""
    imgs = sorted(glob.glob(os.path.join(LES, "images", "*.png")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                  if os.path.splitext(os.path.basename(p))[0].isdigit() else 0)
    pairs = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(LES, "masks_multiclass", stem + ".png")
        if os.path.exists(lp):
            pairs.append((ip, lp))
    items = _load(pairs, _rgb_av_label)
    return items[:-4], items[-4:]          # train, test


def load_fundus_avseg():
    """Fundus-AVSeg: 100 images, annotation/ uses the same colour-coded A/V
    scheme as LES-AV (last 15 held out for test)."""
    imgs = sorted(glob.glob(os.path.join(FUNDUS_AVSEG, "images", "*.png")))
    pairs = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(FUNDUS_AVSEG, "annotation", stem + ".png")
        if os.path.exists(lp):
            pairs.append((ip, lp))
    items = _load(pairs, _rgb_av_label)
    return items[:-15], items[-15:]        # train, test


def augment_geometric(bgr, y):
    """Flips + 90° rotations. Applied to BOTH image (H,W,3) and label (H,W)."""
    if np.random.rand() < 0.5:                       # horizontal flip
        bgr, y = bgr[:, ::-1, :].copy(), y[:, ::-1].copy()
    if np.random.rand() < 0.5:                       # vertical flip
        bgr, y = bgr[::-1, :, :].copy(), y[::-1, :].copy()
    k = np.random.randint(4)                         # 90° rotations
    if k:
        bgr = np.rot90(bgr, k, axes=(0, 1)).copy()
        y = np.rot90(y, k).copy()
    return bgr, y


def augment_photometric(bgr):
    """Simulate the appearance changes disease/different cameras cause.
    IMAGE ONLY — recolouring a pixel does not change its artery/vein label."""
    img = bgr.astype(np.float32)
    if np.random.rand() < 0.7:                       # brightness / contrast
        alpha = np.random.uniform(0.7, 1.3)          # contrast gain
        beta = np.random.uniform(-25, 25)            # brightness offset
        img = np.clip(alpha * img + beta, 0, 255)
    if np.random.rand() < 0.5:                       # gamma (tone) shift
        g = np.random.uniform(0.7, 1.5)
        img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255)
    if np.random.rand() < 0.4:                       # colour cast (HSV hue/sat)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if np.random.rand() < 0.3:                       # blur (media haze / defocus)
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if np.random.rand() < 0.3:                       # sensor noise
        img = img + np.random.normal(0, np.random.uniform(3, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_synthetic_lesions(bgr):
    """Paint a few dark (haemorrhage) / bright (exudate) blobs so the network
    learns NOT to mis-segment them as vessels. IMAGE ONLY — labels unchanged."""
    img = bgr.copy()
    h, w = img.shape[:2]
    for _ in range(np.random.randint(0, 5)):
        cx, cy = np.random.randint(0, w), np.random.randint(0, h)
        rad = np.random.randint(6, 25)
        dark = np.random.rand() < 0.5
        color = ((np.random.randint(20, 60),) * 3 if dark
                 else (np.random.randint(200, 255),) * 3)
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), rad, color, -1)
        img = cv2.addWeighted(overlay, np.random.uniform(0.3, 0.7), img, 0.7, 0)
    return img


def make_sample(bgr, y, train=True):
    """Raw BGR (+label) → model-ready (3,H,W) tensor and label.
    Training path adds geometric + photometric + lesion augmentation; the
    photometric steps run BEFORE preprocess_img so normalisation doesn't
    cancel them out. Eval path just preprocesses."""
    if train:
        bgr, y = augment_geometric(bgr, y)
        bgr = augment_photometric(bgr)
        if np.random.rand() < 0.4:
            bgr = add_synthetic_lesions(bgr)
    return preprocess_img(bgr), y


def load_pretrained_encoder(net, path):
    """Copy every tensor that matches by name+shape from a train_vessel.py
    checkpoint, skipping the final 1x1 `out` conv (2-class vessel vs 4-class
    A/V -- shapes differ there by design)."""
    if not path or not os.path.exists(path):
        return 0
    src = torch.load(path, map_location="cpu")
    dst = net.state_dict()
    xfer = {k: v for k, v in src.items()
            if k in dst and dst[k].shape == v.shape and not k.startswith("out.")}
    dst.update(xfer)
    net.load_state_dict(dst)
    return len(xfer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--base", type=int, default=64)  # must match train_vessel.py's --base for the transfer to line up
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-vessel", default=VESSEL_ENCODER,
                     help="path to a train_vessel.py checkpoint to initialise the encoder from, or '' to skip")
    args = ap.parse_args()

    t0 = time.time()
    drive_train = load_drive("training")
    drive_test = load_drive("test")
    les_train, les_test = load_les()
    fundus_train, fundus_test = load_fundus_avseg()
    train = drive_train + les_train + fundus_train
    print(f"loaded train={len(train)} (DRIVE {len(drive_train)} + LES {len(les_train)} + "
          f"Fundus-AVSeg {len(fundus_train)})  "
          f"test: DRIVE={len(drive_test)} LES={len(les_test)} Fundus-AVSeg={len(fundus_test)}  "
          f"({time.time()-t0:.0f}s)")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)
    out_path = os.path.join(HERE, "av_unet.pth")
    net = UNet(3, 4, base=args.base).to(dev)
    if os.path.exists(out_path):
        net.load_state_dict(torch.load(out_path, map_location=dev))
        print(f"resumed weights from {out_path}", flush=True)
    else:
        n_xfer = load_pretrained_encoder(net, args.init_vessel)
        if n_xfer:
            print(f"  init: transferred {n_xfer} tensors from {os.path.basename(args.init_vessel)}", flush=True)
    # Class weights: background dominates; up-weight vessels, esp. arteries.
    w = torch.tensor([0.05, 2.5, 2.0, 1.0], dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    scaler = torch.amp.GradScaler(dev.type, enabled=(dev.type == "cuda"))

    EPOCHS, BATCH = args.epochs, args.batch_size
    for ep in range(1, EPOCHS + 1):
        net.train()
        order = np.random.permutation(len(train))
        ep_loss = 0.0
        for i in range(0, len(train), BATCH):
            idx = order[i:i + BATCH]
            xs, ys = zip(*[make_sample(*train[j], train=True) for j in idx])
            x = torch.from_numpy(np.stack(xs)).to(dev)
            y = torch.from_numpy(np.stack(ys)).to(dev)
            opt.zero_grad()
            with torch.amp.autocast(dev.type, enabled=(dev.type == "cuda")):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss) * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            bd = evaluate(net, drive_test, dev)
            bl = evaluate(net, les_test, dev)
            bf = evaluate(net, fundus_test, dev)
            print(f"epoch {ep:3d}  loss={ep_loss/len(train):.4f}  "
                  f"A/V bal-acc: DRIVE={bd:.3f} LES={bl:.3f} Fundus-AVSeg={bf:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if ep % 10 == 0 or ep == EPOCHS:
            # Checkpoint periodically -- a long run can get torn down by the
            # host session before it reaches the final save.
            torch.save(net.state_dict(), out_path)
            print(f"  checkpoint saved ({ep}/{EPOCHS})", flush=True)

    print("done ->", out_path)
    for name, test in [("DRIVE", drive_test), ("LES-AV", les_test), ("Fundus-AVSeg", fundus_test)]:
        print_full_report(name, net, test, dev)


CLASSES = ["background", "artery", "vein", "crossing"]


@torch.no_grad()
def evaluate(net, test, dev, full=False):
    """Balanced A/V accuracy on true vessel pixels (artery vs vein only).
    With full=True, also returns the 4x4 confusion matrix (rows=true,
    cols=pred) and per-class precision/recall/F1."""
    net.eval()
    cm = np.zeros((4, 4), dtype=np.int64)
    for bgr, y in test:
        x, _ = make_sample(bgr, y, train=False)      # no augmentation at eval
        pred = net(torch.from_numpy(x[None]).to(dev)).argmax(1)[0].cpu().numpy()
        for t in range(4):
            ymask = y == t
            if not ymask.any():
                continue
            for p in range(4):
                cm[t, p] += int((ymask & (pred == p)).sum())
    tp_a, tot_a = cm[1, 1], cm[1].sum()
    tp_v, tot_v = cm[2, 2], cm[2].sum()
    se = tp_a / (tot_a + 1e-9)         # artery recall
    sp = tp_v / (tot_v + 1e-9)         # vein recall
    bal_acc = (se + sp) / 2
    if not full:
        return bal_acc

    report = {}
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        report[name] = (prec, rec, f1)
    return bal_acc, cm, report


def print_full_report(name, net, test, dev):
    bal_acc, cm, report = evaluate(net, test, dev, full=True)
    print(f"\n{name}  (n={len(test)}, balanced A/V accuracy={bal_acc:.3f})")
    print("  confusion matrix (rows=true, cols=pred):")
    print("           " + "  ".join(f"{c[:8]:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c[:8]:>8} " + "  ".join(f"{cm[i, j]:>8d}" for j in range(4)))
    print("  per-class precision / recall / F1:")
    for c, (prec, rec, f1) in report.items():
        print(f"    {c:<10} {prec:.3f}  {rec:.3f}  {f1:.3f}")


if __name__ == "__main__":
    main()
