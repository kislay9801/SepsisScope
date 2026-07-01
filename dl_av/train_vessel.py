"""Stage 1: pretrain a binary vessel-segmentation U-Net on the full 8-benchmark
retinal-vessel-fundus-dataset-collection/ (GPU; falls back to CPU automatically).

This is NOT the A/V model used for AVR — it only tells vessel from background.
Its purpose is to give train.py's 4-class A/V model a better-initialised
encoder than training A/V from scratch on ~60 labelled images alone (the only
data with artery/vein ground truth is DRIVE_AV + LES-AV; every other benchmark
here has vessel-only masks).

Run:  python dl_av/train_vessel.py
Produces: dl_av/vessel_encoder.pth
"""
import os, time, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet import UNet
from datasets import list_vessel_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
np.random.seed(0)


def preprocess_img(bgr):
    """Green-channel CLAHE + per-image normalisation -> 3-ch float tensor input.
    Same recipe as train.py so the encoder sees a consistent input distribution."""
    g = bgr[:, :, 1]
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    x = g.astype(np.float32) / 255.0
    x = (x - x.mean()) / (x.std() + 1e-6)
    return np.stack([x, x, x], 0)


def augment_geometric(bgr, y):
    if np.random.rand() < 0.5:
        bgr, y = bgr[:, ::-1, :].copy(), y[:, ::-1].copy()
    if np.random.rand() < 0.5:
        bgr, y = bgr[::-1, :, :].copy(), y[::-1, :].copy()
    k = np.random.randint(4)
    if k:
        bgr = np.rot90(bgr, k, axes=(0, 1)).copy()
        y = np.rot90(y, k).copy()
    return bgr, y


def augment_photometric(bgr):
    img = bgr.astype(np.float32)
    if np.random.rand() < 0.7:
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.uniform(-25, 25)
        img = np.clip(alpha * img + beta, 0, 255)
    if np.random.rand() < 0.5:
        g = np.random.uniform(0.7, 1.5)
        img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255)
    if np.random.rand() < 0.3:
        k = int(np.random.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if np.random.rand() < 0.3:
        img = img + np.random.normal(0, np.random.uniform(3, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class VesselDataset(Dataset):
    """Decodes+resizes lazily per-sample (unlike train.py's small DRIVE/LES
    loader) since this collection is 1000+ images up to 3504x2336 -- too big to
    hold pre-resized copies of every sample in RAM."""

    def __init__(self, pairs, size, train):
        self.pairs = pairs
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp, name = self.pairs[idx]
        bgr = cv2.resize(cv2.imread(ip), (self.size, self.size))
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        y = (m > 127).astype(np.int64)
        if self.train:
            bgr, y = augment_geometric(bgr, y)
            bgr = augment_photometric(bgr)
        x = preprocess_img(bgr)
        return torch.from_numpy(x), torch.from_numpy(y), name


def split_per_dataset(pairs, val_frac, seed=0):
    """Held-out split stratified per-dataset, so every benchmark contributes to
    both train and val (matches the collection's own cross-dataset-eval intent)."""
    rng = np.random.RandomState(seed)
    by_ds = {}
    for p in pairs:
        by_ds.setdefault(p[2], []).append(p)
    train, val = [], []
    for items in by_ds.values():
        idx = rng.permutation(len(items))
        n_val = max(1, int(round(len(items) * val_frac)))
        val += [items[i] for i in idx[:n_val]]
        train += [items[i] for i in idx[n_val:]]
    return train, val


def _seg_metrics(tp, fp, fn, tn):
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    sens = tp / (tp + fn + 1e-9)          # recall
    spec = tn / (tn + fp + 1e-9)
    return {"dice": dice, "iou": iou, "prec": prec, "sens": sens, "spec": spec}


@torch.no_grad()
def evaluate(net, val_loader, dev):
    """Per-dataset and overall Dice/IoU/Precision/Sensitivity/Specificity on
    held-out vessel pixels."""
    net.eval()
    counts = {}  # name -> [tp, fp, fn, tn]
    for x, y, names in val_loader:
        x, y = x.to(dev), y.to(dev)
        pred = net(x).argmax(1) == 1
        truth = y == 1
        for i, name in enumerate(names):
            p, t = pred[i], truth[i]
            c = counts.setdefault(name, [0, 0, 0, 0])
            c[0] += int((p & t).sum())        # tp
            c[1] += int((p & ~t).sum())       # fp
            c[2] += int((~p & t).sum())       # fn
            c[3] += int((~p & ~t).sum())      # tn
    per_ds = {k: _seg_metrics(*v) for k, v in counts.items()}
    totals = [sum(c[i] for c in counts.values()) for i in range(4)]
    overall = _seg_metrics(*totals)
    return overall, per_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--base", type=int, default=64)  # keep = train.py's AV UNet base so weights transfer
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "vessel_encoder.pth"))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    t0 = time.time()
    pairs = list_vessel_pairs()
    train_pairs, val_pairs = split_per_dataset(pairs, args.val_frac)
    print(f"train={len(train_pairs)} val={len(val_pairs)}  ({time.time()-t0:.0f}s)")

    train_loader = DataLoader(VesselDataset(train_pairs, args.size, True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=(dev.type == "cuda"))
    val_loader = DataLoader(VesselDataset(val_pairs, args.size, False),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=(dev.type == "cuda"))

    net = UNet(3, 2, base=args.base).to(dev)
    if os.path.exists(args.out):
        net.load_state_dict(torch.load(args.out, map_location=dev))
        print(f"resumed weights from {args.out}", flush=True)
    # Vessels are a small fraction of pixels; up-weight the vessel class.
    w = torch.tensor([0.2, 2.5], dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(5, args.epochs // 3), gamma=0.5)
    scaler = torch.amp.GradScaler(dev.type, enabled=(dev.type == "cuda"))

    for ep in range(1, args.epochs + 1):
        net.train()
        ep_loss, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(dev.type, enabled=(dev.type == "cuda")):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss) * len(x)
            n += len(x)
        sched.step()
        if ep % 2 == 0 or ep == 1 or ep == args.epochs:
            overall, per_ds = evaluate(net, val_loader, dev)
            dice_detail = " ".join(f"{k}={v['dice']:.3f}" for k, v in sorted(per_ds.items()))
            print(f"epoch {ep:3d}/{args.epochs}  loss={ep_loss/n:.4f}  "
                  f"val: dice={overall['dice']:.3f} iou={overall['iou']:.3f} "
                  f"prec={overall['prec']:.3f} sens={overall['sens']:.3f} spec={overall['spec']:.3f}  "
                  f"[dice per-ds: {dice_detail}]  ({time.time()-t0:.0f}s)", flush=True)
        if ep % 5 == 0 or ep == args.epochs:
            # Checkpoint periodically -- a long GPU run can get torn down by the
            # host session before it reaches the final save.
            torch.save(net.state_dict(), args.out)
            print(f"  checkpoint saved ({ep}/{args.epochs})", flush=True)

    print("done ->", args.out)
    overall, per_ds = evaluate(net, val_loader, dev)
    print("\nFinal per-dataset report (dice / iou / precision / sensitivity / specificity):")
    for name, m in sorted(per_ds.items()):
        print(f"  {name:<10} {m['dice']:.3f}  {m['iou']:.3f}  {m['prec']:.3f}  {m['sens']:.3f}  {m['spec']:.3f}")
    print(f"  {'OVERALL':<10} {overall['dice']:.3f}  {overall['iou']:.3f}  "
          f"{overall['prec']:.3f}  {overall['sens']:.3f}  {overall['spec']:.3f}")


if __name__ == "__main__":
    main()
