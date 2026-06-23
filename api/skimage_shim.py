"""
scikit-image shim for Vercel/Lambda environments.

Vercel's Python runtime fails to load scikit-image because it requires
.pyi stub files that are absent in the vendored package bundle.

This module provides identical interfaces for every scikit-image symbol
used by the SepsisScope pipeline, implemented with scipy, numpy, and
opencv-python-headless — all of which work correctly on Vercel.

Injected into sys.modules before the pipeline steps are imported so no
changes are needed in the original step*.py files.
"""

import numpy as np


# ── skimage.filters ───────────────────────────────────────────────────

class filters:
    @staticmethod
    def frangi(image, sigmas=range(1, 10, 2), black_ridges=True,
               alpha=0.5, beta=0.5, gamma=15, **_kwargs):
        """
        Frangi multiscale vessel-enhancement filter (2-D).
        Implemented with scipy.ndimage Gaussian derivatives.

        Parameters match the scikit-image API used in step1_segment_skeleton.py:
          sigmas       — iterable of scale values
          black_ridges — True = dark vessels on bright background (retinal green ch.)
          beta, gamma  — Frangi shape / noise parameters
        """
        from scipy.ndimage import gaussian_filter

        image = np.asarray(image, dtype=np.float64)
        sigma_list = list(sigmas) if hasattr(sigmas, '__iter__') else [sigmas]
        best = np.zeros_like(image)

        for sigma in sigma_list:
            s2 = sigma ** 2

            # Scale-normalised Hessian matrix components (∂²/∂r², ∂²/∂r∂c, ∂²/∂c²)
            Drr = gaussian_filter(image, sigma, order=(2, 0)) * s2
            Drc = gaussian_filter(image, sigma, order=(1, 1)) * s2
            Dcc = gaussian_filter(image, sigma, order=(0, 2)) * s2

            # Eigenvalues of the 2×2 symmetric Hessian at every pixel:
            #   λ = (trace ± √(trace²-4·det)) / 2
            trace = Drr + Dcc
            disc  = np.sqrt(np.maximum(trace**2 - 4 * (Drr * Dcc - Drc**2), 0))

            l1 = (trace - disc) / 2   # algebraically smaller eigenvalue
            l2 = (trace + disc) / 2   # algebraically larger  eigenvalue

            # Frangi vesselness ---
            # black_ridges=True → dark tube on bright BG → both curvatures positive
            # condition for a valid dark ridge pixel: l2 > 0
            if black_ridges:
                valid = l2 > 0
            else:
                valid = l2 < 0

            safe_l2 = np.where(valid, l2, 1.0)           # avoid div-by-zero
            Rb = np.where(valid, (l1 / safe_l2) ** 2, 0) # blob-vs-tube ratio
            S  = l1 ** 2 + l2 ** 2                        # Frobenius norm²

            v = np.where(
                valid,
                np.exp(-Rb / (2 * beta**2)) * (1 - np.exp(-S / (2 * gamma**2))),
                0.0,
            )
            best = np.maximum(best, v)

        return best


# ── skimage.exposure ──────────────────────────────────────────────────

class exposure:
    @staticmethod
    def rescale_intensity(image, in_range="image", out_range=(0, 1)):
        """Min-max rescale to out_range."""
        arr = np.asarray(image, dtype=np.float64)

        if in_range == "image":
            lo, hi = arr.min(), arr.max()
        else:
            lo, hi = float(in_range[0]), float(in_range[1])

        span = hi - lo
        if span < 1e-10:
            return np.full_like(arr, out_range[0], dtype=np.float64)

        scaled = (arr - lo) / span
        out_lo, out_hi = float(out_range[0]), float(out_range[1])
        return scaled * (out_hi - out_lo) + out_lo


# ── skimage.morphology ────────────────────────────────────────────────

class morphology:
    @staticmethod
    def skeletonize(binary_image):
        """
        Zhang–Suen thinning → single-pixel-wide 8-connected centrelines,
        matching scikit-image's skeletonize() for the pipeline's purposes.

        The earlier implementation here was a *morphological* skeleton
        (iterated erosion/subtraction).  That produces a thick, ragged result
        in which ~half the pixels have 3+ neighbours, so Step 1's branch-point
        removal shattered every vessel into sub-min_area fragments and almost
        no segments survived — the primary cause of "insufficient arterioles".
        A true thinning yields clean centrelines with few genuine branch
        points, so whole vessels survive as measurable segments.
        """
        img = (np.asarray(binary_image) > 0).astype(np.uint8)
        if img.sum() == 0:
            return img.astype(bool)

        def _neighbors(im):
            # P2..P9 clockwise from north (scikit-image / Zhang–Suen order)
            up, dn = np.roll(im, 1, 0), np.roll(im, -1, 0)
            P2, P6 = up, dn
            P4 = np.roll(im, -1, 1)
            P8 = np.roll(im, 1, 1)
            P3 = np.roll(up, -1, 1)
            P9 = np.roll(up, 1, 1)
            P5 = np.roll(dn, -1, 1)
            P7 = np.roll(dn, 1, 1)
            return P2, P3, P4, P5, P6, P7, P8, P9

        def _transitions(seq):
            # A(P1): number of 0→1 transitions in the cyclic sequence P2..P9,P2
            a = np.zeros(seq[0].shape, dtype=np.uint8)
            ordered = list(seq) + [seq[0]]
            for k in range(len(seq)):
                a += ((ordered[k] == 0) & (ordered[k + 1] == 1)).astype(np.uint8)
            return a

        while True:
            changed = 0
            for step in (0, 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = _neighbors(img)
                B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                A = _transitions((P2, P3, P4, P5, P6, P7, P8, P9))
                cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
                if step == 0:
                    cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
                else:
                    cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
                # Don't let np.roll's wrap-around delete real border pixels
                cond[0, :] = cond[-1, :] = cond[:, 0] = cond[:, -1] = False
                n = int(cond.sum())
                if n:
                    img[cond] = 0
                    changed += n
            if changed == 0:
                break

        return img.astype(bool)

    @staticmethod
    def remove_small_objects(binary_image, min_size=64, connectivity=1):
        """
        Remove connected components smaller than ``min_size`` pixels.
        Mirrors scikit-image's morphology.remove_small_objects for boolean
        input: returns a bool ndarray with small blobs cleared.
        """
        from scipy.ndimage import label as _sp_label

        arr = np.asarray(binary_image, dtype=bool)
        # connectivity=1 → 4-connected, 2 → 8-connected (scikit-image semantics)
        structure = np.ones((3, 3), dtype=np.int32) if connectivity >= 2 else None
        labeled, n = _sp_label(arr, structure=structure)
        if n == 0:
            return arr
        counts = np.bincount(labeled.ravel())
        too_small = counts < min_size
        too_small[0] = False                      # never touch the background
        return arr & ~too_small[labeled]


# ── skimage.measure ───────────────────────────────────────────────────

class _Region:
    """
    Minimal stand-in for skimage.measure.RegionProperties.
    Provides the attributes accessed by the SepsisScope pipeline:
      .label, .coords, .area, .centroid
    """
    __slots__ = ("label", "coords", "area", "centroid")

    def __init__(self, label_id: int, coords: np.ndarray):
        self.label    = label_id
        self.coords   = coords                      # (N,2) array of (row, col)
        self.area     = len(coords)
        self.centroid = (float(coords[:, 0].mean()),  # (row_mean, col_mean)
                         float(coords[:, 1].mean()))


class measure:
    @staticmethod
    def label(image, connectivity=2):
        """
        Label connected components.
        connectivity=2 → 8-connected (full 3×3 structuring element).
        Returns an int32 labeled array matching scikit-image's output.
        """
        from scipy.ndimage import label as _sp_label

        structure = np.ones((3, 3), dtype=np.int32) if connectivity == 2 else None
        labeled, _ = _sp_label(np.asarray(image, dtype=bool), structure=structure)
        return labeled.astype(np.int32)

    @staticmethod
    def regionprops(labeled_image, intensity_image=None):
        """
        Compute properties of labelled regions.
        Returns a list of _Region objects (label, coords, area, centroid).
        The intensity_image argument is accepted but not used (pipeline only
        reads coords and centroid from regionprops output).
        """
        labeled = np.asarray(labeled_image, dtype=np.int32)
        labels  = np.unique(labeled)
        regions = []
        for lbl in labels:
            if lbl == 0:
                continue
            coords = np.argwhere(labeled == lbl)   # (N, 2) — (row, col)
            regions.append(_Region(int(lbl), coords))
        return regions
