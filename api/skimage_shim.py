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
        Morphological skeletonization via iterated erosion/subtraction.
        Equivalent to scikit-image's skeletonize() for boolean arrays.
        Uses OpenCV for speed; result is a bool ndarray.
        """
        import cv2

        img = np.asarray(binary_image, dtype=np.uint8)
        img = np.where(img > 0, np.uint8(255), np.uint8(0))

        skel   = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded  = cv2.erode(img, kernel)
            opened  = cv2.dilate(eroded, kernel)
            temp    = cv2.subtract(img, opened)
            skel    = cv2.bitwise_or(skel, temp)
            img     = eroded
            if cv2.countNonZero(img) == 0:
                break

        return skel.astype(bool)


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
