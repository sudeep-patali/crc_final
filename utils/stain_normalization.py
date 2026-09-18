"""
stain_normalization.py
Macenko stain normalization implemented from first principles (SVD-based
H&E deconvolution), so it can be defended in a viva as "custom implemented"
rather than a black-box library call.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009.

Usage:
    normalizer = MacenkoNormalizer()
    normalizer.fit(reference_image_rgb)         # fit once on a reference tile
    normalized = normalizer.transform(image_rgb) # apply to any patch
"""

import numpy as np
import cv2


class MacenkoNormalizer:
    def __init__(self, alpha=1.0, beta=0.15, target_od_max=None):
        """
        alpha: percentile cutoff for extreme angle pixels (1st/99th typical)
        beta : OD threshold below which pixels are considered background/white
               and excluded from stain vector estimation
        """
        self.alpha = alpha
        self.beta = beta
        self.stain_matrix_target = None   # (3,2) HE stain vectors of reference
        self.max_c_target = None          # (2,) max concentrations of reference
        self.target_od_max = target_od_max

    # ------------------------------------------------------------------
    @staticmethod
    def _rgb_to_od(img):
        """Convert RGB [0,255] image to optical density."""
        img = img.astype(np.float64)
        img[img == 0] = 1.0  # avoid log(0)
        od = -np.log(img / 255.0)
        return od

    @staticmethod
    def _od_to_rgb(od):
        rgb = 255.0 * np.exp(-od)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _estimate_stain_matrix(self, od_flat):
        """
        od_flat: (N,3) optical density values (background already removed)
        Returns (3,2) stain matrix [hematoxylin | eosin] via SVD + angle method.
        """
        # covariance / SVD on OD space
        cov = np.cov(od_flat.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # take the two largest eigenvectors (span the plane of stain colors)
        top2 = eigvecs[:, [-1, -2]]

        # project OD values onto this plane
        proj = od_flat @ top2

        # angle of each point in the plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        v_min = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v_max = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

        # heuristic: hematoxylin has larger red channel OD relation vs eosin
        if v_min[0] > v_max[0]:
            stain_matrix = np.stack([v_min, v_max], axis=1)
        else:
            stain_matrix = np.stack([v_max, v_min], axis=1)

        # normalize columns to unit length
        stain_matrix = stain_matrix / (np.linalg.norm(stain_matrix, axis=0, keepdims=True) + 1e-8)
        return stain_matrix  # (3,2)

    def _get_concentrations(self, od_flat, stain_matrix):
        # solve OD = stain_matrix @ C  ->  C = pinv(stain_matrix) @ OD
        C, _, _, _ = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)
        return C  # (2, N)

    # ------------------------------------------------------------------
    def fit(self, ref_rgb):
        """Fit reference stain vectors + max concentrations from a reference tile."""
        od = self._rgb_to_od(ref_rgb).reshape(-1, 3)
        mask = np.exp(-od).sum(axis=1) / 3.0  # rough brightness proxy
        od_fg = od[(od > self.beta).any(axis=1)]
        if od_fg.shape[0] < 10:
            od_fg = od  # fallback: use everything if too little tissue

        self.stain_matrix_target = self._estimate_stain_matrix(od_fg)
        C = self._get_concentrations(od_fg, self.stain_matrix_target)
        self.max_c_target = np.percentile(C, 99, axis=1)
        return self

    def transform(self, img_rgb):
        """Normalize a new image to the fitted reference's stain appearance."""
        if self.stain_matrix_target is None:
            raise RuntimeError("Call .fit(reference_image) before .transform().")

        h, w, _ = img_rgb.shape
        od = self._rgb_to_od(img_rgb).reshape(-1, 3)
        od_fg_mask = (od > self.beta).any(axis=1)
        od_fg = od[od_fg_mask]
        if od_fg.shape[0] < 10:
            od_fg = od

        stain_matrix_src = self._estimate_stain_matrix(od_fg)
        C_src = self._get_concentrations(od, stain_matrix_src)
        max_c_src = np.percentile(C_src, 99, axis=1)
        max_c_src[max_c_src == 0] = 1e-6

        # scale concentrations to target's max concentration range
        C_scaled = C_src * (self.max_c_target / max_c_src)[:, None]

        od_normalized = self.stain_matrix_target @ C_scaled  # (3, N)
        rgb_normalized = self._od_to_rgb(od_normalized.T).reshape(h, w, 3)
        return rgb_normalized


def load_or_fit_reference(reference_path):
    """Load a reference tile and fit + return a ready-to-use normalizer."""
    ref_bgr = cv2.imread(reference_path)
    if ref_bgr is None:
        raise FileNotFoundError(f"Reference patch not found: {reference_path}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    normalizer = MacenkoNormalizer()
    normalizer.fit(ref_rgb)
    return normalizer


if __name__ == "__main__":
    # quick self-test with a synthetic image
    import sys
    dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    norm = MacenkoNormalizer().fit(dummy)
    out = norm.transform(dummy)
    print("Stain normalization self-test OK, output shape:", out.shape)
