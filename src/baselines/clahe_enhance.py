from __future__ import annotations

import cv2
import numpy as np


def apply_clahe_lab(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """
    Apply CLAHE on the L channel in LAB color space.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input uint8 BGR image.
    clip_limit : float
        CLAHE clip limit.
    tile_grid_size : int
        Tile size for CLAHE.

    Returns
    -------
    np.ndarray
        Enhanced uint8 BGR image.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR image.")

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced
