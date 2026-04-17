from __future__ import annotations

import numpy as np

from src.baselines.white_balance import gray_world_white_balance


def red_channel_compensation(
    image_bgr: np.ndarray,
    red_gain: float = 1.0,
    apply_white_balance_after_compensation: bool = True,
) -> np.ndarray:
    """
    Simple underwater red-channel compensation.

    The idea is to partially restore the red channel by referencing the
    stronger green/blue channels, then optionally apply gray-world WB.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input uint8 image in BGR order.
    red_gain : float
        Strength of red compensation.
    apply_white_balance_after_compensation : bool
        Whether to apply gray-world white balance after compensation.

    Returns
    -------
    np.ndarray
        Enhanced uint8 image in BGR order.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR image.")

    img = image_bgr.astype(np.float32)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    gb_mean = 0.5 * (g + b)
    r_new = r + red_gain * (gb_mean - r)

    out = img.copy()
    out[:, :, 2] = np.clip(r_new, 0, 255)
    out = out.astype(np.uint8)

    if apply_white_balance_after_compensation:
        out = gray_world_white_balance(out)

    return out
