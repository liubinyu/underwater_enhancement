from __future__ import annotations

import numpy as np


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance in BGR space.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input uint8 image in BGR order.

    Returns
    -------
    np.ndarray
        White-balanced uint8 image in BGR order.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR image.")

    img = image_bgr.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)
    global_mean = channel_means.mean()

    scale = global_mean / (channel_means + 1e-6)
    balanced = img * scale[None, None, :]
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced
