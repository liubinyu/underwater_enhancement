from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np


def _single_scale_retinex(channel: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(channel, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.log1p(channel) - np.log1p(blurred)


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    c_min = np.min(channel)
    c_max = np.max(channel)
    if c_max - c_min < 1e-8:
        return np.zeros_like(channel, dtype=np.uint8)
    out = (channel - c_min) / (c_max - c_min)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return out


def multi_scale_retinex(
    image_bgr: np.ndarray,
    sigmas: Sequence[float] | Iterable[float] = (15, 80, 250),
    gain: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Multi-scale Retinex for underwater enhancement.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input uint8 BGR image.
    sigmas : sequence of float
        Gaussian sigmas used in MSR.
    gain : float
        Gain applied after MSR.
    offset : float
        Offset applied after MSR.

    Returns
    -------
    np.ndarray
        Enhanced uint8 BGR image.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Input image must be an HxWx3 BGR image.")

    img = image_bgr.astype(np.float32) + 1.0
    out_channels = []

    for c in range(3):
        channel = img[:, :, c]
        msr = np.zeros_like(channel, dtype=np.float32)
        sigma_list = list(sigmas)
        for sigma in sigma_list:
            msr += _single_scale_retinex(channel, float(sigma))
        msr /= max(len(sigma_list), 1)
        msr = gain * msr + offset
        out_channels.append(_normalize_channel(msr))

    enhanced = cv2.merge(out_channels)
    return enhanced
