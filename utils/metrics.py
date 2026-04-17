from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image


def _to_rgb_array(image: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        arr = np.asarray(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
    return arr


def entropy(image) -> float:
    arr = cv2.cvtColor(_to_rgb_array(image), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([arr], [0], None, [256], [0, 256]).ravel()
    prob = hist / max(hist.sum(), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def rms_contrast(image) -> float:
    gray = cv2.cvtColor(_to_rgb_array(image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return float(gray.std())


def color_balance_error(image) -> float:
    arr = _to_rgb_array(image).astype(np.float32) / 255.0
    means = arr.reshape(-1, 3).mean(axis=0)
    return float(np.mean(np.abs(means - means.mean())))


def uciqe(image) -> float:
    """UCIQE approximation: chroma std, luminance contrast, saturation mean."""
    rgb = _to_rgb_array(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    sigma_c = np.std(chroma)
    lum = lab[..., 0] / 255.0
    con_l = np.percentile(lum, 99) - np.percentile(lum, 1)
    mean_s = np.mean(hsv[..., 1] / 255.0)
    return float(0.4680 * sigma_c / 100.0 + 0.2745 * con_l + 0.2576 * mean_s)


def uiqm_placeholder(image) -> float:
    """Lightweight proxy for UIQM until a full component implementation is needed."""
    return float(0.4 * uciqe(image) + 0.4 * rms_contrast(image) + 0.2 * (entropy(image) / 8.0))


def compute_metrics(image) -> Dict[str, float]:
    return {
        "entropy": entropy(image),
        "rms_contrast": rms_contrast(image),
        "color_balance_error": color_balance_error(image),
        "uciqe": uciqe(image),
        "uiqm_proxy": uiqm_placeholder(image),
    }

