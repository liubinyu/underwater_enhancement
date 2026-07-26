"""Deterministic image statistics used by the rule-based degradation baseline."""

from __future__ import annotations

import math
from typing import Final

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, uniform_filter

from aqua_align.enhancement import ensure_rgb_uint8


FEATURE_VERSION: Final[str] = "0.1"
EPSILON: Final[float] = 1e-8


def _entropy(gray: np.ndarray) -> float:
    """Return Shannon entropy in bits for an 8-bit grayscale image."""
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0.0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def _local_contrast(gray: np.ndarray, window_size: int = 15) -> float:
    """Return mean local standard deviation over a reflected square window."""
    work = gray.astype(np.float32)
    mean = uniform_filter(work, size=window_size, mode="reflect")
    mean_square = uniform_filter(work * work, size=window_size, mode="reflect")
    variance = np.maximum(mean_square - mean * mean, 0.0)
    return float(np.sqrt(variance).mean())


def _plain_finite(features: dict[str, float | str]) -> dict[str, float | str]:
    """Convert NumPy scalars to finite JSON-safe Python values."""
    output: dict[str, float | str] = {}
    for name, value in features.items():
        if isinstance(value, str):
            output[name] = value
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"feature {name} produced NaN or Infinity")
        output[name] = parsed
    return output


def extract_degradation_features(image: np.ndarray) -> dict[str, float | str]:
    """Extract color, brightness, contrast, sharpness, noise, and haze proxies.

    Args:
        image: Gray, RGB, or RGBA numeric image accepted by
            :func:`aqua_align.enhancement.ensure_rgb_uint8`.

    Returns:
        A JSON-serializable mapping containing only plain Python strings and
        finite floats. Sharpness and haze values are proxies, not causal
        measurements of camera focus or underwater backscatter.
    """
    rgb = ensure_rgb_uint8(image)
    work = rgb.astype(np.float32)
    flattened = work.reshape(-1, 3)
    means = flattened.mean(axis=0)
    medians = np.median(flattened, axis=0)
    maximum_mean = max(float(means.max()), EPSILON)
    strongest_blue_green = max(float(means[1]), float(means[2]))
    red_deficit = max(0.0, strongest_blue_green - float(means[0])) / maximum_mean
    blue_green_dominance = max(0.0, 0.5 * float(means[1] + means[2]) - float(means[0])) / maximum_mean
    red_excess = max(0.0, float(means[0]) - max(float(means[1]), float(means[2]))) / maximum_mean

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    lightness = lab[..., 0]
    percentiles = np.percentile(gray_float, (5.0, 50.0, 95.0))

    laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)
    sobel_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    edge_threshold = float(gradient.mean() + gradient.std())
    edge_density = float(np.mean(gradient > edge_threshold))

    smooth = gaussian_filter(gray_float, sigma=1.0, mode="reflect")
    high_pass = gray_float - smooth
    noise_sigma = float(np.median(np.abs(high_pass - np.median(high_pass))) / 0.6745 / 255.0)
    high_frequency_energy = float(np.mean(np.abs(laplacian)) / 255.0)

    dark_channel = np.min(work, axis=2)
    local_dark_channel = minimum_filter(dark_channel, size=15, mode="reflect")
    dark_channel_mean = float(local_dark_channel.mean() / 255.0)
    local_contrast = _local_contrast(gray)
    contrast_component = float(np.clip(1.0 - gray_float.std() / 64.0, 0.0, 1.0))
    local_component = float(np.clip(1.0 - local_contrast / 32.0, 0.0, 1.0))
    edge_component = float(np.clip(1.0 - edge_density / 0.30, 0.0, 1.0))
    haze_proxy = float(
        np.clip(
            0.30 * dark_channel_mean
            + 0.30 * contrast_component
            + 0.20 * local_component
            + 0.20 * edge_component,
            0.0,
            1.0,
        )
    )

    red_green = work[..., 0] - work[..., 1]
    yellow_blue = 0.5 * (work[..., 0] + work[..., 1]) - work[..., 2]
    colorfulness = float(
        np.sqrt(np.var(red_green) + np.var(yellow_blue))
        + 0.3 * np.sqrt(np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2)
    )

    features: dict[str, float | str] = {
        "feature_version": FEATURE_VERSION,
        "mean_r": float(means[0]),
        "mean_g": float(means[1]),
        "mean_b": float(means[2]),
        "median_r": float(medians[0]),
        "median_g": float(medians[1]),
        "median_b": float(medians[2]),
        "red_deficit": red_deficit,
        "blue_green_dominance": blue_green_dominance,
        "red_excess": red_excess,
        "red_to_green_ratio": float(means[0] / max(float(means[1]), EPSILON)),
        "channel_imbalance": float(means.max() - means.min()),
        "channel_imbalance_normalized": float((means.max() - means.min()) / 255.0),
        "lab_a_mean": float(lab[..., 1].mean() - 128.0),
        "lab_b_mean": float(lab[..., 2].mean() - 128.0),
        "hsv_saturation_mean": float(saturation.mean()),
        "high_saturation_pixel_ratio": float(np.mean(saturation >= 0.80)),
        "colorfulness": colorfulness,
        "mean_brightness": float(gray_float.mean()),
        "median_brightness": float(percentiles[1]),
        "lab_l_mean": float(lightness.mean()),
        "dark_pixel_ratio": float(np.mean(gray_float <= 50.0)),
        "extreme_dark_pixel_ratio": float(np.mean(gray_float <= 20.0)),
        "bright_pixel_ratio": float(np.mean(gray_float >= 200.0)),
        "dynamic_range": float(percentiles[2] - percentiles[0]),
        "gray_std": float(gray_float.std()),
        "gray_percentile_range": float(percentiles[2] - percentiles[0]),
        "rms_contrast": float(gray_float.std() / 255.0),
        "local_contrast_mean": local_contrast,
        "entropy": _entropy(gray),
        "laplacian_variance": float(laplacian.var()),
        "sobel_gradient_mean": float(gradient.mean()),
        "high_frequency_energy": high_frequency_energy,
        "noise_sigma_proxy": noise_sigma,
        "dark_channel_mean": dark_channel_mean,
        "edge_density": edge_density,
        "haze_proxy": haze_proxy,
    }
    return _plain_finite(features)
