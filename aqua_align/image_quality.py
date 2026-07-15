"""Reference and no-reference image quality metrics for RGB uint8 images."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, laplace

from aqua_align.enhancement import ensure_rgb_uint8


def _matching_images(image: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate = ensure_rgb_uint8(image)
    target = ensure_rgb_uint8(reference)
    if candidate.shape != target.shape:
        raise ValueError(f"image shapes must match, got {candidate.shape} and {target.shape}")
    return candidate, target


def psnr(image: np.ndarray, reference: np.ndarray, identical_value: float = 100.0) -> float:
    """Return PSNR in dB for 0..255 RGB inputs, capped for identical images."""
    candidate, target = _matching_images(image, reference)
    cap = float(identical_value)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("identical_value must be finite and positive")
    difference = candidate.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference * difference))
    if mse <= np.finfo(np.float64).eps:
        return cap
    value = 10.0 * np.log10((255.0 * 255.0) / mse)
    return float(min(value, cap))


def ssim(image: np.ndarray, reference: np.ndarray) -> float:
    """Return mean windowed SSIM for matching RGB images of at least 3x3 pixels."""
    candidate, target = _matching_images(image, reference)
    height, width = candidate.shape[:2]
    window_size = min(11, height, width)
    if window_size < 3:
        raise ValueError("SSIM requires images of at least 3x3 pixels")
    if window_size % 2 == 0:
        window_size -= 1

    first = candidate.astype(np.float64)
    second = target.astype(np.float64)
    sigma = max(0.5, 1.5 * window_size / 11.0)
    truncate = max(((window_size - 1) / 2.0) / sigma, 0.5)
    mu_first = gaussian_filter(first, sigma=(sigma, sigma, 0.0), mode="reflect", truncate=truncate)
    mu_second = gaussian_filter(second, sigma=(sigma, sigma, 0.0), mode="reflect", truncate=truncate)
    mu_first_sq = mu_first * mu_first
    mu_second_sq = mu_second * mu_second
    mu_product = mu_first * mu_second
    sigma_first = gaussian_filter(first * first, sigma=(sigma, sigma, 0.0), mode="reflect", truncate=truncate) - mu_first_sq
    sigma_second = gaussian_filter(second * second, sigma=(sigma, sigma, 0.0), mode="reflect", truncate=truncate) - mu_second_sq
    sigma_cross = gaussian_filter(first * second, sigma=(sigma, sigma, 0.0), mode="reflect", truncate=truncate) - mu_product
    sigma_first = np.maximum(sigma_first, 0.0)
    sigma_second = np.maximum(sigma_second, 0.0)

    constant_1 = (0.01 * 255.0) ** 2
    constant_2 = (0.03 * 255.0) ** 2
    numerator = (2.0 * mu_product + constant_1) * (2.0 * sigma_cross + constant_2)
    denominator = (mu_first_sq + mu_second_sq + constant_1) * (
        sigma_first + sigma_second + constant_2
    )
    score = float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float64).eps)))
    return float(np.clip(score, -1.0, 1.0))


def no_reference_metrics(
    image: np.ndarray,
    dark_threshold: int = 15,
    saturated_threshold: int = 250,
) -> dict[str, float]:
    """Compute finite quality descriptors for an RGB image on the 0..255 scale."""
    rgb = ensure_rgb_uint8(image)
    dark = int(dark_threshold)
    saturated = int(saturated_threshold)
    if not 0 <= dark <= 255:
        raise ValueError("dark_threshold must be in [0, 255]")
    if not 0 <= saturated <= 255:
        raise ValueError("saturated_threshold must be in [0, 255]")

    work = rgb.astype(np.float32)
    gray = (0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]).astype(np.float32)
    means = work.reshape(-1, 3).mean(axis=0)
    laplacian_response = laplace(gray.astype(np.float64), mode="reflect")
    red_green = work[..., 0] - work[..., 1]
    yellow_blue = 0.5 * (work[..., 0] + work[..., 1]) - work[..., 2]
    colorfulness = np.sqrt(np.var(red_green) + np.var(yellow_blue)) + 0.3 * np.sqrt(
        np.mean(red_green) ** 2 + np.mean(yellow_blue) ** 2
    )
    metrics = {
        "mean_brightness": float(gray.mean()),
        "grayscale_std": float(gray.std()),
        "laplacian_variance": float(laplacian_response.var()),
        "dark_pixel_ratio": float(np.mean(gray <= dark)),
        "saturated_pixel_ratio": float(np.mean(np.max(rgb, axis=2) >= saturated)),
        "red_mean": float(means[0]),
        "green_mean": float(means[1]),
        "blue_mean": float(means[2]),
        "channel_imbalance": float(means.max() - means.min()),
        "colorfulness": float(colorfulness),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("metric calculation produced NaN or Infinity")
    return metrics


def compute_quality_metrics(
    image: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    dark_threshold: int = 15,
    saturated_threshold: int = 250,
    identical_psnr_db: float = 100.0,
) -> dict[str, Any]:
    """Compute no-reference metrics and optional PSNR/SSIM against a reference."""
    metrics: dict[str, Any] = no_reference_metrics(
        image,
        dark_threshold=dark_threshold,
        saturated_threshold=saturated_threshold,
    )
    metrics["has_reference"] = reference is not None
    metrics["psnr"] = None
    metrics["ssim"] = None
    if reference is not None:
        metrics["psnr"] = psnr(image, reference, identical_value=identical_psnr_db)
        metrics["ssim"] = ssim(image, reference)
    return metrics
