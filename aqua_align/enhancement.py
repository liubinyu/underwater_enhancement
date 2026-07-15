"""Traditional underwater image enhancement methods with a unified RGB API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal environments
    cv2 = None


Array = np.ndarray


def ensure_rgb_uint8(image: Array) -> Array:
    """Return a validated RGB uint8 copy from gray, RGB, or RGBA input.

    Integer inputs are interpreted on the 0..255 scale. Floating-point inputs
    must be finite and use either the 0..1 or 0..255 scale.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape HxW, HxWx1, HxWx3, or HxWx4")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image height and width must be positive")

    if image.ndim == 2:
        array = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        array = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 3:
        array = image
    elif image.shape[2] == 4:
        array = image[..., :3]
    else:
        raise ValueError("image channel count must be 1, 3, or 4")

    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image dtype must be numeric")
    work = array.astype(np.float32, copy=True)
    if not np.isfinite(work).all():
        raise ValueError("image contains NaN or Infinity")
    minimum = float(work.min())
    maximum = float(work.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError("image values must be in [0, 1] or [0, 255]")
    if np.issubdtype(array.dtype, np.floating) and maximum <= 1.0:
        work *= 255.0
    return np.rint(np.clip(work, 0.0, 255.0)).astype(np.uint8)


def _positive_float(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite value greater than zero")
    return parsed


def _validate_max_gain(max_gain: float) -> float:
    gain = _positive_float(max_gain, "max_gain")
    if gain < 1.0:
        raise ValueError("max_gain must be at least 1.0")
    return gain


def gray_world(image: Array, max_gain: float = 4.0) -> Array:
    """Apply gray-world white balance to an RGB image."""
    rgb = ensure_rgb_uint8(image)
    gain_limit = _validate_max_gain(max_gain)
    work = rgb.astype(np.float32)
    means = work.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    gains = np.ones(3, dtype=np.float32)
    nonzero = means > 1e-6
    gains[nonzero] = target / means[nonzero]
    gains = np.clip(gains, 1.0 / gain_limit, gain_limit)
    return np.rint(np.clip(work * gains[None, None, :], 0.0, 255.0)).astype(np.uint8)


def white_patch(image: Array, percentile: float = 99.5, max_gain: float = 4.0) -> Array:
    """Apply percentile white-patch correction independently to RGB channels."""
    rgb = ensure_rgb_uint8(image)
    pct = float(percentile)
    if not np.isfinite(pct) or not 0.0 < pct <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    gain_limit = _validate_max_gain(max_gain)
    work = rgb.astype(np.float32)
    whites = np.percentile(work.reshape(-1, 3), pct, axis=0)
    gains = np.ones(3, dtype=np.float32)
    nonzero = whites > 1e-6
    gains[nonzero] = 255.0 / whites[nonzero]
    gains = np.clip(gains, 1.0 / gain_limit, gain_limit)
    return np.rint(np.clip(work * gains[None, None, :], 0.0, 255.0)).astype(np.uint8)


def clahe(image: Array, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Array:
    """Enhance local luminance contrast using CLAHE in CIE LAB space."""
    rgb = ensure_rgb_uint8(image)
    clip = _positive_float(clip_limit, "clip_limit")
    grid = int(tile_grid_size)
    if grid <= 0:
        raise ValueError("tile_grid_size must be a positive integer")
    if cv2 is not None:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        operator = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        enhanced = operator.apply(lightness)
        return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB)

    work = rgb.astype(np.float32)
    luminance = np.rint(0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]).astype(np.uint8)
    enhanced = _clahe_luminance(luminance, clip_limit=clip, grid_size=grid)
    ratio = (enhanced.astype(np.float32) + 1.0) / (luminance.astype(np.float32) + 1.0)
    return np.rint(np.clip(work * ratio[..., None], 0.0, 255.0)).astype(np.uint8)


def _clahe_luminance(channel: Array, clip_limit: float, grid_size: int) -> Array:
    """Apply tile-wise clipped histogram equalization with bilinear LUT blending."""
    height, width = channel.shape
    tiles_y = min(grid_size, height)
    tiles_x = min(grid_size, width)
    y_edges = np.linspace(0, height, tiles_y + 1, dtype=int)
    x_edges = np.linspace(0, width, tiles_x + 1, dtype=int)
    lookup = np.empty((tiles_y, tiles_x, 256), dtype=np.float32)
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile = channel[y_edges[tile_y] : y_edges[tile_y + 1], x_edges[tile_x] : x_edges[tile_x + 1]]
            histogram = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
            threshold = max(1, int(clip_limit * tile.size / 256.0))
            excess = int(np.maximum(histogram - threshold, 0).sum())
            histogram = np.minimum(histogram, threshold)
            histogram += excess // 256
            remainder = excess % 256
            if remainder:
                indices = np.linspace(0, 255, remainder, dtype=int)
                histogram[indices] += 1
            cumulative = histogram.cumsum()
            nonzero = cumulative[cumulative > 0]
            denominator = int(cumulative[-1] - nonzero[0]) if nonzero.size else 0
            if denominator <= 0:
                lookup[tile_y, tile_x] = np.arange(256, dtype=np.float32)
            else:
                lookup[tile_y, tile_x] = np.clip(
                    (cumulative - nonzero[0]) * (255.0 / denominator), 0.0, 255.0
                )

    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:] - 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:] - 1)
    x_upper = np.searchsorted(x_centers, np.arange(width), side="right").clip(0, tiles_x - 1)
    x_lower = np.maximum(x_upper - 1, 0)
    x_span = np.maximum(x_centers[x_upper] - x_centers[x_lower], 1.0)
    x_weight = np.where(x_upper == x_lower, 0.0, (np.arange(width) - x_centers[x_lower]) / x_span)
    x_weight = np.clip(x_weight, 0.0, 1.0).astype(np.float32)
    result = np.empty_like(channel)
    for y in range(height):
        upper_y = int(np.searchsorted(y_centers, y, side="right").clip(0, tiles_y - 1))
        lower_y = max(upper_y - 1, 0)
        span_y = max(float(y_centers[upper_y] - y_centers[lower_y]), 1.0)
        weight_y = 0.0 if upper_y == lower_y else float(np.clip((y - y_centers[lower_y]) / span_y, 0.0, 1.0))
        values = channel[y]
        top = (1.0 - x_weight) * lookup[lower_y, x_lower, values] + x_weight * lookup[lower_y, x_upper, values]
        bottom = (1.0 - x_weight) * lookup[upper_y, x_lower, values] + x_weight * lookup[upper_y, x_upper, values]
        result[y] = np.rint((1.0 - weight_y) * top + weight_y * bottom).astype(np.uint8)
    return result


def gamma_correction(image: Array, gamma: float) -> Array:
    """Apply power-law correction; gamma below one brightens the image."""
    rgb = ensure_rgb_uint8(image)
    exponent = _positive_float(gamma, "gamma")
    normalized = rgb.astype(np.float32) / 255.0
    corrected = np.power(normalized, exponent, dtype=np.float32) * 255.0
    return np.rint(np.clip(corrected, 0.0, 255.0)).astype(np.uint8)


def simplest_color_balance(image: Array, percent: float = 1.0) -> Array:
    """Stretch each RGB channel after clipping a total percentage of outliers."""
    rgb = ensure_rgb_uint8(image)
    clip_percent = float(percent)
    if not np.isfinite(clip_percent) or not 0.0 <= clip_percent < 100.0:
        raise ValueError("percent must be in [0, 100)")
    work = rgb.astype(np.float32)
    output = np.empty_like(work)
    tail = clip_percent / 2.0
    for channel_index in range(3):
        channel = work[..., channel_index]
        low, high = np.percentile(channel, (tail, 100.0 - tail))
        if high - low <= 1e-6:
            output[..., channel_index] = channel
        else:
            output[..., channel_index] = (channel - low) * (255.0 / (high - low))
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def retinex(
    image: Array,
    sigmas: Sequence[float] = (15.0, 80.0, 250.0),
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> Array:
    """Apply multi-scale Retinex and robustly normalize each RGB channel."""
    rgb = ensure_rgb_uint8(image)
    sigma_values = tuple(_positive_float(sigma, "sigma") for sigma in sigmas)
    if not sigma_values:
        raise ValueError("sigmas must contain at least one positive value")
    low = float(low_percentile)
    high = float(high_percentile)
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    work = rgb.astype(np.float32) + 1.0
    output = np.empty_like(work)
    for channel_index in range(3):
        channel = work[..., channel_index]
        response = np.zeros_like(channel, dtype=np.float32)
        for sigma in sigma_values:
            if cv2 is not None:
                blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
            else:
                blurred = gaussian_filter(channel, sigma=sigma, mode="reflect")
            response += np.log(channel) - np.log(np.maximum(blurred, 1e-6))
        response /= float(len(sigma_values))
        lower, upper = np.percentile(response, (low, high))
        if upper - lower <= 1e-6:
            output[..., channel_index] = rgb[..., channel_index]
        else:
            output[..., channel_index] = (response - lower) * (255.0 / (upper - lower))
    return np.rint(np.clip(output, 0.0, 255.0)).astype(np.uint8)


def white_balance_clahe(
    image: Array,
    max_gain: float = 4.0,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> Array:
    """Apply gray-world white balance followed by luminance CLAHE."""
    balanced = gray_world(image, max_gain=max_gain)
    return clahe(balanced, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


METHODS: dict[str, Callable[..., Array]] = {
    "gray_world": gray_world,
    "white_patch": white_patch,
    "clahe": clahe,
    "simplest_color_balance": simplest_color_balance,
    "retinex": retinex,
    "white_balance_clahe": white_balance_clahe,
}


def apply_enhancement(image: Array, method: str, parameters: dict[str, Any] | None = None) -> Array:
    """Apply a named enhancement method using validated keyword parameters."""
    params = dict(parameters or {})
    if method == "gamma_correction":
        return gamma_correction(image, **params)
    try:
        function = METHODS[method]
    except KeyError as exc:
        supported = ", ".join(sorted((*METHODS, "gamma_correction")))
        raise ValueError(f"unknown enhancement method {method!r}; supported: {supported}") from exc
    return function(image, **params)
