import numpy as np
import pytest

from aqua_align.image_quality import compute_quality_metrics, no_reference_metrics, psnr, ssim


def test_identical_images_have_capped_psnr_and_unit_ssim() -> None:
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    assert psnr(image, image) == 100.0
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-12)


def test_black_versus_white_psnr_is_zero() -> None:
    black = np.zeros((12, 12, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    assert psnr(black, white) == pytest.approx(0.0, abs=1e-12)


def test_black_and_white_no_reference_metrics_are_finite() -> None:
    black = no_reference_metrics(np.zeros((10, 10, 3), dtype=np.uint8))
    white = no_reference_metrics(np.full((10, 10, 3), 255, dtype=np.uint8))
    assert black["mean_brightness"] == 0.0
    assert black["dark_pixel_ratio"] == 1.0
    assert black["saturated_pixel_ratio"] == 0.0
    assert white["mean_brightness"] == 255.0
    assert white["dark_pixel_ratio"] == 0.0
    assert white["saturated_pixel_ratio"] == 1.0
    assert all(np.isfinite(value) for value in (*black.values(), *white.values()))


def test_metrics_without_reference_leave_reference_values_missing() -> None:
    metrics = compute_quality_metrics(np.full((8, 8), 100, dtype=np.uint8))
    assert metrics["has_reference"] is False
    assert metrics["psnr"] is None
    assert metrics["ssim"] is None


def test_ssim_rejects_images_smaller_than_three_pixels() -> None:
    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="at least 3x3"):
        ssim(tiny, tiny)
