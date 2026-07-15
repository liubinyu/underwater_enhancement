import numpy as np
import pytest

from aqua_align.enhancement import (
    clahe,
    gamma_correction,
    gray_world,
    retinex,
    simplest_color_balance,
    white_balance_clahe,
    white_patch,
)


@pytest.fixture
def rgb_image() -> np.ndarray:
    y, x = np.mgrid[0:32, 0:40]
    return np.stack(((x * 6) % 256, (y * 8) % 256, ((x + y) * 4) % 256), axis=2).astype(np.uint8)


@pytest.mark.parametrize(
    "method",
    [
        gray_world,
        white_patch,
        clahe,
        lambda image: gamma_correction(image, 0.8),
        simplest_color_balance,
        lambda image: retinex(image, sigmas=(3.0, 9.0)),
        white_balance_clahe,
    ],
)
def test_enhancement_contract_and_no_in_place_modification(method, rgb_image: np.ndarray) -> None:
    original = rgb_image.copy()
    output = method(rgb_image)
    assert output.shape == rgb_image.shape
    assert output.dtype == np.uint8
    assert output.ndim == 3 and output.shape[2] == 3
    assert int(output.min()) >= 0 and int(output.max()) <= 255
    assert np.array_equal(rgb_image, original)


@pytest.mark.parametrize("gamma", [0.0, -1.0, float("nan"), float("inf")])
def test_gamma_rejects_invalid_values(rgb_image: np.ndarray, gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        gamma_correction(rgb_image, gamma)


def test_empty_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        gray_world(np.empty((0, 0, 3), dtype=np.uint8))


def test_grayscale_and_rgba_are_converted_to_rgb() -> None:
    grayscale = np.full((8, 9), 80, dtype=np.uint8)
    rgba = np.dstack((grayscale, grayscale + 10, grayscale + 20, np.full_like(grayscale, 127)))
    gray_output = gamma_correction(grayscale, 0.8)
    rgba_output = gamma_correction(rgba, 0.8)
    assert gray_output.shape == (8, 9, 3)
    assert rgba_output.shape == (8, 9, 3)
    assert gray_output.dtype == rgba_output.dtype == np.uint8
