import cv2
import numpy as np

from aqua_align.degradation_features import extract_degradation_features


def _color_gradient() -> np.ndarray:
    y, x = np.mgrid[0:96, 0:128]
    return np.stack(
        ((40 + x) % 256, (30 + 2 * y) % 256, (20 + x + y) % 256), axis=2
    ).astype(np.uint8)


def test_features_are_finite_json_scalars_and_input_is_unchanged() -> None:
    image = _color_gradient()
    original = image.copy()
    features = extract_degradation_features(image)
    assert np.array_equal(image, original)
    assert features["feature_version"] == "0.1"
    assert all(isinstance(value, (float, str)) for value in features.values())
    assert all(np.isfinite(value) for value in features.values() if isinstance(value, float))


def test_red_deficit_decreases_when_red_channel_is_strengthened() -> None:
    image = _color_gradient()
    weak_red = image.copy()
    weak_red[..., 0] = weak_red[..., 0] // 4
    strong_red = weak_red.copy()
    strong_red[..., 0] = np.clip(strong_red[..., 0].astype(np.int16) * 3, 0, 255).astype(np.uint8)
    assert extract_degradation_features(strong_red)["red_deficit"] < extract_degradation_features(weak_red)["red_deficit"]


def test_blurring_does_not_increase_sharpness_features() -> None:
    image = _color_gradient()
    blurred = cv2.GaussianBlur(image, (15, 15), 4.0)
    sharp_features = extract_degradation_features(image)
    blur_features = extract_degradation_features(blurred)
    assert blur_features["laplacian_variance"] <= sharp_features["laplacian_variance"]
    assert blur_features["sobel_gradient_mean"] <= sharp_features["sobel_gradient_mean"]


def test_empty_image_is_rejected() -> None:
    try:
        extract_degradation_features(np.empty((0, 0, 3), dtype=np.uint8))
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty image must be rejected")
