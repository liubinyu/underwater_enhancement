from copy import deepcopy

import cv2
import numpy as np

from aqua_align.config import load_config
from aqua_align.degradation_features import extract_degradation_features
from aqua_align.rule_diagnosis import diagnose_features


RULES = load_config("configs/diagnosis_rules.yaml")


def _textured_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(40, 220, size=(128, 128, 3), dtype=np.uint8)


def _diagnose(image: np.ndarray) -> dict:
    return diagnose_features(extract_degradation_features(image), RULES, sample_id="synthetic")


def test_darkening_cannot_reduce_low_light_severity() -> None:
    image = _textured_image()
    dark = np.rint(image.astype(np.float32) * 0.22).astype(np.uint8)
    assert _diagnose(dark)["degradations"]["low_light"]["severity"] >= _diagnose(image)["degradations"]["low_light"]["severity"]


def test_reducing_contrast_cannot_reduce_low_contrast_severity() -> None:
    image = _textured_image()
    low_contrast = np.rint(128.0 + (image.astype(np.float32) - 128.0) * 0.12).astype(np.uint8)
    assert _diagnose(low_contrast)["degradations"]["low_contrast"]["severity"] >= _diagnose(image)["degradations"]["low_contrast"]["severity"]


def test_blurring_cannot_reduce_blur_severity() -> None:
    image = _textured_image()
    blurred = cv2.GaussianBlur(image, (21, 21), 5.0)
    assert _diagnose(blurred)["degradations"]["blur"]["severity"] >= _diagnose(image)["degradations"]["blur"]["severity"]


def test_high_saturation_synthetic_image_is_supported() -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, :48] = (255, 0, 0)
    image[:, 48:] = (0, 255, 0)
    result = _diagnose(image)
    assert result["degradations"]["over_saturation"]["severity"] > 0


def test_added_noise_increases_noise_proxy_and_does_not_reduce_noise_severity() -> None:
    base = np.full((128, 128, 3), 120, dtype=np.uint8)
    rng = np.random.default_rng(7)
    noisy = np.clip(base.astype(np.int16) + rng.normal(0, 35, base.shape), 0, 255).astype(np.uint8)
    base_features, noisy_features = extract_degradation_features(base), extract_degradation_features(noisy)
    assert noisy_features["noise_sigma_proxy"] > base_features["noise_sigma_proxy"]
    assert _diagnose(noisy)["degradations"]["possible_noise"]["severity"] >= _diagnose(base)["degradations"]["possible_noise"]["severity"]


def test_all_severity_and_confidence_values_are_bounded() -> None:
    result = _diagnose(_textured_image())
    for diagnosis in result["degradations"].values():
        assert diagnosis["severity"] in {0, 1, 2, 3}
        assert 0.0 <= diagnosis["confidence"] <= 1.0
        assert diagnosis["feature_checks"]
    assert 0.0 <= result["global_confidence"] <= 1.0
    assert result["annotation_source"] == "rule_based"
    assert result["needs_review"] is True


def test_rule_thresholds_are_config_driven() -> None:
    features = extract_degradation_features(_textured_image())
    changed = deepcopy(RULES)
    changed["degradations"]["low_light"]["indicators"]["mean_brightness"].update(
        {"mild": 255.0, "moderate": 254.0, "severe": 253.0}
    )
    original = diagnose_features(features, RULES)["degradations"]["low_light"]["severity"]
    modified = diagnose_features(features, changed)["degradations"]["low_light"]["severity"]
    assert modified >= original
