from pathlib import Path

import yaml

from aqua_align.strategy_mapping import map_diagnosis_to_strategy


CONFIG = yaml.safe_load(Path("configs/strategy_mapping.yaml").read_text(encoding="utf-8"))


def _diagnosis(**severities: int) -> dict:
    return {"degradations": {key: {"severity": value} for key, value in severities.items()}}


def test_low_light_and_contrast_are_ordered_and_parameterized() -> None:
    result = map_diagnosis_to_strategy(_diagnosis(low_light=2, low_contrast=3), CONFIG)
    operations = [item["operation"] for item in result["recommended_pipeline"]]
    assert operations == ["gamma_correction", "retinex", "clahe"]
    assert result["recommended_pipeline"][0]["parameters"]["gamma"] == 0.75


def test_noise_removes_sharpening_and_caps_clahe() -> None:
    result = map_diagnosis_to_strategy(
        _diagnosis(possible_noise=2, blur=3, low_contrast=3), CONFIG
    )
    steps = result["recommended_pipeline"]
    operations = [item["operation"] for item in steps]
    assert operations[0] == "denoise"
    assert "mild_sharpening" not in operations
    clahe = next(item for item in steps if item["operation"] == "clahe")
    assert clahe["parameters"]["clip_limit"] <= 1.8


def test_red_cast_prevents_red_compensation() -> None:
    result = map_diagnosis_to_strategy(
        _diagnosis(blue_green_color_cast=2, red_color_cast=2), CONFIG
    )
    operations = [item["operation"] for item in result["recommended_pipeline"]]
    assert "white_balance" in operations
    assert "red_channel_compensation" not in operations


def test_no_active_degradation_preserves_original() -> None:
    result = map_diagnosis_to_strategy(_diagnosis(low_light=0), CONFIG)
    assert result["recommended_pipeline"][0]["operation"] == "preserve_original"
    assert result["needs_review"] is True
