import pytest

from aqua_align.config import ConfigError, load_config, resolve_project_path


def test_load_project_config() -> None:
    config = load_config("configs/project.yaml")
    assert config["project"]["name"] == "AquaAlign-VLM"
    assert config["model"]["name"] == "Qwen/Qwen3-VL-4B-Instruct"


def test_load_config_rejects_non_mapping() -> None:
    path = "tests/fixtures/non_mapping.yaml"
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_config(path)


def test_resolve_project_path_is_absolute() -> None:
    resolved = resolve_project_path("data/raw")
    assert resolved.is_absolute()
    assert resolved.name == "raw"
