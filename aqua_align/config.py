"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 YAML mapping without mutating or resolving its values."""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return config


def project_root() -> Path:
    """Return the repository root independent of the process working directory."""
    return Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a configured path against the repository root when it is relative."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(root).expanduser().resolve() if root is not None else project_root()
    return (base / path).resolve()


def project_relative_path(value: str | Path, root: str | Path | None = None) -> str:
    """Return a portable POSIX path relative to the project root.

    Raises:
        ValueError: If ``value`` is outside the selected project root.
    """
    base = Path(root).expanduser().resolve() if root is not None else project_root()
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside project root {base}: {path}") from exc
