from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a Python dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML content must be a dict: {path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
