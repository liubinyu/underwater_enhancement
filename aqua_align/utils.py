"""Shared reproducibility and logging helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(
    name: str = "aqua_align",
    level: str | int = "INFO",
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create an idempotent console logger and an optional UTF-8 file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
