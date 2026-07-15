"""Dataset discovery and preparation helpers for UIEB-style image layouts."""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from aqua_align.config import project_relative_path, project_root


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REFERENCE_HINTS = {
    "reference",
    "references",
    "reference-890",
    "target",
    "targets",
    "ground-truth",
    "ground_truth",
    "gt",
}
RAW_HINTS = {
    "raw",
    "raw-890",
    "input",
    "inputs",
    "underwater",
    "challenging",
    "challenging-60",
    "images",
    "images_gopro",
}


def readable_image(path: str | Path) -> tuple[bool, str | None]:
    """Check that Pillow can decode an image fully without retaining a file handle."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        return False, str(exc)
    return True, None


def _normalized_parts(path: Path, root: Path) -> set[str]:
    parts = path.relative_to(root).parts[:-1]
    return {part.casefold().replace(" ", "_") for part in parts}


def discover_uieb_images(input_dir: str | Path) -> tuple[list[Path], dict[str, Path]]:
    """Discover raw images and reference images in common UIEB directory layouts."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"UIEB input directory not found: {root}")
    images = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().casefold(),
    )
    if not images:
        raise FileNotFoundError(f"no supported images found under: {root}")

    raw_paths: list[Path] = []
    reference_paths: list[Path] = []
    for path in images:
        parts = _normalized_parts(path, root)
        if parts & REFERENCE_HINTS:
            reference_paths.append(path)
        else:
            raw_paths.append(path)

    if not raw_paths:
        raise ValueError("all discovered images were classified as references; no raw images found")
    _raise_duplicate_names(raw_paths, "raw")
    _raise_duplicate_names(reference_paths, "reference")

    references: dict[str, Path] = {}
    for path in reference_paths:
        key = path.stem.casefold()
        if key in references:
            raise ValueError(f"duplicate reference stem {path.stem!r}: {references[key]} and {path}")
        references[key] = path
    return raw_paths, references


def _raise_duplicate_names(paths: list[Path], role: str) -> None:
    names = Counter(path.name.casefold() for path in paths)
    duplicates = sorted(name for name, count in names.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {role} filenames detected: {', '.join(duplicates)}")


def split_sample_ids(
    sample_ids: list[str],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each original sample to exactly one deterministic data split."""
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("ratios must contain three finite non-negative values")
    if not np.isclose(values.sum(), 1.0, atol=1e-8):
        raise ValueError("split ratios must sum to 1.0")

    total = len(sample_ids)
    exact = values * total
    counts = np.floor(exact).astype(int)
    remainder = total - int(counts.sum())
    order = sorted(range(3), key=lambda index: (exact[index] - counts[index], values[index]), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1

    shuffled = list(sample_ids)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    labels = ("train", "val", "test")
    assignments: dict[str, str] = {}
    cursor = 0
    for label, count in zip(labels, counts, strict=True):
        for sample_id in shuffled[cursor : cursor + int(count)]:
            assignments[sample_id] = label
        cursor += int(count)
    return assignments


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "image"


def _save_rgb_jpeg(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output image already exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="JPEG", quality=95, subsampling=0)


def prepare_uieb_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    source_dataset: str = "UIEB",
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Normalize a UIEB-style directory and write portable metadata.csv."""
    log = logger or logging.getLogger(__name__)
    root = project_root()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    project_relative_path(destination, root)
    destination.mkdir(parents=True, exist_ok=True)

    raw_paths, references = discover_uieb_images(input_dir)
    valid_references: dict[str, Path] = {}
    for key, reference_path in references.items():
        readable, error = readable_image(reference_path)
        if not readable:
            log.error("Unreadable reference image %s: %s", reference_path, error)
        else:
            valid_references[key] = reference_path
    raw_stems = {path.stem.casefold() for path in raw_paths}
    for key, reference_path in valid_references.items():
        if key not in raw_stems:
            log.warning("Reference image has no matching raw image: %s", reference_path)

    valid_pairs: list[tuple[Path, Path | None]] = []
    for raw_path in raw_paths:
        readable, error = readable_image(raw_path)
        if not readable:
            log.error("Unreadable raw image %s: %s", raw_path, error)
            continue
        reference = valid_references.get(raw_path.stem.casefold())
        valid_pairs.append((raw_path, reference))
    if not valid_pairs:
        raise ValueError("no readable raw images were found")
    valid_pairs.sort(key=lambda pair: (pair[1] is None, pair[0].as_posix().casefold()))

    sample_ids = [f"sample_{index:04d}" for index in range(1, len(valid_pairs) + 1)]
    assignments = split_sample_ids(sample_ids, ratios=ratios, seed=seed)
    rows: list[dict[str, Any]] = []
    for sample_id, (raw_source, reference_source) in zip(sample_ids, valid_pairs, strict=True):
        raw_destination = destination / "raw" / f"{sample_id}.jpg"
        _save_rgb_jpeg(raw_source, raw_destination, overwrite=overwrite)
        reference_destination: Path | None = None
        if reference_source is not None:
            reference_destination = destination / "reference" / f"{sample_id}.jpg"
            _save_rgb_jpeg(reference_source, reference_destination, overwrite=overwrite)
        rows.append(
            {
                "sample_id": sample_id,
                "raw_image": project_relative_path(raw_destination, root),
                "reference_image": (
                    project_relative_path(reference_destination, root) if reference_destination else ""
                ),
                "has_reference": reference_destination is not None,
                "split": assignments[sample_id],
                "source_dataset": source_dataset,
                "source_filename": raw_source.name,
                "source_stem": _safe_stem(raw_source.stem),
            }
        )

    metadata_path = destination / "metadata.csv"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"metadata already exists; pass --overwrite: {metadata_path}")
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Prepared %d samples at %s", len(rows), metadata_path)
    return rows
