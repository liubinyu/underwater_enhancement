import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from aqua_align.candidates import generate_candidates
from aqua_align.config import load_config, project_root, resolve_project_path
from aqua_align.dataset import prepare_uieb_dataset, split_sample_ids


def _write_synthetic_image(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[0:24, 0:32]
    image = np.stack(
        ((x * 5 + index * 7) % 256, (y * 8 + index * 11) % 256, ((x + y) * 3 + index) % 256),
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def test_split_assigns_each_original_once() -> None:
    sample_ids = [f"sample_{index:04d}" for index in range(20)]
    assignments = split_sample_ids(sample_ids, seed=42)
    assert set(assignments) == set(sample_ids)
    assert set(assignments.values()) == {"train", "val", "test"}
    assert sum(split == "train" for split in assignments.values()) == 14
    assert sum(split == "val" for split in assignments.values()) == 3
    assert sum(split == "test" for split in assignments.values()) == 3


def test_prepare_and_candidate_builders_use_valid_relative_paths() -> None:
    runtime_dir = project_root() / "outputs" / "test_synthetic"
    input_dir = runtime_dir / "synthetic_uieb"
    for index in range(6):
        filename = f"image_{index:02d}.png"
        _write_synthetic_image(input_dir / "raw-890" / filename, index)
        if index < 4:
            _write_synthetic_image(input_dir / "reference-890" / filename, index + 1)

    output_dir = runtime_dir / "processed_uieb"
    rows = prepare_uieb_dataset(input_dir, output_dir, seed=42, overwrite=True)
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(not Path(row["raw_image"]).is_absolute() for row in rows)
    assert all(resolve_project_path(row["raw_image"]).is_file() for row in rows)

    config = load_config(project_root() / "configs" / "data.yaml")
    candidates_csv = runtime_dir / "candidates.csv"
    candidate_rows = generate_candidates(
        output_dir / "metadata.csv",
        config,
        runtime_dir / "candidates",
        output_csv=candidates_csv,
        limit=3,
        overwrite=True,
        workers=2,
    )
    assert candidate_rows
    split_by_sample = {row["sample_id"]: row["split"] for row in rows}
    for row in candidate_rows:
        assert resolve_project_path(row["raw_image"]).is_file()
        assert resolve_project_path(row["candidate_image"]).is_file()
        assert json.loads(row["parameters"]) is not None
        assert row["split"] == split_by_sample[row["sample_id"]]

    with candidates_csv.open("r", newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == len(candidate_rows)
    assert all(resolve_project_path(row["candidate_image"]).exists() for row in persisted)
