"""Prepare UIEB or an equivalent image directory into portable metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.dataset import prepare_uieb_dataset
from aqua_align.utils import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse UIEB preparation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--source-dataset", default="UIEB")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Prepare images and return a nonzero status on any unhandled error."""
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    logger = setup_logger("aqua_align.prepare_uieb", log_file=output_dir / "prepare_uieb.log")
    try:
        rows = prepare_uieb_dataset(
            args.input_dir,
            output_dir,
            seed=args.seed,
            ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
            source_dataset=args.source_dataset,
            overwrite=args.overwrite,
            logger=logger,
        )
    except Exception:
        logger.exception("UIEB preparation failed")
        return 1
    split_counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    logger.info("Split counts: %s", split_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
