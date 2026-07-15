"""Generate traditional underwater enhancement candidates from metadata.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.candidates import generate_candidates
from aqua_align.config import load_config, resolve_project_path
from aqua_align.utils import set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse candidate generation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/candidates.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run candidate generation and preserve full failures in a UTF-8 log."""
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_project_path(args.output_dir)
    logger = setup_logger(
        "aqua_align.generate_candidates",
        log_file=None if args.dry_run else output_dir / "generation.log",
    )
    try:
        config = load_config(resolve_project_path(args.config))
        generate_candidates(
            args.metadata,
            config,
            output_dir,
            output_csv=args.output_csv,
            limit=args.limit,
            overwrite=args.overwrite,
            workers=args.workers,
            dry_run=args.dry_run,
            logger=logger,
        )
    except Exception:
        logger.exception("Candidate generation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
