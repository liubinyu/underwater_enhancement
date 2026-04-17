from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.file_scan import scan_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(ROOT / "data" / "raw" / "images"))
    parser.add_argument("--max_read", type=int, default=200)
    args = parser.parse_args()

    paths = scan_images(args.input_dir)
    print(f"Image count: {len(paths)}")
    print("Examples:")
    for p in paths[:10]:
        print(f"  {p.name}")

    sizes = Counter()
    failures = 0
    for p in paths[: args.max_read]:
        try:
            with Image.open(p) as img:
                sizes[img.size] += 1
        except Exception:
            failures += 1
    print(f"Read for size statistics: {min(len(paths), args.max_read)}")
    print(f"Failed reads: {failures}")
    print("Top sizes:")
    for (w, h), n in sizes.most_common(20):
        print(f"  {w}x{h}: {n}")


if __name__ == "__main__":
    main()
