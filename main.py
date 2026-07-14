from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Underwater image enhancement project entry.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    sub.add_parser("hist")
    train = sub.add_parser("train")
    train.add_argument("--config", default=str(ROOT / "configs" / "physics_guided.yaml"))
    infer = sub.add_parser("infer")
    infer.add_argument("--config", default=str(ROOT / "configs" / "physics_guided.yaml"))
    infer.add_argument("--checkpoint", default="")
    infer.add_argument("--input", default=str(ROOT / "data" / "raw" / "images"))
    args, extra = parser.parse_known_args()

    script = {
        "scan": ROOT / "scripts" / "scan_dataset.py",
        "hist": ROOT / "scripts" / "plot_rgb_hist.py",
        "train": ROOT / "scripts" / "train.py",
        "infer": ROOT / "scripts" / "infer.py",
    }[args.cmd]

    cmd = [sys.executable, str(script)]
    if args.cmd in {"train", "infer"}:
        cmd += ["--config", args.config]
    if args.cmd == "infer":
        if args.checkpoint:
            cmd += ["--checkpoint", args.checkpoint]
        cmd += ["--input", args.input]
    cmd += extra
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
