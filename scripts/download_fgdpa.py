"""Download a small, pinned official checkpoint and verify its SHA256."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Keep downloading independent of torch so it also works before installing it.
COMMIT = "530c692feb99ecf6205bb929118a9167e2809d4d"
SHA256 = "7f278f37eca028e91a11a53b178ed3cb0b5d966fc48ccc7e7082bf8c4c1a8d3c"
URL = f"https://raw.githubusercontent.com/LethyZhang/FGDPA/{COMMIT}/experiments/pretrain/models/model_best_slim.pkl"


def main() -> None:
    target = ROOT / "checkpoints/fgdpa/model_best_slim.pkl"
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"Existing checkpoint has unexpected hash; inspect it before replacing: {target}")
        print(f"Verified existing checkpoint: {target}")
        return
    with urlopen(URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Download SHA256 mismatch; no checkpoint written")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    print(f"Downloaded and verified: {target}")


if __name__ == "__main__":
    main()
