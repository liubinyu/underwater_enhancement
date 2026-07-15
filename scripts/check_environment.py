"""Report AquaAlign-VLM runtime capabilities without changing the environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PACKAGE_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "ms_swift": "ms-swift",
    "peft": "peft",
    "trl": "trl",
}


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version, or ``None`` when unavailable."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect Python, training framework, CUDA, and GPU capability information."""
    versions = {key: package_version(dist) for key, dist in PACKAGE_NAMES.items()}
    report: dict[str, Any] = {
        "python": platform.python_version(),
        **versions,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "bitsandbytes_available": importlib.util.find_spec("bitsandbytes") is not None,
        "bf16_supported": False,
    }

    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        if report["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            report["gpu_name"] = properties.name
            report["gpu_memory_gb"] = round(properties.total_memory / (1024**3), 2)
            is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
            report["bf16_supported"] = bool(is_bf16_supported and is_bf16_supported())

    missing = [distribution for key, distribution in PACKAGE_NAMES.items() if report[key] is None]
    if not report["bitsandbytes_available"]:
        missing.append("bitsandbytes")
    report["missing_packages"] = missing
    report["install_command"] = (
        f"{sys.executable} -m pip install " + " ".join(missing) if missing else None
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    return parser.parse_args()


def main() -> int:
    """Print the environment report and optionally save the same JSON to disk."""
    args = parse_args()
    report = collect_environment()
    rendered = json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
