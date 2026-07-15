from pathlib import Path

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def requirement_lines(path: Path) -> list[str]:
    """Recursively collect requirement specifiers from pip requirement files."""
    collected: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            collected.extend(requirement_lines(path.parent / line[3:].strip()))
        else:
            collected.append(line)
    return collected


def test_requirement_files_are_valid() -> None:
    for filename in ("requirements.txt", "requirements-local.txt", "requirements-server.txt"):
        lines = requirement_lines(ROOT / filename)
        assert lines, f"{filename} must contain dependencies"
        for line in lines:
            Requirement(line)


def test_server_requirements_include_training_stack() -> None:
    names = {
        Requirement(line).name.lower()
        for line in requirement_lines(ROOT / "requirements-server.txt")
    }
    assert {"torch", "ms-swift", "transformers", "peft", "trl", "bitsandbytes"} <= names
