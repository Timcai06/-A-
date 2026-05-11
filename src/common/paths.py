"""Common project paths and filesystem helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """Build an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_parents(paths: list[Path] | tuple[Path, ...]) -> None:
    """Create parent directories for several output files."""
    for path in paths:
        ensure_parent(path)

