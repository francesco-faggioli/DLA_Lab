from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the DLA_1 project root from inside src/dla_lab1."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a project-relative path."""
    value = Path(path)
    if value.is_absolute():
        return value
    return (Path(root) if root is not None else project_root()) / value


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as Path."""
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value
