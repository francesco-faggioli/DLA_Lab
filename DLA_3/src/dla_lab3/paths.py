from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the DLA_3 project root.

    Args:
        None.

    What it does:
        Resolves the directory that contains `src`, `notebooks`, `scripts`,
        `config`, `artifacts` and `checkpoints`.

    Outputs:
        Path pointing to the project root.
    """

    return Path(__file__).resolve().parents[2]


def artifact_dir(*parts: str, create: bool = True) -> Path:
    """Return a path inside `artifacts`.

    Args:
        *parts: Optional subdirectories or filename components.
        create: If True, create the parent directory or the directory itself.

    What it does:
        Centralizes local generated outputs outside notebooks.

    Outputs:
        Path inside the project `artifacts` folder.
    """

    path = project_root() / "artifacts" / Path(*parts)
    if create:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(*parts: str, create: bool = True) -> Path:
    """Return a path inside `checkpoints`.

    Args:
        *parts: Optional subdirectories or filename components.
        create: If True, create the parent directory or the directory itself.

    What it does:
        Keeps model files out of notebook folders.

    Outputs:
        Path inside the project `checkpoints` folder.
    """

    path = project_root() / "checkpoints" / Path(*parts)
    if create:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
    return path
