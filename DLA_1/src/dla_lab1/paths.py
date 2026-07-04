from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Trova la cartella principale del progetto DLA_1.

    Args:
        Nessun argomento.

    Returns:
        Path assoluto della cartella `DLA_1`, ricavato dalla posizione del file corrente.
    """
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """
    Converte un path relativo al progetto in un path assoluto.

    Args:
        path: Path da risolvere. Se e' gia' assoluto viene restituito invariato.
        root: Cartella di riferimento. Se non viene passata, usa `project_root()`.

    Returns:
        Path assoluto pronto per essere usato da script e notebook.
    """
    value = Path(path)
    if value.is_absolute():
        return value
    return (Path(root) if root is not None else project_root()) / value


def ensure_dir(path: str | Path) -> Path:
    """
    Crea una cartella se non esiste.

    Args:
        path: Cartella da creare.

    Returns:
        Path della cartella creata o gia' esistente.
    """
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value
