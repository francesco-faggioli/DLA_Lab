from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Trova la root del laboratorio Lab 2.

    Argomenti:
        Nessun argomento.

    Restituisce:
        Path assoluto della cartella `DLA_2`.
    """
    return Path(__file__).resolve().parents[2]


def artifacts_dir() -> Path:
    """
    Restituisce la cartella degli artefatti generati.

    Argomenti:
        Nessun argomento.

    Restituisce:
        Path della cartella `artifacts`, creata se non esiste.
    """
    path = project_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir(name: str) -> Path:
    """
    Costruisce una sottocartella di output per un esperimento.

    Argomenti:
        name: Nome breve dell'esperimento.

    Restituisce:
        Path della cartella `artifacts/name`, creata se non esiste.
    """
    path = artifacts_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
