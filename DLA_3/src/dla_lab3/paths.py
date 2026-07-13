from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Restituisce la root del progetto DLA 3.

    Argomenti:
        Nessuno.

    Operazione:
        Risolve la directory che contiene `src`, `notebooks`, `scripts`,
        `config`, `artifacts` e `checkpoints`.

    Output:
        Percorso della root del progetto.
    """

    return Path(__file__).resolve().parents[2]


def artifact_dir(*parts: str, create: bool = True) -> Path:
    """Restituisce un percorso interno ad `artifacts`.

    Argomenti:
        *parts: Sottodirectory o componenti del nome file opzionali.
        create: Se True, crea la directory o la relativa directory genitore.

    Operazione:
        Centralizza gli output generati fuori dai notebook. Con `create=True`
        può creare directory sul filesystem.

    Output:
        Percorso interno alla cartella `artifacts`.
    """

    path = project_root() / "artifacts" / Path(*parts)
    if create:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(*parts: str, create: bool = True) -> Path:
    """Restituisce un percorso interno a `checkpoints`.

    Argomenti:
        *parts: Sottodirectory o componenti del nome file opzionali.
        create: Se True, crea la directory o la relativa directory genitore.

    Operazione:
        Mantiene i modelli fuori dalle cartelle dei notebook. Con `create=True`
        può creare directory sul filesystem.

    Output:
        Percorso interno alla cartella `checkpoints`.
    """

    path = project_root() / "checkpoints" / Path(*parts)
    if create:
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
    return path
