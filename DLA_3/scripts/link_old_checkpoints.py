from __future__ import annotations

import argparse
from pathlib import Path


def project_root() -> Path:
    """Restituisce la root del progetto DLA 3.

    Argomenti:
        Nessuno.

    Operazione:
        Risolve la directory genitore di `scripts`.

    Output:
        Percorso della cartella di lavoro del laboratorio.
    """

    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collega vecchi checkpoint alla cartella del laboratorio."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root() / "checkpoints" / "old_external",
        help="Directory contenente i vecchi checkpoint.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="old",
        help="Nome del collegamento simbolico creato nella cartella locale dei checkpoint.",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Directory sorgente dei checkpoint non trovata: {source}")

    checkpoints = project_root() / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    target = checkpoints / args.name
    if target.exists() or target.is_symlink():
        print(f"Il collegamento ai checkpoint esiste già: {target}")
        print(f"Points to: {target.resolve()}")
        return

    target.symlink_to(source, target_is_directory=True)
    print(f"Created checkpoint link: {target}")
    print(f"Points to: {source}")


if __name__ == "__main__":
    main()
