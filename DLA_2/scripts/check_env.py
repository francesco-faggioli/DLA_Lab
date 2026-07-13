from __future__ import annotations

import importlib
import platform
import sys


def version(package: str) -> str:
    """
    Legge la versione installata di un pacchetto.

    Argomenti:
        package: Nome del modulo da importare.

    Restituisce:
        Stringa con la versione, oppure `non installato`.
    """
    try:
        module = importlib.import_module(package)
    except ImportError:
        return "non installato"
    return getattr(module, "__version__", "sconosciuta")


def main() -> None:
    """
    Stampa un riepilogo dell'ambiente Python.

    Argomenti:
        Nessun argomento.

    Restituisce:
        None. Scrive il report su stdout.
    """
    print(f"Interprete: {sys.executable}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Piattaforma: {platform.platform()}")
    for package in [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "datasets",
        "sklearn",
        "peft",
        "open_clip",
        "dla_lab2",
    ]:
        print(f"{package}: {version(package)}")

    try:
        import torch

        print(f"CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
