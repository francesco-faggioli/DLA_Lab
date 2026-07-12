from __future__ import annotations

import importlib
import platform
import sys


def version(package: str) -> str:
    """
    Legge la versione installata di un pacchetto.

    Args:
        package: Nome del modulo da importare.

    Returns:
        Stringa con la versione, oppure `not installed`.
    """
    try:
        module = importlib.import_module(package)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def main() -> None:
    """
    Stampa un riepilogo dell'ambiente Python.

    Args:
        Nessun argomento.

    Returns:
        None. Scrive il report su stdout.
    """
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    for package in ["torch", "transformers", "datasets", "sklearn", "peft", "open_clip"]:
        print(f"{package}: {version(package)}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
