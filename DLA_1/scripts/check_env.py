from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    """
    Stampa un riepilogo dell'ambiente Python e del dataset locale.

    Args:
        Nessun argomento.

    Returns:
        Codice di uscita del processo: 0 se il controllo termina.
    """
    print(f"Python: {platform.python_version()}")
    print(f"Project root: {ROOT}")

    try:
        import torch
        import torchvision

        print(f"PyTorch: {torch.__version__}")
        print(f"torchvision: {torchvision.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name}")
            print(f"VRAM: {props.total_memory / 1024 ** 3:.2f} GB")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")

    try:
        import sklearn
        import yaml

        print(f"scikit-learn: {sklearn.__version__}")
        print(f"PyYAML: {yaml.__version__}")
    except Exception as exc:
        print(f"Package check failed: {exc}")

    for package in ["numpy", "pandas"]:
        result = subprocess.run(
            [sys.executable, "-c", f"import {package}; print({package}.__version__)"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = "ok" if result.returncode == 0 else f"failed rc={result.returncode}"
        details = (result.stdout or result.stderr).strip()
        print(f"{package}: {status} {details}")

    data_root = ROOT / "data" / "gtsrb"
    print(f"Dataset root exists: {data_root.exists()} ({data_root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
