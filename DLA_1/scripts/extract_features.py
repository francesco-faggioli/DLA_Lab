from __future__ import annotations

from pathlib import Path

from dla_lab1.config import load_config
from dla_lab1.experiments import run_feature_svm

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """
    Lancia la baseline di feature extraction e SVM.

    Argomenti:
        Nessun argomento.

    Restituisce:
        Codice di uscita del processo: 0 se feature extraction e SVM terminano.
    """
    config = load_config(ROOT / "config" / "config.yaml")
    run_feature_svm(config, root=ROOT)
    print("Estrazione delle feature e baseline SVM completate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
