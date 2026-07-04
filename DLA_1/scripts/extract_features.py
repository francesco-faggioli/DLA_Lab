from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dla_lab1.config import load_config
from dla_lab1.experiments import run_feature_svm


def main() -> int:
    """
    Lancia la baseline di feature extraction e SVM.

    Args:
        Nessun argomento.

    Returns:
        Codice di uscita del processo: 0 se feature extraction e SVM terminano.
    """
    config = load_config(ROOT / "config" / "config.yaml")
    run_feature_svm(config, root=ROOT)
    print("Feature extraction and SVM baseline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
