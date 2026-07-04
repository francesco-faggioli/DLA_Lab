from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dla_lab1.config import load_config
from dla_lab1.experiments import run_feature_svm, run_finetuning


def parse_args() -> argparse.Namespace:
    """
    Legge gli argomenti da riga di comando per scegliere l'esperimento.

    Args:
        Nessun argomento diretto: usa `sys.argv`.

    Returns:
        Namespace con nome esperimento e path del file di configurazione.
    """
    parser = argparse.ArgumentParser(description="Run a DLA Lab 1 experiment.")
    parser.add_argument(
        "experiment",
        choices=[
            "feature_svm",
            "ex1_3_head_only_adam_ce",
            "finetune_frozen",
            "finetune_layer4",
            "ex3_1_layer4_unfrozen",
            "ex3_1_head_only_aggressive_aug",
            "ex3_1_layer4_aggressive_aug",
            "ex3_1_layer4_conservative_aug",
            "ex3_1_layer4_spatial_aug",
        ],
        help="Experiment defined in config/config.yaml.",
    )
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml")
    return parser.parse_args()


def main() -> int:
    """
    Esegue una run definita in `config.yaml`.

    Args:
        Nessun argomento diretto: usa gli argomenti letti da `parse_args()`.

    Returns:
        Codice di uscita del processo: 0 se la run termina correttamente.
    """
    args = parse_args()
    config = load_config(args.config)

    if args.experiment == "feature_svm":
        result = run_feature_svm(config, root=ROOT)
        print(f"Finished feature_svm. Predictions: {len(result['predictions'])}")
    else:
        result = run_finetuning(config, args.experiment, root=ROOT)
        print(f"Finished {args.experiment}. Epochs run: {len(result['history'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
