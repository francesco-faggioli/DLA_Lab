from __future__ import annotations

import argparse
from pathlib import Path

from dla_lab1.config import load_config
from dla_lab1.experiments import run_feature_svm, run_finetuning

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """
    Legge gli argomenti da riga di comando per scegliere l'esperimento.

    Argomenti:
        Nessun argomento diretto: usa `sys.argv`.

    Restituisce:
        Namespace con nome esperimento e path del file di configurazione.
    """
    parser = argparse.ArgumentParser(description="Esegue un esperimento di DLA Lab 1.")
    parser.add_argument(
        "experiment",
        choices=[
            "feature_svm",
            "ex1_3_head_only_adam_ce",
            "Adam_CrossEntropy",
            "AdamW_CrossEntropy",
            "SGD_CrossEntropy",
            "Adam_WeightedCrossEntropy",
            "AdamW_WeightedCrossEntropy",
            "SGD_WeightedCrossEntropy",
            "Adam_FocalLoss",
            "AdamW_FocalLoss",
            "SGD_FocalLoss",
            "finetune_frozen",
            "finetune_layer4",
            "ex3_1_layer4_unfrozen",
            "ex3_1_head_only_aggressive_aug",
            "ex3_1_layer4_aggressive_aug",
            "ex3_1_layer4_conservative_aug",
            "ex3_1_layer4_spatial_aug",
            "ex3_1_layer4_no_aug_lr1e4_wd05",
            "ex3_1_layer4_safe_aug_lr1e4_wd05",
            "ex3_1_layer4_safe_aug_lr2e4_wd05",
            "ex3_1_layer4_conservative_ls005",
            "ex3_1_layer4_safe_aug_ls005_discriminative",
        ],
        help="Esperimento definito in config/config.yaml.",
    )
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml")
    return parser.parse_args()


def main() -> int:
    """
    Esegue una run definita in `config.yaml`.

    Argomenti:
        Nessun argomento diretto: usa gli argomenti letti da `parse_args()`.

    Restituisce:
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
