from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dla_lab1.config import load_config
from dla_lab1.experiments import run_feature_svm, run_finetuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DLA Lab 1 experiment.")
    parser.add_argument(
        "experiment",
        choices=["feature_svm", "ex1_3_head_only_adam_ce", "finetune_frozen", "finetune_layer4"],
        help="Experiment defined in config/config.yaml.",
    )
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml")
    return parser.parse_args()


def main() -> int:
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
