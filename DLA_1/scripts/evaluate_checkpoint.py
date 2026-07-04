from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dla_lab1.config import load_config
from dla_lab1.data import build_dataloaders
from dla_lab1.evaluate import classification_metrics, predict
from dla_lab1.models import build_classifier
from dla_lab1.train import resolve_device


def parse_args() -> argparse.Namespace:
    """
    Legge da riga di comando il checkpoint da valutare.

    Args:
        Nessun argomento diretto: usa `sys.argv`.

    Returns:
        Namespace con path del checkpoint e path del file di configurazione.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on GTSRB test split.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml")
    return parser.parse_args()


def main() -> int:
    """
    Valuta un checkpoint PyTorch sul test set GTSRB.

    Args:
        Nessun argomento diretto: usa gli argomenti letti da `parse_args()`.

    Returns:
        Codice di uscita del processo: 0 se la valutazione termina correttamente.
    """
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config["project"].get("device", "auto"))

    loaders = build_dataloaders(
        data_root=ROOT / config["paths"]["data_root"],
        image_size=int(config["dataset"]["image_size"]),
        batch_size=int(config["hardware"]["batch_size_finetune_frozen"]),
        val_split=float(config["dataset"]["val_split"]),
        track_size=int(config["dataset"]["track_size"]),
        seed=int(config["project"]["seed"]),
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )
    model_cfg = config["model"]
    model = build_classifier(
        model_name=model_cfg["name"],
        num_classes=int(config["project"]["num_classes"]),
        weights=None,
        freeze_backbone=False,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    y_true, y_pred = predict(model, loaders["test"], device)
    metrics = classification_metrics(y_true.numpy(), y_pred.numpy())
    print(metrics["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
