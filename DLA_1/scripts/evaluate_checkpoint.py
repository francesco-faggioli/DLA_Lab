from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dla_lab1.config import experiment_config, load_config
from dla_lab1.data import build_dataloaders
from dla_lab1.evaluate import classification_metrics, predict
from dla_lab1.models import build_classifier
from dla_lab1.train import resolve_device

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """
    Legge da riga di comando il checkpoint da valutare.

    Argomenti:
        Nessun argomento diretto: usa `sys.argv`.

    Restituisce:
        Namespace con path del checkpoint e path del file di configurazione.
    """
    parser = argparse.ArgumentParser(
        description="Valuta un checkpoint addestrato sullo split di test GTSRB."
    )
    parser.add_argument("checkpoint", type=Path, nargs="?", help="Checkpoint .pt da valutare.")
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml")
    parser.add_argument(
        "--experiment",
        default=None,
        help="Nome opzionale dell'esperimento in config.yaml, usato per ricostruire l'architettura corrispondente.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Valuta un checkpoint PyTorch sul test set GTSRB.

    Argomenti:
        Nessun argomento diretto: usa gli argomenti letti da `parse_args()`.

    Restituisce:
        Codice di uscita del processo: 0 se la valutazione termina correttamente.
    """
    args = parse_args()
    config = load_config(args.config)
    if args.checkpoint is None and args.experiment is None:
        raise ValueError("Provide a checkpoint path or --experiment.")
    checkpoint = args.checkpoint or (ROOT / "checkpoints" / f"{args.experiment}.pt")
    model_config = (
        experiment_config(config, args.experiment)["model"] if args.experiment else config["model"]
    )
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
    model = build_classifier(
        model_name=model_config["name"],
        num_classes=int(config["project"]["num_classes"]),
        weights=None,
        freeze_backbone=False,
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    y_true, y_pred = predict(model, loaders["test"], device)
    metrics = classification_metrics(y_true.numpy(), y_pred.numpy())
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(metrics["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
