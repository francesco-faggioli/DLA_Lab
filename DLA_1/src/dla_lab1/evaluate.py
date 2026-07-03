from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def predict(model, dataloader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Serve a ottenere le predizioni del modello su un DataLoader.

    Restituisce label vere e predizioni su CPU, cosi' possiamo calcolare
    accuracy, classification report e confusion matrix con scikit-learn.
    """
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    with torch.inference_mode():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            logits = model(inputs)
            y_true.append(labels.cpu())
            y_pred.append(logits.argmax(dim=1).cpu())
    return torch.cat(y_true), torch.cat(y_pred)


def classification_metrics(y_true, y_pred, target_names=None) -> dict:
    """
    Serve a calcolare le metriche principali di classificazione.

    La accuracy e' il numero piu' immediato, mentre il classification report
    mostra precision, recall e F1 anche per le singole classi.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def history_to_frame(history) -> pd.DataFrame:
    """
    Serve a trasformare la storia del training in una tabella Pandas.

    La usiamo nel notebook per visualizzare e commentare loss e accuracy.
    """
    return pd.DataFrame([asdict(row) for row in history])


def save_text_report(path: str | Path, report: str) -> Path:
    """
    Serve a salvare su disco un report testuale.

    E' utile quando vogliamo archiviare i risultati di una run senza
    dipendere solo dall'output del notebook.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
