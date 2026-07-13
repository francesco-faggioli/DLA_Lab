from __future__ import annotations

import torch.nn as nn
from torchvision.models import get_model


def build_classifier(
    model_name: str = "resnet18",
    num_classes: int = 43,
    weights: str | None = "DEFAULT",
    freeze_backbone: bool = True,
    unfreeze_layer4: bool = False,
) -> nn.Module:
    """
    Serve a creare un modello pre-addestrato adattato alle 43 classi GTSRB.

    Lo useremo nel notebook di fine-tuning: sostituisce il classificatore finale
    ImageNet con un classificatore per i segnali stradali.

    Argomenti:
        model_name: Nome del modello torchvision, ad esempio `resnet18`.
        num_classes: Numero di classi finali da predire.
        weights: Pesi pre-addestrati da caricare, oppure None.
        freeze_backbone: Se True, congela tutti i layer del backbone.
        unfreeze_layer4: Se True, riattiva l'ultimo blocco ResNet per il fine-tuning selettivo.

    Restituisce:
        Modello PyTorch con layer finale sostituito e parametri addestrabili configurati.
    """
    model = get_model(model_name, weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_layer4 and hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_feature_extractor(
    model_name: str = "resnet18", weights: str | None = "DEFAULT"
) -> nn.Module:
    """
    Serve a usare ResNet-18 come estrattore di feature.

    Rimuove il classificatore finale e restituisce vettori da 512 dimensioni
    per ogni immagine; questi vettori vengono poi usati dalla SVM.

    Argomenti:
        model_name: Nome del modello torchvision da usare come backbone.
        weights: Pesi pre-addestrati da caricare, oppure None.

    Restituisce:
        Modello PyTorch che produce feature invece di logits di classificazione.
    """
    model = get_model(model_name, weights=weights)
    model.fc = nn.Identity()
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Serve a contare parametri totali e addestrabili.

    E' utile per spiegare quanto del modello stiamo davvero allenando.

    Argomenti:
        model: Modello PyTorch da analizzare.

    Restituisce:
        Dizionario con `total` e `trainable`.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
