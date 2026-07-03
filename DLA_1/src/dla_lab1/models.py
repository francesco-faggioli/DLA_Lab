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


def build_feature_extractor(model_name: str = "resnet18", weights: str | None = "DEFAULT") -> nn.Module:
    """
    Serve a usare ResNet-18 come estrattore di feature.

    Rimuove il classificatore finale e restituisce vettori da 512 dimensioni
    per ogni immagine; questi vettori vengono poi usati dalla SVM.
    """
    model = get_model(model_name, weights=weights)
    model.fc = nn.Identity()
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """
    Serve a contare parametri totali e addestrabili.

    E' utile per spiegare quanto del modello stiamo davvero allenando.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
