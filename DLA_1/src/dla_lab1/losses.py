from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Serve a dare piu' peso agli esempi difficili durante il training.

    E' una loss alternativa alla CrossEntropy, utile da provare quando le
    classi sono sbilanciate o quando il modello ignora esempi difficili.

    Args:
        alpha: Fattore moltiplicativo della loss.
        gamma: Intensita' con cui vengono pesati di piu' gli esempi difficili.
        weights: Pesi opzionali per classe.
        reduction: Tipo di riduzione finale: `mean`, `sum` o nessuna.

    Returns:
        Oggetto `nn.Module` usabile come funzione di loss nel training loop.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        """
        Inizializza i parametri della Focal Loss.

        Args:
            alpha: Fattore moltiplicativo della loss.
            gamma: Peso dato agli esempi difficili.
            weights: Pesi opzionali per classe.
            reduction: Riduzione finale della loss.

        Returns:
            None.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if weights is not None:
            self.register_buffer("class_weights", weights.detach().float())
        else:
            self.class_weights = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Serve a calcolare la Focal Loss su un batch.

        Riceve i logits del modello e le label corrette, poi restituisce
        una loss media usabile con `backward()`.

        Args:
            inputs: Logits prodotti dal modello, forma `[batch, num_classes]`.
            targets: Label corrette, forma `[batch]`.

        Returns:
            Tensore scalare se `reduction` e' `mean` o `sum`, altrimenti una loss per esempio.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str, class_weights: torch.Tensor | None = None, device: torch.device | None = None, **kwargs):
    """
    Serve a creare la funzione di loss scelta nella configurazione.

    Nel notebook usiamo soprattutto CrossEntropy, ma la funzione supporta
    anche WeightedCrossEntropy e FocalLoss per gli esperimenti preliminari.

    Args:
        name: Nome della loss: `CrossEntropy`, `WeightedCrossEntropy` o `FocalLoss`.
        class_weights: Pesi per classe, richiesti dalle loss pesate.
        device: Device su cui spostare i pesi.
        **kwargs: Parametri extra, ad esempio `alpha_focal` e `gamma_focal`.

    Returns:
        Funzione di loss PyTorch pronta per il training.
    """
    weights = class_weights.to(device) if class_weights is not None and device is not None else class_weights

    if name == "CrossEntropy":
        return nn.CrossEntropyLoss()
    if name == "WeightedCrossEntropy":
        if weights is None:
            raise ValueError("WeightedCrossEntropy requires class weights.")
        return nn.CrossEntropyLoss(weight=weights)
    if name == "FocalLoss":
        if weights is None:
            raise ValueError("FocalLoss requires class weights.")
        return FocalLoss(
            alpha=kwargs.get("alpha_focal", 1.0),
            gamma=kwargs.get("gamma_focal", 2.0),
            weights=weights,
        )
    raise ValueError(f"Unsupported loss {name!r}.")
