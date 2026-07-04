from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(labels, title: str = "Class distribution"):
    """
    Serve a disegnare la distribuzione delle classi.

    Nel primo notebook lo usiamo per rendere visibile lo sbilanciamento.

    Args:
        labels: Sequenza di label intere.
        title: Titolo del grafico.

    Returns:
        Coppia `(fig, ax)` di Matplotlib.
    """
    counts = np.bincount(np.asarray(labels, dtype=int))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(np.arange(len(counts)), counts)
    ax.set_title(title)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Count")
    ax.set_xticks(np.arange(len(counts)))
    return fig, ax


def plot_training_history(history_frame):
    """
    Serve a disegnare loss e accuracy di train/validation.

    Sara' utile nella parte di fine-tuning.

    Args:
        history_frame: DataFrame con colonne epoch, train/val loss e train/val accuracy.

    Returns:
        Coppia `(fig, axes)` di Matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    axes[0].plot(history_frame["epoch"], history_frame["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_frame["epoch"], history_frame["train_acc"], label="train")
    axes[1].plot(history_frame["epoch"], history_frame["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    return fig, axes


def plot_confusion_matrix(cm, title: str = "Confusion matrix"):
    """
    Serve a visualizzare una matrice di confusione.

    La useremo quando confronteremo gli errori del modello sulle varie classi.

    Args:
        cm: Matrice di confusione.
        title: Titolo del grafico.

    Returns:
        Coppia `(fig, ax)` di Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    return fig, ax


def plot_retrieval_results(query_dataset, gallery_dataset, sim_matrix, query_index: int, k: int = 5):
    """
    Serve a visualizzare una query e le K immagini piu' simili della gallery.

    La usiamo nell'Exercise 3.2 solo come controllo qualitativo: il colore verde
    indica un recupero della stessa classe, il rosso un recupero errato.

    Args:
        query_dataset: Dataset da cui prendere l'immagine query.
        gallery_dataset: Dataset usato come gallery retrieval.
        sim_matrix: Matrice di similarita' query-gallery.
        query_index: Indice della query da visualizzare.
        k: Numero di immagini recuperate da mostrare.

    Returns:
        Coppia `(fig, axes)` di Matplotlib.
    """
    query_img, query_label = query_dataset[query_index]
    topk_scores, topk_indices = torch.topk(sim_matrix[query_index], k=k)

    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\nClass {query_label}", color="blue")
    axes[0].axis("off")

    for position in range(k):
        gallery_index = int(topk_indices[position].item())
        score = float(topk_scores[position].item())
        retrieved_img, retrieved_label = gallery_dataset[gallery_index]
        color = "green" if retrieved_label == query_label else "red"

        axes[position + 1].imshow(retrieved_img)
        axes[position + 1].set_title(
            f"Top {position + 1}\nClass {retrieved_label}\nSim {score:.2f}",
            color=color,
        )
        axes[position + 1].axis("off")

    fig.tight_layout()
    return fig, axes
