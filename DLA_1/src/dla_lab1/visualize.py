from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(labels, title: str = "Class distribution"):
    """
    Serve a disegnare la distribuzione delle classi.

    Nel primo notebook lo usiamo per rendere visibile lo sbilanciamento.
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
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    return fig, ax
