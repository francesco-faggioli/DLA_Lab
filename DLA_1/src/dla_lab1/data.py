from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB

from .transforms import build_transforms


def load_gtsrb(data_root: str | Path, split: str, transform=None, download: bool = False) -> GTSRB:
    """
    Serve a caricare uno split del dataset GTSRB con torchvision.

    Nel notebook lo usiamo sia senza trasformazioni, per fare EDA sulle
    immagini grezze, sia con trasformazioni per preparare le immagini
    all'estrazione delle feature.

    Args:
        data_root: Cartella radice in cui torchvision trova o salva GTSRB.
        split: Split da caricare, `train` oppure `test`.
        transform: Trasformazioni opzionali da applicare alle immagini.
        download: Se True, prova a scaricare il dataset tramite torchvision.

    Returns:
        Dataset torchvision `GTSRB` pronto per essere usato in un DataLoader.
    """
    return GTSRB(root=str(data_root), split=split, transform=transform, download=download)


def gtsrb_base_dir(data_root: str | Path) -> Path:
    """
    Serve a trovare la cartella reale dei file GTSRB locali.

    Torchvision usa come root `data`, ma i CSV e le cartelle estratte sono
    dentro `data/gtsrb`. Questa funzione gestisce entrambi i casi.

    Args:
        data_root: Cartella `data` o direttamente cartella `data/gtsrb`.

    Returns:
        Path della cartella che contiene `GTSRB/Training` e i CSV ufficiali.
    """
    root = Path(data_root)
    candidates = [root, root / "gtsrb"]
    for candidate in candidates:
        if (candidate / "GTSRB" / "Training").exists():
            return candidate
    checked = ", ".join(str(candidate / "GTSRB" / "Training") for candidate in candidates)
    raise FileNotFoundError(
        "Could not find local GTSRB metadata. "
        f"Checked: {checked}. "
        "If you edited src files while the notebook kernel was already running, restart the kernel."
    )


def read_training_metadata(data_root: str | Path) -> pd.DataFrame:
    """
    Serve a leggere i CSV del training set GTSRB in un unico DataFrame.

    Lo usiamo per calcolare dimensioni delle immagini, numero di classi
    e distribuzione delle etichette senza aprire manualmente ogni immagine.

    Args:
        data_root: Cartella radice del dataset.

    Returns:
        DataFrame con le righe dei CSV di training e la classe di appartenenza.
    """
    train_dir = gtsrb_base_dir(data_root) / "GTSRB" / "Training"
    rows: list[dict] = []
    for class_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        csv_path = class_dir / f"GT-{class_dir.name}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, sep=";")
        frame["ClassDir"] = class_dir.name
        rows.extend(frame.to_dict("records"))
    return pd.DataFrame(rows)


def read_test_metadata(data_root: str | Path) -> pd.DataFrame:
    """
    Serve a leggere il CSV del test set ufficiale.

    Lo usiamo per verificare il numero di immagini e per documentare
    la struttura del dataset.

    Args:
        data_root: Cartella radice del dataset.

    Returns:
        DataFrame letto da `GT-final_test.csv`.
    """
    return pd.read_csv(gtsrb_base_dir(data_root) / "GT-final_test.csv", sep=";")


def dataset_targets(dataset) -> np.ndarray:
    """
    Serve a estrarre le label da un dataset torchvision.

    Lo usiamo nelle parti di training/split, per costruire train e validation
    mantenendo la distribuzione delle classi sotto controllo.

    Args:
        dataset: Dataset torchvision o compatibile che restituisce coppie immagine/label.

    Returns:
        Array NumPy con una label intera per ogni immagine del dataset.
    """
    if hasattr(dataset, "_samples"):
        return np.array([int(sample[1]) for sample in dataset._samples])
    if hasattr(dataset, "samples"):
        return np.array([int(sample[1]) for sample in dataset.samples])
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets, dtype=int)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))], dtype=int)


def stratified_track_split(
    labels: Sequence[int],
    val_split: float = 0.20,
    track_size: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Serve a dividere il training set in train e validation.

    GTSRB contiene sequenze brevi di immagini simili tra loro. Per evitare
    che immagini quasi identiche finiscano sia in train sia in validation,
    dividiamo gruppi consecutivi della stessa classe invece dei singoli file.

    Args:
        labels: Label del training set nello stesso ordine del dataset.
        val_split: Percentuale approssimativa da assegnare alla validation.
        track_size: Numero di immagini consecutive trattate come gruppo.
        seed: Seed usato per rendere lo split riproducibile.

    Returns:
        Due array di indici: training e validation.
    """
    labels = np.asarray(labels, dtype=int)
    groups_by_class: dict[int, list[np.ndarray]] = defaultdict(list)

    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        for start in range(0, len(indices), track_size):
            groups_by_class[int(class_id)].append(indices[start : start + track_size])

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_id, groups in groups_by_class.items():
        group_ids = np.arange(len(groups))
        if len(group_ids) <= 1:
            train_parts.extend(groups)
            continue
        train_ids, val_ids = train_test_split(
            group_ids,
            test_size=val_split,
            random_state=seed + int(class_id),
            shuffle=True,
        )
        train_parts.extend(groups[i] for i in train_ids)
        val_parts.extend(groups[i] for i in val_ids)

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(int), val_idx.astype(int)


def split_class_summary(
    labels: Sequence[int],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    num_classes: int,
) -> pd.DataFrame:
    """
    Serve a controllare quante immagini di ogni classe finiscono nello split.

    Nel notebook di fine-tuning lo usiamo per verificare esplicitamente che
    anche la validation contenga almeno un esempio per ciascuna classe GTSRB.

    Args:
        labels: Label complete del dataset di training.
        train_idx: Indici assegnati al training split.
        val_idx: Indici assegnati alla validation split.
        num_classes: Numero totale di classi attese.

    Returns:
        DataFrame con conteggi train/validation e rapporto di validation per classe.
    """
    labels = np.asarray(labels, dtype=int)
    train_labels = labels[np.asarray(train_idx, dtype=int)]
    val_labels = labels[np.asarray(val_idx, dtype=int)]

    train_counts = np.bincount(train_labels, minlength=num_classes)
    val_counts = np.bincount(val_labels, minlength=num_classes)
    summary = pd.DataFrame(
        {
            "class_id": np.arange(num_classes),
            "train_count": train_counts,
            "val_count": val_counts,
            "total_count": train_counts + val_counts,
        }
    )
    summary["val_ratio"] = summary["val_count"] / summary["total_count"]
    return summary


def missing_classes_in_split(split_summary: pd.DataFrame, column: str = "val_count") -> list[int]:
    """
    Serve a trovare eventuali classi assenti da train o validation.

    E' una piccola funzione di controllo: se la lista restituita e' vuota,
    significa che lo split contiene almeno un esempio per ogni classe.

    Args:
        split_summary: DataFrame prodotto da `split_class_summary`.
        column: Colonna da controllare, ad esempio `val_count` o `train_count`.

    Returns:
        Lista degli ID classe con conteggio pari a zero nella colonna scelta.
    """
    missing = split_summary.loc[split_summary[column] == 0, "class_id"]
    return missing.astype(int).tolist()


def class_weights_from_labels(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """
    Serve a calcolare pesi inversamente proporzionali alla frequenza delle classi.

    Lo useremo nelle parti successive se vogliamo compensare lo sbilanciamento
    del dataset con WeightedCrossEntropy o FocalLoss.

    Args:
        labels: Label usate per stimare la frequenza delle classi.
        num_classes: Numero totale di classi.

    Returns:
        Tensore PyTorch con un peso per classe.
    """
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    val_split: float,
    track_size: int,
    seed: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    augmentation: str = "none",
) -> dict[str, DataLoader | np.ndarray | torch.Tensor]:
    """
    Serve a creare DataLoader train, validation e test.

    In questo primo notebook non e' indispensabile, ma sara' utile per la parte
    di fine-tuning per non riscrivere sempre split e preprocessing.

    Args:
        data_root: Cartella radice del dataset.
        image_size: Dimensione finale delle immagini.
        batch_size: Numero di immagini per batch.
        val_split: Percentuale di training set usata per validation.
        track_size: Dimensione dei gruppi consecutivi usati nello split anti-leakage.
        seed: Seed per rendere lo split riproducibile.
        num_workers: Processi usati dal DataLoader per caricare immagini.
        pin_memory: Se True, ottimizza il trasferimento CPU-GPU con CUDA.
        augmentation: Tipo di augmentation da applicare solo al training set.

    Returns:
        Dizionario con DataLoader train/val/test, indici dello split, summary e pesi classe.
    """
    train_transform = build_transforms(image_size=image_size, train=True, augmentation=augmentation)
    eval_transform = build_transforms(image_size=image_size, train=False)

    train_full = load_gtsrb(data_root, split="train", transform=train_transform)
    val_full = load_gtsrb(data_root, split="train", transform=eval_transform)
    test_set = load_gtsrb(data_root, split="test", transform=eval_transform)

    labels = dataset_targets(train_full)
    train_idx, val_idx = stratified_track_split(labels, val_split, track_size, seed)
    split_summary = split_class_summary(labels, train_idx, val_idx, num_classes=43)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2

    train_loader = DataLoader(Subset(train_full, train_idx), shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(Subset(val_full, val_idx), shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "split_summary": split_summary,
        "class_weights": class_weights_from_labels(labels[train_idx], num_classes=43),
    }


def build_retrieval_dataloaders(
    data_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """
    Serve a creare i DataLoader per Exercise 3.2.

    In retrieval usiamo tutto il training set come gallery e tutto il test set
    come query. Non facciamo augmentation, perche' non stiamo allenando il
    modello: vogliamo solo estrarre feature stabili da immagini preprocessate
    come richiesto dai backbone ImageNet.

    Args:
        data_root: Cartella radice del dataset.
        image_size: Dimensione finale delle immagini.
        batch_size: Numero di immagini per batch.
        num_workers: Processi usati dal DataLoader.
        pin_memory: Se True, ottimizza il trasferimento CPU-GPU con CUDA.

    Returns:
        Dizionario con DataLoader `gallery` e `query`.
    """
    transform = build_transforms(image_size=image_size, train=False)

    gallery_set = load_gtsrb(data_root, split="train", transform=transform)
    query_set = load_gtsrb(data_root, split="test", transform=transform)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
        "drop_last": False,
    }
    if num_workers > 0:
        common["persistent_workers"] = True
        common["prefetch_factor"] = 2

    return {
        "gallery": DataLoader(gallery_set, **common),
        "query": DataLoader(query_set, **common),
    }
