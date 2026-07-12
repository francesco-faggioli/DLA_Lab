from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"


@dataclass(frozen=True)
class ClipEvalResult:
    """Risultato sintetico di una valutazione CLIP."""

    method: str
    accuracy: float
    gain_vs_baseline: float | None = None


class CLIPAdapter(nn.Module):
    """
    Adapter leggero applicato alle feature immagine di CLIP.

    Args:
        feat_dim: Dimensione delle feature CLIP.
        bottleneck: Dimensione nascosta dell'MLP.
        alpha: Peso iniziale del ramo residuale.

    Returns:
        Modulo PyTorch che restituisce feature adattate.
    """

    def __init__(self, feat_dim: int = 512, bottleneck: int = 64, alpha: float = 0.6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, feat_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.fc1(x))
        residual = self.fc2(residual)
        return x + torch.sigmoid(self.alpha) * residual


def get_device() -> str:
    """
    Seleziona automaticamente CUDA quando disponibile.

    Args:
        Nessun argomento.

    Returns:
        Stringa `cuda` oppure `cpu`.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_open_clip_model(model_name: str = "ViT-B-16-quickgelu", pretrained: str = "openai", device: str | None = None) -> tuple[Any, Any, Any, str]:
    """
    Carica un modello CLIP da open_clip.

    Args:
        model_name: Nome architettura open_clip.
        pretrained: Checkpoint da usare.
        device: Device opzionale. Se None viene scelto automaticamente.

    Returns:
        Tupla `(model, preprocess, tokenizer, device)`.
    """
    import open_clip

    device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer, device


def load_imagenet_labels(path: str | Path = "imagenet_labels.json", download_if_missing: bool = False) -> list[str]:
    """
    Carica i nomi semplici delle 1000 classi ImageNet.

    Args:
        path: Percorso del file JSON locale.
        download_if_missing: Se True scarica il file quando non esiste.

    Returns:
        Lista di 1000 nomi classe.
    """
    label_path = Path(path)
    if not label_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Missing labels file: {label_path}")
        urllib.request.urlretrieve(IMAGENET_LABELS_URL, label_path)

    with label_path.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list) or len(labels) != 1000:
        raise ValueError("Expected a JSON list with 1000 ImageNet labels.")
    return labels


def load_imagenet_sketch(seed: int = 42, train_fraction: float = 0.8) -> tuple[Any, Any]:
    """
    Carica ImageNet-Sketch e crea split train/validation deterministici.

    Args:
        seed: Seed per lo shuffle.
        train_fraction: Frazione del dataset usata per il train.

    Returns:
        Coppia `(sketch_train, sketch_val)` di Dataset Hugging Face.
    """
    from datasets import load_dataset

    full = load_dataset("imagenet_sketch", split="train", trust_remote_code=True)
    full = full.shuffle(seed=seed)
    train_size = int(train_fraction * len(full))
    return full.select(range(train_size)), full.select(range(train_size, len(full)))


def build_clip_tensor_dataset(hf_dataset: Any, preprocess: Any, num_samples: int | None = None) -> TensorDataset:
    """
    Preprocessa immagini HF in tensori CLIP e conserva le label.

    Args:
        hf_dataset: Dataset con colonne `image` e `label`.
        preprocess: Trasformazione immagine restituita da open_clip.
        num_samples: Numero massimo di esempi da usare. Se None usa tutto.

    Returns:
        TensorDataset `(pixel_values, labels)`.
    """
    n = min(num_samples or len(hf_dataset), len(hf_dataset))
    images: list[torch.Tensor] = []
    labels: list[int] = []

    for idx in tqdm(range(n), desc="Preprocessing images"):
        example = hf_dataset[idx]
        image = example["image"].convert("RGB")
        images.append(preprocess(image))
        labels.append(int(example["label"]))

    return TensorDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long))


@torch.no_grad()
def build_text_features(model: Any, tokenizer: Any, class_names: list[str], prompt_template: str, device: str, batch_size: int = 100) -> torch.Tensor:
    """
    Costruisce le feature testuali normalizzate per tutte le classi.

    Args:
        model: Modello CLIP.
        tokenizer: Tokenizer open_clip.
        class_names: Lista di nomi classe.
        prompt_template: Template con `{}` per inserire il nome classe.
        device: Device PyTorch.
        batch_size: Numero di prompt processati per batch.

    Returns:
        Tensore `(n_classi, feature_dim)` normalizzato.
    """
    model.eval()
    chunks: list[torch.Tensor] = []

    for start in range(0, len(class_names), batch_size):
        end = min(start + batch_size, len(class_names))
        prompts = [prompt_template.format(name) for name in class_names[start:end]]
        tokens = tokenizer(prompts).to(device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        chunks.append(features.cpu())

    return torch.cat(chunks, dim=0).to(device)


@torch.no_grad()
def build_text_features_ensemble(model: Any, tokenizer: Any, class_names: list[str], prompt_templates: list[str], device: str) -> torch.Tensor:
    """
    Media feature testuali ottenute da piu' prompt.

    Args:
        model: Modello CLIP.
        tokenizer: Tokenizer open_clip.
        class_names: Lista di nomi classe.
        prompt_templates: Lista di template prompt.
        device: Device PyTorch.

    Returns:
        Tensore normalizzato `(n_classi, feature_dim)`.
    """
    all_features = [
        build_text_features(model, tokenizer, class_names, template, device).cpu()
        for template in prompt_templates
    ]
    ensemble = torch.stack(all_features, dim=0).mean(dim=0)
    ensemble = ensemble / ensemble.norm(dim=-1, keepdim=True)
    return ensemble.to(device)


@torch.no_grad()
def evaluate_zero_shot(model: Any, loader: DataLoader, text_features: torch.Tensor, device: str) -> float:
    """
    Valuta CLIP in zero-shot classification.

    Args:
        model: Modello CLIP.
        loader: DataLoader che produce `(pixel_values, labels)`.
        text_features: Feature testuali normalizzate delle classi.
        device: Device PyTorch.

    Returns:
        Accuracy come float tra 0 e 1.
    """
    model.eval()
    correct = 0
    total = 0
    scale = model.logit_scale.exp()

    for pixel_values, labels in tqdm(loader, desc="Zero-shot eval", leave=False):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = scale * image_features @ text_features.T
        predictions = logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return correct / total


@torch.no_grad()
def precompute_image_features(model: Any, loader: DataLoader, device: str) -> TensorDataset:
    """
    Precalcola le feature immagine CLIP per addestrare adapter leggeri.

    Args:
        model: Modello CLIP congelato.
        loader: DataLoader che produce `(pixel_values, labels)`.
        device: Device PyTorch.

    Returns:
        TensorDataset `(features, labels)` su CPU.
    """
    model.eval()
    features: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    for pixel_values, labels in tqdm(loader, desc="Image features", leave=False):
        pixel_values = pixel_values.to(device)
        image_features = model.encode_image(pixel_values)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features.append(image_features.cpu())
        labels_all.append(labels.cpu())

    return TensorDataset(torch.cat(features, dim=0), torch.cat(labels_all, dim=0))


@torch.no_grad()
def evaluate_precomputed_features(loader: DataLoader, text_features: torch.Tensor, logit_scale: float, device: str) -> float:
    """
    Valuta una classificazione zero-shot usando feature immagine gia' calcolate.

    Args:
        loader: DataLoader di feature `(features, labels)`.
        text_features: Feature testuali normalizzate delle classi.
        logit_scale: Scala dei logit del modello CLIP.
        device: Device PyTorch.

    Returns:
        Accuracy come float tra 0 e 1.

    Notes:
        Questa funzione evita di rieseguire `model.encode_image()` quando si
        confrontano piu' prompt sugli stessi esempi. E' la scelta piu' efficiente
        per prompt study e confronto finale quando il backbone CLIP resta
        congelato.
    """
    correct = 0
    total = 0
    text_features = text_features.to(device)

    for features, labels in tqdm(loader, desc="Feature eval", leave=False):
        features = features.to(device)
        labels = labels.to(device)
        logits = logit_scale * features @ text_features.T
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return correct / total


def split_tensor_dataset(dataset: TensorDataset, train_fraction: float = 0.9, seed: int = 42) -> tuple[TensorDataset, TensorDataset]:
    """
    Divide un TensorDataset in train e validation.

    Args:
        dataset: Dataset da dividere.
        train_fraction: Frazione da assegnare al train.
        seed: Seed per la divisione.

    Returns:
        Coppia `(train_dataset, val_dataset)`.
    """
    n_train = int(train_fraction * len(dataset))
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_idx = torch.tensor(indices[:n_train], dtype=torch.long)
    val_idx = torch.tensor(indices[n_train:], dtype=torch.long)
    tensors = dataset.tensors
    train_tensors = tuple(tensor[train_idx] for tensor in tensors)
    val_tensors = tuple(tensor[val_idx] for tensor in tensors)
    return TensorDataset(*train_tensors), TensorDataset(*val_tensors)


def train_clip_adapter(
    adapter: CLIPAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    text_features: torch.Tensor,
    logit_scale: float,
    device: str,
    epochs: int = 30,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
) -> list[dict[str, float]]:
    """
    Addestra un CLIPAdapter su feature immagine gia' precalcolate.

    Args:
        adapter: Modulo CLIPAdapter.
        train_loader: DataLoader di feature train `(features, labels)`.
        val_loader: DataLoader di feature validation `(features, labels)`.
        text_features: Feature testuali delle classi.
        logit_scale: Scala dei logit del modello CLIP.
        device: Device PyTorch.
        epochs: Numero di epoche.
        lr: Learning rate.
        weight_decay: Regolarizzazione AdamW.

    Returns:
        Lista di dizionari con loss train e validation per epoca.
    """
    adapter.to(device)
    text_features = text_features.to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        adapter.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(features)
            adapted = adapted / adapted.norm(dim=-1, keepdim=True)
            logits = logit_scale * adapted @ text_features.T
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        adapter.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                adapted = adapter(features)
                adapted = adapted / adapted.norm(dim=-1, keepdim=True)
                logits = logit_scale * adapted @ text_features.T
                val_loss += F.cross_entropy(logits, labels).item()

        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss / max(1, len(train_loader)),
                "val_loss": val_loss / max(1, len(val_loader)),
            }
        )
    return history


@torch.no_grad()
def evaluate_adapter(adapter: CLIPAdapter, loader: DataLoader, text_features: torch.Tensor, logit_scale: float, device: str) -> float:
    """
    Valuta un adapter CLIP su feature immagine precalcolate.

    Args:
        adapter: Adapter addestrato.
        loader: DataLoader di feature `(features, labels)`.
        text_features: Feature testuali normalizzate.
        logit_scale: Scala dei logit CLIP.
        device: Device PyTorch.

    Returns:
        Accuracy come float tra 0 e 1.
    """
    adapter.eval()
    correct = 0
    total = 0

    for features, labels in tqdm(loader, desc="Adapter eval", leave=False):
        features = features.to(device)
        labels = labels.to(device)
        adapted = adapter(features)
        adapted = adapted / adapted.norm(dim=-1, keepdim=True)
        logits = logit_scale * adapted @ text_features.to(device).T
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return correct / total
