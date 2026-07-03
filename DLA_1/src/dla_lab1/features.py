from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .paths import ensure_dir


def extract_features(model, dataloader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Serve a estrarre le feature da una CNN pre-addestrata.

    Passa ogni batch nella rete in modalita' inferenza e salva su CPU
    sia i vettori di feature sia le label corrispondenti.
    """
    model = model.to(device)
    model.eval()

    features = []
    labels = []
    with torch.inference_mode():
        for inputs, target in tqdm(dataloader, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            features.append(model(inputs).detach().cpu())
            labels.append(target.detach().cpu())
    return torch.cat(features), torch.cat(labels)


def save_feature_cache(path: str | Path, **tensors: torch.Tensor) -> Path:
    """
    Serve a salvare su disco feature e label gia' calcolate.

    In questo modo non dobbiamo rieseguire ResNet-18 ogni volta che apriamo
    il notebook.
    """
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(tensors, path)
    return path


def load_feature_cache(path: str | Path) -> dict[str, torch.Tensor]:
    """
    Serve a ricaricare le feature salvate in precedenza.

    E' la strada piu' veloce per rifare solo la parte SVM della baseline.
    """
    return torch.load(path, map_location="cpu")


def cosine_similarity_matrix(query_features: torch.Tensor, gallery_features: torch.Tensor) -> torch.Tensor:
    """
    Serve a calcolare similarita' coseno tra query e gallery.

    Sara' utile nella parte retrieval, non nella baseline SVM.
    """
    query = F.normalize(query_features.float(), dim=1)
    gallery = F.normalize(gallery_features.float(), dim=1)
    return query @ gallery.T


def nearest_mean_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
) -> torch.Tensor:
    """
    Serve a classificare usando il centro medio delle feature per classe.

    E' una baseline training-free per la parte retrieval/NMC.
    """
    classes = torch.unique(train_labels).sort().values
    centroids = []
    for class_id in classes:
        centroids.append(train_features[train_labels == class_id].float().mean(dim=0))
    centroids = torch.stack(centroids)
    sim = cosine_similarity_matrix(query_features, centroids)
    return classes[sim.argmax(dim=1)]
