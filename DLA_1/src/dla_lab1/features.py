from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from .paths import ensure_dir


def extract_features(model, dataloader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Serve a estrarre le feature da una CNN pre-addestrata.

    Passa ogni batch nella rete in modalita' inferenza e salva su CPU
    sia i vettori di feature sia le label corrispondenti.

    Args:
        model: Modello PyTorch usato come estrattore di feature.
        dataloader: DataLoader che restituisce immagini e label.
        device: Device su cui eseguire l'inferenza.

    Returns:
        Tupla `(features, labels)` su CPU, con una riga di feature per immagine.
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

    Args:
        path: File `.pt` in cui salvare la cache.
        **tensors: Tensori da salvare, ad esempio feature e label.

    Returns:
        Path del file salvato.
    """
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(tensors, path)
    return path


def load_feature_cache(path: str | Path) -> dict[str, torch.Tensor]:
    """
    Serve a ricaricare le feature salvate in precedenza.

    E' la strada piu' veloce per rifare solo la parte SVM della baseline.

    Args:
        path: File `.pt` generato da `save_feature_cache`.

    Returns:
        Dizionario con i tensori salvati nella cache.
    """
    return torch.load(path, map_location="cpu")


def cosine_similarity_matrix(query_features: torch.Tensor, gallery_features: torch.Tensor) -> torch.Tensor:
    """
    Serve a calcolare similarita' coseno tra query e gallery.

    Sara' utile nella parte retrieval, non nella baseline SVM.

    Args:
        query_features: Feature delle immagini query.
        gallery_features: Feature delle immagini nella gallery.

    Returns:
        Matrice `[num_query, num_gallery]` con similarita' coseno.
    """
    query = F.normalize(query_features.float(), dim=1)
    gallery = F.normalize(gallery_features.float(), dim=1)
    return query @ gallery.T


def retrieval_precision_at_k(
    sim_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """
    Serve a valutare il ranking retrieval con Precision@K.

    Per ogni query prende le K immagini piu' simili nella gallery e misura
    quante hanno la stessa classe della query. E' la metrica piu' semplice
    per controllare se il retrieval sta recuperando esempi corretti.

    Args:
        sim_matrix: Matrice di similarita' query-gallery.
        query_labels: Label vere delle query.
        gallery_labels: Label vere della gallery.
        k_values: Valori di K da valutare.

    Returns:
        Dizionario `{K: precisione_media}`.
    """
    results = {}
    for k in k_values:
        _, topk_indices = torch.topk(sim_matrix, k=k, dim=1)
        topk_labels = gallery_labels[topk_indices]
        expected = query_labels.unsqueeze(1).expand_as(topk_labels)
        precision = (topk_labels == expected).float().mean(dim=1)
        results[k] = float(precision.mean().item())
    return results


def compare_similarity_precision_at_k(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k_values: tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 256,
) -> dict[str, dict[int, float]]:
    """
    Serve a confrontare Dot Product, Cosine Similarity ed Euclidean Distance.

    Calcola Precision@K direttamente dalle feature, ma lavora a blocchi sulle
    query per non creare tre matrici enormi in memoria. Nel notebook 3.2 lo
    usiamo per calcolare le metriche correnti senza tabelle precompilate.

    Args:
        query_features: Feature delle immagini query.
        gallery_features: Feature delle immagini gallery.
        query_labels: Label vere delle query.
        gallery_labels: Label vere della gallery.
        k_values: Valori di K da valutare.
        chunk_size: Numero di query elaborate per blocco.

    Returns:
        Dizionario annidato `{metrica: {K: precisione_media}}`.
    """
    max_k = max(k_values)
    query_float = query_features.float()
    gallery_float = gallery_features.float()
    query_norm = F.normalize(query_float, dim=1)
    gallery_norm = F.normalize(gallery_float, dim=1)

    correct_sums = {
        "Dot product": {k: 0.0 for k in k_values},
        "Cosine similarity": {k: 0.0 for k in k_values},
        "Euclidean distance": {k: 0.0 for k in k_values},
    }
    num_queries = query_float.shape[0]

    for start in tqdm(range(0, num_queries, chunk_size), desc="Similarity comparison", leave=False):
        end = min(start + chunk_size, num_queries)
        labels = query_labels[start:end]
        scores_by_metric = {
            "Dot product": query_float[start:end] @ gallery_float.T,
            "Cosine similarity": query_norm[start:end] @ gallery_norm.T,
            "Euclidean distance": -torch.cdist(query_float[start:end], gallery_float, p=2),
        }

        for metric_name, scores in scores_by_metric.items():
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            topk_labels = gallery_labels[topk_indices]
            expected = labels.unsqueeze(1).expand_as(topk_labels)
            matches = (topk_labels == expected).float()
            for k in k_values:
                precision = matches[:, :k].mean(dim=1)
                correct_sums[metric_name][k] += float(precision.sum().item())

    return {
        metric_name: {k: value / num_queries for k, value in values.items()}
        for metric_name, values in correct_sums.items()
    }


def retrieval_mean_average_precision(
    sim_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    num_classes: int = 43,
) -> dict[str, object]:
    """
    Serve a calcolare la mean Average Precision del retrieval.

    Per ogni query considera rilevanti le immagini della gallery con la stessa
    classe. Restituisce la mAP globale, calcolata come media sulle query, e la
    AP media per classe, utile per capire quali segnali stradali sono
    rappresentati meglio dal backbone.

    Args:
        sim_matrix: Matrice di similarita' query-gallery.
        query_labels: Label vere delle query.
        gallery_labels: Label vere della gallery.
        num_classes: Numero di classi GTSRB.

    Returns:
        Dizionario con `mAP`, `macro_mAP_by_class` e AP media per classe.
    """
    sim_np = sim_matrix.detach().cpu().numpy()
    query_np = query_labels.detach().cpu().numpy()
    gallery_np = gallery_labels.detach().cpu().numpy()

    all_aps: list[float] = []
    ap_by_class: dict[int, list[float]] = {class_id: [] for class_id in range(num_classes)}
    for row_idx, class_id in enumerate(query_np):
        relevant = (gallery_np == class_id).astype(int)
        ap = float(average_precision_score(relevant, sim_np[row_idx]))
        all_aps.append(ap)
        ap_by_class[int(class_id)].append(ap)

    per_class_ap = {
        class_id: float(np.mean(values)) if values else float("nan")
        for class_id, values in ap_by_class.items()
    }
    valid_scores = [score for score in per_class_ap.values() if not np.isnan(score)]
    return {
        "mAP": float(np.mean(all_aps)),
        "macro_mAP_by_class": float(np.mean(valid_scores)),
        "per_class_ap": per_class_ap,
    }


def class_feature_centroids(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 43,
) -> torch.Tensor:
    """
    Serve a calcolare il vettore medio di feature per ogni classe.

    Questo e' il passaggio di "training" del Nearest-Mean Classifier: non usa
    gradienti, ma riassume ogni classe con il centroide delle sue immagini.

    Args:
        features: Feature delle immagini di training/gallery.
        labels: Label associate alle feature.
        num_classes: Numero di classi da rappresentare.

    Returns:
        Tensore con un centroide medio per ogni classe.
    """
    centroids = []
    for class_id in range(num_classes):
        class_features = features[labels == class_id].float()
        centroids.append(class_features.mean(dim=0))
    return torch.stack(centroids)


def nearest_mean_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
) -> torch.Tensor:
    """
    Serve a classificare usando il centro medio delle feature per classe.

    E' una baseline training-free per la parte retrieval/NMC.

    Args:
        train_features: Feature usate per calcolare i centroidi di classe.
        train_labels: Label delle feature di training/gallery.
        query_features: Feature delle immagini da classificare.

    Returns:
        Tensore con la classe predetta per ogni query.
    """
    classes = torch.unique(train_labels).sort().values
    centroids = class_feature_centroids(train_features, train_labels, num_classes=len(classes))
    sim = cosine_similarity_matrix(query_features, centroids)
    return classes[sim.argmax(dim=1)]
