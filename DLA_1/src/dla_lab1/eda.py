from __future__ import annotations

import pandas as pd


def metadata_summary(metadata: pd.DataFrame) -> dict:
    """
    Serve a riassumere il dataset con pochi numeri chiave.

    Restituisce numero di campioni, numero di classi, classi min/max
    e dimensioni min/max/medie delle immagini.

    Args:
        metadata: DataFrame ottenuto dai CSV GTSRB del training set.

    Returns:
        Dizionario con conteggi delle classi e statistiche sulle dimensioni immagini.
    """
    class_counts = metadata["ClassId"].value_counts().sort_index()
    return {
        "num_samples": int(len(metadata)),
        "num_classes": int(class_counts.shape[0]),
        "min_class_count": int(class_counts.min()),
        "max_class_count": int(class_counts.max()),
        "width_min": int(metadata["Width"].min()),
        "width_max": int(metadata["Width"].max()),
        "width_mean": float(metadata["Width"].mean()),
        "height_min": int(metadata["Height"].min()),
        "height_max": int(metadata["Height"].max()),
        "height_mean": float(metadata["Height"].mean()),
    }


def class_distribution(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Serve a ottenere il numero di immagini per ogni classe.

    Lo usiamo per mostrare e commentare lo sbilanciamento del dataset.

    Args:
        metadata: DataFrame con almeno la colonna `ClassId`.

    Returns:
        DataFrame con colonne `class_id` e `count`, una riga per classe.
    """
    counts = metadata["ClassId"].value_counts().sort_index()
    return counts.rename_axis("class_id").reset_index(name="count")
