from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Serve a leggere il file YAML di configurazione.

    Lo usiamo per non scrivere path, batch size, seed e iperparametri
    direttamente nelle celle del notebook.

    Args:
        path: Percorso del file `config.yaml`.

    Returns:
        Dizionario Python con tutte le sezioni della configurazione.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return data


def merge_dicts(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """
    Serve a fondere due dizionari di configurazione.

    E' utile quando un esperimento modifica solo alcuni valori rispetto
    alla configurazione generale.

    Args:
        base: Configurazione di partenza.
        override: Valori da sovrascrivere o aggiungere.

    Returns:
        Nuovo dizionario ottenuto applicando `override` a `base` in modo ricorsivo.
    """
    result = deepcopy(base)
    if not override:
        return result

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def experiment_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    """
    Serve a costruire la configurazione completa di un esperimento.

    Prende i valori globali e li aggiorna con quelli specifici
    dell'esperimento scelto.

    Args:
        config: Configurazione generale caricata da YAML.
        name: Nome dell'esperimento nella sezione `experiments`.

    Returns:
        Configurazione completa della run, con model/training gia' risolti.
    """
    experiments = config.get("experiments", {})
    if name not in experiments:
        available = ", ".join(sorted(experiments)) or "<none>"
        raise KeyError(f"Unknown experiment {name!r}. Available: {available}")

    exp = experiments[name]
    merged = deepcopy(config)
    merged["experiment_name"] = name
    merged["model"] = merge_dicts(config.get("model", {}), exp.get("model", {}))
    merged["training"] = merge_dicts(config.get("training", {}), exp.get("training", {}))
    merged["experiment"] = exp
    return merged
