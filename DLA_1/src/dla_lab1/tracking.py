from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paths import ensure_dir, resolve_path


def experiment_output_dir(config: dict[str, Any], experiment_name: str, root: str | Path = ".") -> Path:
    """
    Serve a creare la cartella locale dedicata a una run.

    Nel notebook dell'Exercise 2 la usiamo per salvare history, configurazione
    e summary senza dipendere obbligatoriamente da WandB.

    Args:
        config: Configurazione dell'esperimento.
        experiment_name: Nome della run.
        root: Cartella radice per risolvere `artifacts_dir`.

    Returns:
        Path della cartella artifact locale della run.
    """
    artifacts_dir = resolve_path(config["paths"]["artifacts_dir"], root)
    return ensure_dir(artifacts_dir / "runs" / experiment_name)


def history_to_records(history) -> list[dict[str, Any]]:
    """
    Serve a convertire la history del training in una lista di dizionari.

    Accetta sia oggetti `EpochMetrics` sia dizionari, cosi' il salvataggio
    resta semplice anche se cambiamo leggermente il formato interno.

    Args:
        history: Lista di metriche per epoca.

    Returns:
        Lista di dizionari serializzabili.
    """
    records: list[dict[str, Any]] = []
    for row in history:
        if is_dataclass(row):
            records.append(asdict(row))
        elif isinstance(row, dict):
            records.append(dict(row))
        else:
            records.append(dict(row))
    return records


def history_to_dataframe(history) -> pd.DataFrame:
    """
    Serve a trasformare la history in un DataFrame Pandas.

    Il DataFrame e' comodo sia per visualizzare i risultati nel notebook sia
    per salvarli in CSV.

    Args:
        history: Lista di metriche per epoca.

    Returns:
        DataFrame Pandas con una riga per epoca.
    """
    return pd.DataFrame(history_to_records(history))


def summarize_history(history) -> dict[str, Any]:
    """
    Serve a riassumere una run in poche metriche confrontabili.

    Estrae l'epoca migliore secondo `val_acc` e conserva anche l'ultima epoca
    disponibile, utile per capire se il modello stava ancora migliorando.

    Args:
        history: Lista di metriche prodotte dal training.

    Returns:
        Dizionario con migliori metriche di validation e metriche finali.
    """
    frame = history_to_dataframe(history)
    if frame.empty:
        return {}

    best_idx = frame["val_acc"].idxmax()
    best = frame.loc[best_idx]
    last = frame.iloc[-1]
    return {
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "best_val_loss": float(best["val_loss"]),
        "last_epoch": int(last["epoch"]),
        "last_train_acc": float(last["train_acc"]),
        "last_val_acc": float(last["val_acc"]),
    }


def save_config_snapshot(config: dict[str, Any], output_dir: str | Path) -> Path:
    """
    Serve a salvare la configurazione effettivamente usata da una run.

    Questo rende l'esperimento riproducibile: oltre alle metriche, conserviamo
    anche iperparametri, path e impostazioni hardware.

    Args:
        config: Configurazione effettiva della run.
        output_dir: Cartella in cui salvare il file YAML.

    Returns:
        Path del file `config_used.yaml`.
    """
    path = Path(output_dir) / "config_used.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def save_history_csv(history, output_dir: str | Path) -> Path:
    """
    Serve a salvare loss, accuracy e learning rate di ogni epoca.

    Il CSV permette di ricaricare e confrontare le run senza rieseguire
    il training.

    Args:
        history: Lista di metriche per epoca.
        output_dir: Cartella in cui salvare il CSV.

    Returns:
        Path del file `history.csv`.
    """
    path = Path(output_dir) / "history.csv"
    history_to_dataframe(history).to_csv(path, index=False)
    return path


def save_summary_json(summary: dict[str, Any], output_dir: str | Path) -> Path:
    """
    Serve a salvare il riassunto finale della run.

    Il file JSON contiene poche metriche chiave ed e' utile per costruire
    tabelle comparative tra esperimenti.

    Args:
        summary: Dizionario di metriche riassuntive.
        output_dir: Cartella in cui salvare il JSON.

    Returns:
        Path del file `summary.json`.
    """
    path = Path(output_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def save_run_artifacts(
    config: dict[str, Any],
    experiment_name: str,
    history,
    root: str | Path = ".",
) -> dict[str, Any]:
    """
    Serve a salvare tutti gli artifact locali principali di una run.

    Restituisce percorsi e metriche riassuntive, cosi' il notebook puo'
    mostrarli in modo esplicito.

    Args:
        config: Configurazione usata dalla run.
        experiment_name: Nome dell'esperimento.
        history: Metriche per epoca.
        root: Cartella radice del progetto.

    Returns:
        Dizionario con path degli artifact locali e summary della run.
    """
    output_dir = experiment_output_dir(config, experiment_name, root=root)
    summary = summarize_history(history)
    return {
        "output_dir": output_dir,
        "history_path": save_history_csv(history, output_dir),
        "config_path": save_config_snapshot(config, output_dir),
        "summary_path": save_summary_json(summary, output_dir),
        "summary": summary,
    }


def wandb_is_available() -> bool:
    """
    Serve a verificare se WandB e' installato nell'ambiente corrente.

    Il notebook usa questa funzione per spiegare se il logging online puo'
    essere attivato senza installare altre librerie.

    Args:
        Nessun argomento.

    Returns:
        True se il pacchetto `wandb` e' importabile, altrimenti False.
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def init_wandb_run(config: dict[str, Any]):
    """
    Serve ad aprire una run WandB solo se il logging online e' abilitato.

    Non contiene API key: l'utente deve aver gia' fatto `wandb login` oppure
    impostare la chiave tramite variabile d'ambiente.

    Args:
        config: Configurazione della run, inclusa la sezione `wandb`.

    Returns:
        Oggetto run WandB se abilitato, altrimenti None.
    """
    wandb_cfg = config.get("wandb", {})
    training_cfg = config.get("training", {})
    enabled = bool(wandb_cfg.get("enabled", False) or training_cfg.get("use_wandb", False))
    if not enabled:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "WandB logging is enabled, but the `wandb` package is not installed in this environment."
        ) from exc

    return wandb.init(
        project=wandb_cfg.get("project", "DLA_Lab1"),
        entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group", "exercise_2_pipeline"),
        name=config.get("experiment_name", "run"),
        job_type=wandb_cfg.get("job_type", "training"),
        mode=wandb_cfg.get("mode", "online"),
        config=config,
        reinit=True,
    )


def log_epoch_to_wandb(run, metrics: dict[str, Any]) -> None:
    """
    Serve a mandare a WandB le metriche di una singola epoca.

    Se `run` e' `None` non fa nulla, quindi il training resta identico anche
    quando WandB e' disattivato.

    Args:
        run: Oggetto run WandB oppure None.
        metrics: Dizionario di metriche da registrare.

    Returns:
        None.
    """
    if run is not None:
        run.log(metrics)


def log_checkpoint_to_wandb(run, checkpoint_path: str | Path, artifact_name: str) -> None:
    """
    Serve a salvare su WandB il miglior checkpoint della run.

    Lo usiamo solo quando WandB e' attivo: oltre alle metriche per epoca,
    resta disponibile anche il file `.pt` del modello migliore.

    Args:
        run: Oggetto run WandB oppure None.
        checkpoint_path: File checkpoint da caricare come artifact.
        artifact_name: Nome dell'artifact su WandB.

    Returns:
        None.
    """
    if run is None:
        return

    path = Path(checkpoint_path)
    if not path.exists():
        return

    import wandb

    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def finish_wandb_run(run) -> None:
    """
    Serve a chiudere correttamente una run WandB.

    E' separata dal training loop per mantenere opzionale il logging online.

    Args:
        run: Oggetto run WandB oppure None.

    Returns:
        None.
    """
    if run is not None:
        run.finish()
