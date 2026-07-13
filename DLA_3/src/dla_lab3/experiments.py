from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from .a2c import (
    evaluate_a2c_policy,
    load_a2c_checkpoint,
    lunar_a2c_from_env,
    train_a2c_vectorized,
)
from .envs import make_env
from .paths import artifact_dir, checkpoint_dir
from .seed import set_seed


def make_lunar_env_factory(seed_value: int):
    """Crea una factory per un ambiente LunarLander con seed.

    Argomenti:
        seed_value: Seed applicato a reset e spazio delle azioni.

    Operazione:
        Costruisce una closure compatibile con gli ambienti vettoriali Gymnasium.

    Output:
        Callable che restituisce un ambiente `LunarLander-v3` configurato.
    """

    def _factory():
        return make_env(
            "LunarLander-v3",
            seed=seed_value,
            continuous=False,
            enable_wind=False,
        )

    return _factory


def save_json_artifact(name: str, payload: Any) -> Path:
    """Salva un artefatto JSON nella cartella `artifacts` del progetto.

    Argomenti:
        name: Nome del file relativo ad `artifacts`.
        payload: Oggetto serializzabile in JSON.

    Operazione:
        Memorizza i risultati sperimentali fuori dagli output dei notebook.

    Output:
        Percorso dell'artefatto salvato.
    """

    path = artifact_dir(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def resolve_checkpoint(filename: str | None, project_root: str | Path | None = None) -> Path | None:
    """Risolve il nome di un checkpoint nelle posizioni note del progetto.

    Argomenti:
        filename: Nome o percorso del checkpoint.
        project_root: Root opzionale usata come ulteriore posizione di ricerca.

    Operazione:
        Cerca tra checkpoint generati, checkpoint storici collegati, root del progetto e
        percorso corrente.

    Output:
        Percorso esistente oppure None se il checkpoint non viene trovato.
    """

    if filename in {None, ""}:
        return None

    candidates = [
        checkpoint_dir(str(filename), create=False),
        checkpoint_dir("old", str(filename), create=False),
        Path(str(filename)),
    ]
    if project_root is not None:
        candidates.insert(2, Path(project_root) / str(filename))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def checkpoint_variants(filename: str) -> list[str]:
    """Restituisce i nomi dei checkpoint best-eval, best-train e last.

    Argomenti:
        filename: Nome di base del checkpoint.

    Operazione:
        Riproduce i nomi salvati da `train_a2c_vectorized`.

    Output:
        Nomi dei checkpoint candidati.
    """

    path = Path(filename)
    return [
        filename,
        f"{path.stem}_best_train{path.suffix}",
        f"{path.stem}_last{path.suffix}",
    ]


def print_lunar_preset_summary(ll_config: dict[str, Any], selected_experiments: list[str]) -> None:
    """Stampa la tabella dei preset LunarLander in forma leggibile.

    Argomenti:
        ll_config: Sezione `a2c_lunarlander` della configurazione YAML.
        selected_experiments: Preset selezionati per la run corrente del notebook.

    Operazione:
        Mostra motivazione, preset disponibili e preset selezionati.

    Output:
        Nessuno; stampa un riepilogo conciso.
    """

    print("Reference note:")
    print(ll_config["reference"]["note"])
    print("\nAvailable presets:")
    for name, preset in ll_config["experiments"].items():
        marker = "*" if name in selected_experiments else " "
        lr_final_text = "" if preset.get("lr_final") is None else f" -> {preset['lr_final']}"
        print(
            f"{marker} {name}: {preset['description']} | "
            f"timesteps={preset['total_timesteps']} | "
            f"n_envs={preset['n_envs']} | lr={preset['lr']}{lr_final_text} | "
            f"reward_scale={preset['reward_scale']}"
        )


def train_lunar_presets(
    selected_experiments: list[str],
    ll_config: dict[str, Any],
    seed: int,
    run_training: bool,
) -> dict[str, dict[str, Any]]:
    """Addestra i preset A2C LunarLander selezionati.

    Argomenti:
        selected_experiments: Nomi presenti in `ll_config["experiments"]`.
        ll_config: Sezione A2C LunarLander della configurazione.
        seed: Seed casuale di base.
        run_training: Se False, salta il training e restituisce un dizionario vuoto.

    Operazione:
        Esegue A2C vettorizzato per ogni preset e salva i checkpoint best-eval,
        best-train e last in `checkpoints`.

    Output:
        Dizionario che associa i preset alle cronologie di training.
    """

    histories: dict[str, dict[str, Any]] = {}
    if not run_training:
        print("Training skipped. Existing current checkpoints will be evaluated below.")
        return histories

    import gymnasium as gym

    for preset_name in selected_experiments:
        preset = ll_config["experiments"][preset_name]
        expected_paths = [
            checkpoint_dir(filename, create=False)
            for filename in checkpoint_variants(str(preset["checkpoint"]))
        ]
        if bool(ll_config.get("skip_completed_training_presets", False)) and all(
            path.exists() for path in expected_paths
        ):
            print("\n" + "=" * 80)
            print("SKIPPING COMPLETED PRESET:", preset_name)
            print("Existing checkpoints found:")
            for path in expected_paths:
                print("-", path)
            print("=" * 80)
            continue

        print("\n" + "=" * 80)
        print("TRAINING PRESET:", preset_name)
        print(preset["description"])
        print("=" * 80)

        set_seed(seed)
        vec_env = gym.vector.SyncVectorEnv(
            [
                make_lunar_env_factory(seed + int(preset["seed_offset"]) + i)
                for i in range(int(preset["n_envs"]))
            ]
        )
        eval_env = make_env(
            "LunarLander-v3",
            seed=seed + int(preset["seed_offset"]) + 1000,
            continuous=False,
            enable_wind=False,
        )

        obs_scale = torch.ones(8, dtype=torch.float32)
        net = lunar_a2c_from_env(eval_env, hidden_size=64)

        initial_path = resolve_checkpoint(preset.get("initial_checkpoint"))
        if preset.get("initial_checkpoint") is not None and initial_path is not None:
            load_a2c_checkpoint(net, initial_path)
            print("Loaded initial checkpoint:", initial_path)
        elif preset.get("initial_checkpoint") is not None:
            print(
                "Initial checkpoint not found, training from random weights:",
                preset["initial_checkpoint"],
            )

        history = train_a2c_vectorized(
            net,
            vec_env,
            eval_env,
            obs_scale,
            total_timesteps=int(preset["total_timesteps"]),
            n_steps=int(preset["n_steps"]),
            gamma=float(preset["gamma"]),
            lr=float(preset["lr"]),
            lr_final=None if preset.get("lr_final") is None else float(preset["lr_final"]),
            value_coef=float(preset["value_coef"]),
            entropy_coef=float(preset["entropy_coef"]),
            entropy_coef_min=float(preset["entropy_coef_min"]),
            gae_lambda=float(preset["gae_lambda"]),
            reward_scale=float(preset["reward_scale"]),
            normalize_advantage=bool(preset["normalize_advantage"]),
            grad_clip=float(ll_config["grad_clip"]),
            eval_every=int(preset["eval_every"]),
            eval_episodes=int(preset["eval_episodes"]),
            solved_threshold=float(ll_config["solved_threshold"]),
            checkpoint_path=str(checkpoint_dir(preset["checkpoint"])),
            optimizer_name=str(preset["optimizer_name"]),
            stop_when_solved=False,
        )

        vec_env.close()
        eval_env.close()

        history["preset_name"] = preset_name
        history["preset"] = dict(preset)
        histories[preset_name] = history

        print("Best evaluation return:", round(history["best_eval_return"], 2))
        print("Best-eval checkpoint:", history["checkpoint_path"])
        print("Best-train checkpoint:", history["train_checkpoint_path"])
        print("Last checkpoint:", history["last_checkpoint_path"])

    save_json_artifact("a2c_lunarlander_training_histories.json", histories)
    return histories


def collect_current_lunar_candidates(
    selected_experiments: list[str],
    ll_config: dict[str, Any],
    histories: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Raccoglie i checkpoint candidati generati dal flusso corrente.

    Argomenti:
        selected_experiments: Preset considerati per la selezione finale.
        ll_config: Sezione A2C LunarLander della configurazione.
        histories: Cronologie opzionali restituite da `train_lunar_presets`.

    Operazione:
        Raccoglie i checkpoint best-eval, best-train e last dei preset
        selezionati, escludendo intenzionalmente quelli esplorativi storici.

    Output:
        Lista di candidati con nome, fonte, preset e percorso.
    """

    candidates: list[dict[str, Any]] = []

    if histories:
        for preset_name, history in histories.items():
            for key, label in [
                ("checkpoint_path", "best_eval"),
                ("train_checkpoint_path", "best_train"),
                ("last_checkpoint_path", "last"),
            ]:
                path_value = history.get(key)
                if path_value is not None and Path(path_value).exists():
                    candidates.append(
                        {
                            "name": Path(path_value).name,
                            "source": "current_run",
                            "preset": preset_name,
                            "checkpoint_type": label,
                            "path": Path(path_value),
                        }
                    )

    if ll_config.get("include_existing_checkpoints", True):
        for preset_name in selected_experiments:
            preset = ll_config["experiments"][preset_name]
            for filename in checkpoint_variants(preset["checkpoint"]):
                path = checkpoint_dir(filename, create=False)
                if path.exists():
                    candidates.append(
                        {
                            "name": filename,
                            "source": "current_folder",
                            "preset": preset_name,
                            "checkpoint_type": "existing",
                            "path": path,
                        }
                    )

    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        deduped[str(candidate["path"].resolve())] = candidate
    return list(deduped.values())


def evaluate_lunar_checkpoint_candidate(
    candidate: dict[str, Any],
    n_eval: int,
    mode: str,
    temperature: float,
    seed_start: int,
) -> dict[str, Any]:
    """Valuta un checkpoint LunarLander.

    Argomenti:
        candidate: Dizionario del candidato con percorso del checkpoint.
        n_eval: Numero di episodi di valutazione.
        mode: `greedy` oppure `sample`.
        temperature: Temperatura della policy stocastica.
        seed_start: Primo seed di valutazione.

    Operazione:
        Carica il checkpoint, esegue episodi nuovi e raccoglie return, successo,
        lunghezza e frequenze delle azioni.

    Output:
        Dizionario con metadati del candidato e metriche di valutazione.
    """

    env = make_env("LunarLander-v3", seed=seed_start, continuous=False, enable_wind=False)
    obs_scale = torch.ones(8, dtype=torch.float32)
    net = lunar_a2c_from_env(env, hidden_size=64)
    load_a2c_checkpoint(net, candidate["path"])

    metrics = evaluate_a2c_policy(
        env,
        net,
        obs_scale,
        n_eval=n_eval,
        mode=mode,
        temperature=temperature,
        seed_start=seed_start,
        max_episode_steps=1000,
    )
    env.close()

    result = {
        "name": candidate["name"],
        "source": candidate["source"],
        "preset": candidate.get("preset"),
        "checkpoint_type": candidate.get("checkpoint_type"),
        "path": str(candidate["path"]),
        "mode": mode,
        "temperature": temperature,
        "n_eval": n_eval,
        "avg_return": metrics["avg_return"],
        "std_return": metrics["std_return"],
        "min_return": metrics["min_return"],
        "max_return": metrics["max_return"],
        "avg_length": metrics["avg_length"],
        "success_rate": metrics["success_rate"],
        "terminated": metrics["terminated"],
        "truncated": metrics["truncated"],
        "truncation_rate": 100.0 * metrics["truncated"] / n_eval,
        "action_freq": metrics["action_freq"],
        "last_quarter_action_freq": metrics.get("last_quarter_action_freq"),
    }
    for key in [
        "final_abs_x",
        "final_abs_y",
        "final_abs_vx",
        "final_abs_vy",
        "final_abs_angle",
        "final_abs_angular_velocity",
        "final_left_leg_contact_rate",
        "final_right_leg_contact_rate",
    ]:
        if key in metrics:
            result[key] = metrics[key]
    return result


def evaluate_lunar_candidates(
    candidates: list[dict[str, Any]],
    ll_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Valuta tutti i checkpoint LunarLander candidati.

    Argomenti:
        candidates: Checkpoint candidati del flusso corrente.
        ll_config: Sezione A2C LunarLander della configurazione.

    Operazione:
        Esegue episodi nuovi per ciascun candidato e salva la tabella
        come artefatto JSON.

    Output:
        Lista dei dizionari dei risultati di valutazione.
    """

    results = []
    for index, candidate in enumerate(candidates):
        result = evaluate_lunar_checkpoint_candidate(
            candidate,
            n_eval=int(ll_config["selection_episodes"]),
            mode=str(ll_config["selection_mode"]),
            temperature=float(ll_config["selection_temperature"]),
            seed_start=50_000 + index * 1_000,
        )
        results.append(result)
        print(
            f"{result['name']}: avg={result['avg_return']:.2f}, "
            f"std={result['std_return']:.2f}, success={result['success_rate']:.1f}%"
        )

    save_json_artifact("a2c_lunarlander_checkpoint_selection.json", results)
    return results


def select_lunar_checkpoint(
    selection_results: list[dict[str, Any]],
    metric: str = "avg_return",
) -> dict[str, Any]:
    """Seleziona il checkpoint migliore dai risultati di valutazione correnti.

    Argomenti:
        selection_results: Risultati di valutazione dei checkpoint candidati.
        metric: Metrica primaria, normalmente `avg_return`.

    Operazione:
        Seleziona la metrica primaria più alta; a parità preferisce un success rate
        maggiore e poi una deviazione standard inferiore.

    Output:
        Dizionario del risultato selezionato.
    """

    if not selection_results:
        raise RuntimeError("Non è stato valutato alcun checkpoint LunarLander.")

    best_by_metric = max(
        selection_results,
        key=lambda row: (
            row[metric],
            row["success_rate"],
            -row["std_return"],
        ),
    )
    best_by_success = max(
        selection_results,
        key=lambda row: (
            row["success_rate"],
            row["avg_return"],
            -row["std_return"],
        ),
    )

    print("Riepilogo della selezione dei checkpoint:")
    print(
        f"- migliore valore di {metric}: {best_by_metric['name']} "
        f"(avg={best_by_metric['avg_return']:.2f}, "
        f"success={best_by_metric['success_rate']:.1f}%)"
    )
    print(
        f"- migliore percentuale di successo: {best_by_success['name']} "
        f"(avg={best_by_success['avg_return']:.2f}, "
        f"success={best_by_success['success_rate']:.1f}%)"
    )
    if best_by_metric["path"] != best_by_success["path"]:
        print(
            "Il checkpoint finale è selezionato con la metrica primaria configurata. "
            "Quando le due scelte differiscono, va riportata anche l'alternativa "
            "basata sulla percentuale di successo."
        )

    selected = best_by_metric
    print("Checkpoint selezionato:")
    print(json.dumps(selected, indent=2))
    return selected


def run_lunar_temperature_sweep(
    selected_checkpoint: dict[str, Any],
    ll_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Valuta il checkpoint selezionato a più temperature della policy.

    Argomenti:
        selected_checkpoint: Dizionario selezionato da
            `select_lunar_checkpoint`.
        ll_config: Sezione A2C LunarLander della configurazione.

    Operazione:
        Esegue valutazioni stocastiche nuove per ogni temperatura della griglia YAML.

    Output:
        Lista dei risultati di valutazione per temperatura.
    """

    candidate = {
        "name": selected_checkpoint["name"],
        "source": selected_checkpoint["source"],
        "preset": selected_checkpoint.get("preset"),
        "checkpoint_type": selected_checkpoint.get("checkpoint_type"),
        "path": Path(selected_checkpoint["path"]),
    }

    results = []
    for index, temperature in enumerate([float(t) for t in ll_config["temperature_grid"]]):
        result = evaluate_lunar_checkpoint_candidate(
            candidate,
            n_eval=int(ll_config["selection_episodes"]),
            mode="sample",
            temperature=temperature,
            seed_start=70_000 + index * 1_000,
        )
        results.append(result)
        print(
            f"T={temperature:.2f}: avg={result['avg_return']:.2f}, "
            f"std={result['std_return']:.2f}, success={result['success_rate']:.1f}%"
        )

    save_json_artifact("a2c_lunarlander_temperature_sweep.json", results)
    return results


def _policy_selection_key(
    row: dict[str, Any], metric: str
) -> tuple[float, float, float, float, float]:
    """Restituisce una chiave deterministica per ordinare le configurazioni della policy."""

    primary = policy_configuration_score(row, metric)
    return (
        primary,
        float(row["avg_return"]),
        float(row["success_rate"]),
        -float(row.get("truncation_rate", 100.0)),
        -float(row["std_return"]),
    )


def policy_configuration_score(row: dict[str, Any], metric: str) -> float:
    """Calcola il punteggio scalare usato per ordinare le configurazioni finali.

    Argomenti:
        row: Metriche di valutazione della configurazione della policy.
        metric: Nome di una metrica diretta oppure `reliability_score`.

    Operazione:
        Mantiene `avg_return` per il criterio ufficiale di soluzione e aggiunge
        un punteggio di affidabilità per le run non risolutive. Il punteggio
        premia il return medio e gli atterraggi riusciti, penalizzando gli episodi
        troncati.

    Output:
        Punteggio scalare usato da `select_lunar_policy_configuration`.
    """

    if metric == "reliability_score":
        return (
            float(row["avg_return"])
            + 2.0 * float(row["success_rate"])
            - 2.0 * float(row.get("truncation_rate", 0.0))
        )
    if metric not in row:
        raise KeyError(f"Metrica di selezione della policy sconosciuta: {metric}")
    return float(row[metric])


def evaluate_lunar_policy_configurations(
    selection_results: list[dict[str, Any]],
    ll_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Valuta combinazioni di checkpoint, modalità delle azioni e temperatura.

    Argomenti:
        selection_results: Risultati dei checkpoint candidati prodotti da
            `evaluate_lunar_candidates`.
        ll_config: Sezione A2C LunarLander della configurazione.

    Operazione:
        Prende i checkpoint più promettenti, valuta la selezione greedy e
        stocastica delle azioni sulla griglia di temperature configurata e
        memorizza la tabella completa come artefatto.

    Output:
        Lista dei dizionari di valutazione delle configurazioni della policy.
    """

    if not selection_results:
        raise RuntimeError("Non è disponibile alcun risultato di selezione dei checkpoint.")

    top_k = int(ll_config.get("policy_selection_top_k", 4))
    top_by_return = sorted(
        selection_results,
        key=lambda row: (row["avg_return"], row["success_rate"], -row["std_return"]),
        reverse=True,
    )[:top_k]
    top_by_success = sorted(
        selection_results,
        key=lambda row: (row["success_rate"], row["avg_return"], -row["std_return"]),
        reverse=True,
    )[:top_k]

    candidates_by_path: dict[str, dict[str, Any]] = {}
    for reason, rows in [
        (f"primi {top_k} per return medio a T={ll_config['selection_temperature']}", top_by_return),
        (
            f"primi {top_k} per percentuale di successo a T={ll_config['selection_temperature']}",
            top_by_success,
        ),
    ]:
        for row in rows:
            key = str(row["path"])
            candidate = candidates_by_path.setdefault(
                key,
                {
                    "name": row["name"],
                    "source": row["source"],
                    "preset": row.get("preset"),
                    "checkpoint_type": row.get("checkpoint_type"),
                    "path": Path(row["path"]),
                    "candidate_reasons": [],
                },
            )
            candidate["candidate_reasons"].append(reason)

    results: list[dict[str, Any]] = []
    n_eval = int(ll_config.get("policy_selection_episodes", ll_config["selection_episodes"]))
    seed_base = 110_000
    temperatures = [float(t) for t in ll_config["temperature_grid"]]
    include_greedy = bool(ll_config.get("policy_selection_include_greedy", True))

    print("Configurazione della valutazione delle policy:")
    print(
        f"- la prima fase ha valutato {len(selection_results)} checkpoint con "
        f"modalità={ll_config['selection_mode']} e T={float(ll_config['selection_temperature']):.2f}"
    )
    print(f"- la seconda fase valuta {len(candidates_by_path)} checkpoint candidati")
    print(
        f"- per ogni candidato: {'greedy + ' if include_greedy else ''}temperature sample {temperatures}"
    )
    print(f"- episodi per configurazione: {n_eval}")
    print("Checkpoint candidati:")
    for candidate in candidates_by_path.values():
        reasons = "; ".join(candidate["candidate_reasons"])
        print(f"  - {candidate['name']} ({candidate['preset']}): {reasons}")

    for candidate_index, candidate in enumerate(candidates_by_path.values()):
        candidate_for_eval = {
            key: candidate[key] for key in ["name", "source", "preset", "checkpoint_type", "path"]
        }
        if include_greedy:
            result = evaluate_lunar_checkpoint_candidate(
                candidate_for_eval,
                n_eval=n_eval,
                mode="greedy",
                temperature=1.0,
                seed_start=seed_base + candidate_index * 20_000,
            )
            results.append(result)
            print(
                f"{result['name']} | greedy: avg={result['avg_return']:.2f}, "
                f"success={result['success_rate']:.1f}%, trunc={result['truncation_rate']:.1f}%"
            )

        for temp_index, temperature in enumerate(temperatures):
            result = evaluate_lunar_checkpoint_candidate(
                candidate_for_eval,
                n_eval=n_eval,
                mode="sample",
                temperature=temperature,
                seed_start=seed_base + candidate_index * 20_000 + 1_000 + temp_index * 1_000,
            )
            results.append(result)
            print(
                f"{result['name']} | sample T={temperature:.2f}: "
                f"avg={result['avg_return']:.2f}, success={result['success_rate']:.1f}%, "
                f"trunc={result['truncation_rate']:.1f}%"
            )

    for result in results:
        result["reliability_score"] = policy_configuration_score(result, "reliability_score")

    save_json_artifact("a2c_lunarlander_policy_config_selection.json", results)
    return results


def select_lunar_policy_configuration(
    policy_results: list[dict[str, Any]],
    metric: str = "avg_return",
) -> dict[str, Any]:
    """Seleziona checkpoint, modalità delle azioni e temperatura finali.

    Argomenti:
        policy_results: Risultati di `evaluate_lunar_policy_configurations`.
        metric: Metrica primaria usata per la scelta finale.

    Operazione:
        Seleziona una configurazione completa, rendendo temperatura e modalità
        greedy/stocastica parte della selezione del modello, non un controllo
        separato successivo.

    Output:
        Dizionario della configurazione selezionata.
    """

    if not policy_results:
        raise RuntimeError("Non è stata valutata alcuna configurazione della policy.")

    for row in policy_results:
        row["reliability_score"] = policy_configuration_score(row, "reliability_score")

    selected = max(policy_results, key=lambda row: _policy_selection_key(row, metric))
    best_success = max(policy_results, key=lambda row: _policy_selection_key(row, "success_rate"))
    fewest_trunc = min(
        policy_results, key=lambda row: (row.get("truncation_rate", 100.0), -row["avg_return"])
    )
    top_by_metric = sorted(
        policy_results, key=lambda row: _policy_selection_key(row, metric), reverse=True
    )[:5]

    print("Riepilogo delle configurazioni della policy:")
    print(f"- metrica di selezione: {metric}")
    if metric == "reliability_score":
        print("- reliability_score = avg_return + 2 * success_rate - 2 * truncation_rate")
    print("Migliori configurazioni secondo la metrica selezionata:")
    for rank, row in enumerate(top_by_metric, start=1):
        print(
            f"  {rank}. {row['name']} | {row['mode']} | T={row['temperature']:.2f} | "
            f"media={row['avg_return']:.2f} | successo={row['success_rate']:.1f}% | "
            f"troncati={row['truncation_rate']:.1f}% | affidabilità={row['reliability_score']:.2f}"
        )
    print(
        f"- selezionata secondo {metric}: {selected['name']} | modalità={selected['mode']} | "
        f"T={selected['temperature']:.2f} | media={selected['avg_return']:.2f} | "
        f"successo={selected['success_rate']:.1f}% | troncati={selected['truncation_rate']:.1f}% | "
        f"affidabilità={selected['reliability_score']:.2f}"
    )
    print(
        f"- migliore percentuale di successo: {best_success['name']} | modalità={best_success['mode']} | "
        f"T={best_success['temperature']:.2f} | media={best_success['avg_return']:.2f} | "
        f"successo={best_success['success_rate']:.1f}% | troncati={best_success['truncation_rate']:.1f}%"
    )
    print(
        f"- meno episodi troncati: {fewest_trunc['name']} | modalità={fewest_trunc['mode']} | "
        f"T={fewest_trunc['temperature']:.2f} | media={fewest_trunc['avg_return']:.2f} | "
        f"successo={fewest_trunc['success_rate']:.1f}% | troncati={fewest_trunc['truncation_rate']:.1f}%"
    )
    print("Configurazione finale della policy selezionata:")
    print(json.dumps(selected, indent=2))
    return selected


def choose_final_temperature(
    temperature_results: list[dict[str, Any]],
    preferred_temperature: float = 1.0,
) -> float:
    """Sceglie la temperatura finale della policy.

    Argomenti:
        temperature_results: Risultati di `run_lunar_temperature_sweep`.
        preferred_temperature: Temperatura scelta per la stabilità qualitativa.

    Operazione:
        Usa la temperatura preferita quando è stata valutata e rende la scelta
        finale esplicita: una temperatura motivata è preferibile a inseguire
        un massimo rumoroso di un singolo sweep.

    Output:
        Temperatura selezionata.
    """

    if not temperature_results:
        raise RuntimeError("Non è stato fornito alcun risultato per le temperature.")

    best_avg = max(temperature_results, key=lambda row: row["avg_return"])
    best_success = max(
        temperature_results,
        key=lambda row: (
            row["success_rate"],
            row["avg_return"],
            -row["std_return"],
        ),
    )

    print("Riepilogo dello sweep della temperatura:")
    print(
        f"- migliore return medio: T={best_avg['temperature']:.2f} "
        f"(avg={best_avg['avg_return']:.2f}, success={best_avg['success_rate']:.1f}%)"
    )
    print(
        f"- migliore percentuale di successo: T={best_success['temperature']:.2f} "
        f"(avg={best_success['avg_return']:.2f}, success={best_success['success_rate']:.1f}%)"
    )
    print(
        "La temperatura è un parametro di campionamento usato in inferenza: modifica "
        "Categorical(logits / T), ma non aggiorna i pesi del checkpoint."
    )

    evaluated = {float(row["temperature"]) for row in temperature_results}
    if preferred_temperature in evaluated:
        print(
            f"Temperatura fissa selezionata per valutazione finale e rollout visuali: "
            f"{preferred_temperature}"
        )
        return preferred_temperature

    fallback = max(temperature_results, key=lambda row: row["avg_return"])
    print(
        "Temperatura preferita non valutata; selezionato il miglior return medio:",
        fallback["temperature"],
    )
    return float(fallback["temperature"])


def final_lunar_evaluation(
    selected_checkpoint: dict[str, Any],
    ll_config: dict[str, Any],
    temperature: float,
    mode: str = "sample",
) -> dict[str, Any]:
    """Esegue la valutazione finale LunarLander.

    Argomenti:
        selected_checkpoint: Dizionario del checkpoint selezionato.
        ll_config: Sezione A2C LunarLander della configurazione.
        temperature: Temperatura finale della policy stocastica.
        mode: Modalità finale delle azioni: `greedy` o `sample`.

    Operazione:
        Usa un campione più ampio rispetto alla selezione e salva le metriche
        come artefatto JSON.

    Output:
        Metriche della valutazione finale.
    """

    candidate = {
        "name": selected_checkpoint["name"],
        "source": selected_checkpoint["source"],
        "preset": selected_checkpoint.get("preset"),
        "checkpoint_type": selected_checkpoint.get("checkpoint_type"),
        "path": Path(selected_checkpoint["path"]),
    }
    result = evaluate_lunar_checkpoint_candidate(
        candidate,
        n_eval=int(ll_config["final_eval_episodes"]),
        mode=mode,
        temperature=temperature,
        seed_start=90_000,
    )
    save_json_artifact("a2c_lunarlander_final_evaluation.json", result)
    print_lunar_evaluation_result("FINAL LUNARLANDER EVALUATION", result)
    return result


def print_lunar_evaluation_result(title: str, result: dict[str, Any]) -> None:
    """Stampa le metriche di valutazione LunarLander.

    Argomenti:
        title: Titolo della sezione.
        result: Dizionario delle metriche.

    Operazione:
        Formatta le metriche principali per l'output del notebook.

    Output:
        Nessuno; stampa le metriche.
    """

    print(title)
    print("Checkpoint:", result["name"])
    print("Origine:", result["source"])
    print("Preset:", result.get("preset"))
    print("Temperatura:", result["temperature"])
    print("Return medio:", round(result["avg_return"], 2))
    print("Deviazione standard del return:", round(result["std_return"], 2))
    print("Return minimo:", round(result["min_return"], 2))
    print("Return massimo:", round(result["max_return"], 2))
    print("Lunghezza media:", round(result["avg_length"], 2))
    print("Percentuale di successo >= 200:", f"{result['success_rate']:.1f}%")
    print("Episodi terminati:", f"{result['terminated']}/{result['n_eval']}")
    print("Episodi troncati:", f"{result['truncated']}/{result['n_eval']}")
    print("Frequenze delle azioni:", [round(x, 3) for x in result["action_freq"]])
    if result.get("last_quarter_action_freq") is not None:
        print(
            "Frequenze delle azioni nell'ultimo quarto:",
            [round(x, 3) for x in result["last_quarter_action_freq"]],
        )
    if "final_abs_x" in result:
        print("|x| finale:", round(result["final_abs_x"], 3))
        print("|velocità verticale| finale:", round(result["final_abs_vy"], 3))
        print("|angolo| finale:", round(result["final_abs_angle"], 3))
        print(
            "Percentuali finali di contatto delle gambe:",
            f"{result['final_left_leg_contact_rate']:.1f}% / {result['final_right_leg_contact_rate']:.1f}%",
        )


def plot_lunar_selection(
    selection_results: list[dict[str, Any]], solved_threshold: float = 200.0
) -> None:
    """Traccia le metriche di selezione dei checkpoint.

    Argomenti:
        selection_results: Risultati di valutazione dei candidati.
        solved_threshold: Soglia del return medio per risolvere LunarLander.

    Operazione:
        Mostra return medio e success rate di tutti i checkpoint candidati.

    Output:
        Figura Matplotlib mostrata nel notebook.
    """

    names = [row["name"] for row in selection_results]
    avg_returns = [row["avg_return"] for row in selection_results]
    success_rates = [row["success_rate"] for row in selection_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(names, avg_returns, color="tab:blue")
    axes[0].axvline(
        solved_threshold, linestyle="--", color="tab:green", label="Soglia di soluzione"
    )
    axes[0].set_xlabel("Return medio")
    axes[0].set_title("Selezione checkpoint - return")
    axes[0].legend()

    axes[1].barh(names, success_rates, color="tab:orange")
    axes[1].set_xlabel("Percentuale di successo >= 200 (%)")
    axes[1].set_title("Selezione checkpoint - percentuale di successo")

    plt.tight_layout()
    plt.show()


def plot_temperature_sweep(temperature_results: list[dict[str, Any]]) -> None:
    """Traccia i risultati dello sweep della temperatura.

    Argomenti:
        temperature_results: Risultati di valutazione delle temperature.

    Operazione:
        Mostra return medio e success rate in funzione della temperatura
        della policy.

    Output:
        Figura Matplotlib mostrata nel notebook.
    """

    temps = [row["temperature"] for row in temperature_results]
    avg_returns = [row["avg_return"] for row in temperature_results]
    success_rates = [row["success_rate"] for row in temperature_results]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(temps, avg_returns, marker="o", color="tab:blue", label="Return medio")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Return medio", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(temps, success_rates, marker="s", color="tab:green", label="Percentuale di successo")
    ax2.set_ylabel("Percentuale di successo >= 200 (%)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("LunarLander-v3 - temperature sweep")
    fig.tight_layout()
    plt.show()


def print_training_scope(ll_config: dict[str, Any], selected_experiments: list[str]) -> None:
    """Stampa il budget di training di ogni preset selezionato.

    Argomenti:
        ll_config: Sezione A2C LunarLander della configurazione.
        selected_experiments: Preset selezionati per training e valutazione.

    Operazione:
        Esplicita il budget: timestep, ambienti vettoriali, step di rollout,
        ottimizzatore e nome del checkpoint.

    Output:
        Nessuno; stampa un riepilogo tabellare.
    """

    print("Training budget for selected presets:")
    for name in selected_experiments:
        preset = ll_config["experiments"][name]
        updates = int(preset["total_timesteps"]) // (int(preset["n_envs"]) * int(preset["n_steps"]))
        lr_final_text = "" if preset.get("lr_final") is None else f" -> {preset['lr_final']}"
        print(
            f"- {name}: timesteps={preset['total_timesteps']}, "
            f"n_envs={preset['n_envs']}, n_steps={preset['n_steps']}, "
            f"updates~{updates}, lr={preset['lr']}{lr_final_text}, optimizer={preset['optimizer_name']}, "
            f"checkpoint={preset['checkpoint']}"
        )
