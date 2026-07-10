from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    """Create a factory for one seeded LunarLander environment.

    Args:
        seed_value: Seed applied to reset and action space.

    What it does:
        Builds a closure compatible with Gymnasium vector environments.

    Outputs:
        Callable that returns one configured `LunarLander-v3` environment.
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
    """Save a JSON artifact under the project artifacts folder.

    Args:
        name: Output filename relative to `artifacts`.
        payload: JSON-serializable object.

    What it does:
        Stores experiment results outside notebook outputs.

    Outputs:
        Path to the saved artifact.
    """

    path = artifact_dir(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def resolve_checkpoint(filename: str | None, project_root: str | Path | None = None) -> Path | None:
    """Resolve a checkpoint name against known project locations.

    Args:
        filename: Checkpoint filename or path.
        project_root: Optional project root used as an extra search location.

    What it does:
        Searches generated checkpoints, linked old checkpoints, project root and
        the current path.

    Outputs:
        Existing checkpoint path, or None when not found.
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
    """Return best-eval, best-train and last checkpoint filenames.

    Args:
        filename: Base checkpoint filename.

    What it does:
        Mirrors the filenames saved by `train_a2c_vectorized`.

    Outputs:
        Candidate checkpoint filenames.
    """

    path = Path(filename)
    return [
        filename,
        f"{path.stem}_best_train{path.suffix}",
        f"{path.stem}_last{path.suffix}",
    ]


def print_lunar_preset_summary(ll_config: dict[str, Any], selected_experiments: list[str]) -> None:
    """Print the LunarLander preset table in a readable form.

    Args:
        ll_config: `a2c_lunarlander` section of the YAML config.
        selected_experiments: Presets selected for the current notebook run.

    What it does:
        Displays the rationale, available presets and selected presets.

    Outputs:
        None. Prints a concise summary.
    """

    print("Reference note:")
    print(ll_config["reference"]["note"])
    print("\nAvailable presets:")
    for name, preset in ll_config["experiments"].items():
        marker = "*" if name in selected_experiments else " "
        print(
            f"{marker} {name}: {preset['description']} | "
            f"timesteps={preset['total_timesteps']} | "
            f"n_envs={preset['n_envs']} | lr={preset['lr']} | "
            f"reward_scale={preset['reward_scale']}"
        )


def train_lunar_presets(
    selected_experiments: list[str],
    ll_config: dict[str, Any],
    seed: int,
    run_training: bool,
) -> dict[str, dict[str, Any]]:
    """Train selected LunarLander A2C presets.

    Args:
        selected_experiments: Names from `ll_config["experiments"]`.
        ll_config: LunarLander A2C config section.
        seed: Base random seed.
        run_training: If False, skip training and return an empty dictionary.

    What it does:
        Runs vectorized A2C for each selected preset, saving best-eval,
        best-train and last checkpoints under `checkpoints`.

    Outputs:
        Dictionary mapping preset names to training histories.
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
            print("Initial checkpoint not found, training from random weights:", preset["initial_checkpoint"])

        history = train_a2c_vectorized(
            net,
            vec_env,
            eval_env,
            obs_scale,
            total_timesteps=int(preset["total_timesteps"]),
            n_steps=int(preset["n_steps"]),
            gamma=float(preset["gamma"]),
            lr=float(preset["lr"]),
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
    """Collect checkpoint candidates generated by the current workflow.

    Args:
        selected_experiments: Presets considered for final selection.
        ll_config: LunarLander A2C config section.
        histories: Optional histories returned by `train_lunar_presets`.

    What it does:
        Collects best-eval, best-train and last checkpoints for selected
        presets. It intentionally avoids old exploratory checkpoints by default.

    Outputs:
        List of candidate dictionaries with name, source, preset and path.
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
    """Evaluate one LunarLander checkpoint.

    Args:
        candidate: Candidate dictionary with a checkpoint path.
        n_eval: Number of evaluation episodes.
        mode: `greedy` or `sample`.
        temperature: Stochastic policy temperature.
        seed_start: First evaluation seed.

    What it does:
        Loads the checkpoint, runs fresh episodes and collects return, success,
        length and action-frequency metrics.

    Outputs:
        Dictionary with candidate metadata and evaluation metrics.
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
    }
    return result


def evaluate_lunar_candidates(
    candidates: list[dict[str, Any]],
    ll_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate all LunarLander candidate checkpoints.

    Args:
        candidates: Checkpoint candidates from the current workflow.
        ll_config: LunarLander A2C config section.

    What it does:
        Runs fresh evaluation episodes for every candidate and saves the table
        as a JSON artifact.

    Outputs:
        List of evaluation-result dictionaries.
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
    """Select the best checkpoint from current evaluation results.

    Args:
        selection_results: Evaluation results for candidate checkpoints.
        metric: Primary metric, usually `avg_return`.

    What it does:
        Selects the highest primary metric. Ties are broken by higher success
        rate, then lower return standard deviation.

    Outputs:
        The selected result dictionary.
    """

    if not selection_results:
        raise RuntimeError("No LunarLander checkpoint was evaluated.")

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

    print("Checkpoint-selection summary:")
    print(
        f"- best {metric}: {best_by_metric['name']} "
        f"(avg={best_by_metric['avg_return']:.2f}, "
        f"success={best_by_metric['success_rate']:.1f}%)"
    )
    print(
        f"- best success rate: {best_by_success['name']} "
        f"(avg={best_by_success['avg_return']:.2f}, "
        f"success={best_by_success['success_rate']:.1f}%)"
    )
    if best_by_metric["path"] != best_by_success["path"]:
        print(
            "The final checkpoint is selected by the configured primary metric. "
            "Report the success-rate alternative when the two choices differ."
        )

    selected = best_by_metric
    print("Selected checkpoint:")
    print(json.dumps(selected, indent=2))
    return selected


def run_lunar_temperature_sweep(
    selected_checkpoint: dict[str, Any],
    ll_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the selected checkpoint at multiple temperatures.

    Args:
        selected_checkpoint: Result dictionary selected by
            `select_lunar_checkpoint`.
        ll_config: LunarLander A2C config section.

    What it does:
        Runs fresh stochastic evaluations for each temperature in the YAML grid.

    Outputs:
        List of temperature evaluation results.
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


def _policy_selection_key(row: dict[str, Any], metric: str) -> tuple[float, float, float, float, float]:
    """Return a deterministic ordering key for policy configuration selection."""

    primary = float(row[metric])
    return (
        primary,
        float(row["avg_return"]),
        float(row["success_rate"]),
        -float(row.get("truncation_rate", 100.0)),
        -float(row["std_return"]),
    )


def evaluate_lunar_policy_configurations(
    selection_results: list[dict[str, Any]],
    ll_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate checkpoint, action-mode and temperature combinations.

    Args:
        selection_results: Candidate checkpoint results from
            `evaluate_lunar_candidates`.
        ll_config: LunarLander A2C config section.

    What it does:
        Takes the most promising checkpoints from the current run, evaluates
        greedy action selection and stochastic action selection over the
        configured temperature grid, and stores the full table as an artifact.

    Outputs:
        List of policy-configuration evaluation dictionaries.
    """

    if not selection_results:
        raise RuntimeError("No checkpoint selection result is available.")

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
    for row in top_by_return + top_by_success:
        candidates_by_path[str(row["path"])] = {
            "name": row["name"],
            "source": row["source"],
            "preset": row.get("preset"),
            "checkpoint_type": row.get("checkpoint_type"),
            "path": Path(row["path"]),
        }

    results: list[dict[str, Any]] = []
    n_eval = int(ll_config.get("policy_selection_episodes", ll_config["selection_episodes"]))
    seed_base = 110_000
    temperatures = [float(t) for t in ll_config["temperature_grid"]]
    include_greedy = bool(ll_config.get("policy_selection_include_greedy", True))

    for candidate_index, candidate in enumerate(candidates_by_path.values()):
        if include_greedy:
            result = evaluate_lunar_checkpoint_candidate(
                candidate,
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
                candidate,
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

    save_json_artifact("a2c_lunarlander_policy_config_selection.json", results)
    return results


def select_lunar_policy_configuration(
    policy_results: list[dict[str, Any]],
    metric: str = "avg_return",
) -> dict[str, Any]:
    """Select the final checkpoint, action mode and temperature.

    Args:
        policy_results: Results from `evaluate_lunar_policy_configurations`.
        metric: Primary metric used for the final choice.

    What it does:
        Selects a complete policy configuration. This makes temperature and
        greedy/stochastic action selection part of model selection instead of a
        separate afterthought.

    Outputs:
        Selected policy-configuration dictionary.
    """

    if not policy_results:
        raise RuntimeError("No policy configuration was evaluated.")

    selected = max(policy_results, key=lambda row: _policy_selection_key(row, metric))
    best_success = max(policy_results, key=lambda row: _policy_selection_key(row, "success_rate"))
    fewest_trunc = min(policy_results, key=lambda row: (row.get("truncation_rate", 100.0), -row["avg_return"]))

    print("Policy-configuration summary:")
    print(
        f"- selected by {metric}: {selected['name']} | mode={selected['mode']} | "
        f"T={selected['temperature']:.2f} | avg={selected['avg_return']:.2f} | "
        f"success={selected['success_rate']:.1f}% | trunc={selected['truncation_rate']:.1f}%"
    )
    print(
        f"- best success rate: {best_success['name']} | mode={best_success['mode']} | "
        f"T={best_success['temperature']:.2f} | avg={best_success['avg_return']:.2f} | "
        f"success={best_success['success_rate']:.1f}% | trunc={best_success['truncation_rate']:.1f}%"
    )
    print(
        f"- fewest truncated episodes: {fewest_trunc['name']} | mode={fewest_trunc['mode']} | "
        f"T={fewest_trunc['temperature']:.2f} | avg={fewest_trunc['avg_return']:.2f} | "
        f"success={fewest_trunc['success_rate']:.1f}% | trunc={fewest_trunc['truncation_rate']:.1f}%"
    )
    print("Selected final policy configuration:")
    print(json.dumps(selected, indent=2))
    return selected


def choose_final_temperature(
    temperature_results: list[dict[str, Any]],
    preferred_temperature: float = 1.0,
) -> float:
    """Choose the final policy temperature.

    Args:
        temperature_results: Results from `run_lunar_temperature_sweep`.
        preferred_temperature: Temperature chosen for qualitative stability.

    What it does:
        Uses the preferred temperature when it was evaluated. This keeps the
        final choice explicit: temperature 1.0 is often easier to justify than
        chasing a noisy maximum from one sweep.

    Outputs:
        Selected temperature.
    """

    if not temperature_results:
        raise RuntimeError("No temperature result was provided.")

    best_avg = max(temperature_results, key=lambda row: row["avg_return"])
    best_success = max(
        temperature_results,
        key=lambda row: (
            row["success_rate"],
            row["avg_return"],
            -row["std_return"],
        ),
    )

    print("Temperature sweep summary:")
    print(
        f"- best average return: T={best_avg['temperature']:.2f} "
        f"(avg={best_avg['avg_return']:.2f}, success={best_avg['success_rate']:.1f}%)"
    )
    print(
        f"- best success rate: T={best_success['temperature']:.2f} "
        f"(avg={best_success['avg_return']:.2f}, success={best_success['success_rate']:.1f}%)"
    )
    print(
        "Temperature is an inference-time sampling parameter: it changes "
        "Categorical(logits / T), but it does not update checkpoint weights."
    )

    evaluated = {float(row["temperature"]) for row in temperature_results}
    if preferred_temperature in evaluated:
        print(
            f"Selected fixed temperature for final evaluation and visual rollouts: "
            f"{preferred_temperature}"
        )
        return preferred_temperature

    fallback = max(temperature_results, key=lambda row: row["avg_return"])
    print(
        "Preferred temperature not evaluated; selected best average return:",
        fallback["temperature"],
    )
    return float(fallback["temperature"])


def final_lunar_evaluation(
    selected_checkpoint: dict[str, Any],
    ll_config: dict[str, Any],
    temperature: float,
    mode: str = "sample",
) -> dict[str, Any]:
    """Run the final LunarLander evaluation.

    Args:
        selected_checkpoint: Selected checkpoint result dictionary.
        ll_config: LunarLander A2C config section.
        temperature: Final stochastic policy temperature.
        mode: Final action-selection mode: `greedy` or `sample`.

    What it does:
        Runs a larger evaluation sample than the selection step and saves the
        metrics as a JSON artifact.

    Outputs:
        Final evaluation metrics.
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
    """Print LunarLander evaluation metrics.

    Args:
        title: Section title.
        result: Metrics dictionary.

    What it does:
        Formats the most important evaluation metrics for notebook output.

    Outputs:
        None. Prints metrics.
    """

    print(title)
    print("Checkpoint:", result["name"])
    print("Source:", result["source"])
    print("Preset:", result.get("preset"))
    print("Temperature:", result["temperature"])
    print("Average return:", round(result["avg_return"], 2))
    print("Std return:", round(result["std_return"], 2))
    print("Min return:", round(result["min_return"], 2))
    print("Max return:", round(result["max_return"], 2))
    print("Average length:", round(result["avg_length"], 2))
    print("Success rate >= 200:", f"{result['success_rate']:.1f}%")
    print("Terminated episodes:", f"{result['terminated']}/{result['n_eval']}")
    print("Truncated episodes:", f"{result['truncated']}/{result['n_eval']}")
    print("Action frequencies:", [round(x, 3) for x in result["action_freq"]])


def plot_lunar_selection(selection_results: list[dict[str, Any]], solved_threshold: float = 200.0) -> None:
    """Plot checkpoint-selection metrics.

    Args:
        selection_results: Candidate evaluation results.
        solved_threshold: Average return threshold for solving LunarLander.

    What it does:
        Shows average return and success rate for all candidate checkpoints.

    Outputs:
        Matplotlib figure displayed in the notebook.
    """

    names = [row["name"] for row in selection_results]
    avg_returns = [row["avg_return"] for row in selection_results]
    success_rates = [row["success_rate"] for row in selection_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(names, avg_returns, color="tab:blue")
    axes[0].axvline(solved_threshold, linestyle="--", color="tab:green", label="Solved threshold")
    axes[0].set_xlabel("Average return")
    axes[0].set_title("Checkpoint selection - return")
    axes[0].legend()

    axes[1].barh(names, success_rates, color="tab:orange")
    axes[1].set_xlabel("Success rate >= 200 (%)")
    axes[1].set_title("Checkpoint selection - success rate")

    plt.tight_layout()
    plt.show()


def plot_temperature_sweep(temperature_results: list[dict[str, Any]]) -> None:
    """Plot temperature sweep results.

    Args:
        temperature_results: Temperature evaluation results.

    What it does:
        Shows average return and success rate as a function of policy
        temperature.

    Outputs:
        Matplotlib figure displayed in the notebook.
    """

    temps = [row["temperature"] for row in temperature_results]
    avg_returns = [row["avg_return"] for row in temperature_results]
    success_rates = [row["success_rate"] for row in temperature_results]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(temps, avg_returns, marker="o", color="tab:blue", label="Average return")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Average return", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(temps, success_rates, marker="s", color="tab:green", label="Success rate")
    ax2.set_ylabel("Success rate >= 200 (%)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title("LunarLander-v3 - temperature sweep")
    fig.tight_layout()
    plt.show()


def print_training_scope(ll_config: dict[str, Any], selected_experiments: list[str]) -> None:
    """Print how much training each selected preset performs.

    Args:
        ll_config: LunarLander A2C config section.
        selected_experiments: Presets selected for training/evaluation.

    What it does:
        Makes the training budget explicit: timesteps, vector envs, rollout
        steps, optimizer and checkpoint name.

    Outputs:
        None. Prints a table-like summary.
    """

    print("Training budget for selected presets:")
    for name in selected_experiments:
        preset = ll_config["experiments"][name]
        updates = int(preset["total_timesteps"]) // (int(preset["n_envs"]) * int(preset["n_steps"]))
        print(
            f"- {name}: timesteps={preset['total_timesteps']}, "
            f"n_envs={preset['n_envs']}, n_steps={preset['n_steps']}, "
            f"updates~{updates}, lr={preset['lr']}, optimizer={preset['optimizer_name']}, "
            f"checkpoint={preset['checkpoint']}"
        )
