from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from matplotlib import animation

from .a2c import load_a2c_checkpoint, lunar_a2c_from_env
from .envs import make_env, observation_scale
from .experiments import save_json_artifact
from .policy_gradient import policy_from_env, preprocess_observation, select_action


def show_frames(frames: list[Any], interval: int = 30, max_display_frames: int = 300) -> None:
    """Mostra nel notebook i frame RGB di un ambiente.

    Argomenti:
        frames: Array RGB restituiti da Gymnasium con `render_mode="rgb_array"`.
        interval: Ritardo tra frame in millisecondi.
        max_display_frames: Numero massimo di frame mantenuti nell'animazione.

    Operazione:
        Sottocampiona gli episodi lunghi e produce un'animazione HTML nella cella
        corrente. Non scrive file.

    Output:
        Nessuno; visualizza l'animazione tramite IPython.
    """

    if not frames:
        print("Nessun frame raccolto.")
        return

    if len(frames) > max_display_frames:
        step = int(np.ceil(len(frames) / max_display_frames))
        frames = frames[::step]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    image = ax.imshow(frames[0])

    def update(frame):
        image.set_data(frame)
        return (image,)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    plt.close(fig)
    display(HTML(anim.to_jshtml()))


def run_cartpole_visual_episodes(
    checkpoint_path: str | Path,
    model_type: str = "reinforce_policy",
    n_episodes: int = 5,
    action_mode: str = "greedy",
    seed_base: int = 11_112,
    show_inline: bool = True,
    summary_path: str | Path | None = None,
    hidden_size: int = 128,
) -> dict[str, Any]:
    """Esegue episodi visuali CartPole riproducibili senza aggiornare il modello.

    Argomenti:
        checkpoint_path: Checkpoint della policy REINFORCE selezionata.
        model_type: Tipo esplicito di architettura. La funzione accetta
            `reinforce_policy`, usato anche dalla variante value-baseline.
        n_episodes: Numero di episodi visuali da eseguire.
        action_mode: `greedy` oppure `sample`.
        seed_base: Seed del generatore che produce seed distinti per gli episodi.
        show_inline: Se True, mostra localmente ogni animazione nel notebook.
        summary_path: JSON leggero opzionale in cui salvare solo le statistiche.
        hidden_size: Ampiezza dei layer nascosti della policy salvata.

    Restituisce:
        Dizionario con checkpoint, modalita', seed e statistiche per episodio.
        I frame non vengono inclusi nel valore restituito o nel JSON.

    Eccezioni:
        FileNotFoundError: Se il checkpoint non esiste.
        ValueError: Se modello, modalita' o numero di episodi non sono validi.
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint CartPole non trovato: {checkpoint}")
    if model_type != "reinforce_policy":
        raise ValueError("model_type deve essere 'reinforce_policy'")
    if action_mode not in {"sample", "greedy"}:
        raise ValueError("action_mode deve essere 'sample' o 'greedy'")
    if n_episodes <= 0:
        raise ValueError("n_episodes deve essere positivo")

    rng = np.random.default_rng(seed_base)
    episode_seeds = rng.choice(1_000_000, size=n_episodes, replace=False).tolist()
    env = make_env("CartPole-v1", seed=int(episode_seeds[0]), render_mode="rgb_array")

    try:
        policy = policy_from_env(env, hidden_size=hidden_size)
        payload = torch.load(checkpoint, map_location="cpu")
        state_dict = (
            payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        )
        policy.load_state_dict(state_dict)
        policy.eval()
        obs_scale = observation_scale("CartPole-v1")
        visual_results: list[dict[str, Any]] = []

        for episode_index, episode_seed in enumerate(episode_seeds, start=1):
            obs, _ = env.reset(seed=int(episode_seed))
            frames: list[Any] = []
            total_reward = 0.0
            terminated = False
            truncated = False
            length = 0

            with torch.inference_mode():
                for _ in range(500):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    obs_t = preprocess_observation(obs, obs_scale)
                    action, _ = select_action(policy, obs_t, mode=action_mode)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    length += 1
                    if terminated or truncated:
                        final_frame = env.render()
                        if final_frame is not None:
                            frames.append(final_frame)
                        break

            row = {
                "episode": episode_index,
                "seed": int(episode_seed),
                "return": total_reward,
                "length": length,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "action_mode": action_mode,
                "model_type": model_type,
                "checkpoint": checkpoint.name,
                "frame_count": len(frames),
                "frame_shape": list(frames[0].shape) if frames else None,
            }
            visual_results.append(row)
            print(
                f"episode={episode_index} seed={episode_seed} return={total_reward:.0f} "
                f"length={length} terminated={terminated} truncated={truncated}"
            )
            if show_inline:
                print(f"Visual episode {episode_index}/{n_episodes}")
                show_frames(frames, interval=30, max_display_frames=250)

        summary = {
            "model_type": model_type,
            "checkpoint": checkpoint.name,
            "action_mode": action_mode,
            "seed_base": seed_base,
            "episode_seeds": [int(seed) for seed in episode_seeds],
            "n_episodes": n_episodes,
            "episodes": visual_results,
            "average_return": float(np.mean([row["return"] for row in visual_results])),
            "average_length": float(np.mean([row["length"] for row in visual_results])),
        }
        if summary_path is not None:
            output_path = Path(summary_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return summary
    finally:
        env.close()


def run_lunar_visual_episodes(
    checkpoint_path: str | Path,
    temperature: float,
    mode: str = "sample",
    n_episodes: int = 5,
    seed_start: int = 100_000,
    show_inline: bool = True,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Esegue e mostra episodi LunarLander senza includere frame nel JSON.

    Argomenti:
        checkpoint_path: Percorso del checkpoint A2C selezionato.
        temperature: Temperatura usata nel campionamento delle azioni.
        mode: Modalità di selezione, `sample` oppure `greedy`.
        n_episodes: Numero di episodi visuali da eseguire.
        seed_start: Primo seed della sequenza deterministica degli episodi.
        show_inline: Se True, mostra ogni animazione nel notebook.
        summary_path: Percorso JSON opzionale per le sole statistiche leggere.

    Operazione:
        Ogni episodio usa il seed `seed_start + indice`, registra terminazione e
        troncamento, ed esegue la policy in modalità di inferenza. Il checkpoint
        non viene modificato.

    Restituisce:
        Dizionario con configurazione, seed, return e lunghezze degli episodi.
        I frame non sono inclusi nel valore restituito o nell'eventuale JSON.

    Eccezioni:
        FileNotFoundError: Se il checkpoint non esiste.
        ValueError: Se modalità, temperatura o numero di episodi non sono validi.
    """

    if mode not in {"sample", "greedy"}:
        raise ValueError("mode deve essere 'sample' oppure 'greedy'")
    if n_episodes <= 0:
        raise ValueError("n_episodes deve essere positivo")
    if mode == "sample" and temperature <= 0:
        raise ValueError("temperature deve essere positiva in modalita' sample")

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint LunarLander non trovato: {checkpoint}")
    env = make_env(
        "LunarLander-v3",
        seed=seed_start,
        render_mode="rgb_array",
        continuous=False,
        enable_wind=False,
    )
    try:
        net = lunar_a2c_from_env(env, hidden_size=64)
        load_a2c_checkpoint(net, checkpoint)
        visual_results: list[dict[str, Any]] = []

        for ep_idx in range(n_episodes):
            episode_seed = seed_start + ep_idx
            torch.manual_seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            actions: list[int] = []
            frames: list[Any] = []
            terminated = False
            truncated = False

            with torch.inference_mode():
                for _ in range(1000):
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    logits, _ = net.get_logits_and_value(obs_t)
                    if mode == "greedy":
                        action = int(torch.argmax(logits).item())
                    else:
                        dist = torch.distributions.Categorical(logits=logits / temperature)
                        action = int(dist.sample().item())

                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += float(reward)
                    actions.append(action)
                    if terminated or truncated:
                        final_frame = env.render()
                        if final_frame is not None:
                            frames.append(final_frame)
                        break

            action_counts = np.bincount(actions, minlength=4)
            action_freq = (action_counts / action_counts.sum()).tolist()
            row = {
                "episode": ep_idx + 1,
                "seed": episode_seed,
                "return": total_reward,
                "length": len(actions),
                "success": total_reward >= 200.0,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "mode": mode,
                "temperature": temperature,
                "checkpoint": checkpoint.name,
                "action_freq": action_freq,
                "frame_count": len(frames),
                "frame_shape": list(frames[0].shape) if frames else None,
            }
            visual_results.append(row)
            print(
                f"episode={row['episode']} seed={episode_seed} return={row['return']:.2f} "
                f"length={row['length']} terminated={terminated} truncated={truncated} "
                f"success={row['success']}"
            )
            if show_inline:
                print(f"\nVisual episode {ep_idx + 1}/{n_episodes}")
                show_frames(frames, interval=30, max_display_frames=300)

        summary = {
            "checkpoint": checkpoint.name,
            "mode": mode,
            "temperature": temperature,
            "seed_start": seed_start,
            "n_episodes": n_episodes,
            "episodes": visual_results,
            "average_return": float(np.mean([row["return"] for row in visual_results])),
            "average_length": float(np.mean([row["length"] for row in visual_results])),
        }
        if summary_path is None:
            save_json_artifact("a2c_lunarlander_visual_episodes.json", summary)
        else:
            output_path = Path(summary_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return summary
    finally:
        env.close()
