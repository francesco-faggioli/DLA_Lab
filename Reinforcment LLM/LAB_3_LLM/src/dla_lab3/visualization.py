from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from matplotlib import animation

from .a2c import load_a2c_checkpoint, lunar_a2c_from_env
from .envs import make_env
from .experiments import save_json_artifact


def show_frames(frames: list[Any], interval: int = 30, max_display_frames: int = 300) -> None:
    """Display environment RGB frames inside a notebook.

    Args:
        frames: RGB arrays returned by Gymnasium `render_mode="rgb_array"`.
        interval: Delay between frames in milliseconds.
        max_display_frames: Maximum number of frames kept in the animation.

    What it does:
        Downsamples long episodes if needed and renders an HTML animation in the
        current notebook cell.

    Outputs:
        None. Displays an animation.
    """

    if not frames:
        print("No frames collected.")
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


def run_lunar_visual_episodes(
    checkpoint_path: str | Path,
    temperature: float,
    mode: str = "sample",
    n_episodes: int = 5,
    seed_start: int = 100_000,
    show_inline: bool = True,
) -> dict[str, Any]:
    """Run and optionally display LunarLander visual episodes.

    Args:
        checkpoint_path: Selected A2C checkpoint.
        temperature: Stochastic policy temperature.
        mode: `sample` for stochastic actions or `greedy` for argmax actions.
        n_episodes: Number of visual episodes to run.
        seed_start: First episode seed.
        show_inline: If True, display every episode as an inline animation.

    What it does:
        Runs the selected policy for `n_episodes` episodes using
        `render_mode="rgb_array"`, prints each return/length and stores the
        visual statistics as an artifact.

    Outputs:
        Dictionary with episode statistics and collected frame lists.
    """

    env = make_env(
        "LunarLander-v3",
        seed=seed_start,
        render_mode="rgb_array",
        continuous=False,
        enable_wind=False,
    )
    net = lunar_a2c_from_env(env, hidden_size=64)
    load_a2c_checkpoint(net, checkpoint_path)

    visual_results = []
    all_frames = []

    if mode not in {"sample", "greedy"}:
        raise ValueError("mode must be 'sample' or 'greedy'")

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep_idx)
        total_reward = 0.0
        actions = []
        frames = []

        with torch.no_grad():
            for _ in range(1000):
                frames.append(env.render())

                obs_t = torch.tensor(obs, dtype=torch.float32) / torch.ones(8, dtype=torch.float32)
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
                    frames.append(env.render())
                    break

        action_counts = np.bincount(actions, minlength=4)
        action_freq = (action_counts / action_counts.sum()).tolist()
        row = {
            "episode": ep_idx + 1,
            "return": total_reward,
            "length": len(actions),
            "success": total_reward >= 200.0,
            "mode": mode,
            "temperature": temperature,
            "action_freq": action_freq,
        }
        visual_results.append(row)
        all_frames.append(frames)

        print(
            f"episode={row['episode']} return={row['return']:.2f} "
            f"length={row['length']} success={row['success']}"
        )

    env.close()
    save_json_artifact("a2c_lunarlander_visual_episodes.json", visual_results)

    if show_inline:
        for idx, frames in enumerate(all_frames, start=1):
            print(f"\nVisual episode {idx}/{n_episodes}")
            show_frames(frames, interval=30, max_display_frames=300)

    return {"results": visual_results, "frames": all_frames}
