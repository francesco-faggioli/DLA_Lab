from __future__ import annotations

import gymnasium as gym
import torch


def make_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs,
) -> gym.Env:
    """Create and seed a Gymnasium environment.

    Args:
        env_id: Gymnasium environment id, for example `CartPole-v1` or
            `LunarLander-v3`.
        seed: Optional seed used for `reset` and for the action space.
        render_mode: Optional render mode, for example `human`.
        **kwargs: Extra keyword arguments passed to `gym.make`.

    What it does:
        Instantiates the environment and applies the seed consistently.

    Outputs:
        A ready-to-use Gymnasium environment.
    """

    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def observation_scale(env_id: str) -> torch.Tensor:
    """Return a simple observation scale for supported environments.

    Args:
        env_id: Environment id or name containing `CartPole` or `LunarLander`.

    What it does:
        Provides fixed scale values used to normalize observations before they
        are passed to the neural networks. These values come from the official
        Gymnasium observation-space ranges, with finite practical values for
        unbounded CartPole velocities.

    Outputs:
        Float tensor with one scale value per observation dimension.
    """

    if "CartPole" in env_id:
        return torch.tensor([4.8, 5.0, 0.41887903, 5.0], dtype=torch.float32)
    if "LunarLander" in env_id:
        return torch.tensor(
            [2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0],
            dtype=torch.float32,
        )
    raise ValueError(f"No observation scale configured for environment: {env_id}")
