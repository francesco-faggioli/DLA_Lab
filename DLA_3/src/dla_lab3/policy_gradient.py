from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


BaselineMode = Literal["none", "standardize"]
ActionMode = Literal["sample", "greedy"]


@dataclass
class Episode:
    """Container returned by one environment rollout.

    Args:
        observations: Normalized observations visited during the episode.
        actions: Integer actions selected by the policy.
        log_probs: Log probabilities of the selected actions.
        rewards: Raw rewards returned by the environment.
        total_reward: Sum of raw rewards in the episode.
        length: Number of environment steps in the episode.

    What it does:
        Stores all data needed by REINFORCE plus human-readable episode
        statistics.

    Outputs:
        Dataclass instance used by training and evaluation functions.
    """

    observations: list[torch.Tensor]
    actions: list[int]
    log_probs: torch.Tensor
    rewards: list[float]
    total_reward: float
    length: int


@dataclass
class ReinforceConfig:
    """Training configuration for REINFORCE experiments.

    Args:
        gamma: Discount factor for Monte Carlo returns.
        lr_policy: Adam learning rate for the policy network.
        lr_value: Adam learning rate for the value network when used.
        num_episodes: Number of training episodes.
        max_episode_steps: Safety cap for each episode.
        eval_every: Evaluate every N training episodes.
        eval_episodes: Run M evaluation episodes at each evaluation point.
        baseline_mode: `none` or `standardize` for vanilla REINFORCE.
        normalize_advantage: Normalize advantages in value-baseline training.
        entropy_coef: Entropy bonus coefficient for exploration.
        grad_clip: Maximum gradient norm.
        checkpoint_path: Optional path used to save the best evaluated policy.
        save_best: If True, save the best evaluated policy.

    What it does:
        Keeps hyperparameters explicit and reproducible across notebook cells.

    Outputs:
        Dataclass instance passed to training functions.
    """

    gamma: float = 0.99
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    num_episodes: int = 1000
    max_episode_steps: int = 1000
    eval_every: int = 50
    eval_episodes: int = 20
    baseline_mode: BaselineMode = "standardize"
    normalize_advantage: bool = True
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    checkpoint_path: str | None = None
    save_best: bool = True


class PolicyNet(nn.Module):
    """MLP stochastic policy for continuous observations and discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        """Create the policy network.

        Args:
            obs_dim: Number of observation features.
            n_actions: Number of discrete actions.
            hidden_size: Width of the hidden layers.

        What it does:
            Builds a two-layer MLP and maps observations to action logits.

        Outputs:
            A PyTorch module. `forward` returns logits; `action_probs` returns
            normalized action probabilities.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            obs: Normalized observation tensor. Shape can be `(obs_dim,)` or
                `(batch, obs_dim)`.

        What it does:
            Runs the observation through the MLP policy.

        Outputs:
            Raw action logits. A categorical distribution can be built directly
            from these logits.
        """

        return self.net(obs)

    def action_probs(self, obs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute normalized action probabilities.

        Args:
            obs: Normalized observation tensor.
            temperature: Softmax temperature; lower values make the policy more
                deterministic.

        What it does:
            Converts logits into a probability distribution over actions.

        Outputs:
            Tensor of action probabilities with the same leading shape as
            `obs`.
        """

        logits = self.forward(obs) / temperature
        return F.softmax(logits, dim=-1)


class ValueNet(nn.Module):
    """MLP state-value baseline network."""

    def __init__(self, obs_dim: int, hidden_size: int = 128):
        """Create the value network.

        Args:
            obs_dim: Number of observation features.
            hidden_size: Width of the hidden layers.

        What it does:
            Builds a two-layer MLP that estimates the scalar value V(s).

        Outputs:
            A PyTorch module whose `forward` returns one value per observation.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s).

        Args:
            obs: Normalized observation tensor. Shape can be `(obs_dim,)` or
                `(batch, obs_dim)`.

        What it does:
            Runs observations through the value MLP and removes the last
            singleton dimension.

        Outputs:
            Scalar value for one observation or a vector of values for a batch.
        """

        return self.net(obs).squeeze(-1)


def policy_from_env(env, hidden_size: int = 128) -> PolicyNet:
    """Build a policy network from a Gymnasium environment.

    Args:
        env: Environment with Box observation space and Discrete action space.
        hidden_size: Width of the hidden layers.

    What it does:
        Reads observation and action dimensions from the environment.

    Outputs:
        PolicyNet configured for the environment.
    """

    return PolicyNet(env.observation_space.shape[0], env.action_space.n, hidden_size)


def value_from_env(env, hidden_size: int = 128) -> ValueNet:
    """Build a value network from a Gymnasium environment.

    Args:
        env: Environment with Box observation space.
        hidden_size: Width of the hidden layers.

    What it does:
        Reads the observation dimension from the environment.

    Outputs:
        ValueNet configured for the environment.
    """

    return ValueNet(env.observation_space.shape[0], hidden_size)


def preprocess_observation(obs, obs_scale: torch.Tensor) -> torch.Tensor:
    """Convert one observation to a normalized float tensor.

    Args:
        obs: Raw observation returned by Gymnasium.
        obs_scale: Per-feature scale tensor.

    What it does:
        Converts the observation to float32 and divides by the scale tensor.

    Outputs:
        Normalized observation tensor.
    """

    return torch.as_tensor(obs, dtype=torch.float32) / obs_scale


def select_action(
    policy: PolicyNet,
    obs: torch.Tensor,
    mode: ActionMode = "sample",
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor]:
    """Select an action from the current policy.

    Args:
        policy: Policy network.
        obs: Preprocessed observation tensor.
        mode: `sample` for stochastic training, `greedy` for evaluation.
        temperature: Softmax temperature. Lower values make the distribution
            sharper.

    What it does:
        Builds a categorical distribution from policy logits and either samples
        an action or picks the most probable action.

    Outputs:
        Tuple `(action, log_prob)` where `action` is an int and `log_prob` is a
        tensor used by the policy-gradient update.
    """

    logits = policy(obs) / temperature
    dist = Categorical(logits=logits)
    if mode == "greedy":
        action = torch.argmax(logits, dim=-1)
    else:
        action = dist.sample()
    return int(action.item()), dist.log_prob(action).reshape(1)


def run_episode(
    env,
    policy: PolicyNet,
    obs_scale: torch.Tensor,
    max_steps: int = 1000,
    mode: ActionMode = "sample",
    temperature: float = 1.0,
    seed: int | None = None,
) -> Episode:
    """Run one complete episode.

    Args:
        env: Gymnasium environment.
        policy: Policy network used to select actions.
        obs_scale: Per-feature scale tensor for observation normalization.
        max_steps: Safety cap on episode length.
        mode: `sample` for training or stochastic evaluation, `greedy` for
            deterministic evaluation.
        temperature: Policy temperature used by the categorical distribution.
        seed: Optional seed passed to `env.reset`.

    What it does:
        Interacts with the environment until termination, truncation or
        `max_steps`, collecting observations, actions, log probabilities and
        rewards.

    Outputs:
        Episode dataclass with rollout tensors and summary statistics.
    """

    observations: list[torch.Tensor] = []
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []

    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

    for _ in range(max_steps):
        obs_t = preprocess_observation(obs, obs_scale)
        action, log_prob = select_action(policy, obs_t, mode=mode, temperature=temperature)

        observations.append(obs_t)
        actions.append(action)
        log_probs.append(log_prob)

        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))

        if terminated or truncated:
            break

    return Episode(
        observations=observations,
        actions=actions,
        log_probs=torch.cat(log_probs),
        rewards=rewards,
        total_reward=float(np.sum(rewards)),
        length=len(rewards),
    )


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute discounted Monte Carlo returns.

    Args:
        rewards: Reward sequence from one episode.
        gamma: Discount factor.

    What it does:
        Computes `G_t = r_t + gamma r_{t+1} + ...` backward through the episode.

    Outputs:
        Float tensor with one discounted return per timestep.
    """

    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def prepare_policy_target(returns: torch.Tensor, baseline_mode: BaselineMode) -> torch.Tensor:
    """Apply the optional episode-level baseline to returns.

    Args:
        returns: Discounted returns for one episode.
        baseline_mode: `none` keeps raw returns; `standardize` subtracts the
            episode mean and divides by the episode standard deviation.

    What it does:
        Implements the simple baseline discussed in Exercise 2.

    Outputs:
        Tensor used as policy-gradient target.
    """

    if baseline_mode == "none":
        return returns
    if baseline_mode == "standardize":
        std = returns.std()
        if len(returns) > 1 and std > 1e-8:
            return (returns - returns.mean()) / (std + 1e-8)
        return returns - returns.mean()
    raise ValueError(f"Unknown baseline mode: {baseline_mode}")


def evaluate_policy(
    env,
    policy: PolicyNet,
    obs_scale: torch.Tensor,
    episodes: int = 20,
    max_steps: int = 1000,
    mode: ActionMode = "greedy",
    temperature: float = 1.0,
    seed_start: int | None = None,
) -> dict:
    """Evaluate a policy over multiple episodes.

    Args:
        env: Gymnasium environment.
        policy: Policy network.
        obs_scale: Per-feature scale tensor.
        episodes: Number of evaluation episodes.
        max_steps: Safety cap on each episode.
        mode: `greedy` or `sample`.
        temperature: Temperature used if actions are sampled.
        seed_start: Optional first seed. Episode `i` uses `seed_start + i`.

    What it does:
        Runs independent episodes without gradient tracking and summarizes total
        rewards and episode lengths.

    Outputs:
        Dictionary with mean/std/min/max return, mean length and raw arrays.
    """

    was_training = policy.training
    policy.eval()

    returns = []
    lengths = []
    with torch.no_grad():
        for idx in range(episodes):
            ep_seed = None if seed_start is None else seed_start + idx
            episode = run_episode(
                env,
                policy,
                obs_scale,
                max_steps=max_steps,
                mode=mode,
                temperature=temperature,
                seed=ep_seed,
            )
            returns.append(episode.total_reward)
            lengths.append(episode.length)

    if was_training:
        policy.train()

    returns_np = np.asarray(returns, dtype=np.float32)
    lengths_np = np.asarray(lengths, dtype=np.float32)
    return {
        "avg_return": float(returns_np.mean()),
        "std_return": float(returns_np.std()),
        "min_return": float(returns_np.min()),
        "max_return": float(returns_np.max()),
        "avg_length": float(lengths_np.mean()),
        "returns": returns,
        "lengths": lengths,
    }


def _checkpoint_payload(policy: PolicyNet, config: ReinforceConfig, metrics: dict) -> dict:
    return {
        "model_state_dict": policy.state_dict(),
        "config": asdict(config),
        "metrics": metrics,
    }


def reinforce(
    policy: PolicyNet,
    env,
    eval_env,
    obs_scale: torch.Tensor,
    config: ReinforceConfig,
) -> dict:
    """Train a policy with vanilla REINFORCE.

    Args:
        policy: Policy network to train.
        env: Training environment.
        eval_env: Environment used for periodic evaluation.
        obs_scale: Per-feature scale tensor.
        config: ReinforceConfig with hyperparameters and logging settings.

    What it does:
        Runs one sampled episode per update, computes Monte Carlo returns,
        applies the optional standardization baseline, updates the policy with
        Adam, and periodically evaluates the greedy policy.

    Outputs:
        Dictionary containing episode returns, losses, evaluation metrics and
        best checkpoint information.
    """

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)
    best_eval_return = -float("inf")

    history = {
        "episode_returns": [],
        "episode_lengths": [],
        "policy_losses": [],
        "eval_episodes": [],
        "eval_avg_returns": [],
        "eval_std_returns": [],
        "eval_avg_lengths": [],
        "best_eval_return": best_eval_return,
        "checkpoint_path": config.checkpoint_path,
    }

    policy.train()
    for episode_idx in range(config.num_episodes):
        episode = run_episode(
            env,
            policy,
            obs_scale,
            max_steps=config.max_episode_steps,
            mode="sample",
        )

        returns = compute_returns(episode.rewards, config.gamma)
        target = prepare_policy_target(returns, config.baseline_mode)

        logits = policy(torch.stack(episode.observations))
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()

        loss = -(episode.log_probs * target.detach()).mean() - config.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        optimizer.step()

        history["episode_returns"].append(episode.total_reward)
        history["episode_lengths"].append(episode.length)
        history["policy_losses"].append(float(loss.item()))

        if episode_idx % config.eval_every == 0:
            metrics = evaluate_policy(
                eval_env,
                policy,
                obs_scale,
                episodes=config.eval_episodes,
                max_steps=config.max_episode_steps,
                mode="greedy",
            )
            history["eval_episodes"].append(episode_idx)
            history["eval_avg_returns"].append(metrics["avg_return"])
            history["eval_std_returns"].append(metrics["std_return"])
            history["eval_avg_lengths"].append(metrics["avg_length"])

            if metrics["avg_return"] > best_eval_return:
                best_eval_return = metrics["avg_return"]
                history["best_eval_return"] = best_eval_return
                if config.save_best and config.checkpoint_path is not None:
                    path = Path(config.checkpoint_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(_checkpoint_payload(policy, config, metrics), path)

            print(
                f"episode={episode_idx:5d} "
                f"train_return={episode.total_reward:8.2f} "
                f"eval_return={metrics['avg_return']:8.2f} "
                f"eval_length={metrics['avg_length']:7.1f}"
            )

    policy.eval()
    return history


def reinforce_with_value_baseline(
    policy: PolicyNet,
    value_net: ValueNet,
    env,
    eval_env,
    obs_scale: torch.Tensor,
    config: ReinforceConfig,
) -> dict:
    """Train REINFORCE with a learned state-value baseline.

    Args:
        policy: Policy network to train.
        value_net: Value network trained to estimate discounted returns.
        env: Training environment.
        eval_env: Environment used for periodic evaluation.
        obs_scale: Per-feature scale tensor.
        config: ReinforceConfig with hyperparameters and logging settings.

    What it does:
        Uses `G_t - V(s_t)` as the policy advantage and trains the value network
        with mean squared error against Monte Carlo returns.

    Outputs:
        Dictionary containing training curves, evaluation metrics and checkpoint
        information.
    """

    opt_policy = torch.optim.Adam(policy.parameters(), lr=config.lr_policy)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=config.lr_value)
    best_eval_return = -float("inf")

    history = {
        "episode_returns": [],
        "episode_lengths": [],
        "policy_losses": [],
        "value_losses": [],
        "eval_episodes": [],
        "eval_avg_returns": [],
        "eval_std_returns": [],
        "eval_avg_lengths": [],
        "best_eval_return": best_eval_return,
        "checkpoint_path": config.checkpoint_path,
    }

    policy.train()
    value_net.train()

    for episode_idx in range(config.num_episodes):
        episode = run_episode(
            env,
            policy,
            obs_scale,
            max_steps=config.max_episode_steps,
            mode="sample",
        )

        obs_tensor = torch.stack(episode.observations)
        returns = compute_returns(episode.rewards, config.gamma)
        values = value_net(obs_tensor)
        advantage = returns - values.detach()

        if config.normalize_advantage and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        logits = policy(obs_tensor)
        dist = Categorical(logits=logits)
        entropy = dist.entropy().mean()

        policy_loss = -(episode.log_probs * advantage.detach()).mean() - config.entropy_coef * entropy
        value_loss = F.mse_loss(values, returns)

        opt_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
        opt_policy.step()

        opt_value.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.grad_clip)
        opt_value.step()

        history["episode_returns"].append(episode.total_reward)
        history["episode_lengths"].append(episode.length)
        history["policy_losses"].append(float(policy_loss.item()))
        history["value_losses"].append(float(value_loss.item()))

        if episode_idx % config.eval_every == 0:
            metrics = evaluate_policy(
                eval_env,
                policy,
                obs_scale,
                episodes=config.eval_episodes,
                max_steps=config.max_episode_steps,
                mode="greedy",
            )
            history["eval_episodes"].append(episode_idx)
            history["eval_avg_returns"].append(metrics["avg_return"])
            history["eval_std_returns"].append(metrics["std_return"])
            history["eval_avg_lengths"].append(metrics["avg_length"])

            if metrics["avg_return"] > best_eval_return:
                best_eval_return = metrics["avg_return"]
                history["best_eval_return"] = best_eval_return
                if config.save_best and config.checkpoint_path is not None:
                    path = Path(config.checkpoint_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(_checkpoint_payload(policy, config, metrics), path)

            print(
                f"episode={episode_idx:5d} "
                f"train_return={episode.total_reward:8.2f} "
                f"eval_return={metrics['avg_return']:8.2f} "
                f"eval_length={metrics['avg_length']:7.1f}"
            )

    policy.eval()
    value_net.eval()
    return history
