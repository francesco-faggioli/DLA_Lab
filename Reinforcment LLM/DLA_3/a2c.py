import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# OBS SCALE HELPERS
# ============================================================

def cartpole_obs_scale():
    return torch.tensor([4.8, 5.0, 0.418, 5.0], dtype=torch.float32)


def lunarlander_obs_scale():
    return torch.tensor([2.5, 2.5, 10.0, 10.0, 6.2832, 10.0, 1.0, 1.0], dtype=torch.float32)

def lunarlander_obs_scale_raw():
    return torch.ones(8, dtype=torch.float32)


# ============================================================
# NETWORK
# ============================================================

class A2CNet(nn.Module):
    """
    Shared-trunk Actor-Critic network.
    forward() returns (action_probs, state_value).
    """

    def __init__(self, env):
        super().__init__()

        n_obs     = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.trunk = nn.Sequential(
            nn.Linear(n_obs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor_head  = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

    def forward(self, s, temperature=1.0):
        features = self.trunk(s)
        probs    = F.softmax(self.actor_head(features) / temperature, dim=-1)
        value    = self.critic_head(features).squeeze(-1)
        return probs, value
    
    def get_logits_and_value(self, s):
        features = self.trunk(s)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_probs(self, s, temperature=1.0):
        probs, _ = self.forward(s, temperature=temperature)
        return probs

    def get_value(self, s):
        _, value = self.forward(s)
        return value


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class A2CNetLunarStable(nn.Module):
    """
    Actor-Critic network for LunarLander with orthogonal initialization.
    Kept separate from A2CNet to preserve the previous experiments.
    """

    def __init__(self, env):
        super().__init__()

        n_obs = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.trunk = nn.Sequential(
            layer_init(nn.Linear(n_obs, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        self.actor_head = layer_init(nn.Linear(128, n_actions), std=0.01)
        self.critic_head = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, s, temperature=1.0):
        features = self.trunk(s)
        logits = self.actor_head(features)
        probs = F.softmax(logits / temperature, dim=-1)
        value = self.critic_head(features).squeeze(-1)
        return probs, value

    def get_logits_and_value(self, s):
        features = self.trunk(s)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_probs(self, s, temperature=1.0):
        probs, _ = self.forward(s, temperature=temperature)
        return probs

    def get_value(self, s):
        _, value = self.forward(s)
        return value

class A2CNetLunarSeparate(nn.Module):
    """
    Actor-Critic network for LunarLander with separate actor and critic networks.
    This is closer to common A2C implementations, where policy and value
    functions do not share all hidden layers.
    """

    def __init__(self, env):
        super().__init__()

        n_obs = env.observation_space.shape[0]
        n_actions = env.action_space.n

        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, s, temperature=1.0):
        logits = self.actor(s)
        probs = F.softmax(logits / temperature, dim=-1)
        value = self.critic(s).squeeze(-1)
        return probs, value

    def get_logits_and_value(self, s):
        logits = self.actor(s)
        value = self.critic(s).squeeze(-1)
        return logits, value

    def get_action_probs(self, s, temperature=1.0):
        probs, _ = self.forward(s, temperature=temperature)
        return probs

    def get_value(self, s):
        return self.critic(s).squeeze(-1)

# ============================================================
# EPISODE RUNNERS
# ============================================================

def _preprocess(obs, obs_scale):
    return torch.tensor(obs, dtype=torch.float32) / obs_scale


def run_episode(env, net, obs_scale,
                reward_scale=1.0, clip_rewards=False, maxlen=1000):
    """
    Run one stochastic episode.

    Returns
    -------
    observations : list[Tensor]
    actions      : list[int]
    rewards      : Tensor
    next_obs     : np.ndarray
    terminated   : bool
    truncated    : bool
    """
    observations = []
    actions = []
    rewards_list = []

    obs, _ = env.reset()

    terminated = False
    truncated = False

    for _ in range(maxlen):
        obs_t = _preprocess(obs, obs_scale)
        probs, _ = net(obs_t)
        dist = Categorical(probs)
        action = dist.sample()

        observations.append(obs_t)
        actions.append(action.item())

        obs, reward, terminated, truncated, _ = env.step(action.item())

        if clip_rewards:
            reward = float(np.clip(reward, -10.0, 10.0))

        rewards_list.append(reward / reward_scale)

        if terminated or truncated:
            break

    return (
        observations,
        actions,
        torch.tensor(rewards_list, dtype=torch.float32),
        obs,
        terminated,
        truncated,
    )


def run_episode_greedy(env, net, obs_scale, maxlen=1000):
    """
    Run one deterministic (greedy) episode — used for evaluation only.

    Returns
    -------
    observations, actions, rewards (raw, unscaled)
    """
    observations, actions, rewards = [], [], []

    obs, _ = env.reset()

    for _ in range(maxlen):
        obs_t  = _preprocess(obs, obs_scale)
        probs  = net.get_action_probs(obs_t, temperature=1e-8)
        action = probs.argmax().item()

        observations.append(obs_t)
        actions.append(action)

        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)

        if term or trunc:
            break

    return observations, actions, rewards


# ============================================================
# RETURN HELPERS
# ============================================================

def compute_mc_returns(rewards, gamma, bootstrap_value=0.0):
    """
    Monte Carlo returns with optional bootstrap (for truncated episodes).
    Coerente con compute_returns() di reinforce.py.
    """
    returns = torch.zeros_like(rewards, dtype=torch.float32)

    if torch.is_tensor(bootstrap_value):
        R = bootstrap_value.detach()
    else:
        R = torch.tensor(float(bootstrap_value), dtype=torch.float32, device=rewards.device)

    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R

    return returns


def compute_td0_targets(rewards, values, next_value, done, gamma):
    """
    One-step TD targets:
        target_t = r_t + gamma * V(s_{t+1})
    """
    next_values = torch.zeros_like(values)
    if len(values) > 1:
        next_values[:-1] = values[1:].detach()

    if done:
        next_values[-1] = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    else:
        next_values[-1] = next_value.detach()

    targets = rewards + gamma * next_values
    return targets


def compute_nstep_returns(rewards, values, gamma, n_step, next_value, done):
    T = len(rewards)

    returns = torch.zeros(T, device=values.device)

    for t in range(T):

        G = 0.0
        discount = 1.0

        for k in range(n_step):

            idx = t + k

            if idx >= T:
                break

            G += discount * rewards[idx]
            discount *= gamma

        bootstrap_idx = t + n_step

        if bootstrap_idx < T:
            G += discount * values[bootstrap_idx].detach()

        elif not done:
            G += discount * next_value.detach()

        returns[t] = G

    return returns


def compute_gae(rewards, values, next_value, done, gamma, gae_lambda=0.95):
    """
    Generalized Advantage Estimation (Schulman et al., 2016).

    GAE(γ,λ) is an exponentially-weighted average of k-step advantage
    estimators, trading off bias (low λ → TD(0)) vs variance (λ→1 → MC).
    The standard choice λ=0.95 with γ=0.99 works well for most envs.

        δ_t  = r_t + γ V(s_{t+1}) - V(s_t)          (TD residual)
        Â_t  = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

    Returns
    -------
    advantages : (T,) Tensor   — raw (un-normalised) GAE advantages
    returns    : (T,) Tensor   — V-targets = advantages + V(s_t)
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)

    # Build next-step values: V(s_1), V(s_2), ..., V(s_{T-1}), V(s_T)
    next_values = torch.zeros(T, dtype=torch.float32)
    if T > 1:
        next_values[:-1] = values[1:].detach()

    if done:
        next_values[-1] = 0.0
    else:
        next_values[-1] = next_value.detach()

    # TD residuals
    deltas = rewards + gamma * next_values - values.detach()

    # Backward pass: Â_t = δ_t + (γλ) Â_{t+1}
    gae = 0.0
    for t in reversed(range(T)):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values.detach()
    return advantages, returns


# ============================================================
# A2C TRAINING LOOP
# ============================================================

def a2c(net, env, obs_scale,
        gamma=0.99,
        lr=3e-4,
        lr_critic=1e-3,
        value_coef=0.25,
        entropy_coef=0.05,
        entropy_coef_min=0.001,
        num_episodes=1000,
        eval_every=50,
        eval_episodes=10,
        reward_scale=1.0,
        clip_rewards=False,
        solved_threshold=200.0,
        env_render=None,
        return_mode="td0",
        n_step=20,
        gae_lambda=0.95,
        optimizer="adam",
        env_name="a2c",
        checkpoint_path=None,
        normalize_advantage=True,
        grad_clip=0.5,
        verbose=True,
        stop_when_solved=False):

    # --------------------------------------------------------
    # 0. Basic setup
    # --------------------------------------------------------
    return_mode = return_mode.lower()
    optimizer = optimizer.lower()

    if return_mode not in ["mc", "td0", "nstep", "gae"]:
        raise ValueError("return_mode must be one of: 'mc', 'td0', 'nstep', 'gae'")

    if optimizer not in ["adam", "rmsprop"]:
        raise ValueError("optimizer must be either 'adam' or 'rmsprop'")

    if checkpoint_path is None:
        checkpoint_path = f"best_a2c_{env_name}.pt"

    # --------------------------------------------------------
    # 1. Optimizer
    # --------------------------------------------------------
    actor_params = list(net.trunk.parameters()) + list(net.actor_head.parameters())
    critic_params = list(net.critic_head.parameters())

    if optimizer == "rmsprop":
        opt = torch.optim.RMSprop(
            [
                {"params": actor_params, "lr": lr},
                {"params": critic_params, "lr": lr_critic},
            ],
            eps=1e-5,
            alpha=0.99
        )
    else:
        opt = torch.optim.Adam(
            [
                {"params": actor_params, "lr": lr},
                {"params": critic_params, "lr": lr_critic},
            ]
        )

    # --------------------------------------------------------
    # 2. Containers for statistics
    # --------------------------------------------------------
    solved = False
    best_avg_return = -float("inf")
    best_eval_return = -float("inf")

    running_rewards = [0.0]
    episode_returns = []
    episode_lengths = []
    eval_avg_returns = []
    eval_avg_lengths = []
    actor_losses = []
    critic_losses = []
    entropy_values = []

    net.train()

    # --------------------------------------------------------
    # 3. Training loop
    # --------------------------------------------------------
    for episode in range(num_episodes):

        # Linear entropy decay
        current_entropy_coef = max(
            entropy_coef_min,
            entropy_coef - (entropy_coef - entropy_coef_min) * episode / num_episodes
        )

        # ----------------------------------------------------
        # 3.1 Collect one stochastic episode
        # ----------------------------------------------------
        observations, actions, rewards, next_obs, terminated, truncated = run_episode(
            env,
            net,
            obs_scale,
            reward_scale=reward_scale,
            clip_rewards=clip_rewards,
        )

        done = terminated or truncated
        terminal_for_bootstrap = terminated

        T = len(rewards)
        episode_return = rewards.sum().item() * reward_scale

        episode_returns.append(episode_return)
        episode_lengths.append(T)

        # ----------------------------------------------------
        # 3.2 Fresh forward pass
        # ----------------------------------------------------
        obs_tensor = torch.stack(observations)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        action_probs, values = net(obs_tensor)

        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()

        # ----------------------------------------------------
        # 3.3 Compute returns / targets
        # ----------------------------------------------------
        with torch.no_grad():
            next_value = (
                values.new_tensor(0.0)
                if terminal_for_bootstrap
                else net.get_value(_preprocess(next_obs, obs_scale))
            )

        if return_mode == "mc":
            returns = compute_mc_returns(
                rewards,
                gamma,
                bootstrap_value=next_value
            )
            advantage = returns - values.detach()

        elif return_mode == "td0":
            returns = compute_td0_targets(
                rewards,
                values,
                next_value,
                terminal_for_bootstrap,
                gamma
            )
            advantage = returns - values.detach()

        elif return_mode == "nstep":
            returns = compute_nstep_returns(
                rewards,
                values,
                gamma,
                n_step,
                next_value,
                terminal_for_bootstrap
            )
            advantage = returns - values.detach()

        elif return_mode == "gae":
            advantage, returns = compute_gae(
                rewards,
                values,
                next_value,
                terminal_for_bootstrap,
                gamma,
                gae_lambda=gae_lambda
            )

        # ----------------------------------------------------
        # 3.4 Normalize advantage
        # ----------------------------------------------------
        if normalize_advantage and len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # ----------------------------------------------------
        # 3.5 Compute losses
        # ----------------------------------------------------
        actor_loss = -(log_probs * advantage).mean() - current_entropy_coef * entropy
        critic_loss = F.smooth_l1_loss(values, returns)
        loss = actor_loss + value_coef * critic_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropy_values.append(entropy.item())

        # ----------------------------------------------------
        # 3.6 Running reward
        # ----------------------------------------------------
        running_rewards.append(
            0.05 * episode_return + 0.95 * running_rewards[-1]
        )

        # ----------------------------------------------------
        # 3.7 Logging and checkpointing
        # ----------------------------------------------------
        if episode % 50 == 0:
            avg50 = np.mean(episode_returns[-50:])
            std50 = np.std(episode_returns[-50:])

            if avg50 > best_avg_return:
                best_avg_return = avg50

                if verbose:
                    print(
                        f"📈 New best TRAINING average: "
                        f"AvgReturn = {best_avg_return:.2f}"
                    )

            if verbose:
                print(
                    f"\nEpisode: {episode}"
                    f"\n  Return:            {episode_return:.2f}"
                    f"\n  Avg Return (±std): {avg50:.2f} ± {std50:.2f}"
                    f"\n  Episode Length:    {T}"
                    f"\n  Actor Loss:        {actor_loss.item():.4f}"
                    f"\n  Critic Loss:       {critic_loss.item():.4f}"
                    f"\n  Entropy:           {entropy.item():.4f}"
                    f"\n  Entropy coef:      {current_entropy_coef:.5f}"
                    f"\n  Running Reward:    {running_rewards[-1]:.2f}"
                )

        # ----------------------------------------------------
        # 3.8 Periodic greedy evaluation
        # ----------------------------------------------------
        if episode % eval_every == 0:
            net.eval()

            with torch.no_grad():
                eval_eps = [
                    run_episode_greedy(env, net, obs_scale)[2]
                    for _ in range(eval_episodes)
                ]

            net.train()

            avg_eval_return = np.mean([sum(r) for r in eval_eps])
            avg_eval_length = np.mean([len(r) for r in eval_eps])

            eval_avg_returns.append(avg_eval_return)
            eval_avg_lengths.append(avg_eval_length)

            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                torch.save(net.state_dict(), checkpoint_path)

                if verbose:
                    print(
                    f"💾 New best EVAL model saved to {checkpoint_path}! "
                    f"EvalReturn = {best_eval_return:.2f}"
                )

            if verbose:
                print(
                    f"  [EVAL ep.{episode}] "
                    f"AvgReturn: {avg_eval_return:.1f} | "
                    f"AvgLength: {avg_eval_length:.1f}"
                )

            if not solved and avg_eval_return >= solved_threshold:
                if verbose:
                    print(f"  ✅ SOLVED at episode {episode}!")
                solved = True

                if stop_when_solved:
                    break

        # ----------------------------------------------------
        # 3.9 Optional rendering
        # ----------------------------------------------------
        if env_render is not None and episode % 100 == 0:
            net.eval()
            run_episode(env_render, net, obs_scale)
            net.train()

    net.eval()

    return {
    "running_rewards": running_rewards,
    "episode_returns": episode_returns,
    "episode_lengths": episode_lengths,
    "actor_losses": actor_losses,
    "critic_losses": critic_losses,
    "entropy_values": entropy_values,
    "eval_avg_returns": eval_avg_returns,
    "eval_avg_lengths": eval_avg_lengths,
    "solved": solved,
    "best_avg_return": best_avg_return,
    "best_eval_return": best_eval_return,
    "checkpoint_path": checkpoint_path,
    "return_mode": return_mode,
    "env_name": env_name,
}

def a2c_rollout(net, env, obs_scale,
                gamma=0.99,
                lr=7e-4,
                lr_critic=7e-4,
                value_coef=0.5,
                entropy_coef=0.01,
                entropy_coef_min=0.0005,
                total_updates=2500,
                rollout_steps=64,
                eval_every=100,
                eval_episodes=5,
                reward_scale=100.0,
                clip_rewards=False,
                solved_threshold=200.0,
                return_mode="gae",
                gae_lambda=0.95,
                optimizer="rmsprop",
                env_name="lunarlander_rollout",
                checkpoint_path=None,
                normalize_advantage=True,
                grad_clip=0.5,
                verbose=True,
                stop_when_solved=True,
                eval_env=None):
    """
    Rollout-based Advantage Actor-Critic.

    Instead of waiting for the end of a full episode, this version collects
    short rollouts of fixed length and performs one update per rollout.
    This is more suitable for environments such as LunarLander, where episodes
    can last up to 1000 steps.
    """

    optimizer = optimizer.lower()

    if optimizer not in ["adam", "rmsprop"]:
        raise ValueError("optimizer must be either 'adam' or 'rmsprop'")

    if checkpoint_path is None:
        checkpoint_path = f"best_a2c_{env_name}.pt"

    if eval_env is None:
        eval_env = env

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    actor_params = list(net.trunk.parameters()) + list(net.actor_head.parameters())
    critic_params = list(net.critic_head.parameters())

    if optimizer == "rmsprop":
        opt = torch.optim.RMSprop(
            [
                {"params": actor_params, "lr": lr},
                {"params": critic_params, "lr": lr_critic},
            ],
            eps=1e-5,
            alpha=0.99
        )
    else:
        opt = torch.optim.Adam(
            [
                {"params": actor_params, "lr": lr},
                {"params": critic_params, "lr": lr_critic},
            ]
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    completed_returns = []
    completed_lengths = []

    actor_losses = []
    critic_losses = []
    entropy_values = []

    eval_avg_returns = []
    eval_avg_lengths = []

    best_eval_return = -float("inf")
    solved = False

    obs, _ = env.reset()
    current_ep_return = 0.0
    current_ep_length = 0

    net.train()

    # --------------------------------------------------------
    # Main rollout loop
    # --------------------------------------------------------
    for update in range(total_updates):

        current_entropy_coef = max(
            entropy_coef_min,
            entropy_coef - (entropy_coef - entropy_coef_min) * update / total_updates
        )

        obs_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        entropy_list = []
        masks_list = []

        # ----------------------------------------------------
        # Collect rollout
        # ----------------------------------------------------
        for _ in range(rollout_steps):

            obs_t = _preprocess(obs, obs_scale)

            probs, value = net(obs_t)
            dist = Categorical(probs)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())

            raw_reward = reward

            if clip_rewards:
                reward = float(np.clip(reward, -10.0, 10.0))

            scaled_reward = reward / reward_scale

            obs_list.append(obs_t)
            rewards_list.append(torch.tensor(scaled_reward, dtype=torch.float32))
            values_list.append(value)
            log_probs_list.append(dist.log_prob(action))
            entropy_list.append(dist.entropy())

            done = terminated or truncated

            # mask = 0 when episode ends, otherwise 1
            masks_list.append(torch.tensor(0.0 if done else 1.0, dtype=torch.float32))

            current_ep_return += raw_reward
            current_ep_length += 1

            if done:
                completed_returns.append(current_ep_return)
                completed_lengths.append(current_ep_length)

                obs, _ = env.reset()
                current_ep_return = 0.0
                current_ep_length = 0
            else:
                obs = next_obs

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------
        rewards = torch.stack(rewards_list)
        values = torch.stack(values_list).squeeze(-1)
        log_probs = torch.stack(log_probs_list)
        entropies = torch.stack(entropy_list)
        masks = torch.stack(masks_list)

        # ----------------------------------------------------
        # Bootstrap value
        # ----------------------------------------------------
        with torch.no_grad():
            next_value = net.get_value(_preprocess(obs, obs_scale)).detach()

        # ----------------------------------------------------
        # GAE over rollout
        # ----------------------------------------------------
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = torch.tensor(0.0, dtype=torch.float32)

        for t in reversed(range(rollout_steps)):

            if t == rollout_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].detach()

            delta = rewards[t] + gamma * next_val * masks[t] - values[t].detach()
            gae = delta + gamma * gae_lambda * masks[t] * gae

            advantages[t] = gae
            returns[t] = advantages[t] + values[t].detach()

        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------
        actor_loss = -(log_probs * advantages.detach()).mean()
        entropy_bonus = entropies.mean()
        actor_loss = actor_loss - current_entropy_coef * entropy_bonus

        critic_loss = F.smooth_l1_loss(values, returns)

        loss = actor_loss + value_coef * critic_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropy_values.append(entropy_bonus.item())

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        if update % 50 == 0:

            if len(completed_returns) > 0:
                avg50 = np.mean(completed_returns[-50:])
                std50 = np.std(completed_returns[-50:])
                avg_len50 = np.mean(completed_lengths[-50:])
            else:
                avg50 = current_ep_return
                std50 = 0.0
                avg_len50 = current_ep_length

            if verbose:
                print(
                    f"\nUpdate: {update}"
                    f"\n  Completed episodes: {len(completed_returns)}"
                    f"\n  Avg Return (±std):  {avg50:.2f} ± {std50:.2f}"
                    f"\n  Avg Length:         {avg_len50:.1f}"
                    f"\n  Actor Loss:         {actor_loss.item():.4f}"
                    f"\n  Critic Loss:        {critic_loss.item():.4f}"
                    f"\n  Entropy:            {entropy_bonus.item():.4f}"
                    f"\n  Entropy coef:       {current_entropy_coef:.5f}"
                )

        # ----------------------------------------------------
        # Periodic greedy evaluation
        # ----------------------------------------------------
        if update % eval_every == 0:

            net.eval()

            with torch.no_grad():
                eval_eps = [
                    run_episode_greedy(eval_env, net, obs_scale)[2]
                    for _ in range(eval_episodes)
                ]

            net.train()

            avg_eval_return = np.mean([sum(r) for r in eval_eps])
            avg_eval_length = np.mean([len(r) for r in eval_eps])

            eval_avg_returns.append(avg_eval_return)
            eval_avg_lengths.append(avg_eval_length)

            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                torch.save(net.state_dict(), checkpoint_path)

                if verbose:
                    print(
                        f"💾 New best EVAL model saved to {checkpoint_path}! "
                        f"EvalReturn = {best_eval_return:.2f}"
                    )

            if verbose:
                print(
                    f"  [EVAL update.{update}] "
                    f"AvgReturn: {avg_eval_return:.1f} | "
                    f"AvgLength: {avg_eval_length:.1f}"
                )

            if not solved and avg_eval_return >= solved_threshold:
                solved = True

                if verbose:
                    print(f"  ✅ SOLVED at update {update}!")

                if stop_when_solved:
                    break

    net.eval()

    return {
        "completed_returns": completed_returns,
        "completed_lengths": completed_lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_values": entropy_values,
        "eval_avg_returns": eval_avg_returns,
        "eval_avg_lengths": eval_avg_lengths,
        "best_eval_return": best_eval_return,
        "checkpoint_path": checkpoint_path,
        "solved": solved,
        "env_name": env_name,
    }

def a2c_vectorized(net, vec_env, eval_env, obs_scale,
                   gamma=0.995,
                   lr=8.3e-4,
                   value_coef=0.5,
                   entropy_coef=1e-5,
                   total_timesteps=200_000,
                   n_steps=5,
                   eval_every=250,
                   eval_episodes=10,
                   reward_scale=1.0,
                   clip_rewards=False,
                   solved_threshold=200.0,
                   gae_lambda=0.95,
                   optimizer="rmsprop",
                   checkpoint_path="best_a2c_lunarlander_vectorized.pt",
                   normalize_advantage=False,
                   grad_clip=0.5,
                   verbose=True,
                   stop_when_solved=True):
    """
    Vectorized rollout-based A2C.

    This version collects short rollouts from multiple parallel environments.
    It is closer to the standard A2C formulation used in reliable implementations
    such as Stable-Baselines3.

    Parameters
    ----------
    net : A2CNet
        Actor-Critic network.
    vec_env : gymnasium.vector.VectorEnv
        Vectorized training environment.
    eval_env : gymnasium.Env
        Single environment used for greedy evaluation.
    obs_scale : torch.Tensor
        Observation scaling tensor.
    gamma : float
        Discount factor.
    lr : float
        Learning rate.
    value_coef : float
        Weight of the critic loss.
    entropy_coef : float
        Entropy regularization coefficient.
    total_timesteps : int
        Total number of environment steps.
    n_steps : int
        Number of rollout steps per environment before each update.
    eval_every : int
        Evaluation frequency in number of updates.
    eval_episodes : int
        Number of greedy episodes for each evaluation.
    reward_scale : float
        Reward scaling factor.
    clip_rewards : bool
        Whether to clip rewards.
    solved_threshold : float
        Score threshold used to mark the environment as solved.
    gae_lambda : float
        Lambda parameter for GAE.
    optimizer : str
        "rmsprop" or "adam".
    checkpoint_path : str
        Path used to save the best model.
    normalize_advantage : bool
        Whether to normalize advantages.
    grad_clip : float
        Maximum gradient norm.
    verbose : bool
        Whether to print logs.
    stop_when_solved : bool
        Whether to stop training when the solved threshold is reached.

    Returns
    -------
    dict
        Training and evaluation statistics.
    """

    optimizer = optimizer.lower()

    if optimizer not in ["rmsprop", "adam"]:
        raise ValueError("optimizer must be either 'rmsprop' or 'adam'")

    n_envs = vec_env.num_envs
    total_updates = total_timesteps // (n_envs * n_steps)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    if optimizer == "rmsprop":
        opt = torch.optim.RMSprop(
            net.parameters(),
            lr=lr,
            eps=1e-5,
            alpha=0.99
        )
    else:
        opt = torch.optim.Adam(
            net.parameters(),
            lr=lr
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    completed_returns = []
    completed_lengths = []

    current_returns = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=np.int32)

    actor_losses = []
    critic_losses = []
    entropy_values = []

    eval_avg_returns = []
    eval_avg_lengths = []

    best_eval_return = -float("inf")
    solved = False

    # --------------------------------------------------------
    # Initial reset
    # --------------------------------------------------------
    obs, _ = vec_env.reset()

    net.train()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for update in range(total_updates):

        # Linear learning-rate decay, similar to "lin_0.00083"
        progress_remaining = 1.0 - (update / total_updates)
        current_lr = lr * progress_remaining

        for param_group in opt.param_groups:
            param_group["lr"] = current_lr

        obs_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        entropies_list = []
        masks_list = []

        # ----------------------------------------------------
        # Collect rollout: n_steps x n_envs
        # ----------------------------------------------------
        for _ in range(n_steps):

            obs_t = _preprocess(obs, obs_scale)

            probs, values = net(obs_t)
            dist = Categorical(probs)
            actions = dist.sample()

            next_obs, rewards, terminated, truncated, _ = vec_env.step(
                actions.cpu().numpy()
            )

            dones = np.logical_or(terminated, truncated)

            raw_rewards = rewards.copy()

            if clip_rewards:
                rewards = np.clip(rewards, -10.0, 10.0)

            rewards_scaled = rewards / reward_scale

            obs_list.append(obs_t)
            actions_list.append(actions)
            rewards_list.append(torch.tensor(rewards_scaled, dtype=torch.float32))
            values_list.append(values)
            log_probs_list.append(dist.log_prob(actions))
            entropies_list.append(dist.entropy())

            # In this simplified implementation, both terminated and truncated
            # episodes are treated as rollout boundaries.
            masks_list.append(
                torch.tensor(1.0 - dones.astype(np.float32), dtype=torch.float32)
            )

            current_returns += raw_rewards
            current_lengths += 1

            for env_idx, done in enumerate(dones):
                if done:
                    completed_returns.append(float(current_returns[env_idx]))
                    completed_lengths.append(int(current_lengths[env_idx]))

                    current_returns[env_idx] = 0.0
                    current_lengths[env_idx] = 0

            obs = next_obs

        # ----------------------------------------------------
        # Stack rollout tensors
        # Shape: [n_steps, n_envs]
        # ----------------------------------------------------
        rewards_tensor = torch.stack(rewards_list)
        values_tensor = torch.stack(values_list)
        log_probs_tensor = torch.stack(log_probs_list)
        entropies_tensor = torch.stack(entropies_list)
        masks_tensor = torch.stack(masks_list)

        # ----------------------------------------------------
        # Bootstrap value
        # ----------------------------------------------------
        with torch.no_grad():
            next_obs_t = _preprocess(obs, obs_scale)
            _, next_values = net(next_obs_t)
            next_values = next_values.detach()

        # ----------------------------------------------------
        # GAE computation
        # ----------------------------------------------------
        advantages = torch.zeros_like(rewards_tensor)
        last_gae = torch.zeros(n_envs, dtype=torch.float32)

        for step in reversed(range(n_steps)):

            if step == n_steps - 1:
                next_value = next_values
            else:
                next_value = values_tensor[step + 1].detach()

            delta = (
                rewards_tensor[step]
                + gamma * next_value * masks_tensor[step]
                - values_tensor[step].detach()
            )

            last_gae = (
                delta
                + gamma * gae_lambda * masks_tensor[step] * last_gae
            )

            advantages[step] = last_gae

        returns = advantages + values_tensor.detach()

        # ----------------------------------------------------
        # Flatten rollout
        # ----------------------------------------------------
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values_tensor.reshape(-1)
        log_probs_flat = log_probs_tensor.reshape(-1)
        entropies_flat = entropies_tensor.reshape(-1)

        if normalize_advantage and len(advantages_flat) > 1:
            advantages_flat = (
                advantages_flat - advantages_flat.mean()
            ) / (advantages_flat.std() + 1e-8)

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------
        actor_loss = -(log_probs_flat * advantages_flat.detach()).mean()
        entropy_bonus = entropies_flat.mean()
        actor_loss = actor_loss - entropy_coef * entropy_bonus

        critic_loss = F.smooth_l1_loss(values_flat, returns_flat)

        loss = actor_loss + value_coef * critic_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropy_values.append(entropy_bonus.item())

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        if update % 50 == 0:

            if len(completed_returns) > 0:
                avg50 = np.mean(completed_returns[-50:])
                std50 = np.std(completed_returns[-50:])
                avg_len50 = np.mean(completed_lengths[-50:])
            else:
                avg50 = 0.0
                std50 = 0.0
                avg_len50 = 0.0

            timesteps_done = (update + 1) * n_envs * n_steps

            if verbose:
                print(
                    f"\nUpdate: {update}"
                    f"\n  Timesteps:          {timesteps_done}"
                    f"\n  Completed episodes: {len(completed_returns)}"
                    f"\n  Avg Return (±std):  {avg50:.2f} ± {std50:.2f}"
                    f"\n  Avg Length:         {avg_len50:.1f}"
                    f"\n  Actor Loss:         {actor_loss.item():.4f}"
                    f"\n  Critic Loss:        {critic_loss.item():.4f}"
                    f"\n  Entropy:            {entropy_bonus.item():.4f}"
                    f"\n  LR:                 {current_lr:.6f}"
                )

        # ----------------------------------------------------
        # Greedy evaluation
        # ----------------------------------------------------
        if update % eval_every == 0:

            net.eval()

            with torch.no_grad():
                eval_eps = [
                    run_episode_greedy(eval_env, net, obs_scale)[2]
                    for _ in range(eval_episodes)
                ]

            net.train()

            avg_eval_return = np.mean([sum(r) for r in eval_eps])
            avg_eval_length = np.mean([len(r) for r in eval_eps])

            eval_avg_returns.append(avg_eval_return)
            eval_avg_lengths.append(avg_eval_length)

            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                torch.save(net.state_dict(), checkpoint_path)

                if verbose:
                    print(
                        f"💾 New best EVAL model saved to {checkpoint_path}! "
                        f"EvalReturn = {best_eval_return:.2f}"
                    )

            if verbose:
                print(
                    f"  [EVAL update.{update}] "
                    f"AvgReturn: {avg_eval_return:.1f} | "
                    f"AvgLength: {avg_eval_length:.1f}"
                )

            if not solved and avg_eval_return >= solved_threshold:
                solved = True

                if verbose:
                    print(f"  ✅ SOLVED at update {update}!")

                if stop_when_solved:
                    break

    net.eval()

    return {
        "completed_returns": completed_returns,
        "completed_lengths": completed_lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_values": entropy_values,
        "eval_avg_returns": eval_avg_returns,
        "eval_avg_lengths": eval_avg_lengths,
        "best_eval_return": best_eval_return,
        "checkpoint_path": checkpoint_path,
        "solved": solved,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "n_steps": n_steps,
    }

def a2c_vectorized_lunarlander_stable(
        net,
        vec_env,
        eval_env,
        obs_scale,
        gamma=0.995,
        lr=2.5e-4,
        value_coef=0.5,
        entropy_coef=0.005,
        entropy_coef_min=0.0005,
        total_timesteps=500_000,
        n_steps=64,
        eval_every=50,
        eval_episodes=5,
        reward_scale=100.0,
        clip_rewards=False,
        solved_threshold=200.0,
        gae_lambda=0.95,
        optimizer="adam",
        checkpoint_path="best_a2c_lunarlander_vectorized_stable.pt",
        normalize_advantage=True,
        grad_clip=0.5,
        verbose=True,
        stop_when_solved=True):
    """
    Stabilized vectorized A2C for LunarLander-v3.

    This version is kept separate from the previous A2C implementations
    in order to document the experimental progression:
    - episodic A2C,
    - rollout A2C,
    - vectorized A2C,
    - stabilized vectorized A2C for LunarLander.

    Main stabilizations:
    - vectorized environments;
    - GAE;
    - reward scaling;
    - advantage normalization;
    - logits-based categorical distribution;
    - entropy decay;
    - minimum learning rate;
    - gradient clipping.
    """

    optimizer = optimizer.lower()

    if optimizer not in ["adam", "rmsprop"]:
        raise ValueError("optimizer must be either 'adam' or 'rmsprop'")

    n_envs = vec_env.num_envs
    total_updates = total_timesteps // (n_envs * n_steps)

    if optimizer == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        opt = torch.optim.RMSprop(
            net.parameters(),
            lr=lr,
            eps=1e-5,
            alpha=0.99
        )

    completed_returns = []
    completed_lengths = []

    current_returns = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=np.int32)

    actor_losses = []
    critic_losses = []
    entropy_values = []

    eval_avg_returns = []
    eval_avg_lengths = []

    best_eval_return = -float("inf")
    solved = False

    best_train_return = -float("inf")

    train_checkpoint_path = checkpoint_path.replace(".pt", "_best_train.pt")
    last_checkpoint_path = checkpoint_path.replace(".pt", "_last.pt")

    obs, _ = vec_env.reset()

    net.train()

    for update in range(total_updates):

        # Learning rate decay with floor
        progress_remaining = 1.0 - (update / total_updates)
        min_lr = lr * 0.1
        current_lr = max(min_lr, lr * progress_remaining)

        for param_group in opt.param_groups:
            param_group["lr"] = current_lr

        # Entropy decay with floor
        current_entropy_coef = max(
            entropy_coef_min,
            entropy_coef - (entropy_coef - entropy_coef_min) * update / total_updates
        )

        rewards_list = []
        values_list = []
        log_probs_list = []
        entropies_list = []
        masks_list = []

        # ----------------------------------------------------
        # Collect rollout: n_steps x n_envs
        # ----------------------------------------------------
        for _ in range(n_steps):

            obs_t = _preprocess(obs, obs_scale)

            logits, values = net.get_logits_and_value(obs_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()

            next_obs, rewards, terminated, truncated, _ = vec_env.step(
                actions.cpu().numpy()
            )

            dones = np.logical_or(terminated, truncated)

            raw_rewards = rewards.copy()

            if clip_rewards:
                rewards = np.clip(rewards, -10.0, 10.0)

            rewards_scaled = rewards / reward_scale

            rewards_list.append(torch.tensor(rewards_scaled, dtype=torch.float32))
            values_list.append(values)
            log_probs_list.append(dist.log_prob(actions))
            entropies_list.append(dist.entropy())

            masks_list.append(
                torch.tensor(1.0 - dones.astype(np.float32), dtype=torch.float32)
            )

            current_returns += raw_rewards
            current_lengths += 1

            for env_idx, done in enumerate(dones):
                if done:
                    completed_returns.append(float(current_returns[env_idx]))
                    completed_lengths.append(int(current_lengths[env_idx]))

                    current_returns[env_idx] = 0.0
                    current_lengths[env_idx] = 0

            obs = next_obs

        # ----------------------------------------------------
        # Stack rollout tensors
        # ----------------------------------------------------
        rewards_tensor = torch.stack(rewards_list)        # [n_steps, n_envs]
        values_tensor = torch.stack(values_list)          # [n_steps, n_envs]
        log_probs_tensor = torch.stack(log_probs_list)    # [n_steps, n_envs]
        entropies_tensor = torch.stack(entropies_list)    # [n_steps, n_envs]
        masks_tensor = torch.stack(masks_list)            # [n_steps, n_envs]

        # ----------------------------------------------------
        # Bootstrap value
        # ----------------------------------------------------
        with torch.no_grad():
            next_obs_t = _preprocess(obs, obs_scale)
            _, next_values = net.get_logits_and_value(next_obs_t)
            next_values = next_values.detach()

        # ----------------------------------------------------
        # GAE
        # ----------------------------------------------------
        advantages = torch.zeros_like(rewards_tensor)
        last_gae = torch.zeros(n_envs, dtype=torch.float32)

        for step in reversed(range(n_steps)):

            if step == n_steps - 1:
                next_value = next_values
            else:
                next_value = values_tensor[step + 1].detach()

            delta = (
                rewards_tensor[step]
                + gamma * next_value * masks_tensor[step]
                - values_tensor[step].detach()
            )

            last_gae = (
                delta
                + gamma * gae_lambda * masks_tensor[step] * last_gae
            )

            advantages[step] = last_gae

        returns = advantages + values_tensor.detach()

        # ----------------------------------------------------
        # Flatten rollout
        # ----------------------------------------------------
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values_tensor.reshape(-1)
        log_probs_flat = log_probs_tensor.reshape(-1)
        entropies_flat = entropies_tensor.reshape(-1)

        if normalize_advantage and len(advantages_flat) > 1:
            advantages_flat = (
                advantages_flat - advantages_flat.mean()
            ) / (advantages_flat.std() + 1e-8)

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------
        actor_loss = -(log_probs_flat * advantages_flat.detach()).mean()
        entropy_bonus = entropies_flat.mean()

        actor_loss = actor_loss - current_entropy_coef * entropy_bonus

        critic_loss = F.smooth_l1_loss(values_flat, returns_flat)

        loss = actor_loss + value_coef * critic_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        entropy_values.append(entropy_bonus.item())

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        if update % 50 == 0:

            if len(completed_returns) > 0:
                avg50 = np.mean(completed_returns[-50:])
                std50 = np.std(completed_returns[-50:])
                avg_len50 = np.mean(completed_lengths[-50:])
            else:
                avg50 = 0.0
                std50 = 0.0
                avg_len50 = 0.0

            timesteps_done = (update + 1) * n_envs * n_steps


            if len(completed_returns) >= 50 and avg50 > best_train_return:
                best_train_return = avg50
                torch.save(net.state_dict(), train_checkpoint_path)

                if verbose:
                    print(
                        f"📈 New best TRAINING model saved to {train_checkpoint_path}! "
                        f"TrainAvgReturn = {best_train_return:.2f}"
                    )

            if verbose:
                print(
                    f"\nUpdate: {update}"
                    f"\n  Timesteps:          {timesteps_done}"
                    f"\n  Completed episodes: {len(completed_returns)}"
                    f"\n  Avg Return (±std):  {avg50:.2f} ± {std50:.2f}"
                    f"\n  Avg Length:         {avg_len50:.1f}"
                    f"\n  Actor Loss:         {actor_loss.item():.4f}"
                    f"\n  Critic Loss:        {critic_loss.item():.4f}"
                    f"\n  Entropy:            {entropy_bonus.item():.4f}"
                    f"\n  Entropy coef:       {current_entropy_coef:.5f}"
                    f"\n  LR:                 {current_lr:.6f}"
                )

        # ----------------------------------------------------
        # Greedy evaluation
        # ----------------------------------------------------
        if update % eval_every == 0:

            torch.save(net.state_dict(), last_checkpoint_path)

            net.eval()

            with torch.no_grad():
                eval_eps = [
                    run_episode_greedy(eval_env, net, obs_scale)[2]
                    for _ in range(eval_episodes)
                ]

            net.train()

            avg_eval_return = np.mean([sum(r) for r in eval_eps])
            avg_eval_length = np.mean([len(r) for r in eval_eps])

            eval_avg_returns.append(avg_eval_return)
            eval_avg_lengths.append(avg_eval_length)

            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                torch.save(net.state_dict(), checkpoint_path)

                if verbose:
                    print(
                        f"💾 New best EVAL model saved to {checkpoint_path}! "
                        f"EvalReturn = {best_eval_return:.2f}"
                    )

            if verbose:
                print(
                    f"  [EVAL update.{update}] "
                    f"AvgReturn: {avg_eval_return:.1f} | "
                    f"AvgLength: {avg_eval_length:.1f}"
                )

            if not solved and avg_eval_return >= solved_threshold:
                solved = True

                if verbose:
                    print(f"  ✅ SOLVED at update {update}!")

                if stop_when_solved:
                    break
    
    
    torch.save(net.state_dict(), last_checkpoint_path)
    net.eval()

    return {
        "completed_returns": completed_returns,
        "completed_lengths": completed_lengths,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "entropy_values": entropy_values,
        "eval_avg_returns": eval_avg_returns,
        "eval_avg_lengths": eval_avg_lengths,
        "best_eval_return": best_eval_return,
        "checkpoint_path": checkpoint_path,
        "solved": solved,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "best_train_return": best_train_return,
        "train_checkpoint_path": train_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
    }