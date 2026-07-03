import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.distributions import Categorical


# Given an environment, observation, and policy, sample from pi(a | obs). Returns the
# selected action and the log probability of that action (needed for policy gradient).
def select_action(env, obs, policy, temperature=1.0):
    dist = Categorical(policy(obs, temperature=temperature))
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))


# Utility to compute the discounted total reward. Torch doesn't like flipped arrays, so we need to
# .copy() the final numpy array.
def compute_returns(rewards, gamma):
    return np.flip(np.cumsum([gamma**(i+1)*r for (i, r) in enumerate(rewards)][::-1]), 0).copy()


# Given an environment and a policy, run it up to the maximum number of steps.
def run_episode(env, policy, maxlen=500):
    observations = []
    actions = []
    log_probs = []
    rewards = []

    OBS_SCALE = torch.tensor([4.8, 5.0, 0.418, 5.0], dtype=torch.float32)

    (obs, info) = env.reset()
    for i in range(maxlen):
        obs = torch.tensor(obs, dtype=torch.float32) / OBS_SCALE
        (action, log_prob) = select_action(env, obs, policy)
        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)

        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, torch.cat(log_probs), rewards)


# Deterministic action selection — always picks the most probable action.
def select_action_greedy(env, obs, policy):
    probs = policy(obs, temperature=1e-8)
    action = probs.argmax().item()
    return action


# Run an episode with greedy (deterministic) policy — used for evaluation only.
def run_episode_greedy(env, policy, maxlen=500):
    observations = []
    actions = []
    rewards = []

    OBS_SCALE = torch.tensor([4.8, 5.0, 0.418, 5.0], dtype=torch.float32)

    (obs, info) = env.reset()
    for i in range(maxlen):
        obs = torch.tensor(obs, dtype=torch.float32) / OBS_SCALE
        action = select_action_greedy(env, obs, policy)
        observations.append(obs)
        actions.append(action)

        (obs, reward, term, trunc, info) = env.step(action)
        rewards.append(reward)
        if term or trunc:
            break
    return (observations, actions, rewards)


# A simple, but generic, policy network with one hidden layer.
class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)

    def forward(self, s, temperature=1.0):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s) / temperature, dim=-1)
        return s

class ValueNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        return self.fc2(s).squeeze(-1)



def evaluate_policy_greedy(env, policy, eval_episodes=10):
    policy.eval()

    returns = []
    lengths = []

    with torch.no_grad():
        for _ in range(eval_episodes):
            _, _, rewards = run_episode_greedy(env, policy)
            returns.append(sum(rewards))
            lengths.append(len(rewards))

    policy.train()

    return np.mean(returns), np.mean(lengths)


def reinforce(
    policy,
    env,
    env_render=None,
    gamma=0.98,
    num_episodes=100,
    eval_every=50,
    eval_episodes=10,
    standardize=True,
    checkpoint_path="best_policy.pt",
    log_every=50,
    save_best_on_eval=True
):
    # Optimizer
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # Checkpoint folder
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Best model tracking
    best_eval_return = -float("inf")
    best_train_return = -float("inf")

    # Metrics
    running_rewards = [0.0]
    episode_returns = []
    episode_lengths = []
    eval_episode_indices = []
    eval_avg_returns = []
    eval_avg_lengths = []
    losses = []

    policy.train()

    for episode in range(num_episodes):

        # === Run training episode ===
        observations, _, log_probs, rewards = run_episode(env, policy)

        episode_return = sum(rewards)
        episode_length = len(rewards)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # === Discounted returns ===
        returns = torch.tensor(
            compute_returns(rewards, gamma),
            dtype=torch.float32
        )

        # === Advantage ===
        advantage = returns - returns.mean()

        if standardize and advantage.std() > 1e-8:
            advantage = advantage / (advantage.std() + 1e-8)

        # === Policy loss ===
        opt.zero_grad()

        obs_tensor = torch.stack(observations)
        dist = Categorical(policy(obs_tensor))
        entropy = dist.entropy().mean()

        loss = (-log_probs * advantage).sum() - 0.01 * entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt.step()

        losses.append(loss.item())

        # === Running reward ===
        running_rewards.append(
            0.05 * episode_return + 0.95 * running_rewards[-1]
        )

        # === Periodic evaluation ===
        if episode % eval_every == 0:
            avg_eval_return, avg_eval_length = evaluate_policy_greedy(
                env,
                policy,
                eval_episodes=eval_episodes
            )

            eval_episode_indices.append(episode)
            eval_avg_returns.append(avg_eval_return)
            eval_avg_lengths.append(avg_eval_length)

            print(
                f"  [EVAL ep.{episode}] "
                f"AvgReturn: {avg_eval_return:.1f} | "
                f"AvgLength: {avg_eval_length:.1f}"
            )

            # Save best model based on greedy evaluation
            if save_best_on_eval and avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                torch.save(policy.state_dict(), checkpoint_path)
                print(f"💾 New best eval model saved! AvgEvalReturn = {best_eval_return:.2f}")

        # === Logging ===
        if episode % log_every == 0:
            train_avg_return = np.mean(episode_returns[-log_every:])
            train_std_return = np.std(episode_returns[-log_every:])

            if not save_best_on_eval and train_avg_return > best_train_return:
                best_train_return = train_avg_return
                torch.save(policy.state_dict(), checkpoint_path)
                print(f"💾 New best train model saved! AvgTrainReturn = {best_train_return:.2f}")

            print(f"""
Episode: {episode}
Train Return: {episode_return:.2f}
Train Avg Return (last {log_every}): {train_avg_return:.2f}
Train Std Return (last {log_every}): {train_std_return:.2f}
Episode Length: {episode_length}
Loss: {loss.item():.4f}
Running Reward: {running_rewards[-1]:.2f}
""")

        # === Optional rendering during training ===
        if env_render and episode % 100 == 0:
            policy.eval()
            run_episode_greedy(env_render, policy)
            policy.train()

    policy.eval()

    return {
        "running_rewards": running_rewards,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "eval_episode_indices": eval_episode_indices,
        "eval_avg_returns": eval_avg_returns,
        "eval_avg_lengths": eval_avg_lengths,
        "best_eval_return": best_eval_return,
        "checkpoint_path": checkpoint_path
    }

def reinforce_baseline(
    policy,
    value_net,
    env,
    env_render=None,
    gamma=0.98,
    num_episodes=100,
    eval_every=50,
    eval_episodes=10,
    checkpoint_path="best_policy_baseline.pt",
    log_every=50,
    normalize_advantage=True,
    save_best_on_eval=True
):
    # Optimizers
    opt_policy = torch.optim.Adam(policy.parameters(), lr=3e-4)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=1e-3)

    # Checkpoint folder
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Best model tracking
    best_eval_return = -float("inf")

    # Metrics
    running_rewards = [0.0]
    episode_returns = []
    episode_lengths = []
    eval_episode_indices = []
    eval_avg_returns = []
    eval_avg_lengths = []
    policy_losses = []
    value_losses = []

    policy.train()
    value_net.train()

    for episode in range(num_episodes):

        # === Run training episode ===
        observations, _, log_probs, rewards = run_episode(env, policy)

        episode_return = sum(rewards)
        episode_length = len(rewards)

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # === Discounted returns ===
        returns = torch.tensor(
            compute_returns(rewards, gamma),
            dtype=torch.float32
        )

        obs_tensor = torch.stack(observations)

        # === Value baseline ===
        values = value_net(obs_tensor)
        advantage = returns - values.detach()

        if normalize_advantage and advantage.std() > 1e-8:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # === Policy update ===
        opt_policy.zero_grad()

        dist = Categorical(policy(obs_tensor))
        entropy = dist.entropy().mean()

        policy_loss = (-log_probs * advantage).sum() - 0.01 * entropy

        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        opt_policy.step()

        # === Value update ===
        opt_value.zero_grad()

        value_loss = F.mse_loss(values, returns)

        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
        opt_value.step()

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        # === Running reward ===
        running_rewards.append(
            0.05 * episode_return + 0.95 * running_rewards[-1]
        )

        # === Periodic evaluation ===
        if episode % eval_every == 0:
            avg_eval_return, avg_eval_length = evaluate_policy_greedy(
                env,
                policy,
                eval_episodes=eval_episodes
            )

            eval_episode_indices.append(episode)
            eval_avg_returns.append(avg_eval_return)
            eval_avg_lengths.append(avg_eval_length)

            print(
                f"  [EVAL ep.{episode}] "
                f"AvgReturn: {avg_eval_return:.1f} | "
                f"AvgLength: {avg_eval_length:.1f}"
            )

            if save_best_on_eval and avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                torch.save(policy.state_dict(), checkpoint_path)
                print(f"💾 New best baseline model saved! AvgEvalReturn = {best_eval_return:.2f}")

        # === Logging ===
        if episode % log_every == 0:
            train_avg_return = np.mean(episode_returns[-log_every:])
            train_std_return = np.std(episode_returns[-log_every:])

            print(f"""
Episode: {episode}
Train Return: {episode_return:.2f}
Train Avg Return (last {log_every}): {train_avg_return:.2f}
Train Std Return (last {log_every}): {train_std_return:.2f}
Episode Length: {episode_length}
Policy Loss: {policy_loss.item():.4f}
Value Loss: {value_loss.item():.4f}
Running Reward: {running_rewards[-1]:.2f}
""")

        # === Optional rendering ===
        if env_render and episode % 100 == 0:
            policy.eval()
            run_episode_greedy(env_render, policy)
            policy.train()

    policy.eval()
    value_net.eval()

    return {
        "running_rewards": running_rewards,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "losses": policy_losses,
        "eval_episode_indices": eval_episode_indices,
        "eval_avg_returns": eval_avg_returns,
        "eval_avg_lengths": eval_avg_lengths,
        "best_eval_return": best_eval_return,
        "checkpoint_path": checkpoint_path
    }