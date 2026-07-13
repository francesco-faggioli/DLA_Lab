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
    """Contenitore restituito da un rollout dell'ambiente.

    Argomenti:
        observations: Osservazioni normalizzate visitate durante l'episodio.
        actions: Azioni intere selezionate dalla policy.
        log_probs: Log-probabilità delle azioni selezionate.
        rewards: Reward grezzi restituiti dall'ambiente.
        total_reward: Somma dei reward grezzi nell'episodio.
        length: Numero di step dell'ambiente nell'episodio.

    Operazione:
        Memorizza i dati necessari a REINFORCE e statistiche di episodio
        leggibili.

    Output:
        Istanza dataclass usata da training e valutazione.
    """

    observations: list[torch.Tensor]
    actions: list[int]
    log_probs: torch.Tensor
    rewards: list[float]
    total_reward: float
    length: int


@dataclass
class ReinforceConfig:
    """Configurazione di training per gli esperimenti REINFORCE.

    Argomenti:
        gamma: Fattore di sconto dei return Monte Carlo.
        lr_policy: Learning rate Adam della rete di policy.
        lr_value: Learning rate Adam della rete di valore, se usata.
        num_episodes: Numero di episodi di training.
        max_episode_steps: Limite di sicurezza per ciascun episodio.
        eval_every: Valuta ogni N episodi di training.
        eval_episodes: Esegue M episodi a ogni punto di valutazione.
        baseline_mode: `none` o `standardize` per REINFORCE standard.
        normalize_advantage: Normalizza gli advantage nel training con value baseline.
        entropy_coef: Coefficiente del bonus di entropia per l'esplorazione.
        grad_clip: Norma massima del gradiente.
        checkpoint_path: Percorso opzionale per salvare la miglior policy valutata.
        save_best: Se True, salva la miglior policy valutata.

    Operazione:
        Mantiene gli iperparametri espliciti e riproducibili tra le celle.

    Output:
        Istanza dataclass passata alle funzioni di training.
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
    """Policy MLP stocastica per osservazioni continue e azioni discrete."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        """Crea la rete della policy.

        Argomenti:
            obs_dim: Numero di feature dell'osservazione.
            n_actions: Numero di azioni discrete.
            hidden_size: Ampiezza dei layer nascosti.

        Operazione:
            Costruisce un MLP a due layer e mappa le osservazioni nei logit delle azioni.

        Output:
            Modulo PyTorch: `forward` restituisce logit e `action_probs`
            probabilità normalizzate delle azioni.
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
        """Calcola i logit delle azioni.

        Argomenti:
            obs: Tensore dell'osservazione normalizzata. La forma può essere `(obs_dim,)` oppure
                `(batch, obs_dim)`.

        Operazione:
            Propaga l'osservazione nella policy MLP.

        Output:
            Logit grezzi delle azioni, utilizzabili direttamente per costruire una
            distribuzione categorica.
        """

        return self.net(obs)

    def action_probs(self, obs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Calcola le probabilità normalizzate delle azioni.

        Argomenti:
            obs: Tensore dell'osservazione normalizzata.
            temperature: Temperatura softmax; valori inferiori rendono la policy più
                deterministica.

        Operazione:
            Converte i logit in una distribuzione di probabilità sulle azioni.

        Output:
            Tensore delle probabilità con la stessa forma iniziale di
            `obs`.
        """

        logits = self.forward(obs) / temperature
        return F.softmax(logits, dim=-1)


class ValueNet(nn.Module):
    """Rete MLP per la baseline del valore di stato."""

    def __init__(self, obs_dim: int, hidden_size: int = 128):
        """Crea la rete di valore.

        Argomenti:
            obs_dim: Numero di feature dell'osservazione.
            hidden_size: Ampiezza dei layer nascosti.

        Operazione:
            Costruisce un MLP a due layer che stima il valore scalare V(s).

        Output:
            Modulo PyTorch il cui `forward` restituisce un valore per osservazione.
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
        """Stima il valore di stato V(s).

        Argomenti:
            obs: Tensore dell'osservazione normalizzata. La forma può essere `(obs_dim,)` oppure
                `(batch, obs_dim)`.

        Operazione:
            Propaga le osservazioni nella rete di valore e rimuove l'ultima
            dimensione singleton.

        Output:
            Valore scalare per un'osservazione o vettore di valori per un batch.
        """

        return self.net(obs).squeeze(-1)


def policy_from_env(env, hidden_size: int = 128) -> PolicyNet:
    """Costruisce una rete di policy da un ambiente Gymnasium.

    Argomenti:
        env: Ambiente con osservazioni Box e azioni Discrete.
        hidden_size: Ampiezza dei layer nascosti.

    Operazione:
        Legge dall'ambiente le dimensioni di osservazioni e azioni.

    Output:
        `PolicyNet` configurata per l'ambiente.
    """

    return PolicyNet(env.observation_space.shape[0], env.action_space.n, hidden_size)


def value_from_env(env, hidden_size: int = 128) -> ValueNet:
    """Costruisce una rete di valore da un ambiente Gymnasium.

    Argomenti:
        env: Ambiente con spazio delle osservazioni Box.
        hidden_size: Ampiezza dei layer nascosti.

    Operazione:
        Legge dall'ambiente la dimensione delle osservazioni.

    Output:
        `ValueNet` configurata per l'ambiente.
    """

    return ValueNet(env.observation_space.shape[0], hidden_size)


def preprocess_observation(obs, obs_scale: torch.Tensor) -> torch.Tensor:
    """Converte un'osservazione in un tensore float normalizzato.

    Argomenti:
        obs: Osservazione grezza restituita da Gymnasium.
        obs_scale: Tensore di scala per feature.

    Operazione:
        Converte l'osservazione in float32 e la divide per il tensore di scala.

    Output:
        Tensore dell'osservazione normalizzata.
    """

    return torch.as_tensor(obs, dtype=torch.float32) / obs_scale


def select_action(
    policy: PolicyNet,
    obs: torch.Tensor,
    mode: ActionMode = "sample",
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor]:
    """Seleziona un'azione dalla policy corrente.

    Argomenti:
        policy: Rete di policy.
        obs: Tensore dell'osservazione preprocessata.
        mode: `sample` per training stocastico, `greedy` per valutazione.
        temperature: Temperatura softmax; valori inferiori rendono la distribuzione
            più concentrata.

    Operazione:
        Costruisce una distribuzione categorica dai logit e campiona un'azione
        oppure sceglie quella più probabile.

    Output:
        Tupla `(action, log_prob)`, composta da un'azione intera e dalla
        log-probabilità usata nell'aggiornamento policy-gradient.
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
    """Esegue un episodio completo.

    Argomenti:
        env: Ambiente Gymnasium.
        policy: Rete di policy usata per selezionare le azioni.
        obs_scale: Tensore di scala per normalizzare le osservazioni.
        max_steps: Limite di sicurezza della lunghezza dell'episodio.
        mode: `sample` per training o valutazione stocastica, `greedy` per
            valutazione deterministica.
        temperature: Temperatura usata dalla distribuzione categorica.
        seed: Seed opzionale passato a `env.reset`.

    Operazione:
        Interagisce con l'ambiente fino a terminazione, troncamento o
        `max_steps`, raccogliendo osservazioni, azioni, log-probabilità e
        reward.

    Output:
        Dataclass `Episode` con tensori del rollout e statistiche di sintesi.
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
    """Calcola i return Monte Carlo scontati.

    Argomenti:
        rewards: Sequenza dei reward di un episodio.
        gamma: Fattore di sconto.

    Operazione:
        Calcola a ritroso `G_t = r_t + gamma r_{t+1} + ...` nell'episodio.

    Output:
        Tensore float con un return scontato per timestep.
    """

    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def prepare_policy_target(returns: torch.Tensor, baseline_mode: BaselineMode) -> torch.Tensor:
    """Applica ai return la baseline opzionale a livello di episodio.

    Argomenti:
        returns: Return scontati di un episodio.
        baseline_mode: `none` mantiene i return grezzi; `standardize` sottrae la media
            dell'episodio e divide per la deviazione standard.

    Operazione:
        Implementa la baseline semplice discussa nell'Esercizio 2.

    Output:
        Tensore usato come target policy-gradient.
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
    """Valuta una policy su più episodi.

    Argomenti:
        env: Ambiente Gymnasium.
        policy: Rete di policy.
        obs_scale: Tensore di scala per feature.
        episodes: Numero di episodi di valutazione.
        max_steps: Limite di sicurezza per episodio.
        mode: `greedy` oppure `sample`.
        temperature: Temperatura usata se le azioni vengono campionate.
        seed_start: Primo seed opzionale; l'episodio `i` usa `seed_start + i`.

    Operazione:
        Esegue episodi indipendenti senza gradienti e riassume reward totali
        e lunghezze.

    Output:
        Dizionario con media, deviazione, minimo e massimo del return, lunghezza
        media e array grezzi.
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
    """Addestra una policy con REINFORCE standard.

    Argomenti:
        policy: Rete di policy da addestrare.
        env: Ambiente di training.
        eval_env: Ambiente usato per la valutazione periodica.
        obs_scale: Tensore di scala per feature.
        config: `ReinforceConfig` con iperparametri e impostazioni di logging.

    Operazione:
        Esegue un episodio campionato per aggiornamento, calcola i return Monte Carlo,
        applica la standardizzazione opzionale, aggiorna la policy con Adam e
        valuta periodicamente la policy greedy.

    Output:
        Dizionario con return per episodio, loss, metriche di valutazione e
        informazioni sul checkpoint migliore.
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
    """Addestra REINFORCE con una baseline di valore appresa.

    Argomenti:
        policy: Rete di policy da addestrare.
        value_net: Rete di valore addestrata a stimare i return scontati.
        env: Ambiente di training.
        eval_env: Ambiente usato per la valutazione periodica.
        obs_scale: Tensore di scala per feature.
        config: `ReinforceConfig` con iperparametri e impostazioni di logging.

    Operazione:
        Usa `G_t - V(s_t)` come advantage e addestra la rete di valore
        con errore quadratico medio sui return Monte Carlo.

    Output:
        Dizionario con curve di training, metriche di valutazione e informazioni
        sui checkpoint.
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

        policy_loss = (
            -(episode.log_probs * advantage.detach()).mean() - config.entropy_coef * entropy
        )
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
