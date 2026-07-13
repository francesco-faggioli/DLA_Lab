from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ActionMode = Literal["greedy", "sample"]


@dataclass
class A2CConfig:
    """Configurazione di training per gli esperimenti Advantage Actor-Critic.

    Argomenti:
        gamma: Fattore di sconto.
        lr: Learning rate dell'ottimizzatore.
        value_coef: Peso della loss del critic.
        entropy_coef: Coefficiente del bonus di entropia.
        entropy_coef_min: Coefficiente minimo di entropia dopo il decadimento lineare.
        num_episodes: Numero di episodi per il training con un singolo ambiente.
        max_episode_steps: Limite di sicurezza per ciascun episodio.
        eval_every: Valuta ogni N episodi o aggiornamenti vettorizzati.
        eval_episodes: Numero di episodi usati a ogni valutazione.
        gae_lambda: Parametro lambda della Generalized Advantage Estimation.
        normalize_advantage: Se True, standardizza gli advantage prima
            dell'aggiornamento dell'actor.
        grad_clip: Norma massima del gradiente.
        solved_threshold: Return medio di valutazione usato come criterio di soluzione.
        checkpoint_path: Percorso opzionale per salvare il miglior modello di valutazione.
        save_best: Se True, salva il miglior modello di valutazione.
        stop_when_solved: Se True, interrompe il training dopo il raggiungimento della soglia.

    Operazione:
        Memorizza in un unico oggetto esplicito tutti gli iperparametri necessari
        per riprodurre gli esperimenti A2C.

    Output:
        Istanza dataclass passata alle funzioni di training A2C.
    """

    gamma: float = 0.99
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_min: float = 0.001
    num_episodes: int = 800
    max_episode_steps: int = 1000
    eval_every: int = 25
    eval_episodes: int = 20
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    grad_clip: float = 0.5
    solved_threshold: float = 475.0
    checkpoint_path: str | None = None
    save_best: bool = True
    stop_when_solved: bool = True


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Inizializza un layer lineare con pesi ortogonali.

    Argomenti:
        layer: Layer lineare da inizializzare.
        std: Guadagno usato dall'inizializzazione ortogonale.
        bias_const: Valore costante assegnato al bias.

    Operazione:
        Applica un'inizializzazione comune nelle implementazioni actor-critic.

    Output:
        Lo stesso layer, inizializzato in-place.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class A2CNet(nn.Module):
    """Rete actor-critic con trunk condiviso per ambienti ad azioni discrete."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        """Crea un modello actor-critic con rappresentazione condivisa.

        Argomenti:
            obs_dim: Numero di feature dell'osservazione.
            n_actions: Numero di azioni discrete.
            hidden_size: Ampiezza dei layer nascosti.

        Operazione:
            Costruisce un estrattore condiviso e due teste: una per i logit
            della policy e una per la stima del valore di stato.

        Output:
            Modulo PyTorch; `get_logits_and_value` restituisce logit della policy e
            valori del critic.
        """

        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_size, n_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def get_logits_and_value(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Restituisce logit della policy e valori di stato per uno stato o un batch."""

        features = self.trunk(states)
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values

    def forward(
        self, states: torch.Tensor, temperature: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.get_logits_and_value(states)
        return F.softmax(logits / temperature, dim=-1), values

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        _, values = self.get_logits_and_value(states)
        return values


class A2CNetLunarSeparate(nn.Module):
    """Reti actor e critic separate per LunarLander."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64):
        """Crea il modello actor-critic per LunarLander.

        Argomenti:
            obs_dim: Numero di feature dell'osservazione LunarLander.
            n_actions: Numero di azioni discrete LunarLander.
            hidden_size: Ampiezza dei layer nascosti di actor e critic.

        Operazione:
            Usa due MLP indipendenti, scelta che nelle prove esplorative ha ridotto
            l'interferenza tra actor e critic.

        Output:
            Modulo PyTorch compatibile con le funzioni A2C di valutazione e training
            definite in questo file.
        """

        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

    def get_logits_and_value(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Restituisce logit della policy e valori di stato per uno stato o un batch."""

        logits = self.actor(states)
        values = self.critic(states).squeeze(-1)
        return logits, values

    def forward(
        self, states: torch.Tensor, temperature: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.get_logits_and_value(states)
        return F.softmax(logits / temperature, dim=-1), values

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)


def a2c_from_env(env, hidden_size: int = 128) -> A2CNet:
    """Costruisce una rete A2C condivisa da un ambiente Gymnasium.

    Argomenti:
        env: Ambiente con osservazioni continue e spazio delle azioni
            space.
        hidden_size: Ampiezza dei layer nascosti.

    Operazione:
        Legge dall'ambiente le dimensioni di osservazioni e azioni.

    Output:
        Istanza di `A2CNet`.
    """

    return A2CNet(env.observation_space.shape[0], env.action_space.n, hidden_size)


def lunar_a2c_from_env(env, hidden_size: int = 64) -> A2CNetLunarSeparate:
    """Costruisce la rete actor-critic separata usata per LunarLander.

    Argomenti:
        env: Ambiente LunarLander.
        hidden_size: Ampiezza dei layer nascosti di actor e critic.

    Operazione:
        Legge dall'ambiente le dimensioni di osservazioni e azioni.

    Output:
        Istanza di `A2CNetLunarSeparate`.
    """

    return A2CNetLunarSeparate(env.observation_space.shape[0], env.action_space.n, hidden_size)


def _preprocess(obs, obs_scale: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32) / obs_scale


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcola i target della Generalized Advantage Estimation.

    Argomenti:
        rewards: Tensore dei reward con forma `[T]` oppure `[T, n_envs]`.
        values: Valori del critic con la stessa forma iniziale dei reward.
        next_value: Valore di bootstrap dopo il rollout.
        masks: Maschera di continuazione; usa 0 quando lo stato successivo è terminale.
        gamma: Fattore di sconto.
        gae_lambda: Parametro di traccia GAE.

    Operazione:
        Calcola stime dell'advantage a varianza ridotta e i target corrispondenti.

    Output:
        Tupla `(advantages, returns)`.
    """

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)

    for step in reversed(range(rewards.shape[0])):
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * masks[step] - values[step]
        last_gae = delta + gamma * gae_lambda * masks[step] * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns


def run_a2c_episode(
    env,
    net: nn.Module,
    obs_scale: torch.Tensor,
    mode: ActionMode = "sample",
    temperature: float = 1.0,
    max_episode_steps: int = 1000,
    seed: int | None = None,
) -> dict:
    """Esegue un episodio con una policy A2C.

    Argomenti:
        env: Ambiente Gymnasium.
        net: Rete actor-critic.
        obs_scale: Tensore di normalizzazione delle osservazioni.
        mode: `sample` per azioni stocastiche o `greedy` per l'argmax.
        temperature: Temperatura softmax usata soltanto in modalità stocastica.
        max_episode_steps: Limite di sicurezza del rollout.
        seed: Seed opzionale per il reset.

    Operazione:
        Esegue nell'ambiente una policy actor-critic addestrata o parzialmente
        addestrata e registra le statistiche dell'episodio.

    Output:
        Dizionario con return, lunghezza, azioni selezionate e flag terminali.
    """

    if mode not in {"sample", "greedy"}:
        raise ValueError("mode deve essere 'sample' oppure 'greedy'")

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    actions: list[int] = []
    observations = [np.asarray(obs, dtype=np.float32).copy()]
    terminated = False
    truncated = False

    net.eval()
    with torch.no_grad():
        for _ in range(max_episode_steps):
            obs_t = _preprocess(obs, obs_scale)
            logits, _ = net.get_logits_and_value(obs_t)

            if mode == "greedy":
                action = int(torch.argmax(logits).item())
            else:
                dist = Categorical(logits=logits / temperature)
                action = int(dist.sample().item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            actions.append(action)
            observations.append(np.asarray(obs, dtype=np.float32).copy())

            if terminated or truncated:
                break

    return {
        "return": total_reward,
        "length": len(actions),
        "actions": actions,
        "final_observation": observations[-1].tolist(),
        "terminated": terminated,
        "truncated": truncated,
    }


def evaluate_a2c_policy(
    env,
    net: nn.Module,
    obs_scale: torch.Tensor,
    n_eval: int = 100,
    mode: ActionMode = "greedy",
    temperature: float = 1.0,
    seed_start: int = 10_000,
    max_episode_steps: int = 1000,
) -> dict:
    """Valuta una policy A2C su più episodi.

    Argomenti:
        env: Ambiente Gymnasium di valutazione.
        net: Policy actor-critic da valutare.
        obs_scale: Tensore di normalizzazione delle osservazioni.
        n_eval: Numero di episodi di valutazione.
        mode: `greedy` oppure `sample`.
        temperature: Temperatura della policy stocastica.
        seed_start: Primo seed degli episodi di valutazione deterministici.
        max_episode_steps: Limite di sicurezza per ciascun episodio.

    Operazione:
        Esegue episodi di valutazione indipendenti e aggrega le metriche
        di prestazione richieste dal laboratorio.

    Output:
        Dizionario con return medio, lunghezza media e metriche di dispersione,
        percentuale di successo, conteggi terminali e frequenze delle azioni.
    """

    episodes = [
        run_a2c_episode(
            env,
            net,
            obs_scale,
            mode=mode,
            temperature=temperature,
            max_episode_steps=max_episode_steps,
            seed=seed_start + idx,
        )
        for idx in range(n_eval)
    ]

    returns = np.array([ep["return"] for ep in episodes], dtype=np.float64)
    lengths = np.array([ep["length"] for ep in episodes], dtype=np.float64)
    actions = [action for ep in episodes for action in ep["actions"]]
    n_actions = getattr(env.action_space, "n", 0)
    action_freq = None
    last_quarter_action_freq = None
    if n_actions and actions:
        counts = np.bincount(actions, minlength=n_actions)
        action_freq = counts / counts.sum()
        last_quarter_actions = [
            action for ep in episodes for action in ep["actions"][int(0.75 * len(ep["actions"])) :]
        ]
        if last_quarter_actions:
            last_counts = np.bincount(last_quarter_actions, minlength=n_actions)
            last_quarter_action_freq = last_counts / last_counts.sum()

    final_observations = np.array([ep["final_observation"] for ep in episodes], dtype=np.float64)
    final_diagnostics = {}
    if final_observations.ndim == 2 and final_observations.shape[1] >= 6:
        final_diagnostics = {
            "final_abs_x": float(np.abs(final_observations[:, 0]).mean()),
            "final_abs_y": float(np.abs(final_observations[:, 1]).mean()),
            "final_abs_vx": float(np.abs(final_observations[:, 2]).mean()),
            "final_abs_vy": float(np.abs(final_observations[:, 3]).mean()),
            "final_abs_angle": float(np.abs(final_observations[:, 4]).mean()),
            "final_abs_angular_velocity": float(np.abs(final_observations[:, 5]).mean()),
        }
        if final_observations.shape[1] >= 8:
            final_diagnostics["final_left_leg_contact_rate"] = float(
                final_observations[:, 6].mean() * 100.0
            )
            final_diagnostics["final_right_leg_contact_rate"] = float(
                final_observations[:, 7].mean() * 100.0
            )

    result = {
        "returns": returns.tolist(),
        "lengths": lengths.tolist(),
        "avg_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "avg_length": float(lengths.mean()),
        "success_rate": float((returns >= 200.0).mean() * 100.0),
        "terminated": int(sum(ep["terminated"] for ep in episodes)),
        "truncated": int(sum(ep["truncated"] for ep in episodes)),
        "action_freq": None if action_freq is None else action_freq.tolist(),
        "last_quarter_action_freq": (
            None if last_quarter_action_freq is None else last_quarter_action_freq.tolist()
        ),
    }
    result.update(final_diagnostics)
    return result


def train_a2c_single_env(
    net: nn.Module,
    train_env,
    eval_env,
    obs_scale: torch.Tensor,
    config: A2CConfig,
) -> dict:
    """Addestra A2C su un singolo ambiente.

    Argomenti:
        net: Rete actor-critic.
        train_env: Ambiente usato per i rollout stocastici di training.
        eval_env: Ambiente usato per la valutazione greedy.
        obs_scale: Tensore di normalizzazione delle osservazioni.
        config: `A2CConfig` con iperparametri e impostazioni dei checkpoint.

    Operazione:
        Raccoglie un episodio completo, stima gli advantage con GAE,
        aggiorna congiuntamente actor e critic e valuta periodicamente la policy.

    Output:
        Dizionario storico con return di training, loss, metriche di valutazione e
        percorso del checkpoint migliore.
    """

    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    best_eval_return = -float("inf")
    solved = False

    history = {
        "episode_returns": [],
        "episode_lengths": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropy_values": [],
        "eval_episodes": [],
        "eval_avg_returns": [],
        "eval_avg_lengths": [],
        "best_eval_return": best_eval_return,
        "checkpoint_path": config.checkpoint_path,
        "solved": False,
        "config": asdict(config),
    }

    if config.checkpoint_path is not None:
        Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for episode in range(1, config.num_episodes + 1):
        obs, _ = train_env.reset()
        observations: list[torch.Tensor] = []
        actions: list[int] = []
        rewards: list[float] = []
        masks: list[float] = []
        terminated = False
        truncated = False
        next_obs = obs

        net.eval()
        for _ in range(config.max_episode_steps):
            obs_t = _preprocess(obs, obs_scale)
            logits, _ = net.get_logits_and_value(obs_t)
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())

            next_obs, reward, terminated, truncated, _ = train_env.step(action)

            observations.append(obs_t)
            actions.append(action)
            rewards.append(float(reward))
            masks.append(0.0 if terminated else 1.0)

            obs = next_obs
            if terminated or truncated:
                break

        obs_tensor = torch.stack(observations)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)

        net.train()
        logits, values = net.get_logits_and_value(obs_tensor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()

        with torch.no_grad():
            if terminated:
                next_value = values.new_tensor(0.0)
            else:
                next_value = net.get_value(_preprocess(next_obs, obs_scale)).detach()
            advantages, returns = compute_gae(
                rewards_tensor,
                values.detach(),
                next_value,
                masks_tensor,
                config.gamma,
                config.gae_lambda,
            )

        if config.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        progress = episode / max(1, config.num_episodes)
        current_entropy_coef = max(
            config.entropy_coef_min,
            config.entropy_coef - (config.entropy_coef - config.entropy_coef_min) * progress,
        )

        actor_loss = -(log_probs * advantages.detach()).mean() - current_entropy_coef * entropy
        critic_loss = F.smooth_l1_loss(values, returns)
        loss = actor_loss + config.value_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        optimizer.step()

        history["episode_returns"].append(float(sum(rewards)))
        history["episode_lengths"].append(len(rewards))
        history["actor_losses"].append(float(actor_loss.item()))
        history["critic_losses"].append(float(critic_loss.item()))
        history["entropy_values"].append(float(entropy.item()))

        if episode % config.eval_every == 0 or episode == 1:
            metrics = evaluate_a2c_policy(
                eval_env,
                net,
                obs_scale,
                n_eval=config.eval_episodes,
                mode="greedy",
                seed_start=20_000 + episode,
                max_episode_steps=config.max_episode_steps,
            )
            avg_eval_return = metrics["avg_return"]

            history["eval_episodes"].append(episode)
            history["eval_avg_returns"].append(avg_eval_return)
            history["eval_avg_lengths"].append(metrics["avg_length"])

            if avg_eval_return > best_eval_return:
                best_eval_return = avg_eval_return
                history["best_eval_return"] = best_eval_return
                if config.save_best and config.checkpoint_path is not None:
                    torch.save(
                        {
                            "model_state_dict": net.state_dict(),
                            "config": asdict(config),
                            "metrics": metrics,
                        },
                        config.checkpoint_path,
                    )

            if avg_eval_return >= config.solved_threshold:
                solved = True
                history["solved"] = True
                if config.stop_when_solved:
                    break

    net.eval()
    history["solved"] = solved
    return history


def train_a2c_vectorized(
    net: nn.Module,
    vec_env,
    eval_env,
    obs_scale: torch.Tensor,
    total_timesteps: int = 250_000,
    n_steps: int = 64,
    gamma: float = 0.995,
    lr: float = 2.5e-4,
    lr_final: float | None = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.005,
    entropy_coef_min: float = 0.0005,
    gae_lambda: float = 0.95,
    reward_scale: float = 100.0,
    normalize_advantage: bool = True,
    grad_clip: float = 0.5,
    eval_every: int = 50,
    eval_episodes: int = 10,
    solved_threshold: float = 200.0,
    checkpoint_path: str | None = None,
    optimizer_name: Literal["adam", "rmsprop"] = "adam",
    stop_when_solved: bool = False,
) -> dict:
    """Addestra A2C con ambienti di rollout paralleli.

    Argomenti:
        net: Rete actor-critic.
        vec_env: Ambiente vettoriale Gymnasium usato per raccogliere i rollout.
        eval_env: Singolo ambiente usato per la valutazione periodica.
        obs_scale: Tensore di normalizzazione delle osservazioni.
        total_timesteps: Budget totale di step dell'ambiente.
        n_steps: Lunghezza del rollout per ambiente prima di ogni aggiornamento.
        gamma: Fattore di sconto.
        lr: Learning rate dell'ottimizzatore.
        lr_final: Learning rate finale dello scheduler lineare. Se None,
            lo scheduler mantiene il precedente limite minimo del 10% per compatibilità.
        value_coef: Peso della loss del critic.
        entropy_coef: Coefficiente iniziale del bonus di entropia.
        entropy_coef_min: Coefficiente minimo di entropia dopo il decadimento.
        gae_lambda: Parametro di traccia GAE.
        reward_scale: Divisore applicato ai reward prima dell'ottimizzazione.
        normalize_advantage: Se True, standardizza gli advantage del rollout.
        grad_clip: Norma massima del gradiente.
        eval_every: Valuta ogni N aggiornamenti.
        eval_episodes: Numero di episodi per valutazione.
        solved_threshold: Return medio considerato risolutivo.
        checkpoint_path: Percorso opzionale del miglior checkpoint di valutazione.
        optimizer_name: Ottimizzatore dell'aggiornamento. Le migliori prove LunarLander
            con actor e critic separati usavano `rmsprop`.
        stop_when_solved: Se True, si arresta dopo aver raggiunto la soglia.

    Operazione:
        Implementa la variante A2C pratica usata per LunarLander: rollout
        vettorizzati, GAE, scaling dei reward, decadimento dell'entropia e gradient clipping.

    Output:
        Dizionario storico con return completati, loss, metriche di valutazione
        e percorsi dei checkpoint.
    """

    n_envs = vec_env.num_envs
    total_updates = max(1, total_timesteps // (n_envs * n_steps))
    lr_floor = lr * 0.1 if lr_final is None else float(lr_final)
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, eps=1e-5, alpha=0.99)
    else:
        raise ValueError("optimizer_name deve essere 'adam' oppure 'rmsprop'")

    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    current_returns = np.zeros(n_envs, dtype=np.float32)
    current_lengths = np.zeros(n_envs, dtype=np.int32)

    best_eval_return = -float("inf")
    best_train_return = -float("inf")
    solved = False
    train_checkpoint_path = None
    last_checkpoint_path = None

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        train_checkpoint_path = str(
            checkpoint.with_name(checkpoint.stem + "_best_train" + checkpoint.suffix)
        )
        last_checkpoint_path = str(
            checkpoint.with_name(checkpoint.stem + "_last" + checkpoint.suffix)
        )

    history = {
        "completed_returns": completed_returns,
        "completed_lengths": completed_lengths,
        "actor_losses": [],
        "critic_losses": [],
        "entropy_values": [],
        "eval_updates": [],
        "eval_avg_returns": [],
        "eval_avg_lengths": [],
        "best_eval_return": best_eval_return,
        "best_train_return": best_train_return,
        "checkpoint_path": checkpoint_path,
        "train_checkpoint_path": train_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "solved": False,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "lr": lr,
        "lr_final": lr_floor,
        "optimizer_name": optimizer_name,
    }

    obs, _ = vec_env.reset()
    net.train()

    for update in range(total_updates):
        progress = update / max(1, total_updates - 1)
        current_lr = lr_floor + (lr - lr_floor) * (1.0 - progress)
        current_entropy_coef = max(
            entropy_coef_min,
            entropy_coef - (entropy_coef - entropy_coef_min) * progress,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        rewards_list = []
        values_list = []
        log_probs_list = []
        entropies_list = []
        masks_list = []

        for _ in range(n_steps):
            obs_t = _preprocess(obs, obs_scale)
            logits, values = net.get_logits_and_value(obs_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()

            next_obs, rewards, terminated, truncated, _ = vec_env.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            rewards_list.append(torch.tensor(rewards / reward_scale, dtype=torch.float32))
            values_list.append(values)
            log_probs_list.append(dist.log_prob(actions))
            entropies_list.append(dist.entropy())
            masks_list.append(torch.tensor(1.0 - dones.astype(np.float32), dtype=torch.float32))

            current_returns += rewards
            current_lengths += 1
            for env_idx, done in enumerate(dones):
                if done:
                    completed_returns.append(float(current_returns[env_idx]))
                    completed_lengths.append(int(current_lengths[env_idx]))
                    current_returns[env_idx] = 0.0
                    current_lengths[env_idx] = 0

            obs = next_obs

        rewards_tensor = torch.stack(rewards_list)
        values_tensor = torch.stack(values_list)
        log_probs_tensor = torch.stack(log_probs_list)
        entropies_tensor = torch.stack(entropies_list)
        masks_tensor = torch.stack(masks_list)

        with torch.no_grad():
            _, next_values = net.get_logits_and_value(_preprocess(obs, obs_scale))
            advantages, returns = compute_gae(
                rewards_tensor,
                values_tensor.detach(),
                next_values.detach(),
                masks_tensor,
                gamma,
                gae_lambda,
            )

        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        values_flat = values_tensor.reshape(-1)
        log_probs_flat = log_probs_tensor.reshape(-1)
        entropies_flat = entropies_tensor.reshape(-1)

        if normalize_advantage and len(advantages_flat) > 1:
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (
                advantages_flat.std() + 1e-8
            )

        actor_loss = -(log_probs_flat * advantages_flat.detach()).mean()
        entropy_bonus = entropies_flat.mean()
        actor_loss = actor_loss - current_entropy_coef * entropy_bonus
        critic_loss = F.smooth_l1_loss(values_flat, returns_flat)
        loss = actor_loss + value_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        history["actor_losses"].append(float(actor_loss.item()))
        history["critic_losses"].append(float(critic_loss.item()))
        history["entropy_values"].append(float(entropy_bonus.item()))

        if len(completed_returns) >= 50:
            avg50 = float(np.mean(completed_returns[-50:]))
            if checkpoint_path is not None and avg50 > best_train_return:
                best_train_return = avg50
                history["best_train_return"] = best_train_return
                torch.save(net.state_dict(), train_checkpoint_path)

        if update % eval_every == 0 or update == total_updates - 1:
            if last_checkpoint_path is not None:
                torch.save(net.state_dict(), last_checkpoint_path)

            metrics = evaluate_a2c_policy(
                eval_env,
                net,
                obs_scale,
                n_eval=eval_episodes,
                mode="greedy",
                seed_start=30_000 + update,
            )
            history["eval_updates"].append(update)
            history["eval_avg_returns"].append(metrics["avg_return"])
            history["eval_avg_lengths"].append(metrics["avg_length"])

            if metrics["avg_return"] > best_eval_return:
                best_eval_return = metrics["avg_return"]
                history["best_eval_return"] = best_eval_return
                if checkpoint_path is not None:
                    torch.save(net.state_dict(), checkpoint_path)

            if metrics["avg_return"] >= solved_threshold:
                solved = True
                history["solved"] = True
                if stop_when_solved:
                    break

    if last_checkpoint_path is not None:
        torch.save(net.state_dict(), last_checkpoint_path)
    net.eval()
    history["solved"] = solved
    return history


def load_a2c_checkpoint(net: nn.Module, checkpoint_path: str | Path) -> nn.Module:
    """Carica un checkpoint A2C in una rete.

    Argomenti:
        net: Rete con la stessa architettura del checkpoint.
        checkpoint_path: Percorso di un checkpoint salvato come state dict grezzo
            oppure come dizionario contenente `model_state_dict`.

    Operazione:
        Supporta sia i checkpoint esplorativi storici sia il formato pulito
        usato nel progetto.

    Output:
        La rete in input caricata con i pesi e impostata in modalità eval.
    """

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    net.load_state_dict(state_dict)
    net.eval()
    return net
