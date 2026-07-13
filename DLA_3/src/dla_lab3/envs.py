from __future__ import annotations

import gymnasium as gym
import torch


def make_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    **kwargs,
) -> gym.Env:
    """Crea un ambiente Gymnasium e ne imposta il seed.

    Argomenti:
        env_id: Identificativo Gymnasium, per esempio `CartPole-v1` o
            `LunarLander-v3`.
        seed: Seed opzionale usato per `reset` e per lo spazio delle azioni.
        render_mode: Modalità di rendering opzionale, per esempio `human`.
        **kwargs: Argomenti aggiuntivi passati a `gym.make`.

    Operazione:
        Istanzia l'ambiente e applica il seed in modo coerente. Non chiude
        automaticamente l'ambiente restituito.

    Output:
        Ambiente Gymnasium pronto per l'uso.
    """

    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def observation_scale(env_id: str) -> torch.Tensor:
    """Restituisce la scala delle osservazioni per gli ambienti supportati.

    Argomenti:
        env_id: Identificativo contenente `CartPole` o `LunarLander`.

    Operazione:
        Fornisce valori fissi per normalizzare le osservazioni prima delle reti.
        Per le velocità CartPole non limitate usa valori pratici finiti.

    Output:
        Tensore float con un valore per dimensione dell'osservazione.

    Eccezioni:
        ValueError: Se l'ambiente non è supportato.
    """

    if "CartPole" in env_id:
        return torch.tensor([4.8, 5.0, 0.41887903, 5.0], dtype=torch.float32)
    if "LunarLander" in env_id:
        return torch.tensor(
            [2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0],
            dtype=torch.float32,
        )
    raise ValueError(f"Nessuna scala delle osservazioni configurata per l'ambiente: {env_id}")
