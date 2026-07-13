from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    """Calcola una media mobile semplice.

    Argomenti:
        values: Sequenza di valori numerici.
        window: Numero di valori per finestra.

    Operazione:
        Applica una convoluzione uniforme per smussare curve RL rumorose.

    Output:
        Array NumPy con la media mobile; è vuoto se i valori sono meno di `window`.
    """

    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < window:
        return np.asarray([], dtype=np.float32)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_training_returns(
    history: dict, title: str, solved_threshold: float | None = None, window: int = 50
) -> None:
    """Traccia i return grezzi per episodio e la loro media mobile.

    Argomenti:
        history: Cronologia restituita dalle funzioni di training.
        title: Titolo del grafico.
        solved_threshold: Linea di riferimento orizzontale opzionale.
        window: Ampiezza della media mobile.

    Operazione:
        Visualizza return rumorosi e tendenza smussata sul device CPU.

    Output:
        Nessuno; mostra una figura Matplotlib senza scrivere file.
    """

    returns = history["episode_returns"]
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.35, label="Episode return")
    ma = moving_average(returns, window=window)
    if len(ma) > 0:
        plt.plot(
            range(window - 1, window - 1 + len(ma)), ma, linewidth=2, label=f"Moving avg ({window})"
        )
    if solved_threshold is not None:
        plt.axhline(
            solved_threshold, linestyle="--", color="tab:green", label="Soglia di soluzione"
        )
    plt.title(title)
    plt.xlabel("Training episode")
    plt.ylabel("Total reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_evaluation(history: dict, title: str, solved_threshold: float | None = None) -> None:
    """Traccia reward e lunghezza delle valutazioni periodiche.

    Argomenti:
        history: Cronologia restituita dalle funzioni di training.
        title: Prefisso del titolo.
        solved_threshold: Linea di riferimento opzionale per il reward.

    Operazione:
        Mostra reward totale medio e lunghezza media sugli episodi di valutazione.

    Output:
        Nessuno; mostra una figura Matplotlib senza scrivere file.
    """

    x = history["eval_episodes"]
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, history["eval_avg_returns"], marker="o", markersize=4)
    if solved_threshold is not None:
        plt.axhline(solved_threshold, linestyle="--", color="tab:green")
    plt.title(f"{title} - average return")
    plt.xlabel("Training episode")
    plt.ylabel("Reward totale medio")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x, history["eval_avg_lengths"], marker="o", markersize=4, color="tab:orange")
    plt.title(f"{title} - average length")
    plt.xlabel("Training episode")
    plt.ylabel("Lunghezza media dell'episodio")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
