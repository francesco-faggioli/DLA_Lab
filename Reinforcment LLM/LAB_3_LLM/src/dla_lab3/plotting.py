from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    """Compute a simple moving average.

    Args:
        values: Sequence of numeric values.
        window: Number of values per averaging window.

    What it does:
        Applies a uniform convolution to smooth noisy RL curves.

    Outputs:
        NumPy array with the moving average. Returns an empty array if there are
        fewer values than `window`.
    """

    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < window:
        return np.asarray([], dtype=np.float32)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_training_returns(history: dict, title: str, solved_threshold: float | None = None, window: int = 50) -> None:
    """Plot raw episode returns and a moving average.

    Args:
        history: Training history returned by the training functions.
        title: Plot title.
        solved_threshold: Optional horizontal reference line.
        window: Moving-average window.

    What it does:
        Visualizes noisy episode returns and their smoothed trend.

    Outputs:
        None. It displays a matplotlib figure.
    """

    returns = history["episode_returns"]
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.35, label="Episode return")
    ma = moving_average(returns, window=window)
    if len(ma) > 0:
        plt.plot(range(window - 1, window - 1 + len(ma)), ma, linewidth=2, label=f"Moving avg ({window})")
    if solved_threshold is not None:
        plt.axhline(solved_threshold, linestyle="--", color="tab:green", label="Solved threshold")
    plt.title(title)
    plt.xlabel("Training episode")
    plt.ylabel("Total reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_evaluation(history: dict, title: str, solved_threshold: float | None = None) -> None:
    """Plot periodic evaluation reward and episode length.

    Args:
        history: Training history returned by the training functions.
        title: Plot title prefix.
        solved_threshold: Optional horizontal reference line for reward.

    What it does:
        Plots the two metrics requested by the exercise: average total reward
        and average episode length over evaluation episodes.

    Outputs:
        None. It displays a matplotlib figure.
    """

    x = history["eval_episodes"]
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, history["eval_avg_returns"], marker="o", markersize=4)
    if solved_threshold is not None:
        plt.axhline(solved_threshold, linestyle="--", color="tab:green")
    plt.title(f"{title} - average return")
    plt.xlabel("Training episode")
    plt.ylabel("Average total reward")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x, history["eval_avg_lengths"], marker="o", markersize=4, color="tab:orange")
    plt.title(f"{title} - average length")
    plt.xlabel("Training episode")
    plt.ylabel("Average episode length")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
