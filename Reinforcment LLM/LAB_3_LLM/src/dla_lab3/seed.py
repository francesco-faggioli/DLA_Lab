from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds.

    Args:
        seed: Integer seed used by Python, NumPy and PyTorch.

    What it does:
        Reduces run-to-run variability in experiments. Reinforcement learning
        remains stochastic because environments and sampled actions can still
        produce high-variance trajectories.

    Outputs:
        None. The function changes global random states.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
