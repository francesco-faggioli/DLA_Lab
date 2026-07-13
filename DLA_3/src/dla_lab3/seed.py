from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Imposta i seed di Python, NumPy e PyTorch.

    Argomenti:
        seed: Seed intero condiviso dai tre generatori.

    Operazione:
        Riduce la variabilità tra run. Il reinforcement learning resta stocastico
        perché ambienti e azioni campionate possono produrre traiettorie diverse.

    Output:
        Nessuno. Modifica lo stato globale dei generatori casuali e, se
        disponibile, quello di tutti i device CUDA.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
