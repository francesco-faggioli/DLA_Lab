# DLA 3 notebook order

Linux or WSL is recommended. Run or inspect the final notebooks in this order:

1. `01_cartpole_reinforce_evaluation.ipynb` — REINFORCE and periodic multi-episode evaluation.
2. `02_cartpole_value_baseline.ipynb` — raw/standardized returns and learned value baseline.
3. `03_a2c_cartpole_lunarlander.ipynb` — A2C validation on CartPole and final LunarLander study.

The final report is [`../README.md`](../README.md) and lightweight result evidence is under [`../results/`](../results/). Training, large sweeps, and rendering are opt-in through `RUN_*` flags. Checkpoint evaluation is performed only when a local checkpoint exists.

The exploratory notebook is preserved at `../exploratory/00_esperimenti_di_prova_a2c.ipynb`. It is not part of the final execution order and its early trials are not used as headline results.
