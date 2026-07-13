# Ordine dei notebook DLA 3

Si consiglia Linux o WSL. Eseguire o consultare i notebook finali nel seguente ordine:

1. `01_cartpole_reinforce_evaluation.ipynb` — REINFORCE e valutazione periodica su più episodi.
2. `02_cartpole_value_baseline.ipynb` — return grezzi/standardizzati e value baseline appresa.
3. `03_a2c_cartpole_lunarlander.ipynb` — validazione A2C su CartPole e studio finale LunarLander.

La relazione finale è in [`../README.md`](../README.md) e le evidenze leggere dei risultati si trovano in [`../results/`](../results/). Training, sweep estesi e rendering sono attivabili esplicitamente tramite flag `RUN_*`. La valutazione dei checkpoint viene eseguita soltanto quando il checkpoint locale esiste.
