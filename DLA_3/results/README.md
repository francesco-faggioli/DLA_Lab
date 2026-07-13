# Evidenze dei risultati del Lab 3

Questi file espongono le evidenze numeriche leggere alla base della relazione, mentre i checkpoint restano ignorati.

| File | Contenuto | Fonte primaria |
| --- | --- | --- |
| `cartpole_evaluation.csv` | Valutazione periodica delle tre varianti REINFORCE | Notebook della value baseline eseguito |
| `method_summary.csv` | Metriche principali CartPole e LunarLander | Notebook eseguiti |
| `lunarlander_final_evaluation.json` | Aggregato completo su 200 episodi | Notebook A2C eseguito e artefatto JSON locale |
| `lunarlander_temperature_sweep.csv` | Variabilità del return e success rate tra temperature per il checkpoint selezionato | Artefatto JSON locale della configurazione della policy |
| `lunarlander_policy_mode_comparison.csv` | Policy greedy rispetto alla policy stocastica selezionata | Artefatto JSON locale della configurazione della policy |
| `cartpole_visual_episodes.json` | Cinque rollout qualitativi greedy, seed e flag terminali, senza frame | Esecuzione headless diretta di `run_cartpole_visual_episodes` con `cartpole_reinforce_value_baseline.pt` |
| `lunarlander_visual_episodes.json` | Cinque rollout qualitativi campionati, seed e flag terminali, senza frame | Controllo headless precedente con lo stesso checkpoint, modalità sampled e `T=0.85`; distinto dalla valutazione finale a `T=0.75` |

Il JSON finale memorizza intenzionalmente soltanto il nome del file di checkpoint, non il percorso WSL assoluto obsoleto presente nell'artefatto locale.

Il precedente `a2c_lunarlander_temperature_sweep.json` descriveva una vecchia valutazione e non viene deliberatamente usato. Lo sweep versionato deriva da `a2c_lunarlander_policy_config_selection.json`, l'evidenza usata dal notebook finale per selezionare la temperatura `0.75`.

I due JSON visuali sono stati generati durante la revisione finale della repository. Documentano soltanto brevi controlli qualitativi e non modificano né ampliano le metriche principali di valutazione.
