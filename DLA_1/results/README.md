# Evidenze dei risultati del Lab 1

Questi file leggeri rendono ispezionabili le metriche riportate senza pubblicare dataset, log W&B o checkpoint dei modelli.

| File | Contenuto | Fonte primaria |
| --- | --- | --- |
| `dataset_summary.json` | Cardinalità, sbilanciamento e dimensioni delle immagini GTSRB | Output di `01_eda_and_feature_baseline.ipynb` |
| `test_metrics.csv` | Metriche finali di classificazione e retrieval | Output dei notebook eseguiti |
| `run_validation_summary.csv` | Confronto in validazione tra le run salvate della pipeline | File locali `artifacts/runs/*/summary.json` |
| `baseline_history.csv` | Cronologia del fine-tuning head-only | Artefatto locale della pipeline e output del notebook |
| `improved_history.csv` | Cronologia della run migliorata selezionata | Artefatto locale della pipeline e output del notebook |

L'accuracy di test `0.8025` è preservata dall'output finale di test del notebook dei miglioramenti. I file di validazione descrivono le run denominate della pipeline e non devono essere interpretati come una nuova valutazione sul test.
