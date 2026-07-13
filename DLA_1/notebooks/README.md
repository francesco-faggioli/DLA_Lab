# Ordine dei notebook DLA 1

Eseguire o consultare i notebook finali nel seguente ordine:

1. `01_eda_and_feature_baseline.ipynb` — EDA GTSRB e baseline con feature ResNet-18/SVM.
2. `02_finetuning_pipeline.ipynb` — baseline di fine-tuning head-only.
3. `02b_pipeline_consolidation.ipynb` — pipeline modulare, configurazione, artefatti e W&B facoltativo.
4. `03_improvements_and_retrieval.ipynb` — fine-tuning selettivo e studio dei miglioramenti.
5. `03b_retrieval_training_free_classification.ipynb` — cosine retrieval, Precision@K, mAP e NMC.

La relazione finale è in [`../README.md`](../README.md). Le evidenze leggere si trovano in [`../results/`](../results/). Training lunghi ed estrazione delle feature richiedono flag espliciti `RUN_* = True`; W&B resta disattivato a meno di impostare `ENABLE_WANDB = True` dopo un login manuale.
