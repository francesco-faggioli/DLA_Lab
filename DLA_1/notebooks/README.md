# DLA 1 notebook order

Run or inspect the final notebooks in this order:

1. `01_eda_and_feature_baseline.ipynb` — GTSRB EDA and ResNet-18 feature/SVM baseline.
2. `02_finetuning_pipeline.ipynb` — head-only fine-tuning baseline.
3. `02b_pipeline_consolidation.ipynb` — modular pipeline, configuration, artifacts, and optional W&B.
4. `03_improvements_and_retrieval.ipynb` — selective fine-tuning and improvement study.
5. `03b_retrieval_training_free_classification.ipynb` — cosine retrieval, Precision@K, mAP, and NMC.

The final report is [`../README.md`](../README.md). Lightweight evidence is under [`../results/`](../results/). Long training and feature extraction require explicit `RUN_* = True` flags; W&B is disabled unless `ENABLE_WANDB = True` is set after a manual login.

The historical trial notebook was moved to `../exploratory/Esperimenti_di_prova.ipynb`. It is retained for provenance but is not part of this execution order and is not a source for headline metrics.
