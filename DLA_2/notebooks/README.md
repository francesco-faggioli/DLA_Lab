# Ordine dei notebook DLA 2

Eseguire o consultare i notebook finali nel seguente ordine:

1. `01_sentiment_dataset_tokenizer_baseline.ipynb` — dataset, tokenizer, feature DistilBERT congelate e SVM.
2. `02_distilbert_full_finetuning.ipynb` — fine-tuning completo per sequence classification.
3. `03_efficient_finetuning_sentiment.ipynb` — LoRA e congelamento parziale.
4. `04_clip_adapter_imagenet_sketch.ipynb` — CLIP zero-shot, studio dei prompt e CLIP-Adapter.

La relazione finale è in [`../README.md`](../README.md) e le evidenze numeriche pubbliche si trovano in [`../results/`](../results/). Nella modalità rapida di consegna, download dei modelli, estrazione delle feature, preprocessing e training vengono saltati a meno che il relativo flag `RUN_*` sia abilitato esplicitamente. Gli output già eseguiti restano visibili.

Le run originali hanno usato `DLA2026-transformers` per i notebook 01–03 e `clip_lora` per il notebook 04. Il `requirements.txt` nella root è la lista canonica delle dipendenze per la consegna.
