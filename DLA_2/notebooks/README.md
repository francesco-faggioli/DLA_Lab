# DLA 2 notebook order

Run or inspect the final notebooks in this order:

1. `01_sentiment_dataset_tokenizer_baseline.ipynb` — dataset, tokenizer, frozen DistilBERT features, and SVM.
2. `02_distilbert_full_finetuning.ipynb` — full sequence-classification fine-tuning.
3. `03_efficient_finetuning_sentiment.ipynb` — LoRA and partial freezing.
4. `04_clip_adapter_imagenet_sketch.ipynb` — zero-shot CLIP, prompt study, and CLIP-Adapter.

The final report is [`../README.md`](../README.md) and the public numerical evidence is under [`../results/`](../results/). In the submitted quick mode, model downloads, feature extraction, preprocessing, and training are skipped unless the corresponding `RUN_*` flag is explicitly enabled. Existing executed outputs remain visible.

The original runs used `DLA2026-transformers` for notebooks 01–03 and `clip_lora` for notebook 04. The root `requirements.txt` is the canonical submission dependency list.
