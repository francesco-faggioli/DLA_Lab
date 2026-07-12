# Lab 2 Notebook Structure

The original notebooks remain in `DLA_Lab2/` for traceability. The clean notebooks in this folder are the intended final report structure.

1. `01_sentiment_dataset_tokenizer_baseline.ipynb`
   - Exercise 1.1 dataset loading and exploration;
   - Exercise 1.2 DistilBERT tokenizer/model inspection;
   - Exercise 1.3 frozen DistilBERT feature extraction plus Linear SVM baseline.

2. `02_distilbert_full_finetuning.ipynb`
   - Exercise 2.1 token preprocessing with `Dataset.map`;
   - Exercise 2.2 sequence classification model setup;
   - Exercise 2.3 Hugging Face `Trainer` fine-tuning and final evaluation.

3. `03_efficient_finetuning_sentiment.ipynb`
   - Exercise 3.1 efficient fine-tuning;
   - LoRA for sequence classification;
   - partial freezing baseline;
   - comparison by metrics and trainable parameter percentage.

4. `04_clip_adapter_imagenet_sketch.ipynb`
   - Exercise 3.2 CLIP zero-shot evaluation;
   - ImageNet-Sketch domain shift;
   - prompt study;
   - CLIP-Adapter parameter-efficient adaptation.

Keep code cells short. Reusable logic belongs in `src/dla_lab2`.

Kernel mapping:

- Notebooks 01, 02 and 03: `DLA2026-transformers`.
- Notebook 04: `clip_lora`.

Before final submission, rerun the notebooks and update the final markdown conclusions with the actual metrics from that run.
