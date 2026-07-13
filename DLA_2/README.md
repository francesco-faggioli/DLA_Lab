# DLA 2 — Transformers, Efficient Fine-tuning, and CLIP

## Overview

This laboratory contains two distinct studies. The first performs binary sentiment classification on Rotten Tomatoes with DistilBERT, moving from frozen features to full and parameter-efficient fine-tuning. The second evaluates CLIP on ImageNet-Sketch and trains a small CLIP-Adapter while keeping the vision-language backbone frozen.

The two accuracy groups are reported separately because they belong to different datasets and prediction spaces. All values below are linked to executed notebook outputs mirrored in [`results/`](results/).

**Final index:** [`DLA_2.ipynb`](DLA_2.ipynb)

**Official assignment:** [`ASSIGNMENT.md`](ASSIGNMENT.md)

**Notebook guide:** [`notebooks/README.md`](notebooks/README.md)

## Objectives and assignment coverage

| Requirement | Implementation | Evidence | Status |
| --- | --- | --- | ---: |
| Exercise 1.1: dataset | Load and inspect Rotten Tomatoes splits and examples | [`01_sentiment_dataset_tokenizer_baseline.ipynb`](notebooks/01_sentiment_dataset_tokenizer_baseline.ipynb) | Completed |
| Exercise 1.2: Transformer/tokenizer | Inspect token IDs, masks, decoded tokens, hidden state | Same notebook | Completed |
| Exercise 1.3: stable baseline | DistilBERT `[CLS]` features + linear SVM | Test accuracy `0.7946` | Completed |
| Exercise 2: full fine-tuning | Dynamic padding, sequence-classification head, Trainer | [`02_distilbert_full_finetuning.ipynb`](notebooks/02_distilbert_full_finetuning.ipynb) | Completed |
| Exercise 3.1: efficient fine-tuning | LoRA and partial freezing | [`03_efficient_finetuning_sentiment.ipynb`](notebooks/03_efficient_finetuning_sentiment.ipynb) | Completed |
| Exercise 3.2: CLIP adaptation | Prompt study and CLIP-Adapter on ImageNet-Sketch | [`04_clip_adapter_imagenet_sketch.ipynb`](notebooks/04_clip_adapter_imagenet_sketch.ipynb) | Completed |

## Theoretical background

DistilBERT maps tokenized text to contextual hidden states. `input_ids` select vocabulary entries; `attention_mask` distinguishes real tokens from padding. The first hidden vector corresponds to `[CLS]` and is used here as a fixed sentence representation for the SVM baseline. For end-to-end classification, `AutoModelForSequenceClassification` adds a task head and updates the backbone with cross-entropy.

LoRA keeps the pretrained matrices frozen and learns low-rank updates. This implementation targets DistilBERT attention projections `q_lin` and `v_lin`. Partial freezing is a simpler control: embeddings and the first four Transformer blocks remain frozen while later blocks and the classifier are optimized.

CLIP aligns image and text embeddings. Zero-shot classification compares each image to text prototypes derived from prompts. CLIP-Adapter inserts a small residual MLP on frozen image features, providing domain adaptation without updating the CLIP encoders.

## Part I — Rotten Tomatoes sentiment analysis

### Dataset

The Hugging Face dataset contains one text field and a binary label (`0` negative, `1` positive). The submitted run observed the following fixed splits:

| Split | Rows | Labels |
| --- | ---: | --- |
| Train | 8,530 | 0 and 1 |
| Validation | 1,066 | 0 and 1 |
| Test | 1,066 | 0 and 1 |

The source corpus contains 5,331 positive and 5,331 negative sentences overall. Both labels are present in each observed split; the notebook did not save a separate per-split class-count table, so no more granular balance claim is made. Representative examples are sampled with seed `42`. Source: [`dataset_summary.csv`](results/dataset_summary.csv) and the [dataset card](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).

### Tokenization and preprocessing

The tokenizer is loaded with `distilbert/distilbert-base-uncased`. The notebooks inspect decoded `[CLS]`, `[SEP]`, and `[PAD]` tokens, the shape of `input_ids` and `attention_mask`, and the final hidden state of shape `(batch, sequence_length, 768)`. Dataset preprocessing uses a maximum length of `256` and leaves padding to `DataCollatorWithPadding`, so each batch is padded only to its longest sequence. No independent sequence-length distribution was saved; truncation risk beyond 256 tokens is therefore a documented limitation rather than an estimated quantity.

### Stable baseline: frozen DistilBERT + SVM

The baseline extracts the first-token hidden representation from all three splits, producing 768-dimensional features, and fits a linear SVM. DistilBERT remains frozen.

| Parameter | Value |
| --- | ---: |
| Model | `distilbert-base-uncased` |
| Feature | Last hidden state at token index 0 (`[CLS]`) |
| Feature dimension | 768 |
| Batch size | 32 |
| SVM | Linear, C=1.0 |
| Seed | 42 |

| Split | Accuracy | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.8180 | 0.8142 | 0.8317 | 0.7974 |
| Test | **0.7946** | 0.7908 | 0.8054 | 0.7767 |

The validation-to-test drop is `0.0235`, so the stable baseline provides a useful but not perfectly transferable reference. It is computationally attractive because the expensive Transformer forward pass can be cached and the classifier is small.

### Full DistilBERT fine-tuning

`AutoModelForSequenceClassification` adds a randomly initialized pre-classifier and binary head. Hugging Face `Trainer` handles optimization and periodic validation; `DataCollatorWithPadding` performs dynamic padding. Accuracy, F1, precision, and recall are computed with scikit-learn.

| Parameter | Value |
| --- | ---: |
| Epochs | 3 |
| Train / evaluation batch size | 32 / 32 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warm-up steps | 81 of 801 total steps |
| Evaluation / save cadence | Once per epoch |
| Model selection | Validation metric recorded by Trainer |
| Seed | 42 |

![DistilBERT validation history](figures/distilbert_validation_history.svg)

Validation accuracy rose from `0.8368` to `0.8527`, while validation loss was lowest at epoch 2 (`0.3577`) and increased to `0.3872` at epoch 3. This divergence is a mild overfitting signal: the decision boundary improved in accuracy while confidence calibration degraded. The final test result was accuracy `0.8443`, F1 `0.8428`, precision `0.8509`, and recall `0.8349`.

### Efficient fine-tuning

| Parameter | Full fine-tuning | LoRA | Partial freezing |
| --- | ---: | ---: | ---: |
| Epochs | 3 | 3 | 3 |
| Learning rate | 2e-5 | 1e-3 | 2e-5 |
| Batch size | 32 | 32 | 32 |
| Weight decay | 0.01 | 0.01 | 0.01 |
| LoRA rank / alpha / dropout | — | 8 / 16 / 0.1 | — |
| Frozen blocks | None | Backbone except LoRA + head | Embeddings + first 4 blocks |
| Trainable parameters | 66,955,010 | 739,586 | 14,767,874 |
| Trainable share | 100% | **1.09%** | 22.06% |

![Trainable parameter share](figures/sentiment_trainable_parameters.svg)

LoRA is the clearest efficiency result: it uses roughly one twentieth of the trainable share of partial freezing and approximately one hundredth of full fine-tuning. This table measures trainable parameters, not wall-clock time or peak memory; those costs were not systematically logged.

### Quantitative comparison

| Method | Test accuracy | Test F1 | Difference from full accuracy |
| --- | ---: | ---: | ---: |
| DistilBERT `[CLS]` + SVM | 0.7946 | 0.7908 | -0.0497 |
| Full fine-tuning | **0.8443** | **0.8428** | — |
| LoRA | 0.8386 | 0.8349 | -0.0056 |
| Partial freezing | 0.8377 | 0.8348 | -0.0066 |

![Sentiment method comparison](figures/sentiment_test_accuracy.svg)

Full fine-tuning achieved the highest accuracy, but the margin over LoRA was only `0.0056`. Partial freezing reached almost the same score as LoRA while updating about twenty times more parameters. The fixed-feature SVM was clearly weaker, which indicates that task-specific adaptation of contextual layers was useful.

No confusion matrix or per-example prediction file was saved in the final sentiment notebooks. Precision, recall, and F1 are available and reported, but a confusion matrix is not reconstructed from aggregate metrics.

## Part II — CLIP on ImageNet-Sketch

### Dataset and domain shift

ImageNet-Sketch contains hand-drawn representations of ImageNet classes. The run used an `80/20` split with `40,711` train and `10,178` external-validation images. Adapter training used a reproducible 5,000-image subset, split internally into `4,500` training and `500` adapter-validation feature vectors. The external validation set remained complete.

This task is deliberately out-of-domain: CLIP sees sketches rather than natural photographs. Its 1,000-class accuracy is therefore not comparable to the binary Rotten Tomatoes accuracy.

### Zero-shot protocol

The model is OpenCLIP `ViT-B-16-quickgelu` with OpenAI weights. Five prompt templates were tested. The best single prompt, `a hand-drawn sketch of a {}`, reached `0.4759`; the worst tested prompt reached `0.4622`. Averaging text features over the five prompts produced `0.4789`, only `+0.0029` over the selected single-prompt baseline.

### CLIP-Adapter configuration

| Parameter | Value |
| --- | ---: |
| CLIP backbone | `ViT-B-16-quickgelu`, OpenAI weights |
| Image-feature dimension | 512 |
| Frozen components | CLIP image and text encoders |
| Adapter bottlenecks | 64 and 128 |
| Residual alpha | 0.6 |
| Optimizer learning rate | 2e-3 |
| Epochs | 30 |
| Image preprocessing batch | 64 |
| Adapter training batch | 256 |
| Seed | 42 |

### CLIP results

| Method | Accuracy | Gain over single-prompt zero-shot |
| --- | ---: | ---: |
| Zero-shot CLIP | 0.4759 | — |
| Zero-shot + prompt ensemble | 0.4789 | +0.0029 |
| Adapter, bottleneck 64 | 0.5144 | +0.0385 |
| Adapter 64 + prompt ensemble | 0.5067 | +0.0308 |
| Adapter, bottleneck 128 | **0.5241** | **+0.0481** |
| Adapter 128 + prompt ensemble | 0.5193 | +0.0433 |

![CLIP comparison](figures/clip_accuracy_comparison.svg)

The 128-unit adapter gave the strongest external-validation score. Prompt ensembling improved zero-shot CLIP slightly but reduced both adapter scores, so it was not retained as the best configuration.

![CLIP-Adapter training loss](figures/clip_adapter_training_loss_tail.svg)

In the final five epochs, the 128-unit adapter's training loss remained near `0.094`, while its validation loss remained near `2.18`. The 64-unit model showed the same qualitative gap. This is evidence of overfitting to the 4,500 feature-training examples, even though both adapters improved external top-1 accuracy over zero-shot CLIP. The result is useful but should not be read as fully calibrated adaptation.

## What worked, what did not, and why

**Worked:** full DistilBERT adaptation improved over fixed features; LoRA retained most of that gain with very few trainable parameters; dynamic padding avoided global maximum-length padding; CLIP-Adapter improved sketch-domain top-1 accuracy without updating CLIP.

**Did not work as well:** the frozen `[CLS]` SVM lagged fine-tuning; partial freezing offered no accuracy advantage over LoRA despite many more trainable parameters; prompt ensembling did not improve the adapters; adapter validation loss showed substantial overfitting.

**Problems resolved:** dataset-loading fallbacks support the local Hugging Face cache; Trainer construction and Unicode/progress output were centralized; feature preprocessing avoids recomputing the CLIP image encoder for every prompt; expensive training calls are controlled in the final quick-inspection path.

## Limitations

- Results use one seed; uncertainty across fine-tuning runs was not estimated.
- No wall-clock, VRAM, energy, or inference-latency benchmark was saved, so efficiency conclusions are parameter-count conclusions.
- Sequence-length statistics and sentiment confusion matrices were not preserved.
- CLIP used only 5,000 training images and two bottleneck sizes; the search is not exhaustive.
- Dataset/model downloads and Hugging Face checkpoints are not versioned.
- ImageNet-Sketch results are external-validation measurements, not a hidden test-set benchmark.

## Reproducibility

The canonical defaults are in [`config/lab2_defaults.yaml`](config/lab2_defaults.yaml). In quick mode, training is skipped and the saved output/evidence remains available for inspection. Full mode requires setting the corresponding notebook `RUN_*` flag, downloading models/datasets, and using a CUDA-capable environment for practical execution time.

The sentiment notebooks were run in the `DLA2026-transformers` environment and the CLIP notebook in `clip_lora`; the root `requirements.txt` unifies their Python packages for submission. Regenerate figures with `python tools/build_report_assets.py` from the repository root.

## File structure

| Path | Purpose |
| --- | --- |
| `DLA_2.ipynb` | GitHub-readable laboratory index |
| `notebooks/01_*` | Dataset, tokenizer, and stable sentiment baseline |
| `notebooks/02_*` | Full DistilBERT fine-tuning |
| `notebooks/03_*` | LoRA and partial freezing |
| `notebooks/04_*` | CLIP and adapter study |
| `src/dla_lab2/` | Reusable sentiment and CLIP helpers |
| `config/lab2_defaults.yaml` | Main experiment defaults |
| `results/` | Versioned numerical evidence |
| `figures/` | Reproducible report figures |

## Main dependencies

| Library | Purpose | Used in |
| --- | --- | --- |
| PyTorch | Model execution and optimization | All experiments |
| Hugging Face Datasets | Dataset loading and mapping | Rotten Tomatoes and ImageNet-Sketch |
| Transformers | DistilBERT, tokenizer, Trainer, dynamic collator | Sentiment notebooks |
| PEFT | LoRA layers and model wrapping | Efficient fine-tuning |
| scikit-learn | Linear SVM and classification metrics | Baseline and evaluation |
| OpenCLIP | CLIP model, tokenizer, preprocessing | ImageNet-Sketch experiment |
| pandas / NumPy | Result tables and feature handling | All notebooks |

## Local modules and main functions

| Function/class | Source file | Purpose | Inputs | Outputs | Used in |
| --- | --- | --- | --- | --- | --- |
| `load_rotten_tomatoes` | [`sentiment.py`](src/dla_lab2/sentiment.py) | Load Hub data or verified local cache | Dataset identifier | DatasetDict | Notebooks 01–03 |
| `extract_cls_features_with_pipeline` | [`sentiment.py`](src/dla_lab2/sentiment.py) | Batch DistilBERT feature extraction | Texts, model, tokenizer, batch settings | NumPy feature matrix | Stable baseline |
| `build_training_arguments` | [`sentiment.py`](src/dla_lab2/sentiment.py) | Build compatible Trainer settings | Output path and hyperparameters | `TrainingArguments` | Full/efficient fine-tuning |
| `lora_sequence_classifier_init` | [`sentiment.py`](src/dla_lab2/sentiment.py) | Attach LoRA to DistilBERT attention | Rank, alpha, dropout, targets | PEFT sequence classifier | Notebook 03 |
| `partial_freezing_sequence_classifier_init` | [`sentiment.py`](src/dla_lab2/sentiment.py) | Freeze embeddings and early blocks | Model ID, number of frozen layers | Sequence classifier | Notebook 03 |
| `CLIPAdapter` | [`clip_utils.py`](src/dla_lab2/clip_utils.py) | Residual bottleneck adapter | Feature dimension, bottleneck, alpha | Adapted image features | Notebook 04 |
| `precompute_image_features` | [`clip_utils.py`](src/dla_lab2/clip_utils.py) | Cache frozen CLIP image embeddings | CLIP model, loader, device | TensorDataset | Prompt and adapter study |
| `train_clip_adapter` | [`clip_utils.py`](src/dla_lab2/clip_utils.py) | Optimize only adapter parameters | Adapter, loaders, text features, settings | Epoch history | Notebook 04 |

## AI use

AI assistance supported debugging, repository organization, explanation of efficient fine-tuning, and documentation. Metrics come from executed notebooks, not AI output. See [`../AI_USAGE.md`](../AI_USAGE.md).

## Sources

- [Rotten Tomatoes dataset card](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- [DistilBERT documentation](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [Hugging Face Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [OpenCLIP repository](https://github.com/mlfoundations/open_clip)
- [ImageNet-Sketch dataset and paper repository](https://github.com/HaohanWang/ImageNet-Sketch)

## Conclusion

The sentiment study shows a clear performance/efficiency trade-off: full fine-tuning is best by accuracy, while LoRA is close with only `1.09%` trainable parameters. The CLIP study shows that a small adapter can improve an out-of-domain sketch task, but its loss curves also expose overfitting. Both positive and negative findings are retained because they explain the method choices more accurately than a single headline score.
