# DLA 1 - GTSRB Computer Vision Laboratory

## Overview

This laboratory studies GTSRB traffic-sign classification through exploratory analysis, pretrained CNN feature extraction, a classical SVM baseline, ResNet-18 fine-tuning, pipeline consolidation and retrieval-based classification.

## Official assignment

- GitHub notebook: [`ASSIGNMENT.ipynb`](ASSIGNMENT.ipynb)
- Markdown copy: [`ASSIGNMENT.md`](ASSIGNMENT.md)

## Assignment coverage

| Assignment requirement | Notebook section | Status | Evidence |
| --- | --- | ---: | --- |
| Exercise 1.1 - EDA GTSRB | `01_eda_and_feature_baseline.ipynb` | Completed | Dataset statistics and class distribution |
| Exercise 1.2 - stable baseline | `01_eda_and_feature_baseline.ipynb` | Completed | ResNet-18 features + SVM, test accuracy 0.6412 |
| Exercise 1.3 - fine-tuning baseline | `02_finetuning_pipeline.ipynb` | Completed | Baseline fine-tuning, test accuracy 0.5038 |
| Exercise 2 - reproducible pipeline | `02b_pipeline_consolidation.ipynb` | Completed | Config-driven pipeline, local artifacts and W&B logs |
| Exercise 3.1 - improve fine-tuning | `03_improvements_and_retrieval.ipynb` | Completed | Improved model, test accuracy 0.8025 |
| Exercise 3.2 - retrieval/NMC | `03b_retrieval_training_free_classification.ipynb` | Completed | Precision@1 0.4812 and NMC accuracy 0.4185 |

## Results summary

| Experiment | Metric | Observed value |
| --- | --- | ---: |
| ResNet-18 feature extractor + SVM | Test accuracy | 0.6412 |
| Fine-tuning baseline | Test accuracy | 0.5038 |
| Improved fine-tuning | Test accuracy | 0.8025 |
| Retrieval, cosine similarity | Precision@1 | 0.4812 |
| Retrieval, nearest-mean classifier | Test accuracy | 0.4185 |

![Best validation accuracy summary](figures/gtsrb_validation_accuracy_summary.svg)

## Weights & Biases usage

Weights & Biases was used to monitor training and validation metrics during multiple fine-tuning runs. Local W&B summaries were inspected to document learning rate, training accuracy, validation accuracy and validation loss. These logs are generated experiment evidence and are ignored by Git.

## Main dependencies and imports

| Package/import | Purpose | Used in |
| --- | --- | --- |
| `torch` | Neural-network training, tensors and checkpoint loading | Fine-tuning, evaluation |
| `torchvision` | GTSRB dataset and pretrained ResNet backbones | Dataset and feature extraction |
| `sklearn.svm.SVC` | Classical classifier over extracted features | Stable baseline |
| `pandas` | Result tables and summaries | Notebooks and reports |
| `matplotlib` | Plots and visual checks | EDA and training curves |
| `wandb` | Optional experiment tracking | Pipeline consolidation |

## Local functions and source files

| Function/class | Defined in | Purpose | Main inputs | Main outputs | Used in |
| --- | --- | --- | --- | --- | --- |
| `load_config` | `src/dla_lab1/config.py` | Load YAML configuration | config path | dict config | All notebooks |
| `build_dataloaders` | `src/dla_lab1/data.py` | Build train/validation/test loaders | data root, split config | dataloaders and metadata | Fine-tuning |
| `build_classifier` | `src/dla_lab1/models.py` | Create ResNet classifier | model name, num classes | PyTorch model | Fine-tuning |
| `run_finetuning` | `src/dla_lab1/experiments.py` | Execute configured experiment | config, experiment name | model, history, artifacts | Exercises 1.3/3.1 |
| `extract_features` | `src/dla_lab1/features.py` | Extract CNN embeddings | model, dataloader | feature tensors | Baseline/retrieval |
| `classification_metrics` | `src/dla_lab1/evaluate.py` | Compute accuracy and report | y_true, y_pred | metrics dict | Evaluation |

## External sources and references

| Source | Used for | Adaptation or contribution |
| --- | --- | --- |
| Official assignment | Requirements | Converted into `ASSIGNMENT.ipynb`. |
| GTSRB benchmark | Dataset | Traffic-sign classification data. |
| Torchvision documentation | Dataset/model APIs | GTSRB wrapper and pretrained ResNet. |
| Scikit-learn documentation | SVM and metrics | Classical baseline and reports. |
| Weights & Biases documentation | Experiment tracking | Optional monitoring of Lab 1 runs. |
