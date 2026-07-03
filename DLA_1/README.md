# DLA Lab 1 - GTSRB Transfer Learning

This folder contains the Lab 1 work on the German Traffic Sign Recognition Benchmark (GTSRB).

The refactored project separates reusable Python code from notebooks:

- `config/`: reproducible experiment configuration.
- `src/dla_lab1/`: reusable data, model, training and evaluation utilities.
- `scripts/`: command-line entry points for environment checks, feature extraction and experiments.
- `notebooks/`: clean report notebooks organized by exercise.
- `data/`, `artifacts/`, `checkpoints/`, `wandb/`: local/generated folders ignored by Git.

## Reproducibility Notes

- Dataset: GTSRB, 43 classes.
- Local train split source: `data/gtsrb/GTSRB/Training`.
- Local test split source: `data/gtsrb/GTSRB/Final_Test` plus `data/gtsrb/GT-final_test.csv`.
- Default image size: `64x64`.
- Default hardware profile: NVIDIA RTX 2050 with 4 GB VRAM.
- Default training path: ResNet-18 transfer learning.

## Suggested Workflow

1. Run `scripts/check_env.py`.
2. Run `notebooks/01_eda_and_feature_baseline.ipynb`.
3. Run `notebooks/02_finetuning_pipeline.ipynb`.
4. Run `notebooks/03_improvements_and_retrieval.ipynb`.
5. Keep final interpretation, limitations and AI/tooling disclosure in notebook markdown.

## Environment Recommendation

The current local `DLA2026` environment is Python 3.14.3. That is newer than the most common PyTorch/scientific-Python teaching stack. If `scripts/check_env.py` reports native import failures for NumPy or pandas, recreate the environment from `environment.yml` with Python 3.12 before running long experiments.

## Academic Integrity and AI Disclosure

The course code of conduct allows LLM-assisted work only when transparent. Any final submission should state:

- which AI tools were used;
- whether they helped with code generation, refactoring, debugging or writing;
- which outputs were manually checked and modified;
- external sources used for datasets, pretrained models and libraries.

## Local Files Not Tracked by Git

The following are intentionally excluded:

- datasets and downloaded archives;
- model checkpoints and cached feature tensors;
- W&B logs;
- Python and notebook caches;
- local secrets such as API keys.
