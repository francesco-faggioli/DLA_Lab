from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "OneDrive" / "Desktop"
if not DESKTOP.exists():
    DESKTOP = Path.home() / "Desktop"
ASSIGNMENT_NOTE = (
    "> Official assignment provided by the course instructor.  \n"
    "> This file is included to document the requirements addressed by the submitted work."
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_text(text: str) -> str:
    text = text.strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def svg_bar_chart(path: Path, title: str, rows: list[tuple[str, float]], x_label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    row_h = 42
    left = 330
    top = 70
    bar_w = 540
    height = top + len(rows) * row_h + 70
    max_val = max([value for _, value in rows] + [1.0])
    max_axis = 1.0 if max_val <= 1 else math.ceil(max_val / 50) * 50
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="30" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{title}</text>',
    ]
    for index, (label, value) in enumerate(rows):
        y = top + index * row_h
        bar = 0 if max_axis == 0 else bar_w * value / max_axis
        shown = f"{value:.3f}" if max_axis <= 1.0 else f"{value:.1f}"
        parts.extend(
            [
                f'<text x="30" y="{y + 24}" font-family="Arial" font-size="15" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y + 8}" width="{bar_w}" height="24" rx="3" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y + 8}" width="{bar:.1f}" height="24" rx="3" fill="#2563eb"/>',
                f'<text x="{left + bar_w + 15}" y="{y + 25}" font-family="Arial" font-size="15" fill="#111827">{shown}</text>',
            ]
        )
    if x_label:
        parts.append(f'<text x="{left}" y="{height - 25}" font-family="Arial" font-size="13" fill="#4b5563">{x_label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def generate_assignments() -> None:
    for lab, source_name, title in [
        ("DLA_1", "1.txt", "Official Assignment - DLA 1"),
        ("DLA_2", "2.txt", "Official Assignment - DLA 2"),
        ("DLA_3", "3.txt", "Official Assignment - DLA 3"),
    ]:
        body = clean_text(read_text(DESKTOP / source_name))
        full = f"# {title}\n\n{ASSIGNMENT_NOTE}\n\n{body}"
        write_md(ROOT / lab / "ASSIGNMENT.md", full)
        write_notebook(ROOT / lab / "ASSIGNMENT.ipynb", [md_cell(f"# {title}\n\n{ASSIGNMENT_NOTE}"), md_cell(body)])


def generate_code_of_conduct() -> None:
    body = clean_text(read_text(DESKTOP / "Linee_Guida.txt"))
    if not body.lstrip().startswith("#"):
        body = "# DLA Code of Conduct\n\n" + body
    write_md(ROOT / "CODE_OF_CONDUCT.md", body)
    write_notebook(ROOT / "CODE_OF_CONDUCT.ipynb", [md_cell(body)])


def generate_figures() -> None:
    run_root = ROOT / "DLA_1" / "artifacts" / "runs"
    if run_root.exists():
        rows = []
        for summary in sorted(run_root.glob("*/summary.json")):
            data = json.loads(summary.read_text(encoding="utf-8"))
            label = summary.parent.name.replace("ex3_1_", "").replace("ex1_3_", "baseline_")
            rows.append((label, float(data.get("best_val_acc", 0))))
        svg_bar_chart(
            ROOT / "DLA_1" / "figures" / "gtsrb_validation_accuracy_summary.svg",
            "DLA 1 - best validation accuracy",
            sorted(rows, key=lambda item: item[1], reverse=True)[:9],
            "Accuracy from saved summary.json files",
        )

    svg_bar_chart(
        ROOT / "DLA_2" / "figures" / "lab2_accuracy_summary.svg",
        "DLA 2 - selected observed accuracies",
        [
            ("DistilBERT full fine-tuning", 0.844278),
            ("LoRA", 0.838649),
            ("Partial freezing", 0.837711),
            ("DistilBERT features + SVM", 0.794559),
            ("CLIP-Adapter b=128", 0.524072),
            ("Zero-shot CLIP", 0.475928),
        ],
        "Accuracy values read from notebook outputs",
    )

    final_eval = ROOT / "DLA_3" / "artifacts" / "a2c_lunarlander_final_evaluation.json"
    if final_eval.exists():
        data = json.loads(final_eval.read_text(encoding="utf-8"))
        svg_bar_chart(
            ROOT / "DLA_3" / "figures" / "lunarlander_final_evaluation.svg",
            "DLA 3 - LunarLander final evaluation",
            [
                ("Average return", float(data.get("avg_return", 0))),
                ("Success rate (%)", float(data.get("success_rate", 0))),
                ("Truncation rate (%)", float(data.get("truncation_rate", 0))),
            ],
            "Saved 200-episode evaluation",
        )

    temp_file = ROOT / "DLA_3" / "artifacts" / "a2c_lunarlander_temperature_sweep.json"
    if temp_file.exists():
        rows = [
            (f"T={row.get('temperature')}", float(row.get("avg_return", 0)))
            for row in json.loads(temp_file.read_text(encoding="utf-8"))
        ]
        svg_bar_chart(
            ROOT / "DLA_3" / "figures" / "lunarlander_temperature_sweep.svg",
            "DLA 3 - temperature sweep",
            sorted(rows, key=lambda item: item[0])[:12],
            "Average return by inference temperature",
        )


def sanitize_notebooks() -> None:
    replacements = {
        "RUN_TRAINING = True": "RUN_TRAINING = False",
        "ENABLE_WANDB = True": "ENABLE_WANDB = False",
        "RUN_EXTRACTION = True": "RUN_EXTRACTION = False",
        "RUN_CARTPOLE_TRAINING = True": "RUN_CARTPOLE_TRAINING = False",
        "RUN_LUNAR_EXPERIMENTS = True": "RUN_LUNAR_EXPERIMENTS = False",
        "DLA_Lab2": "DLA_2",
        "Reinforcment LLM\\LAB_3_LLM": "DLA_3",
        "Reinforcment LLM/LAB_3_LLM": "DLA_3",
        "Imposta RUN_TRAINING = False per rilanciare": "Imposta RUN_TRAINING = True per rilanciare",
    }
    notebook_paths = (
        list((ROOT / "DLA_1" / "notebooks").glob("*.ipynb"))
        + list((ROOT / "DLA_2" / "notebooks").glob("*.ipynb"))
        + list((ROOT / "DLA_3" / "notebooks").glob("*.ipynb"))
    )
    note = (
        "> **Execution note**\n>\n"
        "> Gli output visibili sono stati prodotti durante le esecuzioni finali o di validazione del laboratorio. "
        "Nella versione di consegna i training costosi sono disattivati di default quando sono controllati da flag; "
        "checkpoint e artefatti salvati vengono usati per consultazione rapida."
    )
    reference_tables = {
        "DLA_1": """## Referenced functions and source files

| Function/class | Defined in | Purpose |
| --- | --- | --- |
| `load_config` | `src/dla_lab1/config.py` | Caricamento configurazione del laboratorio. |
| `build_dataloaders` / `build_retrieval_dataloaders` | `src/dla_lab1/data.py` | Preparazione split, loader e pipeline GTSRB. |
| `run_finetuning` | `src/dla_lab1/experiments.py` | Esecuzione controllata degli esperimenti di fine-tuning. |
| `classification_metrics` | `src/dla_lab1/evaluate.py` | Calcolo metriche di classificazione. |
| `extract_features` | `src/dla_lab1/features.py` | Estrazione feature per baseline e retrieval. |
""",
        "DLA_2": """## Referenced functions and source files

| Function/class | Defined in | Purpose |
| --- | --- | --- |
| `load_rotten_tomatoes` | `src/dla_lab2/sentiment.py` | Caricamento dataset sentiment. |
| `extract_cls_features_with_pipeline` | `src/dla_lab2/sentiment.py` | Feature extraction da DistilBERT. |
| `build_trainer` | `src/dla_lab2/sentiment.py` | Configurazione Hugging Face Trainer. |
| `CLIPAdapter` | `src/dla_lab2/clip_utils.py` | Adattatore leggero per feature CLIP. |
""",
        "DLA_3": """## Referenced functions and source files

| Function/class | Defined in | Purpose |
| --- | --- | --- |
| `reinforce` / `reinforce_with_value_baseline` | `src/dla_lab3/policy_gradient.py` | Training policy-gradient su CartPole. |
| `A2CConfig` / `train_a2c_vectorized` | `src/dla_lab3/a2c.py` | Configurazione e training A2C. |
| `evaluate_lunar_candidates` | `src/dla_lab3/experiments.py` | Valutazione checkpoint LunarLander. |
| `run_lunar_visual_episodes` | `src/dla_lab3/visualization.py` | Rollout visuali finali. |
""",
    }
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "markdown"
        )
        if "Execution note" not in markdown_text and path.name not in {"Esperimenti_di_prova.ipynb", "00_esperimenti_di_prova_a2c.ipynb"}:
            notebook["cells"].insert(1, md_cell(note))
            changed = True

        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            updated = source
            for old, new in replacements.items():
                updated = updated.replace(old, new)
            if updated != source:
                cell["source"] = updated.splitlines(keepends=True)
                changed = True

            kept_outputs = []
            for output in cell.get("outputs", []):
                text = ""
                if "text" in output:
                    raw = output["text"]
                    text = "".join(raw) if isinstance(raw, list) else raw
                elif "data" in output and "text/plain" in output["data"]:
                    raw = output["data"]["text/plain"]
                    text = "".join(raw) if isinstance(raw, list) else raw
                lower = text.lower()
                if "wandb.login" in lower or "_netrc" in lower or "currently logged in as" in lower:
                    changed = True
                    continue
                if "C:\\Users\\checc\\OneDrive\\Desktop\\DLA\\DLA_Lab\\" in text or "/mnt/c/Users/checc/OneDrive/Desktop/DLA/DLA_Lab/" in text:
                    text = text.replace("C:\\Users\\checc\\OneDrive\\Desktop\\DLA\\DLA_Lab\\", "")
                    text = text.replace("c:\\Users\\checc\\OneDrive\\Desktop\\DLA\\DLA_Lab\\", "")
                    text = text.replace("/mnt/c/Users/checc/OneDrive/Desktop/DLA/DLA_Lab/", "")
                    for old, new in replacements.items():
                        text = text.replace(old, new)
                    if "text" in output:
                        output["text"] = text.splitlines(keepends=True)
                    elif "data" in output and "text/plain" in output["data"]:
                        output["data"]["text/plain"] = text.splitlines(keepends=True)
                    changed = True
                kept_outputs.append(output)
            if kept_outputs != cell.get("outputs", []):
                cell["outputs"] = kept_outputs

        if "Referenced functions and source files" not in markdown_text:
            lab_key = "DLA_3" if "DLA_3" in str(path) else "DLA_2" if "DLA_2" in str(path) else "DLA_1"
            notebook["cells"].append(md_cell(reference_tables[lab_key]))
            changed = True

        if changed:
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_index_notebooks() -> None:
    write_notebook(
        ROOT / "DLA_1" / "DLA_1.ipynb",
        [
            md_cell("# DLA 1 - Computer Vision on GTSRB\n\nAutore: Francesco Faggioli  \nCorso: Deep Learning Applications\n\n[Consegna ufficiale](ASSIGNMENT.ipynb)"),
            md_cell("> **Execution note**\n>\n> Questo notebook è un indice GitHub-ready del primo laboratorio. I risultati completi sono conservati nei notebook dettagliati in `notebooks/`."),
            md_cell("""## Notebook del laboratorio

| Notebook | Role | Main evidence |
| --- | --- | --- |
| `notebooks/01_eda_and_feature_baseline.ipynb` | EDA e baseline feature extraction + SVM | Baseline test accuracy 0.6412 |
| `notebooks/02_finetuning_pipeline.ipynb` | Fine-tuning baseline | Test accuracy 0.5038 |
| `notebooks/02b_pipeline_consolidation.ipynb` | Pipeline, configurazioni e logging | Configurazione riproducibile e W&B locale |
| `notebooks/03_improvements_and_retrieval.ipynb` | Miglioramento fine-tuning | Test accuracy 0.8025 |
| `notebooks/03b_retrieval_training_free_classification.ipynb` | Retrieval/NMC | NMC accuracy 0.4185, Precision@1 0.4812 |
| `notebooks/Esperimenti_di_prova.ipynb` | Prove conservate | Archivio di esperimenti non finali |
"""),
            md_cell("## Figura riassuntiva\n\n![Validation accuracy summary](figures/gtsrb_validation_accuracy_summary.svg)"),
        ],
    )
    write_notebook(
        ROOT / "DLA_2" / "DLA_2.ipynb",
        [
            md_cell("# DLA 2 - Transformers and Sentiment Analysis\n\nAutore: Francesco Faggioli  \nCorso: Deep Learning Applications\n\n[Consegna ufficiale](ASSIGNMENT.ipynb)"),
            md_cell("> **Execution note**\n>\n> Questo notebook è un indice GitHub-ready del secondo laboratorio. Gli output eseguiti sono nei notebook dettagliati in `notebooks/`."),
            md_cell("""## Notebook del laboratorio

| Notebook | Role | Main evidence |
| --- | --- | --- |
| `notebooks/01_sentiment_dataset_tokenizer_baseline.ipynb` | Dataset, tokenizer, feature baseline | Test accuracy 0.7946 |
| `notebooks/02_distilbert_full_finetuning.ipynb` | Fine-tuning completo | Test accuracy 0.8443 |
| `notebooks/03_efficient_finetuning_sentiment.ipynb` | LoRA e partial freezing | LoRA test accuracy 0.8386 |
| `notebooks/04_clip_adapter_imagenet_sketch.ipynb` | CLIP zero-shot e adapter | CLIP-Adapter b=128 accuracy 0.5241 |
"""),
            md_cell("## Figura riassuntiva\n\n![Lab 2 accuracy summary](figures/lab2_accuracy_summary.svg)"),
        ],
    )
    write_notebook(
        ROOT / "DLA_3" / "Lab_3.ipynb",
        [
            md_cell("# Lab 3 - Deep Reinforcement Learning\n\nAutore: Francesco Faggioli  \nCorso: Deep Learning Applications\n\n[Consegna ufficiale](ASSIGNMENT.ipynb)"),
            md_cell("> **Execution note**\n>\n> Questo notebook è l'indice principale richiesto per il terzo laboratorio. I notebook eseguiti restano in `notebooks/`."),
            md_cell("""## Notebook del laboratorio

| Notebook | Role | Main evidence |
| --- | --- | --- |
| `notebooks/01_cartpole_reinforce_evaluation.ipynb` | REINFORCE e metriche periodiche | Best eval return 489.35 |
| `notebooks/02_cartpole_value_baseline.ipynb` | Baseline con value network | Value baseline final eval return 500.0 |
| `notebooks/03_a2c_cartpole_lunarlander.ipynb` | A2C CartPole/LunarLander | LunarLander avg return 165.76, success 56.0% |
| `notebooks/00_esperimenti_di_prova_a2c.ipynb` | Prove conservate | Esperimenti A2C esplorativi |
"""),
            md_cell("## Figure riassuntive\n\n![LunarLander final evaluation](figures/lunarlander_final_evaluation.svg)\n\n![Temperature sweep](figures/lunarlander_temperature_sweep.svg)"),
        ],
    )


def assignment_tables() -> dict[str, str]:
    return {
        "DLA_1": """| Assignment requirement | Notebook section | Status | Evidence |
| --- | --- | ---: | --- |
| Exercise 1.1 - EDA GTSRB | `01_eda_and_feature_baseline.ipynb` | Completed | Dataset statistics and class distribution |
| Exercise 1.2 - stable baseline | `01_eda_and_feature_baseline.ipynb` | Completed | ResNet-18 features + SVM, test accuracy 0.6412 |
| Exercise 1.3 - fine-tuning baseline | `02_finetuning_pipeline.ipynb` | Completed | Baseline fine-tuning, test accuracy 0.5038 |
| Exercise 2 - reproducible pipeline | `02b_pipeline_consolidation.ipynb` | Completed | Config-driven pipeline, local artifacts and W&B logs |
| Exercise 3.1 - improve fine-tuning | `03_improvements_and_retrieval.ipynb` | Completed | Improved model, test accuracy 0.8025 |
| Exercise 3.2 - retrieval/NMC | `03b_retrieval_training_free_classification.ipynb` | Completed | Precision@1 0.4812 and NMC accuracy 0.4185 |""",
        "DLA_2": """| Assignment requirement | Notebook section | Status | Evidence |
| --- | --- | ---: | --- |
| Exercise 1.1 - dataset splits | `01_sentiment_dataset_tokenizer_baseline.ipynb` | Completed | Rotten Tomatoes splits loaded and inspected |
| Exercise 1.2 - DistilBERT/tokenizer | `01_sentiment_dataset_tokenizer_baseline.ipynb` | Completed | Tokenizer/model outputs inspected |
| Exercise 1.3 - stable baseline | `01_sentiment_dataset_tokenizer_baseline.ipynb` | Completed | DistilBERT CLS features + SVM, test accuracy 0.7946 |
| Exercise 2 - full fine-tuning | `02_distilbert_full_finetuning.ipynb` | Completed | Test accuracy 0.8443 |
| Exercise 3.1 - efficient fine-tuning | `03_efficient_finetuning_sentiment.ipynb` | Completed | LoRA and partial freezing compared |
| Exercise 3.2 - CLIP adaptation | `04_clip_adapter_imagenet_sketch.ipynb` | Completed | CLIP-Adapter b=128 accuracy 0.5241 |""",
        "DLA_3": """| Assignment requirement | Notebook section | Status | Evidence |
| --- | --- | ---: | --- |
| Exercise 1 - improve REINFORCE evaluation | `01_cartpole_reinforce_evaluation.ipynb` | Completed | Periodic evaluation returns and episode lengths |
| Exercise 2 - value baseline | `02_cartpole_value_baseline.ipynb` | Completed | Value baseline reaches final eval return 500.0 |
| Exercise 3.1 - A2C CartPole/LunarLander | `03_a2c_cartpole_lunarlander.ipynb` | Completed | CartPole solved; LunarLander avg return 165.76, success 56.0% |""",
    }


def write_readmes() -> None:
    tables = assignment_tables()
    write_md(ROOT / "README.md", """# Deep Learning Applications - Laboratory Portfolio

Course: Deep Learning Applications  
Author: Francesco Faggioli

This repository contains the final laboratory work for the Deep Learning Applications course. It is not a single Reinforcement Learning project: only the third laboratory concerns Deep Reinforcement Learning.

## Laboratories

| Lab | Topic | Main methods | Main notebook | Report |
| --- | --- | --- | --- | --- |
| DLA 1 | GTSRB image classification | Feature extraction, SVM, ResNet-18 fine-tuning, retrieval/NMC | [`DLA_1.ipynb`](DLA_1/DLA_1.ipynb) | [`README.md`](DLA_1/README.md) |
| DLA 2 | Transformer and CLIP experiments | DistilBERT, SVM, full fine-tuning, LoRA, partial freezing, CLIP-Adapter | [`DLA_2.ipynb`](DLA_2/DLA_2.ipynb) | [`README.md`](DLA_2/README.md) |
| DLA 3 | Deep Reinforcement Learning | REINFORCE, value baseline, A2C, LunarLander evaluation | [`Lab_3.ipynb`](DLA_3/Lab_3.ipynb) | [`README.md`](DLA_3/README.md) |

## Main results

| Lab | Result | Evidence |
| --- | --- | --- |
| DLA 1 | Improved ResNet-18 fine-tuning reached test accuracy 0.8025; feature baseline reached 0.6412. | Lab 1 notebooks and `artifacts/runs/*/summary.json` |
| DLA 2 | Full DistilBERT fine-tuning reached test accuracy 0.8443; LoRA reached 0.8386 with about 1.09% trainable parameters. | Lab 2 notebooks |
| DLA 3 | CartPole was solved by the value-baseline run; LunarLander final A2C evaluation reached average return 165.76 and success rate 56.0%. | `DLA_3/artifacts/a2c_lunarlander_final_evaluation.json` |

## Quick inspection mode

Open the main notebooks and README files on GitHub. Expensive training is disabled by default where notebooks expose execution flags. The displayed outputs were preserved from completed runs or reconstructed from saved artifacts.

## Full execution mode

Run the detailed notebooks in each `notebooks/` folder after installing dependencies and downloading datasets. Long training cells should be enabled manually by setting the relevant `RUN_*` flags to `True`.

## Installation

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

For Lab 3, Linux or WSL is recommended because Gymnasium Box2D environments can be fragile on native Windows.

## Dataset and checkpoint policy

Public datasets, downloaded archives, W&B logs, generated checkpoints and heavy artifacts are ignored by Git. Lightweight reports, configuration files, notebooks and README documentation are kept in the repository.

## Code of Conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) and the GitHub-renderable notebook [`CODE_OF_CONDUCT.ipynb`](CODE_OF_CONDUCT.ipynb).

## Use of AI-assisted tools

See [`AI_USAGE.md`](AI_USAGE.md). ChatGPT/OpenAI Codex supported clarification, debugging, code organization, documentation and consistency checks. Outputs and final decisions remain the author's responsibility.

## References

Each laboratory README lists its specific sources, datasets and documentation references.
""")

    write_md(ROOT / "AI_USAGE.md", """# Use of AI-Assisted Tools

## Tools used

ChatGPT and OpenAI Codex were used as AI-assisted tools during preparation of this repository.

## Scope of assistance

The tools supported conceptual clarification, interpretation of exercise requirements, debugging, code organization, naming, comments, docstrings, repository cleanup, README drafting and consistency checks between notebooks, outputs and documentation.

## Laboratory-specific use

### DLA 1

AI assistance supported the organization of the GTSRB pipeline, interpretation of baseline/fine-tuning results, documentation of W&B usage and cleanup of output logs. Weights & Biases was used in Lab 1 as an experiment-tracking tool for training and validation metrics.

### DLA 2

AI assistance supported organization of DistilBERT, LoRA, partial-freezing and CLIP-Adapter experiments, plus documentation of Hugging Face and CLIP dependencies.

### DLA 3

AI assistance supported review of REINFORCE, value-baseline and A2C experiments. It also supported discussion of LunarLander A2C hyperparameters using the RL Baselines3 Zoo tuned A2C reference for `LunarLander-v3`, as reported in notebook output and README. Final selections were evaluated through saved experiment outputs rather than accepted automatically.

## Human verification

The final code, notebooks and documentation remain the responsibility of the author. Experiments shown in notebooks were executed by the author, and AI-generated suggestions were checked against code, outputs, assignments and external sources. AI was not used to invent results or modify metrics.

## Limitations and responsibility

AI tools can produce incorrect explanations, non-optimal code or unsupported claims. The submitted repository should be understood, reviewed and explainable by the author.

## External sources consulted

| Source | Used for | Notes |
| --- | --- | --- |
| Official assignment notebooks in each lab | Requirements | Converted from instructor-provided text files. |
| PyTorch, Torchvision, Hugging Face, PEFT, Gymnasium and W&B documentation | Library usage | Used as implementation/documentation references. |
| RL Baselines3 Zoo tuned A2C hyperparameters for LunarLander-v3 | Lab 3 parameter discussion | Used as a reference point; final behavior was verified experimentally. |
| GianniMoretti/DeepLearningApplicationLAB | Repository presentation reference | Used only as high-level example for README organization and AI disclosure style. |
""")

    lab_readmes = {
        "DLA_1": f"""# DLA 1 - GTSRB Computer Vision Laboratory

## Overview

This laboratory studies GTSRB traffic-sign classification through exploratory analysis, pretrained CNN feature extraction, a classical SVM baseline, ResNet-18 fine-tuning, pipeline consolidation and retrieval-based classification.

## Official assignment

- GitHub notebook: [`ASSIGNMENT.ipynb`](ASSIGNMENT.ipynb)
- Markdown copy: [`ASSIGNMENT.md`](ASSIGNMENT.md)

## Assignment coverage

{tables['DLA_1']}

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
""",
        "DLA_2": f"""# DLA 2 - Transformers, Sentiment Analysis and CLIP

## Overview

This laboratory uses the Hugging Face ecosystem to build sentiment-analysis baselines and fine-tuned DistilBERT models. It also includes a CLIP/ImageNet-Sketch extension using a parameter-efficient adapter.

## Official assignment

- GitHub notebook: [`ASSIGNMENT.ipynb`](ASSIGNMENT.ipynb)
- Markdown copy: [`ASSIGNMENT.md`](ASSIGNMENT.md)

## Assignment coverage

{tables['DLA_2']}

## Results summary

| Experiment | Metric | Observed value |
| --- | --- | ---: |
| DistilBERT CLS features + SVM | Test accuracy | 0.7946 |
| Full DistilBERT fine-tuning | Test accuracy | 0.8443 |
| LoRA fine-tuning | Test accuracy | 0.8386 |
| Partial freezing | Test accuracy | 0.8377 |
| Zero-shot CLIP on ImageNet-Sketch | Accuracy | 0.4759 |
| CLIP-Adapter bottleneck=128 | Accuracy | 0.5241 |

![Lab 2 accuracy summary](figures/lab2_accuracy_summary.svg)

## Main dependencies and imports

| Package/import | Purpose | Used in |
| --- | --- | --- |
| `datasets` | Rotten Tomatoes dataset loading | Sentiment analysis |
| `transformers` | DistilBERT tokenizer/model and Trainer | Feature extraction and fine-tuning |
| `peft` | LoRA configuration | Efficient fine-tuning |
| `sklearn` | SVM and classification metrics | Baseline and evaluation |
| `torch` | Tensor/model execution | All experiments |
| `open_clip` | CLIP model and preprocessing | ImageNet-Sketch extension |

## Local functions and source files

| Function/class | Defined in | Purpose | Main inputs | Main outputs | Used in |
| --- | --- | --- | --- | --- | --- |
| `load_rotten_tomatoes` | `src/dla_lab2/sentiment.py` | Load dataset splits | dataset id/cache | dataset dict | Exercise 1 |
| `extract_cls_features_with_pipeline` | `src/dla_lab2/sentiment.py` | Extract CLS embeddings | texts, model, tokenizer | feature matrix | Baseline |
| `build_training_arguments` | `src/dla_lab2/sentiment.py` | Configure Trainer | output dir, hyperparameters | TrainingArguments | Fine-tuning |
| `lora_sequence_classifier_init` | `src/dla_lab2/sentiment.py` | Build LoRA model | model id, LoRA config | PEFT model | Exercise 3.1 |
| `CLIPAdapter` | `src/dla_lab2/clip_utils.py` | Parameter-efficient adapter | CLIP features | adapted features | Exercise 3.2 |

## External sources and references

| Source | Used for | Adaptation or contribution |
| --- | --- | --- |
| Official assignment | Requirements | Converted into `ASSIGNMENT.ipynb`. |
| Hugging Face Datasets | Dataset loading | Rotten Tomatoes splits. |
| Hugging Face Transformers | DistilBERT and Trainer | Tokenization/fine-tuning pipeline. |
| Hugging Face PEFT | LoRA | Efficient fine-tuning comparison. |
| Rotten Tomatoes dataset card | Dataset description | Sentiment dataset source. |
| open_clip repository | CLIP model loading | ImageNet-Sketch experiment. |
| ImageNet-Sketch | External validation dataset | CLIP domain-shift evaluation. |
""",
        "DLA_3": f"""# Lab 3 - Deep Reinforcement Learning

## Overview

This laboratory is the only Deep Reinforcement Learning part of the repository. It covers REINFORCE on CartPole, a learned value baseline, and A2C experiments on CartPole and LunarLander.

## Official assignment

- GitHub notebook: [`ASSIGNMENT.ipynb`](ASSIGNMENT.ipynb)
- Markdown copy: [`ASSIGNMENT.md`](ASSIGNMENT.md)

## Assignment coverage

{tables['DLA_3']}

## Results summary

| Experiment | Metric | Observed value |
| --- | --- | ---: |
| REINFORCE standardized returns | Best eval return | 489.35 |
| REINFORCE with value baseline | Final eval return | 500.00 |
| A2C CartPole | Average greedy return | 494.51 |
| A2C LunarLander final evaluation | Average return | 165.76 |
| A2C LunarLander final evaluation | Success rate | 56.0% |

![LunarLander final evaluation](figures/lunarlander_final_evaluation.svg)

![Temperature sweep](figures/lunarlander_temperature_sweep.svg)

## Main dependencies and imports

| Package/import | Purpose | Used in |
| --- | --- | --- |
| `torch` | Policy/value networks and optimization | REINFORCE and A2C |
| `gymnasium` | CartPole and LunarLander environments | Environment interaction |
| `numpy` | Metrics and rollout statistics | Training/evaluation |
| `matplotlib` | Training and selection plots | Notebook figures |
| `pygame` / Box2D extras | Rendering and LunarLander support | Visual checks and environment setup |

## Local functions and source files

| Function/class | Defined in | Purpose | Main inputs | Main outputs | Used in |
| --- | --- | --- | --- | --- | --- |
| `PolicyNet` / `ValueNet` | `src/dla_lab3/policy_gradient.py` | Policy and value models | observations | action probabilities/value | Exercises 1-2 |
| `reinforce` | `src/dla_lab3/policy_gradient.py` | REINFORCE training | policy, env, config | history/checkpoint | Exercise 1 |
| `reinforce_with_value_baseline` | `src/dla_lab3/policy_gradient.py` | REINFORCE with learned baseline | policy, value net | history/checkpoint | Exercise 2 |
| `A2CConfig` / `train_a2c_vectorized` | `src/dla_lab3/a2c.py` | A2C setup and training | envs, net, hyperparameters | history/checkpoint | Exercise 3 |
| `evaluate_lunar_policy_configurations` | `src/dla_lab3/experiments.py` | Compare checkpoint/mode/temperature settings | candidates, config | evaluation table | Exercise 3 |
| `run_lunar_visual_episodes` | `src/dla_lab3/visualization.py` | Visual rollout summaries | checkpoint, mode, temperature | rollout metrics | Exercise 3 |

## AI-assisted parameter discussion

AI assistance was used to interpret and discuss A2C hyperparameter choices for LunarLander. The exact source recovered in the notebook output is the RL Baselines3 Zoo tuned A2C reference for `LunarLander-v3`, which reports settings such as `n_envs=8`, `n_timesteps=2e5`, `gamma=0.995`, `n_steps=5`, `learning_rate=0.00083` and `ent_coef=0.00001`. The final choice was not copied blindly: candidate checkpoints, action-selection modes and inference temperatures were evaluated with saved experiment code and JSON artifacts.

## External sources and references

| Source | Used for | Adaptation or contribution |
| --- | --- | --- |
| Official assignment | Requirements | Converted into `ASSIGNMENT.ipynb`. |
| Professor REINFORCE CartPole reference | Starting conceptual reference | Refactored and extended with evaluation/checkpointing. |
| Gymnasium CartPole documentation | Environment specification | Observation/action space and solved criterion. |
| Gymnasium LunarLander documentation | Environment specification | Box2D environment and success threshold. |
| RL Baselines3 Zoo A2C LunarLander-v3 tuned reference | Hyperparameter comparison | Used as a reference point for A2C settings. |
| PyTorch documentation | Model/optimizer/checkpoint conventions | Local implementations. |
""",
    }
    for lab, content in lab_readmes.items():
        write_md(ROOT / lab / "README.md", content)


def write_requirements_gitignore_archive() -> None:
    write_md(ROOT / "requirements.txt", """jupyter
ipykernel
numpy
pandas
matplotlib
scikit-learn
torch
torchvision
tqdm
pyyaml
wandb
datasets
transformers
peft
accelerate
open_clip_torch
gymnasium
pygame
box2d-py
""")
    gitignore_path = ROOT / ".gitignore"
    gitignore = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""
    additions = """
# Python caches and test caches
.pytest_cache/
.mypy_cache/

# Editors
.vscode/
.idea/

# Local generated documentation archives
_archive/obsolete_local_settings/

# Experiment logs
logs/
"""
    if "# Editors" not in gitignore:
        write_md(gitignore_path, gitignore.rstrip() + "\n" + additions)
    write_md(ROOT / "_archive" / "README.md", """# Archive

This folder contains local or obsolete material that should not be treated as final laboratory output.

| Archived item | Reason | Replacement |
| --- | --- | --- |
| `obsolete_local_settings/reinforcment_llm_vscode/` | VS Code settings from the old misspelled Lab 3 folder name. | `DLA_3/` and root-level documentation. |

Old notebooks and duplicated files already deleted from the working tree remain recoverable from Git history and are listed in the final Codex report.
""")


def main() -> None:
    generate_assignments()
    generate_code_of_conduct()
    generate_figures()
    sanitize_notebooks()
    generate_index_notebooks()
    write_readmes()
    write_requirements_gitignore_archive()
    print("submission documentation generated")


if __name__ == "__main__":
    main()
