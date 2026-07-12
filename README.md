# Deep Learning Applications - Laboratory Portfolio

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
.venv\Scripts\activate
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
