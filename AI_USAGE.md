# Use of AI-Assisted Tools

## Tools and scope

ChatGPT and OpenAI Codex were used as support tools during the three laboratories and the final repository review. Assistance covered:

- conceptual clarification of deep-learning and reinforcement-learning topics;
- interpretation and decomposition of the official assignments;
- debugging and review of Python/notebook code;
- organization of reusable functions, modules, scripts, and configurations;
- comments and docstrings;
- diagnosis of path, environment, dependency, and notebook-state problems;
- design and review of README structure;
- repository organization, naming, links, and Git consistency checks;
- review of whether reported metrics were traceable to notebook outputs or saved artifacts;
- discussion of A2C hyperparameters and evaluation choices in DLA 3.

AI assistance was not treated as an experimental instrument. A generated explanation or proposed value was not accepted as evidence unless it was checked against code, an executed output, a configuration, a saved metric, or an external source.

## Laboratory-specific use

### DLA 1

AI support was used to organize the GTSRB pipeline, separate feature extraction from fine-tuning, review retrieval metrics, improve comments/documentation, and inspect the consistency of local artifacts and W&B summaries. Weights & Biases itself was used as an experiment-tracking service for selected runs; it is not an AI system. Its public role in this repository is documented through exported histories and summaries because raw local W&B directories are not versioned.

### DLA 2

AI support was used to review DistilBERT tokenization and Trainer code, structure the LoRA/partial-freezing comparison, debug local Hugging Face environment issues, and document the CLIP-Adapter experiment. The final accuracy, F1, precision, recall, loss, and parameter-count values come from executed notebooks.

### DLA 3

AI support was used to discuss REINFORCE variance, the learned value baseline, A2C updates, evaluation design, and LunarLander hyperparameters. The exact external parameter reference was the [RL Baselines3 Zoo A2C configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml). It was used as a starting point, not copied as a complete solution. The local implementation, training budgets, checkpoint variants, policy modes, temperatures, and final selection were reviewed and evaluated experimentally.

AI suggestions were not accepted automatically. In particular, the final LunarLander configuration was selected from repeated checkpoint and policy evaluations, and unsuccessful refinements remain documented.

## Human verification and responsibility

The author:

- executed the experiments represented by the preserved notebook outputs;
- reviewed AI-generated code and explanations;
- checked final metrics against notebooks and saved evidence;
- retained responsibility for all scientific and engineering decisions;
- remains responsible for being able to explain the code, methods, results, and limitations.

No metric, plot, hyperparameter, source, or completed exercise was invented to improve the presentation. The final repository review did not change algorithms, final-experiment seeds, dataset splits, optimization settings, or headline experimental results. New seed lists were introduced only for the requested five-episode CartPole and LunarLander qualitative visual checks; their directly executed summaries are versioned separately and are not treated as benchmark results.

## Limitations of AI assistance

AI systems can produce incorrect code, unsupported scientific explanations, fabricated references, and inappropriate parameter suggestions. Their outputs require independent verification. This repository therefore links external references directly and keeps uncertainty, negative results, and unresolved evidence gaps visible.

## External sources consulted

| Source | Use | Verification role |
| --- | --- | --- |
| [PyTorch documentation](https://pytorch.org/docs/stable/index.html) | Tensors, models, optimization, distributions | API and conceptual reference |
| [Torchvision documentation](https://pytorch.org/vision/stable/index.html) | GTSRB and pretrained ResNet models | Dataset/model API reference |
| [Weights & Biases documentation](https://docs.wandb.ai/) | Optional experiment tracking | Logging and artifact-workflow reference |
| [Rotten Tomatoes dataset card](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | Dataset structure and provenance | Split and label reference |
| [DistilBERT documentation](https://huggingface.co/docs/transformers/model_doc/distilbert) | Tokenizer/model behaviour | Transformer API reference |
| [PEFT LoRA documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora) | LoRA configuration | Efficient fine-tuning reference |
| [OpenCLIP repository](https://github.com/mlfoundations/open_clip) | CLIP loading and preprocessing | Implementation API reference |
| [ImageNet-Sketch repository](https://github.com/HaohanWang/ImageNet-Sketch) | Out-of-domain image dataset | Dataset/paper reference |
| [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | Environment specification | State, action, reward, termination reference |
| [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | Environment specification | State, action, reward, success reference |
| [RL Baselines3 Zoo A2C YAML](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml) | DLA 3 hyperparameter discussion | External comparison point |

## Academic-integrity statement

AI tools assisted the work but do not replace authorship, understanding, or accountability. The author is responsible for the submission and for explaining every part of it. See also [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
