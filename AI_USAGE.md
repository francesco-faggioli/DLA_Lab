# Use of AI-Assisted Tools

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
