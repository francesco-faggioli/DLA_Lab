# DLA Lab 3 - Policy Gradient Reinforcement Learning

This folder contains the clean, refactored version of the Deep Reinforcement Learning laboratory.

The old notebooks are intentionally left unchanged in `../DLA_3`. The final work should be developed in this folder.

## Structure

- `src/dla_lab3/`: reusable code for environments, seeding, policy/value networks, REINFORCE training, evaluation and plots.
- `notebooks/`: clean report notebooks, one per exercise phase.
- `scripts/`: environment check and checkpoint inspection/linking utilities.
- `config/`: default hyperparameters and experiment settings.
- `checkpoints/`: generated model checkpoints.
- `artifacts/`: generated plots, tables or local outputs.

## Notebook Workflow

Run the notebooks in order:

0. `notebooks/00_esperimenti_di_prova_a2c.ipynb` optional, documents the exploratory A2C runs.
1. `notebooks/01_cartpole_reinforce_evaluation.ipynb`
2. `notebooks/02_cartpole_value_baseline.ipynb`
3. `notebooks/03_a2c_cartpole_lunarlander.ipynb`

The first notebook covers Exercise 1. The second notebook covers Exercise 2. The third notebook covers Exercise 3.1. The optional notebook 0 preserves the main trial runs from the old exploratory A2C notebook.

## Environment Recommendation

Use the Linux/Ubuntu environment prepared for the lab:

```bash
cd /mnt/c/Users/checc/OneDrive/Desktop/DLA/DLA_Lab/Reinforcment\ LLM/LAB_3_LLM
conda activate DRL
python scripts/check_env.py
jupyter lab
```

The key dependencies are:

- Python 3.10+ or 3.12;
- PyTorch;
- Gymnasium with Box2D support;
- Matplotlib;
- Pygame;
- JupyterLab/Jupyter;

Suggested conda command:

```bash
conda create -n DRL -c conda-forge python=3.12 torch "gymnasium[box2d]" matplotlib pygame jupyterlab jupyter ipykernel
```

## External Requirements Checked

- Professor repository: https://gitlab.com/bagdanov/reinforce-cartpole
- CartPole official documentation: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- LunarLander official documentation: https://gymnasium.farama.org/environments/box2d/lunar_lander/

Relevant environment facts used in the notebooks:

- `CartPole-v1`: observation shape `(4,)`, action space `Discrete(2)`, default solved threshold `500`.
- `LunarLander-v3`: observation shape `(8,)`, action space `Discrete(4)`, solved threshold `200`.
- `LunarLander-v3` is a Box2D environment, so the Box2D extra is required.

## Old Checkpoints

The old checkpoints are expected at:

```bash
/home/francescofaggioli/DRL_lab
```

From this Windows sandbox the WSL filesystem is not visible, so checkpoint analysis should be run inside Ubuntu:

```bash
cd /mnt/c/Users/checc/OneDrive/Desktop/DLA/DLA_Lab/Reinforcment\ LLM/LAB_3_LLM
conda activate DRL
python scripts/link_old_checkpoints.py --source /home/francescofaggioli/DRL_lab
python scripts/inspect_checkpoints.py --checkpoint-dir /home/francescofaggioli/DRL_lab
```

After running the link script, the old checkpoints are also reachable from the working folder at:

```bash
checkpoints/old
```

## Methodological Choices

Exercise 1 uses REINFORCE on CartPole and adds periodic evaluation. The required metrics are average total reward and average episode length over `M` evaluation episodes every `N` training episodes.

Exercise 2 compares standardization of returns with a learned value baseline. The value baseline estimates `V(s)` and the policy update uses `G_t - V(s_t)`.

Exercise 3.1 implements Advantage Actor-Critic (A2C). CartPole is used first as a validation environment because a correct implementation should solve it reliably. LunarLander is then analyzed with the stronger vectorized A2C setup developed during the exploratory work. The final notebook reports both successful and unsuccessful attempts because this is useful evidence for explaining training instability.

The default LunarLander presets in `config/lab3_defaults.yaml` now generate new LAB_3_LLM checkpoints instead of reusing old exploratory checkpoints:

- `separate_current_long`: 1,000,000 environment steps with separate actor/critic heads, RMSprop, `gamma=0.995`, `n_steps=5`, `reward_scale=1.0`.
- `landing_refine_current_long`: 300,000 additional steps from the current long checkpoint with lower learning rate.
- `landing_precision_current`: additional low-learning-rate refinement from the current best-train refinement checkpoint, intended to reduce excessive stochasticity near landing.

The short presets remain in the YAML file for quick smoke tests, but they should not be used as the final result because their training budget is much smaller.

Notebook 03 now selects a complete final policy configuration: checkpoint, action-selection mode (`greedy` or `sample`) and temperature. Temperature remains an inference-time parameter; it changes the stochastic sampler through `logits / temperature`, but it does not update model weights. The selected configuration is used for the final evaluation and the visual rollouts.

## Academic Integrity and AI Disclosure

The DLA Code of Conduct permits AI-assisted work only with transparency.

Suggested disclosure text:

> ChatGPT/Codex was used to help restructure the laboratory into clean notebooks, reusable Python functions and README documentation. The generated code was manually inspected and must be rerun in the declared Linux/Conda environment. Experimental results, interpretations and final responsibility remain with the student.

External code and documentation used:

- The professor's `reinforce-cartpole` repository was used as the starting reference for REINFORCE.
- Gymnasium documentation was used for environment specifications and solved thresholds.
- PyTorch documentation conventions are followed for model, optimizer and checkpoint usage.

## Submission Notes

- Do not submit old exploratory notebooks as the final report.
- Keep final conclusions in the clean notebooks.
- Keep generated checkpoints under `checkpoints/`.
- Keep local plots and tables under `artifacts/`.
- If a run fails or does not solve LunarLander, report it honestly and explain the likely causes.
