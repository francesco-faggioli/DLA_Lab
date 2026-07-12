# Lab 3 - Deep Reinforcement Learning

## Overview

This laboratory is the only Deep Reinforcement Learning part of the repository. It covers REINFORCE on CartPole, a learned value baseline, and A2C experiments on CartPole and LunarLander.

## Official assignment

- GitHub notebook: [`ASSIGNMENT.ipynb`](ASSIGNMENT.ipynb)
- Markdown copy: [`ASSIGNMENT.md`](ASSIGNMENT.md)

## Assignment coverage

| Assignment requirement | Notebook section | Status | Evidence |
| --- | --- | ---: | --- |
| Exercise 1 - improve REINFORCE evaluation | `01_cartpole_reinforce_evaluation.ipynb` | Completed | Periodic evaluation returns and episode lengths |
| Exercise 2 - value baseline | `02_cartpole_value_baseline.ipynb` | Completed | Value baseline reaches final eval return 500.0 |
| Exercise 3.1 - A2C CartPole/LunarLander | `03_a2c_cartpole_lunarlander.ipynb` | Completed | CartPole solved; LunarLander avg return 165.76, success 56.0% |

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
