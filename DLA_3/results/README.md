# Lab 3 result evidence

These files expose the lightweight numerical evidence behind the report while checkpoints remain ignored.

| File | Content | Primary source |
| --- | --- | --- |
| `cartpole_evaluation.csv` | Periodic evaluation of the three REINFORCE variants | Executed value-baseline notebook |
| `method_summary.csv` | CartPole and LunarLander headline metrics | Executed notebooks |
| `lunarlander_final_evaluation.json` | Complete 200-episode aggregate | Executed A2C notebook and local JSON artifact |
| `lunarlander_temperature_sweep.csv` | Return variability and success rate across temperatures for the selected checkpoint | Local policy-configuration JSON artifact |
| `lunarlander_policy_mode_comparison.csv` | Greedy policy versus the selected stochastic policy | Local policy-configuration JSON artifact |
| `cartpole_visual_episodes.json` | Five qualitative greedy rollouts, seeds and terminal flags; no frames | Direct headless execution of `run_cartpole_visual_episodes` with `cartpole_reinforce_value_baseline.pt` |
| `lunarlander_visual_episodes.json` | Five qualitative sampled rollouts, seeds and terminal flags; no frames | Direct headless execution of `run_lunar_visual_episodes` with the final checkpoint, sampled mode and `T=0.75` |

The final JSON intentionally stores only the checkpoint file name, not the obsolete absolute WSL path found in the local artifact.

The older `a2c_lunarlander_temperature_sweep.json` described a previous evaluation and is deliberately not used. The versioned sweep comes from `a2c_lunarlander_policy_config_selection.json`, which is the evidence used by the final notebook to select temperature `0.75`.

The two visual JSON files were generated during the final repository review. They document short qualitative checks only and are not used to change or augment the headline evaluation metrics.
