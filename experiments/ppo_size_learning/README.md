# PPO Size Learning Experiment

This experiment compares two simple maze-learning approaches on generated mazes
with the same three sizes used in the repo:

- small: `16x16`
- medium: `25x25`
- large: `35x35`

Models:

- `legacy_simple_ppo`: the original small actor-critic style from `maze_1/multi_ppo.py`
  with 8 state features, wall penalties, Manhattan-distance reward shaping, and PPO.
- `compact_ppo`: the current compact PPO from `simple/compact_original_feature_ppo.py`
  with 8 state features, action masks, BFS expert behavior cloning, and PPO.

Outputs are written to:

- `data/generated_mazes.json`
- `data/training_curves.csv`
- `data/summary.csv`
- `plots/*.png`
- `report.md`

Run:

```powershell
.\.venv\Scripts\python.exe experiments\ppo_size_learning\run_experiments.py
.\.venv\Scripts\python.exe experiments\ppo_size_learning\plot_results.py
```

