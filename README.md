# ML Maze

The repo is organized around the compact PPO implementation as the default path.

## Current Default

- `simple/compact_original_feature_ppo.py` trains the compact model.
- `simple/run_compact_policy.py` loads the saved compact checkpoint.
- `simple/weights/compact_original_feature_ppo.pt` is the preferred checkpoint.

Run the compact checkpoint on the larger mazes:

```powershell
.\.venv\Scripts\python.exe simple\run_compact_policy.py --print-paths
```

## Folders

- `simple/` contains the compact PPO and legacy/simple checkpoints.
- `complex/` contains the larger feature-expanded PPO experiments and their checkpoints.
- `maze_1/`, `maze_2/`, and `maze_3/` contain the maze definitions and older local training scripts.
- `docs/` contains the compact model explanation and evaluation notes.
