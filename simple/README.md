# Simple Compact PPO

This folder is the default path going forward.

- `compact_original_feature_ppo.py` trains the compact PPO model.
- `run_compact_policy.py` loads the saved compact checkpoint and evaluates selected mazes.
- `weights/compact_original_feature_ppo.pt` is the current best simple checkpoint.

Run the saved policy on the bigger mazes:

```powershell
.\.venv\Scripts\python.exe simple\run_compact_policy.py --print-paths
```

Retrain the compact model:

```powershell
.\.venv\Scripts\python.exe simple\compact_original_feature_ppo.py
```
