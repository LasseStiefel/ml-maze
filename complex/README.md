# Complex PPO Experiments

This folder keeps the larger feature-expanded PPO experiments and their weights.

- `ultimate_multi_ppo.py` adds richer distance/progress features and a distance-prior action boost.
- `ultimate_multi_ppo_original_features.py` keeps the larger architecture but uses the original compact features.
- `weights/` contains their matching checkpoints.

These models performed well, but the repo now treats the compact version in `simple/` as the default because it reached the same maze results with less machinery.
