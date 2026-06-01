# Compact PPO Explanation

The compact PPO is the preferred model because it solved all discovered mazes, including `maze_2` and `maze_3`, with the smallest successful design.

## What It Sees

Each maze state is encoded with 8 features:

- current `x` and `y`, normalized by maze size
- exit direction as `(exit_x - x)` and `(exit_y - y)`, normalized by maze size
- four wall/open indicators for up, down, left, and right

That keeps the input small, but it still tells the policy where it is, where the exit is, and which moves are physically possible.

## Why It Works

The model is not relying on raw random PPO exploration alone. Its strongest ingredients are:

- BFS distance maps from every reachable cell to the exit
- action masks that remove wall moves from the policy distribution
- expert masks that mark actions that reduce BFS distance
- behavior cloning before PPO, so the policy starts near shortest-path behavior
- PPO fine-tuning with a small progress reward, repeat penalty, timeout penalty, and value learning
- task embeddings, so one shared model can remember maze-specific behavior

In practice, the behavior cloning stage teaches the shortest-path skeleton, and PPO mostly stabilizes the policy/value estimates.

## Current Compact Result

The saved compact checkpoint is:

```text
simple/weights/compact_original_feature_ppo.pt
```

It solved 7/7 discovered mazes optimally in greedy evaluation:

| Maze | Shortest path | Compact path |
|---|---:|---:|
| `maze_1/maze_1.py` | 30 | 30 |
| `maze_1/maze_11.py` | 30 | 30 |
| `maze_1/maze_12.py` | 30 | 30 |
| `maze_1/maze_13.py` | 60 | 60 |
| `maze_1/maze_14.py` | 108 | 108 |
| `maze_2/maze_2.py` | 120 | 120 |
| `maze_3/maze_3.py` | 276 | 276 |

## Zero-Shot Big Maze Test

I also tested the harder question: train without `maze_2` and `maze_3`, then evaluate those big mazes as unseen holdouts.

Because the normal compact model uses task embeddings, a truly unseen maze has no trained embedding. For this test I used the same compact 8 input features, action masks, behavior cloning, and PPO loop, but removed the task embedding so the policy had to rely on shared maze-solving behavior.

Result:

| Training mazes | Held-out mazes | Holdout result |
|---|---|---|
| `maze_1`, `maze_11`, `maze_12`, `maze_13`, `maze_14` | `maze_2`, `maze_3` | 0/2 solved |

The trained policy still solved only 3/5 small training mazes in that short run, and on the held-out big mazes it looped:

| Maze | Shortest path | Zero-shot path before loop | Result |
|---|---:|---:|---|
| `maze_2/maze_2.py` | 120 | 31 | looped |
| `maze_3/maze_3.py` | 276 | 41 | looped |

So the compact checkpoint is strong when trained on the target mazes, but the current compact representation does not yet prove reliable zero-shot generalization to larger unseen mazes.

Use this command to run the saved policy on the larger mazes:

```powershell
.\.venv\Scripts\python.exe simple\run_compact_policy.py --print-paths
```
