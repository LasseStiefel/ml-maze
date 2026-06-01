# Compact Big-Maze Holdout Experiment

Question: can the compact approach solve big mazes without training on big mazes?

## Setup

I added:

```text
experiments/compact_holdout_big_mazes.py
```

The experiment trains on only:

- `maze_1/maze_1.py`
- `maze_1/maze_11.py`
- `maze_1/maze_12.py`
- `maze_1/maze_13.py`
- `maze_1/maze_14.py`

It holds out:

- `maze_2/maze_2.py`
- `maze_3/maze_3.py`

The usual compact checkpoint has task embeddings, so unseen mazes do not have trained maze IDs. For a fairer zero-shot test, this experiment removes the task embedding and keeps the compact 8-feature observation, action masks, BFS expert behavior cloning, and PPO update loop.

Command used:

```powershell
.\.venv\Scripts\python.exe experiments\compact_holdout_big_mazes.py --bc-epochs 80 --updates 60 --rollout-steps 256 --eval-every 10 --print-paths
```

## Result

The policy did not solve the big held-out mazes.

| Split | Solved | Optimal |
|---|---:|---:|
| Small training mazes | 3/5 | 3/5 |
| Big held-out mazes | 0/2 | 0/2 |

Held-out details:

| Maze | Shortest path | Greedy path length | Result |
|---|---:|---:|---|
| `maze_2/maze_2.py` | 120 | 31 | looped |
| `maze_3/maze_3.py` | 276 | 41 | looped |

## Conclusion

The current compact model is excellent when the target maze is included in training, but it does not yet solve larger unseen mazes reliably. To make that work, the next experiment should train a no-task-embedding or maze-invariant policy on many generated mazes across sizes, then evaluate on held-out larger mazes.

## Additional Transfer Tests

I also tried the two reverse/mixed transfer questions.

### Biggest Maze to Smaller Mazes

Question: if the policy trains only on the biggest maze, can it solve the smaller ones?

Command:

```powershell
.\.venv\Scripts\python.exe experiments\compact_holdout_big_mazes.py --split biggest_to_smaller --bc-epochs 1000 --updates 0 --rollout-steps 512 --eval-every 10
```

Result:

| Training maze | Training result | Held-out mazes | Held-out solved |
|---|---|---|---:|
| `maze_3/maze_3.py` | optimal, 276/276 | `maze_1/*`, `maze_2/maze_2.py` | 0/6 |

Held-out details:

| Maze | Shortest path | Greedy path before loop | Result |
|---|---:|---:|---|
| `maze_1/maze_1.py` | 30 | 13 | looped |
| `maze_1/maze_11.py` | 30 | 10 | looped |
| `maze_1/maze_12.py` | 30 | 6 | looped |
| `maze_1/maze_13.py` | 60 | 17 | looped |
| `maze_1/maze_14.py` | 108 | 31 | looped |
| `maze_2/maze_2.py` | 120 | 38 | looped |

The policy can memorize the biggest maze, but that does not become a general smaller-maze solver.

### Small Mazes Plus Biggest Maze to Medium Maze

Question: if the policy trains on small mazes and the biggest maze, can it solve the medium maze?

Command:

```powershell
.\.venv\Scripts\python.exe experiments\compact_holdout_big_mazes.py --split small_biggest_to_medium --bc-epochs 1000 --updates 0 --rollout-steps 512 --eval-every 10
```

Result:

| Training mazes | Training solved | Held-out maze | Held-out solved |
|---|---:|---|---:|
| `maze_1/*` plus `maze_3/maze_3.py` | 2/6 | `maze_2/maze_2.py` | 0/1 |

Held-out detail:

| Maze | Shortest path | Greedy path before loop | Result |
|---|---:|---:|---|
| `maze_2/maze_2.py` | 120 | 33 | looped |

This is another sign that the 8-feature compact no-task policy is not learning a robust transfer rule from the current hand-authored mazes.
