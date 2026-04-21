# ML-Maze — Teaching Deep Q-Networks with a Grid Maze

A beginner-friendly project that teaches **Deep Reinforcement Learning (DQN)**
by training an AI agent to solve a maze — the same maze you can play yourself.

---

## Table of Contents

1. [What this project does](#what-this-project-does)
2. [Project structure](#project-structure)
3. [The three layers explained](#the-three-layers-explained)
4. [How the files connect to each other](#how-the-files-connect-to-each-other)
5. [Key variables and what they mean](#key-variables-and-what-they-mean)
6. [How to install and run](#how-to-install-and-run)
7. [Understanding the output](#understanding-the-output)
8. [The DQN paper](#the-dqn-paper)
9. [What was changed and what was added](#what-was-changed-and-what-was-added)

---

## What this project does

There are three mazes of increasing size. For **Maze 1** (16×16), we connect
the playable game to a neural network that learns — on its own, through trial
and error — how to reach the exit.

The technique used is called **Deep Q-Network (DQN)**, introduced by DeepMind
in 2015. Instead of Atari games, we apply it to a grid maze. The core ideas
are identical.

```
Human plays maze_1.py  ──────────────────────────────────► fun game
                                                          (no ML involved)

AI trains with dqn_train.py + maze_env.py ───────────────► agent learns
                                                          to solve maze
```

---

## Project structure

```
ml-maze/
│
├── maze_1/                   ← Maze 1: 16×16  (DQN implemented here)
│   ├── maze_1.py             │  Human-playable game (curses terminal UI)
│   ├── q_learning.py         │  Stub for classic Q-learning (tabular)
│   ├── maze_env.py           │  NEW — environment wrapper for the DQN agent
│   └── dqn_train.py          │  NEW — DQN agent, training loop, data collection
│
├── maze_2/                   ← Maze 2: 25×25  (human game only for now)
│   └── maze_2.py
│
├── maze_3/                   ← Maze 3: 35×35  (human game only for now)
│   └── maze_3.py
│
├── Papers/
│   └── DQNNaturePaper.pdf    ← The original DeepMind DQN paper (Mnih et al. 2015)
│
└── dopamine/                 ← Google's RL research framework (not used yet)
```

After training, a `maze_1/results/` folder is created automatically:

```
maze_1/results/
├── training_log.csv      ← per-episode numbers (reward, steps, loss, epsilon)
├── q_network.keras       ← saved neural network weights
└── training_plot.png     ← 4-panel learning curve chart
```

---

## The three layers explained

Think of the project as three independent layers that sit on top of each other:

```
┌──────────────────────────────────────────────────────────┐
│  LAYER 1 — maze_1.py  (Human game)                       │
│                                                          │
│  A normal terminal game. You move with WASD or arrows.   │
│  Uses curses to draw on screen.                          │
│  Has nothing to do with machine learning.                │
└──────────────────────────────────────────────────────────┘
          ↕  shares the same wall layout
┌──────────────────────────────────────────────────────────┐
│  LAYER 2 — maze_env.py  (Environment)                    │
│                                                          │
│  Rebuilds the same maze in a format the AI can use.      │
│  No screen, no keyboard. Just numbers in, numbers out.   │
│  Exposes two functions: reset() and step(action).        │
└──────────────────────────────────────────────────────────┘
          ↕  calls reset() and step()
┌──────────────────────────────────────────────────────────┐
│  LAYER 3 — dqn_train.py  (Learning algorithm)            │
│                                                          │
│  The neural network (Q-network) that learns to play.     │
│  Tries actions, collects rewards, adjusts its weights.   │
│  Runs 1000 episodes and saves everything it learns.      │
└──────────────────────────────────────────────────────────┘
```

These layers do not interfere with each other. You can play the human game
while training runs — they are completely separate programs.

---

## How the files connect to each other

### maze_env.py → what it provides to dqn_train.py

```
maze_env.py                         dqn_train.py
───────────                         ────────────
MazeEnv.reset()          ──────►    obs, _ = env.reset()
MazeEnv.step(action)     ──────►    obs, reward, terminated, truncated, _ = env.step(a)
OBS_SIZE = 256           ──────►    Input shape of the neural network
NUM_ACTIONS = 4          ──────►    Output size of the neural network
```

### What the agent "sees" (the observation)

The environment converts the 16×16 grid into a flat array of 256 numbers:

```
Cell value meanings
───────────────────
0.0  →  empty floor
1.0  →  wall
2.0  →  player (current position)
3.0  →  exit

Example (tiny 4×4 slice):
[ 2.0, 0.0, 1.0, 0.0,     ← player at (0,0), wall at (2,0)
  0.0, 0.0, 0.0, 0.0,
  1.0, 0.0, 0.0, 1.0,
  0.0, 0.0, 0.0, 3.0 ]    ← exit at (3,3)
```

### What the agent "does" (the action)

```
0 → move up
1 → move down
2 → move left
3 → move right
```

### What the agent "feels" (the reward)

```
+50.0  →  reached the exit  (big reward for the goal)
+1.0   →  moved closer to exit  (encourages progress)
-1.0   →  moved further from exit  (discourages wandering)
-0.5   →  bumped into a wall  (small penalty)
```

---

## Key variables and what they mean

These are all defined at the top of `dqn_train.py` so you can easily change them.

| Variable | Default | What it controls |
|---|---|---|
| `EPISODES` | 1000 | How many complete maze runs to train for |
| `MAX_STEPS` | 1024 | Maximum moves per episode before giving up |
| `GAMMA` | 0.99 | **Discount factor** — how much the agent values future rewards vs immediate ones. 0 = only care about now, 1 = care equally about all future rewards |
| `ALPHA` | 0.001 | **Learning rate** — how big each gradient update step is. Too high = unstable, too low = learns very slowly |
| `EPSILON_START` | 1.0 | Agent starts fully random (pure exploration) |
| `EPSILON_MIN` | 0.05 | Agent always keeps 5% randomness even after training |
| `EPSILON_DECAY` | 0.995 | Each episode, epsilon is multiplied by this. Controls how fast exploration → exploitation |
| `BUFFER_SIZE` | 20000 | How many past experiences to store in memory |
| `BATCH_SIZE` | 64 | How many experiences to learn from in each update step |
| `TRAIN_START` | 500 | Wait this many episodes before starting gradient updates (fills the replay buffer first) |
| `TARGET_UPDATE_C` | 10 | Copy online network → target network every this many episodes |

### What is epsilon-greedy?

```
ε = 1.0  →  agent acts randomly 100% of the time  (explores everything)
ε = 0.5  →  agent acts randomly 50% of the time
ε = 0.05 →  agent acts randomly  5% of the time  (mostly uses what it learned)
```

At the start we want the agent to explore the maze freely. As training
progresses, epsilon decays so the agent increasingly uses its learned Q-values.

### What is the Bellman equation (the core update)?

The loss that trains the network comes from this equation:

```
Target  y  =  reward  +  γ × max( Q_target(next_state) )

Loss  =  MSE( Q_online(state, action)  vs  y )
```

In plain English:
- **What did I get?** (`reward`)
- **Plus: what is the best I can get from here?** (`γ × max Q_target`)
- **Did my network predict that correctly?** If not, nudge the weights.

### What is the target network?

Without it, you would be chasing a moving target — the network you are training
is also the one producing the training labels, which causes oscillation.

The **target network** is a frozen copy. It provides stable labels.
Every `TARGET_UPDATE_C` episodes its weights are replaced with the online
network's current weights.

```
Online network  →  being trained every step
Target network  →  frozen, only updated every 10 episodes
                   used only to compute the Bellman target y
```

---

## How to install and run

### Requirements

```bash
pip install tensorflow numpy matplotlib
```

Python 3.10 or later is recommended.

### Play the maze yourself (no ML)

```bash
cd maze_1
python3 maze_1.py
```

Controls: `WASD` or arrow keys. `R` to restart. `Q` or `Esc` to quit.

### Train the DQN agent

```bash
cd maze_1
python3 dqn_train.py
```

You will see a progress table printed every 50 episodes:

```
 Episode    Reward   Steps      Loss  Epsilon  Solved
------------------------------------------------------------
      50    -45.20     312    0.0000   0.7783   False
     100    -38.11     280    0.0000   0.6065   False
     550     12.44     180    0.2341   0.0653    True
    1000     44.88      62    0.0891   0.0500    True
```

Results are saved automatically to `maze_1/results/`.

### Changing the number of episodes

Open `dqn_train.py` and edit line 44:

```python
EPISODES = 1000   # change this number
```

---

## Understanding the output

### training_log.csv

One row per episode with these columns:

| Column | Meaning |
|---|---|
| `episode` | Episode number |
| `total_reward` | Sum of all rewards in this episode |
| `steps` | How many moves the agent made |
| `avg_loss` | Average Bellman MSE loss this episode |
| `epsilon` | Exploration rate at end of episode |
| `solved` | 1 if agent reached the exit, 0 if it ran out of steps |

### training_plot.png

Four charts:

1. **Total reward per episode** — should trend upward as the agent improves
2. **Steps per episode** — should trend downward (fewer moves to solve)
3. **Bellman MSE loss** — typically rises then falls as the network stabilises
4. **Epsilon decay** — exponential curve from 1.0 down to 0.05

The blue/red/green lines are raw per-episode values. The bold lines are
50-episode rolling averages to show the trend more clearly.

---

## The DQN paper

The algorithm implemented here is based on:

> Mnih, V., Kavukcuoglu, K., Silver, D., et al.  
> **"Human-level control through deep reinforcement learning"**  
> *Nature*, 518, 529–533 (2015).

The paper's original application was Atari video games. The key differences
in our implementation:

| Paper (Atari) | This project (Maze) |
|---|---|
| CNN processes raw pixels | Dense layers process a flat grid array |
| 84×84×4 grayscale frames | 256-element float array |
| 18 possible actions | 4 actions (up/down/left/right) |
| Reward clipped to [-1, 1] | Reward shaped to encourage progress |
| Millions of frames | 1000 episodes |

The core algorithm — Q-network, target network, experience replay,
epsilon-greedy, Bellman update — is identical to the paper.

---

## What was changed and what was added

### Original files (unchanged)

| File | What it does |
|---|---|
| `maze_1/maze_1.py` | 16×16 human-playable maze game |
| `maze_1/q_learning.py` | Stub for tabular Q-learning (not yet complete) |
| `maze_2/maze_2.py` | 25×25 human-playable maze game |
| `maze_3/maze_3.py` | 35×35 human-playable maze game |

None of these files were modified.

### New files added

| File | What it does |
|---|---|
| `maze_1/maze_env.py` | Wraps Maze 1 into a gym-style environment for the agent |
| `maze_1/dqn_train.py` | Full DQN implementation: Q-network, target network, replay buffer, training loop, CSV logging, plotting |
| `README.md` | This file |
