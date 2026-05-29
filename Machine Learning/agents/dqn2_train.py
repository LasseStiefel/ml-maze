"""
DQN2  —  Deep Q-Network (PyTorch implementation).

Ported from ml-semester-project/agents/dqn.py.

Key DQN ideas implemented here:
  1. Q-network   — maps one-hot state → Q-values for all actions (PyTorch)
  2. Target network — separate, frozen copy; synced every TARGET_UPDATE_FREQ steps
  3. Experience replay — random mini-batch from a circular buffer
  4. Epsilon-greedy exploration — ε decays from 1.0 → 0.01

State encoding: integer index  y * WIDTH + x  →  one-hot vector of size (WIDTH*HEIGHT).

Select the maze via MAZE_ID in config.py (1, 2, or 3).

Run:
    python agents/dqn2_train.py

Outputs (saved to Results/DQN2/):
    training_DQN2_maze<N>.csv        — per-episode reward, steps, loss, epsilon
    dqn2_model_maze<N>.pt            — trained model weights (PyTorch)
    training_plot_DQN2_maze<N>.png   — learning curve
"""

from __future__ import annotations

import importlib
import os
import sys
import csv
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Hyper-parameters  (including maze selection)
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DQN2, MAZE_ID

EPISODES           = DQN2.EPISODES
MAX_STEPS          = DQN2.MAX_STEPS

GAMMA              = DQN2.GAMMA
HIDDEN_SIZE        = DQN2.HIDDEN_SIZE
LEARNING_RATE      = DQN2.LEARNING_RATE

EPSILON_START      = DQN2.EPSILON_START
EPSILON_END        = DQN2.EPSILON_END
EPSILON_DECAY      = DQN2.EPSILON_DECAY

BUFFER_SIZE        = DQN2.BUFFER_SIZE
BATCH_SIZE         = DQN2.BATCH_SIZE
TARGET_UPDATE_FREQ = DQN2.TARGET_UPDATE_FREQ


# ---------------------------------------------------------------------------
# Maze environment  —  resolved from MAZE_ID in config.py
# ---------------------------------------------------------------------------
_MAZE_INFO = {
    1: ("maze_1", "maze_env",    "16×16"),
    2: ("maze_2", "maze_2_env",  "25×25"),
    3: ("maze_3", "maze_3_env",  "35×35"),
}

if MAZE_ID not in _MAZE_INFO:
    raise ValueError(f"MAZE_ID must be 1, 2, or 3 — got {MAZE_ID!r}")

_maze_dir, _maze_module, MAZE_SIZE_LABEL = _MAZE_INFO[MAZE_ID]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", _maze_dir))
_env_mod    = importlib.import_module(_maze_module)
MazeEnv     = _env_mod.MazeEnv
OBS_SIZE    = _env_mod.OBS_SIZE
NUM_ACTIONS = _env_mod.NUM_ACTIONS
WIDTH       = _env_mod.WIDTH


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Machine Learning", "Results", "DQN2")
os.makedirs(RESULTS_DIR, exist_ok=True)

_SUFFIX = f"_maze{MAZE_ID}"


# ---------------------------------------------------------------------------
# 1. Q-Network  (source: ml-semester-project/agents/dqn.py — unchanged)
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Two-layer feedforward network that approximates Q(s, a)."""

    def __init__(self, state_size: int, action_size: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. Replay Buffer  (source: ml-semester-project/agents/dqn.py — unchanged)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 3. DQN Agent  (source: ml-semester-project/agents/dqn.py — adapted to use
#                module-level config constants instead of the other project's
#                config module; update() now returns the loss for logging)
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    Deep Q-Network (DQN) agent.

    Improvements over tabular Q-Learning:
    1. Neural network approximates Q(s,a) — scales to large/continuous state spaces
    2. Experience replay — breaks correlations between consecutive samples
    3. Target network — separate network for stable TD targets, synced every N steps

    Same off-policy update logic as Q-Learning, but:
        loss = MSE(Q(s,a),  r + γ * max_a' Q_target(s', a'))
    """

    def __init__(self, state_size: int, action_size: int):
        self.state_size  = state_size
        self.action_size = action_size
        self.gamma              = GAMMA
        self.epsilon            = EPSILON_START
        self.epsilon_end        = EPSILON_END
        self.epsilon_decay      = EPSILON_DECAY
        self.batch_size         = BATCH_SIZE
        self.target_update_freq = TARGET_UPDATE_FREQ

        self.device = torch.device("cpu")

        # Online network (trained every step) and target network (synced periodically)
        self.online_net = QNetwork(state_size, action_size, HIDDEN_SIZE).to(self.device)
        self.target_net = QNetwork(state_size, action_size, HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn   = nn.MSELoss()
        self.replay    = ReplayBuffer(BUFFER_SIZE)
        self.steps     = 0

    def _state_tensor(self, state: int) -> torch.Tensor:
        """One-hot encode integer state for the network input."""
        vec = np.zeros(self.state_size, dtype=np.float32)
        vec[state] = 1.0
        return torch.tensor(vec, device=self.device).unsqueeze(0)

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            q_vals = self.online_net(self._state_tensor(state))
        return int(q_vals.argmax().item())

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float | None:
        """
        Store transition, then perform one Bellman update if the buffer is warm.
        Returns the scalar MSE loss, or None if the buffer is not yet full enough.
        """
        self.replay.push(state, action, reward, next_state, done)
        self.steps += 1

        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Build tensors
        state_vecs      = torch.zeros(self.batch_size, self.state_size, device=self.device)
        next_state_vecs = torch.zeros(self.batch_size, self.state_size, device=self.device)
        for i, (s, ns) in enumerate(zip(states, next_states)):
            state_vecs[i, s]      = 1.0
            next_state_vecs[i, ns] = 1.0

        actions_t = torch.tensor(actions, dtype=torch.long,    device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t   = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        # Current Q values for taken actions
        q_values = self.online_net(state_vecs).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q  = self.target_net(next_state_vecs).max(1).values
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "steps":      self.steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self.steps   = ckpt.get("steps", 0)


# ---------------------------------------------------------------------------
# State helper  —  map (x, y) player position to integer cell index
# ---------------------------------------------------------------------------
def _pos_to_state(player: tuple[int, int]) -> int:
    """Convert (x, y) position to flat index  y * WIDTH + x."""
    x, y = player
    return y * WIDTH + x


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train() -> tuple:
    env   = MazeEnv(max_steps=MAX_STEPS)
    agent = DQNAgent(state_size=OBS_SIZE, action_size=NUM_ACTIONS)

    # ---- Data collectors (what we'll plot / export) ----
    episode_rewards:  list[float] = []
    episode_steps:    list[int]   = []
    episode_losses:   list[float] = []
    episode_epsilons: list[float] = []

    csv_path = os.path.join(RESULTS_DIR, f"training_DQN2{_SUFFIX}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "avg_loss", "epsilon", "solved"])

        print(f"Training DQN2 (PyTorch) on Maze {MAZE_ID} ({MAZE_SIZE_LABEL})")
        print(f"{'Episode':>8}  {'Reward':>8}  {'Steps':>6}  {'Loss':>8}  {'Epsilon':>7}  {'Solved':>6}")
        print("-" * 60)

        for ep in range(1, EPISODES + 1):
            env.reset()
            state        = _pos_to_state(env._player)
            total_reward = 0.0
            losses: list[float] = []
            solved = False

            for _ in range(MAX_STEPS):
                action = agent.select_action(state)
                _, reward, terminated, truncated, _ = env.step(action)
                next_state = _pos_to_state(env._player)

                done = terminated or truncated
                loss = agent.update(state, action, reward, next_state, done)
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward

                if terminated:
                    solved = True
                if done:
                    break

            agent.decay_epsilon()

            avg_loss = float(np.mean(losses)) if losses else 0.0

            episode_rewards.append(total_reward)
            episode_steps.append(env._steps)
            episode_losses.append(avg_loss)
            episode_epsilons.append(agent.epsilon)

            writer.writerow([ep, f"{total_reward:.2f}", env._steps,
                             f"{avg_loss:.4f}", f"{agent.epsilon:.4f}", int(solved)])

            if ep % 50 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                print(f"{ep:>8}  {avg_r:>8.2f}  {env._steps:>6}  "
                      f"{avg_loss:>8.4f}  {agent.epsilon:>7.4f}  {str(solved):>6}")

    print("\nTraining complete.")
    print(f"Log saved  → {csv_path}")

    # ---- Save trained model ----
    model_path = os.path.join(RESULTS_DIR, f"dqn2_model{_SUFFIX}.pt")
    agent.save(model_path)
    print(f"Model saved → {model_path}")

    return agent, episode_rewards, episode_steps, episode_losses, episode_epsilons


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------

def _build_maze_data():
    """Assemble a MazeData from the module-level env-module globals."""
    from visualisation import MazeData
    return MazeData(
        width      = _env_mod.WIDTH,
        height     = _env_mod.HEIGHT,
        start      = _env_mod.START,
        exit       = _env_mod.EXIT_CELL,
        walls      = _env_mod.WALLS,
        spikes     = getattr(_env_mod, 'SPIKES', frozenset()),
        maze_id    = MAZE_ID,
        size_label = MAZE_SIZE_LABEL,
    )


def _run_vis_episode(agent: DQNAgent) -> list[dict]:
    """Run one fully-greedy episode and collect per-step animation frames."""
    env = MazeEnv(max_steps=MAX_STEPS)
    env.reset()
    path = [env._player]

    saved_eps     = agent.epsilon
    agent.epsilon = 0.0   # greedy — no random moves

    frames  = []
    _ANAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

    for _ in range(MAX_STEPS):
        prev_pos = env._player
        state    = _pos_to_state(env._player)
        action   = agent.select_action(state)
        _, reward, terminated, truncated, _ = env.step(action)

        moved = env._player != prev_pos
        path.append(env._player)

        frames.append({
            'path':   list(path),
            'agent':  env._player,
            'step':   env._steps,
            'dist':   env._manhattan(),
            'reward': reward,
            'action': _ANAMES.get(action, '?'),
            'moved':  moved,
            'exit':   terminated and reward > 0,
        })
        if terminated or truncated:
            break

    agent.epsilon = saved_eps
    return frames


def visualise(
    agent:            DQNAgent,
    episode_rewards:  list,
    episode_steps:    list,
    episode_losses:   list,
    episode_epsilons: list,
) -> None:
    """Generate maze layout PNG, episode animation GIF, and training curves."""
    from visualisation import (
        save_maze_layout,
        save_episode_animation,
        save_dqn2_training_plot,
    )

    print('\nGenerating visualisations...')
    maze = _build_maze_data()

    save_maze_layout(maze, RESULTS_DIR, _SUFFIX)

    frames   = _run_vis_episode(agent)
    gif_path = os.path.join(RESULTS_DIR, f'episode_animation{_SUFFIX}.gif')
    save_episode_animation(maze, frames, gif_path, fps=6)

    save_dqn2_training_plot(
        episode_rewards, episode_steps, episode_losses, episode_epsilons,
        MAZE_ID, MAZE_SIZE_LABEL, RESULTS_DIR, _SUFFIX,
    )
    print('Visualisation complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print()
    agent, ep_rewards, ep_steps, ep_losses, ep_epsilons = train()
    visualise(agent, ep_rewards, ep_steps, ep_losses, ep_epsilons)
