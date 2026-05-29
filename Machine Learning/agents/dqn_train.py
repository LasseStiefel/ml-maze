"""
Deep Q-Network (DQN)  —  TensorFlow implementation.

Based on: Mnih et al., "Human-level control through deep reinforcement
learning", Nature 2015.

Key DQN ideas implemented here:
  1. Q-network  — maps state -> Q-values for all actions
  2. Target network — separate, frozen copy; synced every C steps
  3. Experience replay — random mini-batch from a circular buffer
  4. Epsilon-greedy exploration — ε decays from 1.0 → 0.05

Select the maze via MAZE_ID in config.py (1, 2, or 3).

Run:
    python dqn_train.py

Outputs (saved to Results/):
    training_DQN_maze<N>.csv   — per-episode reward, steps, loss, epsilon
    q_network_maze<N>.keras    — trained model weights
    training_plot_maze<N>.png  — learning curve
"""

from __future__ import annotations

import importlib
import os
import sys
import csv
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Enable GPU via DirectML on Windows (install: pip install tensorflow-directml-plugin)
# For NVIDIA/CUDA, this block also prevents OOM by enabling memory growth.
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    for _gpu in _gpus:
        tf.config.experimental.set_memory_growth(_gpu, True)


# ---------------------------------------------------------------------------
# Hyper-parameters  (including maze selection)
# ---------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DQN, MAZE_ID

EPISODES        = DQN.EPISODES
MAX_STEPS       = DQN.MAX_STEPS       # max moves per episode

GAMMA           = DQN.GAMMA           # discount factor
ALPHA           = DQN.LEARNING_RATE   # Adam learning rate

EPSILON_START   = DQN.EPSILON_START   # initial exploration rate
EPSILON_MIN     = DQN.EPSILON_MIN     # floor for exploration
EPSILON_DECAY   = DQN.EPSILON_DECAY   # multiplicative decay per episode

BUFFER_SIZE     = DQN.BUFFER_SIZE     # replay buffer capacity
BATCH_SIZE      = DQN.BATCH_SIZE      # mini-batch size
TRAIN_START     = DQN.TRAIN_START     # episodes before we start training
TARGET_UPDATE_C = DQN.TARGET_UPDATE_C # sync target network every C episodes

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

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Machine Learning", "Results", "DQN")
os.makedirs(RESULTS_DIR, exist_ok=True)

_SUFFIX = f"_maze{MAZE_ID}"


# ---------------------------------------------------------------------------
# 1. Build the Q-network
# ---------------------------------------------------------------------------
def build_q_network() -> keras.Model:
    """
    A simple fully-connected network.

    Input  : flat grid observation  → shape (OBS_SIZE,)
    Hidden : Dense(256, relu) → Dense(128, relu) → Dense(64, relu)
    Output : Q-value per action     → shape (4,)

    Why not a CNN?
    The paper uses a CNN because Atari frames are raw pixels.
    Here our "observation" is already a structured float array, so Dense
    layers are simpler and just as effective for these grid mazes.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(OBS_SIZE,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64,  activation="relu"),
        keras.layers.Dense(NUM_ACTIONS, activation="linear"),   # raw Q-values
    ], name="q_network")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ALPHA),
        loss="mse",   # mean-squared Bellman error
    )
    return model


# ---------------------------------------------------------------------------
# 2. Experience Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Circular buffer that stores transitions (s, a, r, s', done).

    Random sampling breaks temporal correlations that would destabilise
    gradient updates if we trained on consecutive frames (as in the paper).
    """

    def __init__(self, capacity: int) -> None:
        self._buf: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Return a batch as stacked numpy arrays."""
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# 3. DQN Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    Encapsulates the Q-network, target network, replay buffer, and the
    core Bellman update step.
    """

    def __init__(self) -> None:
        self.q_net     = build_q_network()    # online network (trained every step)
        self.target_net = build_q_network()   # frozen copy (updated every C episodes)
        self._sync_target()                   # start them identical

        self.buffer  = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START

    # ------------------------------------------------------------------
    # Action selection — epsilon-greedy
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        With probability ε pick a random action (explore).
        Otherwise pick argmax Q(s, ·) from the online network (exploit).
        """
        if np.random.rand() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)

        q_values = self.q_net(state[np.newaxis], training=False)  # shape (1, 4)
        return int(tf.argmax(q_values[0]).numpy())

    # ------------------------------------------------------------------
    # Learning step — Bellman update on a mini-batch
    # ------------------------------------------------------------------

    def learn(self) -> float:
        """
        Sample a random mini-batch from the replay buffer and do one
        gradient step.

        Bellman target:
            y = r                         if terminal
            y = r + γ · max_a' Q_target(s', a')   otherwise

        Loss = MSE( Q_online(s, a)  vs  y )

        Returns the scalar loss value for logging.
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # --- compute targets using the TARGET network (frozen) ---
        next_q = self.target_net(next_states, training=False)       # (B, 4)
        max_next_q = tf.reduce_max(next_q, axis=1).numpy()          # (B,)

        targets = rewards + GAMMA * max_next_q * (1.0 - dones)      # (B,)

        # --- compute current Q predictions from ONLINE network ---
        with tf.GradientTape() as tape:
            all_q = self.q_net(states, training=True)               # (B, 4)

            # Select only the Q-value for the action that was taken
            action_mask = tf.one_hot(actions, NUM_ACTIONS)          # (B, 4)
            q_taken = tf.reduce_sum(all_q * action_mask, axis=1)   # (B,)

            loss = tf.reduce_mean(tf.square(targets - q_taken))     # scalar MSE

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        self.q_net.optimizer.apply_gradients(
            zip(grads, self.q_net.trainable_variables)
        )

        return float(loss.numpy())

    # ------------------------------------------------------------------
    # Target network sync
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Copy weights from online → target network."""
        self.target_net.set_weights(self.q_net.get_weights())

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    env   = MazeEnv(max_steps=MAX_STEPS)
    agent = DQNAgent()

    # ---- Data collectors (what we'll plot / export) ----
    episode_rewards: list[float] = []
    episode_steps:   list[int]   = []
    episode_losses:  list[float] = []
    episode_epsilons: list[float] = []

    csv_path = os.path.join(RESULTS_DIR, f"training_DQN{_SUFFIX}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "avg_loss", "epsilon", "solved"])

        print(f"Training DQN on Maze {MAZE_ID} ({MAZE_SIZE_LABEL})")
        print(f"{'Episode':>8}  {'Reward':>8}  {'Steps':>6}  {'Loss':>8}  {'Epsilon':>7}  {'Solved':>6}")
        print("-" * 60)

        for ep in range(1, EPISODES + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            losses: list[float] = []
            solved = False

            for _ in range(MAX_STEPS):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                done = terminated or truncated
                agent.buffer.push(obs, action, reward, next_obs, done)

                obs = next_obs
                total_reward += reward

                # Only start learning once we have enough experience
                if len(agent.buffer) >= BATCH_SIZE and ep > TRAIN_START:
                    loss = agent.learn()
                    losses.append(loss)

                if terminated:
                    solved = True

                if done:
                    break

            # End of episode
            agent.decay_epsilon()

            # Sync target network every C episodes
            if ep % TARGET_UPDATE_C == 0:
                agent._sync_target()

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
    model_path = os.path.join(RESULTS_DIR, f"q_network{_SUFFIX}.keras")
    agent.q_net.save(model_path)
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
    env    = MazeEnv(max_steps=MAX_STEPS)
    obs, _ = env.reset()
    path   = [env._player]

    saved_eps     = agent.epsilon
    agent.epsilon = 0.0   # greedy — no random moves

    frames  = []
    _ANAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}

    for _ in range(MAX_STEPS):
        prev_pos = env._player
        action   = agent.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

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
        save_dqn_training_plot,
    )

    print('\nGenerating visualisations...')
    maze = _build_maze_data()

    save_maze_layout(maze, RESULTS_DIR, _SUFFIX)

    frames   = _run_vis_episode(agent)
    gif_path = os.path.join(RESULTS_DIR, f'episode_animation{_SUFFIX}.gif')
    save_episode_animation(maze, frames, gif_path, fps=6)

    save_dqn_training_plot(
        episode_rewards, episode_steps, episode_losses, episode_epsilons,
        MAZE_ID, MAZE_SIZE_LABEL, RESULTS_DIR, _SUFFIX,
    )
    print('Visualisation complete.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
    print()
    agent, ep_rewards, ep_steps, ep_losses, ep_epsilons = train()
    visualise(agent, ep_rewards, ep_steps, ep_losses, ep_epsilons)
