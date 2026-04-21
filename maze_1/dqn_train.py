"""
Deep Q-Network (DQN) for Maze 1  —  TensorFlow implementation.

Based on: Mnih et al., "Human-level control through deep reinforcement
learning", Nature 2015.

Key DQN ideas implemented here:
  1. Q-network  — maps state -> Q-values for all actions
  2. Target network — separate, frozen copy; synced every C steps
  3. Experience replay — random mini-batch from a circular buffer
  4. Epsilon-greedy exploration — ε decays from 1.0 → 0.05

Run:
    python dqn_train.py

Outputs (saved to maze_1/results/):
    training_log.csv   — per-episode reward, steps, loss, epsilon
    q_network.keras    — trained model weights
    training_plot.png  — learning curve
"""

from __future__ import annotations

import os
import csv
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt

from maze_env import MazeEnv, OBS_SIZE, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPISODES        = 1000          # total training episodes
MAX_STEPS       = 16 * 16 * 4  # max moves per episode (1024)

GAMMA           = 0.99          # discount factor
ALPHA           = 0.001         # Adam learning rate

EPSILON_START   = 1.0           # initial exploration rate
EPSILON_MIN     = 0.05          # floor for exploration
EPSILON_DECAY   = 0.995         # multiplicative decay per episode

BUFFER_SIZE     = 20_000        # replay buffer capacity
BATCH_SIZE      = 64            # mini-batch size
TRAIN_START     = 500           # episodes before we start training
TARGET_UPDATE_C = 10            # sync target network every C episodes

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Build the Q-network
# ---------------------------------------------------------------------------
def build_q_network() -> keras.Model:
    """
    A simple fully-connected network.

    Input  : flat grid observation  → shape (256,)
    Hidden : Dense(256, relu) → Dense(128, relu) → Dense(64, relu)
    Output : Q-value per action     → shape (4,)

    Why not a CNN?
    The paper uses a CNN because Atari frames are raw pixels.
    Here our "observation" is already a structured float array, so Dense
    layers are simpler and just as effective for a 16×16 grid.
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

    csv_path = os.path.join(RESULTS_DIR, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "avg_loss", "epsilon", "solved"])

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
    model_path = os.path.join(RESULTS_DIR, "q_network.keras")
    agent.q_net.save(model_path)
    print(f"Model saved → {model_path}")

    # ---- Plot learning curves ----
    _plot(episode_rewards, episode_steps, episode_losses, episode_epsilons)


# ---------------------------------------------------------------------------
# 5. Plotting / data visualisation
# ---------------------------------------------------------------------------

def _rolling_mean(values: list[float], window: int = 50) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        out[i] = np.mean(values[i - window + 1 : i + 1])
    return out


def _plot(
    rewards: list[float],
    steps: list[int],
    losses: list[float],
    epsilons: list[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("DQN Training — Maze 1 (16×16)", fontsize=14)

    eps_range = range(1, len(rewards) + 1)

    # --- (a) Episode reward ---
    ax = axes[0, 0]
    ax.plot(eps_range, rewards, alpha=0.3, color="steelblue", label="raw")
    ax.plot(eps_range, _rolling_mean(rewards), color="steelblue", linewidth=2, label="50-ep avg")
    ax.set_title("Total reward per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- (b) Steps per episode ---
    ax = axes[0, 1]
    ax.plot(eps_range, steps, alpha=0.3, color="coral", label="raw")
    ax.plot(eps_range, _rolling_mean(steps), color="coral", linewidth=2, label="50-ep avg")
    ax.set_title("Steps per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- (c) Training loss ---
    ax = axes[1, 0]
    ax.plot(eps_range, losses, alpha=0.3, color="green", label="raw")
    ax.plot(eps_range, _rolling_mean(losses), color="green", linewidth=2, label="50-ep avg")
    ax.set_title("Average Bellman MSE loss per episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- (d) Epsilon decay ---
    ax = axes[1, 1]
    ax.plot(eps_range, epsilons, color="purple", linewidth=2)
    ax.set_title("Epsilon (exploration rate)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "training_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved  → {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")
    print()
    train()
