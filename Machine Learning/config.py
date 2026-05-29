"""
Centralised hyperparameter configuration for all agents.

Usage
-----
from config import DQN, DQN2, PPO, SARSA, QLearning, SAC, MAZE_ID
"""

from __future__ import annotations
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Maze selection  —  set to 1, 2, or 3
# ---------------------------------------------------------------------------
#   Maze 1 : 16×16  (256 observations)
#   Maze 2 : 25×25  (625 observations)
#   Maze 3 : 35×35  (1225 observations, includes spikes)
MAZE_ID = 3


# ---------------------------------------------------------------------------
# Shared reward constants (used across all agents)
# ---------------------------------------------------------------------------
MAZE_FINISHED = 50   # reward for reaching the exit
WALL_PENALTY  = -1    # penalty for hitting a wall
STEP_PENALTY  = -0.01    # penalty per normal step


# ---------------------------------------------------------------------------
# DQN  —  Deep Q-Network
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _DQNConfig:
    # Training
    EPISODES:        int   = 4000
    MAX_STEPS:       int   = 16 * 16 * 4   # max moves per episode

    # Bellman
    GAMMA:           float = 0.995           # discount factor
    LEARNING_RATE:   float = 0.0005          # Adam learning rate

    # Exploration (epsilon-greedy)
    EPSILON_START:   float = 1.0
    EPSILON_MIN:     float = 0.05
    EPSILON_DECAY:   float = 0.997          # multiplicative decay per episode

    # Replay buffer
    BUFFER_SIZE:     int   = 50000
    BATCH_SIZE:      int   = 64
    TRAIN_START:     int   = 1500            # episodes before learning starts

    # Target network
    TARGET_UPDATE_C: int   = 10             # sync target every C episodes

DQN = _DQNConfig()


# ---------------------------------------------------------------------------
# DQN2  —  Deep Q-Network (PyTorch implementation)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _DQN2Config:
    # Training
    EPISODES:           int   = 1000
    MAX_STEPS:          int   = 16 * 16 * 4   # max moves per episode

    # Bellman
    GAMMA:              float = 0.95           # discount factor

    # Network
    HIDDEN_SIZE:        int   = 128            # neurons per hidden layer (two layers)
    LEARNING_RATE:      float = 0.001          # Adam learning rate

    # Exploration (epsilon-greedy)
    EPSILON_START:      float = 1.0
    EPSILON_END:        float = 0.01           # floor for exploration
    EPSILON_DECAY:      float = 0.997          # multiplicative decay per episode

    # Replay buffer
    BUFFER_SIZE:        int   = 20_000
    BATCH_SIZE:         int   = 64
    TARGET_UPDATE_FREQ: int   = 200            # sync target net every N agent steps

DQN2 = _DQN2Config()


# ---------------------------------------------------------------------------
# PPO  —  Proximal Policy Optimization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _PPOConfig:
    # Training
    EPISODES:       int   = 400
    ROLLOUT_STEPS:  int   = 2048           # steps collected before each update

    # Bellman / advantage
    GAMMA:          float = 0.95           # discount factor
    GAE_LAMBDA:     float = 0.95           # GAE smoothing parameter

    # Clipping
    CLIP_EPSILON:   float = 0.2            # PPO clip ratio

    # Optimiser
    LEARNING_RATE:  float = 3e-4

    # Update loop
    UPDATE_EPOCHS:  int   = 6              # passes over each rollout
    MINIBATCH_SIZE: int   = 128

    # Loss coefficients
    ENTROPY_COEF:   float = 0.1            # encourages exploration
    VALUE_COEF:     float = 0.5            # scales critic loss

PPO = _PPOConfig()


# ---------------------------------------------------------------------------
# SARSA  —  On-policy TD control
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SARSAConfig:
    # Training
    EPISODES:      int   = 5000

    # TD update
    ALPHA:         float = 0.1             # learning rate
    GAMMA:         float = 0.95            # discount factor

    # Exploration (epsilon-greedy)
    EPSILON:       float = 1.0
    EPSILON_DECAY: float = 0.995
    MIN_EPSILON:   float = 0.05

SARSA = _SARSAConfig()


# ---------------------------------------------------------------------------
# Q-Learning  —  Off-policy tabular TD
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _QLearningConfig:
    # Training
    EPISODES:      int   = 5000

    # TD update
    ALPHA:         float = 0.1             # learning rate
    GAMMA:         float = 0.95            # discount factor

    # Exploration (epsilon-greedy)
    EPSILON:       float = 1.0
    EPSILON_DECAY: float = 0.995
    MIN_EPSILON:   float = 0.05

QLearning = _QLearningConfig()


# ---------------------------------------------------------------------------
# SAC  —  Soft Actor-Critic  (stable-baselines3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SACConfig:
    # Training duration
    TIMESTEPS:         int   = 10_000        # total environment steps to train for
    MAX_STEPS:         int   = 16 * 16 * 4  # max steps per episode

    # Bellman
    GAMMA:             float = 0.99          # discount factor

    # Optimiser
    LEARNING_RATE:     float = 3e-4          # actor, critic, and alpha networks

    # Entropy regularisation
    ALPHA:             float = 0.2           # initial entropy temperature
    AUTO_TUNE_ALPHA:   bool  = True          # auto-tune ALPHA during training
    TARGET_ENTROPY:    float = -1.0          # target entropy (used when auto-tuning)

    # Replay buffer
    BUFFER_SIZE:       int   = 200_000       # diverse experiences → stable gradients
    BATCH_SIZE:        int   = 256
    TRAIN_START:       int   = 1_000        # steps of random exploration before learning

    # Target network (soft update)
    TAU:               float = 0.005         # θ_target = τ·θ + (1-τ)·θ_target
    TARGET_UPDATE_INTERVAL: int = 1

    # Network architecture
    HIDDEN_SIZE:       int   = 256           # neurons per hidden layer  (two layers)

    # Logging & checkpointing
    RECORD_EVERY:      int   = 1             # run greedy eval every N training episodes

    # Reward shaping
    STEP_PENALTY:      float = -0.1          # constant cost per step
    WALL_PENALTY:      float = -3.0          # hitting a wall
    CLOSER_REWARD:     float =  1.0          # BFS distance decreased
    FARTHER_PENALTY:   float = -0.3          # BFS distance increased
    EXPLORE_BONUS:     float =  0.5          # first visit to a cell
    REVISIT_BASE:      float =  1.0          # penalty magnitude per revisit (×n visits)
    REVISIT_CAP:       float = 20.0          # cap on revisit penalty
    PROX_THRESHOLD:    int   =  5            # proximity bonus when BFS dist ≤ this
    EXIT_REWARD:       float = 200.0         # reaching the exit (overrides all shaping)
    SPIKE_PENALTY:     float = -10.0         # stepping on a spike — maze 3 only

SAC = _SACConfig()
