from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


ACTIONS = [
    ("up", (0, -1)),
    ("down", (0, 1)),
    ("left", (-1, 0)),
    ("right", (1, 0)),
]

BASE_FEATURES = 14
VALID_FEATURE_START = BASE_FEATURES
WALL_FEATURE_START = VALID_FEATURE_START + len(ACTIONS)
PROGRESS_FEATURE_START = WALL_FEATURE_START + len(ACTIONS)
RAW_PROGRESS_FEATURE_START = PROGRESS_FEATURE_START + len(ACTIONS)
NEIGHBOR_DISTANCE_FEATURE_START = RAW_PROGRESS_FEATURE_START + len(ACTIONS)


@dataclass
class TrainConfig:
    seed: int = 7
    hidden_size: int = 192
    task_embedding_size: int = 16
    bc_epochs: int = 300
    bc_batch_size: int = 512
    bc_learning_rate: float = 1e-3
    updates: int = 160
    rollout_steps: int = 512
    update_epochs: int = 3
    minibatch_size: int = 1024
    learning_rate: float = 2.5e-4
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.18
    entropy_start: float = 0.025
    entropy_end: float = 0.004
    value_coef: float = 0.5
    max_grad_norm: float = 0.75
    target_kl: float = 0.04
    finish_reward: float = 20.0
    step_penalty: float = -0.03
    progress_reward: float = 0.20
    repeat_penalty: float = -0.015
    timeout_penalty: float = -5.0
    distance_prior_scale: float = 3.0
    eval_every: int = 10
    early_stop_patience: int = 3


@dataclass
class MazeSource:
    path: Path
    module: Any
    maze: Any
    name: str


@dataclass
class MazeTask:
    index: int
    name: str
    path: Path
    maze: Any
    distance_map: dict[tuple[int, int], int]
    states: list[tuple[int, int]]
    state_to_row: dict[tuple[int, int], int]
    obs: torch.Tensor
    masks: torch.Tensor
    expert_masks: torch.Tensor
    value_targets: torch.Tensor
    max_distance: int
    shortest_length: int
    start_row: int
    max_steps: int


@dataclass
class Rollout:
    obs: list[torch.Tensor]
    task_ids: list[int]
    masks: list[torch.Tensor]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    last_value: torch.Tensor
    episode_rewards: list[float]
    episode_wins: list[bool]
    episode_lengths: list[int]


@dataclass
class EvalResult:
    task_name: str
    path: list[tuple[int, int]]
    reached_exit: bool
    looped: bool
    shortest_length: int

    @property
    def path_length(self) -> int:
        return len(self.path) - 1

    @property
    def optimal(self) -> bool:
        return self.reached_exit and self.path_length == self.shortest_length


class ActorCritic(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_size: int,
        task_count: int,
        hidden_size: int,
        task_embedding_size: int,
    ) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(task_count, task_embedding_size)
        trunk_input = input_size + task_embedding_size

        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(
        self,
        obs: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded_tasks = self.task_embedding(task_ids)
        hidden = self.trunk(torch.cat([obs, embedded_tasks], dim=-1))
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values


def apply_action_mask(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)


def policy_logits_and_value(
    model: ActorCritic,
    obs: torch.Tensor,
    task_ids: torch.Tensor,
    masks: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, values = model(obs, task_ids)
    distance_prior = obs[
        :,
        RAW_PROGRESS_FEATURE_START : RAW_PROGRESS_FEATURE_START + len(ACTIONS),
    ]
    logits = logits + config.distance_prior_scale * distance_prior
    return apply_action_mask(logits, masks), values


def pick_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def natural_key(path: Path) -> tuple[str, ...]:
    return tuple(path.parts)


def should_skip_path(path: Path, root: Path) -> bool:
    ignored_parts = {".git", ".venv", ".vendor", "__pycache__"}
    relative = path.relative_to(root)
    return any(part in ignored_parts or part.startswith(".") for part in relative.parts)


def load_module_from_path(path: Path) -> Any:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    module_name = f"discovered_maze_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_maze(module: Any) -> Any:
    width = int(module.DEFAULT_WIDTH)
    height = int(module.DEFAULT_HEIGHT)
    start = tuple(module.START)
    exit_cell = tuple(module.resolve_exit(width, height))

    return module.Maze(
        width=width,
        height=height,
        start=start,
        exit=exit_cell,
        walls=module.build_walls(width, height, start, exit_cell),
    )


def has_maze_api(module: Any) -> bool:
    required_names = [
        "DEFAULT_WIDTH",
        "DEFAULT_HEIGHT",
        "START",
        "resolve_exit",
        "build_walls",
        "Maze",
    ]
    return all(hasattr(module, name) for name in required_names)


def discover_mazes(root: Path) -> list[MazeSource]:
    sources: list[MazeSource] = []

    for path in sorted(root.rglob("maze_*.py"), key=natural_key):
        if should_skip_path(path, root):
            continue

        try:
            module = load_module_from_path(path)
            if not has_maze_api(module):
                continue
            maze = build_maze(module)
        except Exception as exc:
            print(f"Skipping {path.relative_to(root)}: {exc}")
            continue

        if not hasattr(maze, "is_open"):
            continue

        sources.append(
            MazeSource(
                path=path,
                module=module,
                maze=maze,
                name=str(path.relative_to(root)),
            )
        )

    return sources


def build_distance_map(maze: Any) -> dict[tuple[int, int], int]:
    queue: deque[tuple[int, int]] = deque([maze.exit])
    distances = {maze.exit: 0}

    while queue:
        x, y = queue.popleft()
        current_distance = distances[(x, y)]

        for _name, (dx, dy) in ACTIONS:
            next_state = (x + dx, y + dy)
            if maze.is_open(next_state) and next_state not in distances:
                distances[next_state] = current_distance + 1
                queue.append(next_state)

    return distances


def encode_state(
    maze: Any,
    state: tuple[int, int],
    distance_map: dict[tuple[int, int], int],
    max_distance: int,
    max_width: int,
    max_height: int,
) -> torch.Tensor:
    x, y = state
    ex, ey = maze.exit
    width_scale = max(1, maze.width - 1)
    height_scale = max(1, maze.height - 1)
    global_width_scale = max(1, max_width)
    global_height_scale = max(1, max_height)
    distance_scale = max(1, max_distance)
    current_distance = distance_map[state]

    valid_moves = []
    wall_bits = []
    progress = []
    raw_progress = []
    neighbor_distances = []

    for _name, (dx, dy) in ACTIONS:
        next_state = (x + dx, y + dy)
        is_valid = maze.is_open(next_state) and next_state in distance_map
        valid_moves.append(1.0 if is_valid else 0.0)
        wall_bits.append(0.0 if is_valid else 1.0)

        if is_valid:
            next_distance = distance_map[next_state]
            distance_delta = current_distance - next_distance
            progress.append(distance_delta / distance_scale)
            raw_progress.append(float(max(-1, min(1, distance_delta))))
            neighbor_distances.append(next_distance / distance_scale)
        else:
            progress.append(-1.0)
            raw_progress.append(-1.0)
            neighbor_distances.append(1.0)

    manhattan = abs(ex - x) + abs(ey - y)
    euclidean = math.sqrt((ex - x) ** 2 + (ey - y) ** 2)
    max_manhattan = max(1, maze.width + maze.height - 2)
    max_euclidean = max(1.0, math.sqrt(width_scale**2 + height_scale**2))
    degree = sum(valid_moves) / len(ACTIONS)

    features = [
        x / width_scale,
        y / height_scale,
        ex / width_scale,
        ey / height_scale,
        (ex - x) / width_scale,
        (ey - y) / height_scale,
        abs(ex - x) / width_scale,
        abs(ey - y) / height_scale,
        maze.width / global_width_scale,
        maze.height / global_height_scale,
        current_distance / distance_scale,
        1.0 - current_distance / distance_scale,
        manhattan / max_manhattan,
        euclidean / max_euclidean,
        *valid_moves,
        *wall_bits,
        *progress,
        *raw_progress,
        *neighbor_distances,
        degree,
        1.0 if state == maze.start else 0.0,
        1.0 if state == maze.exit else 0.0,
    ]
    return torch.tensor(features, dtype=torch.float32)


def optimal_value_from_distance(distance: int, config: TrainConfig) -> float:
    if distance <= 0:
        return 0.0

    shaped_step_reward = config.step_penalty + config.progress_reward
    if abs(config.gamma - 1.0) < 1e-8:
        return shaped_step_reward * distance + config.finish_reward

    discounted_steps = shaped_step_reward * (1.0 - config.gamma**distance) / (
        1.0 - config.gamma
    )
    discounted_finish = (config.gamma ** (distance - 1)) * config.finish_reward
    return discounted_steps + discounted_finish


def build_task(
    index: int,
    source: MazeSource,
    root: Path,
    max_width: int,
    max_height: int,
    config: TrainConfig,
) -> MazeTask:
    maze = source.maze
    distance_map = build_distance_map(maze)
    if maze.start not in distance_map:
        raise ValueError(f"{source.name} has no route from start to exit.")

    states = sorted(distance_map, key=lambda cell: (cell[1], cell[0]))
    state_to_row = {state: row for row, state in enumerate(states)}
    max_distance = max(distance_map.values(), default=1)
    shortest_length = distance_map[maze.start]

    obs_rows = []
    mask_rows = []
    expert_rows = []
    value_rows = []

    for state in states:
        x, y = state
        current_distance = distance_map[state]
        obs_rows.append(
            encode_state(maze, state, distance_map, max_distance, max_width, max_height)
        )

        mask = []
        expert_mask = []
        for _name, (dx, dy) in ACTIONS:
            next_state = (x + dx, y + dy)
            is_valid = maze.is_open(next_state) and next_state in distance_map
            mask.append(is_valid)
            expert_mask.append(
                is_valid and distance_map[next_state] == current_distance - 1
            )

        mask_rows.append(mask)
        expert_rows.append(expert_mask)
        value_rows.append(optimal_value_from_distance(current_distance, config))

    max_steps = max(32, min(maze.width * maze.height * 4, shortest_length * 4 + 20))

    return MazeTask(
        index=index,
        name=source.name,
        path=source.path.relative_to(root),
        maze=maze,
        distance_map=distance_map,
        states=states,
        state_to_row=state_to_row,
        obs=torch.stack(obs_rows),
        masks=torch.tensor(mask_rows, dtype=torch.bool),
        expert_masks=torch.tensor(expert_rows, dtype=torch.bool),
        value_targets=torch.tensor(value_rows, dtype=torch.float32),
        max_distance=max_distance,
        shortest_length=shortest_length,
        start_row=state_to_row[maze.start],
        max_steps=max_steps,
    )


def build_tasks(
    sources: list[MazeSource],
    root: Path,
    config: TrainConfig,
) -> list[MazeTask]:
    max_width = max(source.maze.width for source in sources)
    max_height = max(source.maze.height for source in sources)
    tasks = []

    for index, source in enumerate(sources):
        tasks.append(build_task(index, source, root, max_width, max_height, config))

    return tasks


def move_tasks_to_device(tasks: list[MazeTask], device: torch.device) -> None:
    for task in tasks:
        task.obs = task.obs.to(device)
        task.masks = task.masks.to(device)
        task.expert_masks = task.expert_masks.to(device)
        task.value_targets = task.value_targets.to(device)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def build_bc_dataset(
    tasks: list[MazeTask],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_rows = []
    task_rows = []
    mask_rows = []
    expert_rows = []
    value_rows = []

    for task in tasks:
        valid_rows = task.expert_masks.any(dim=1)
        count = int(valid_rows.sum().item())
        if count == 0:
            continue

        obs_rows.append(task.obs[valid_rows])
        task_rows.append(torch.full((count,), task.index, dtype=torch.long, device=device))
        mask_rows.append(task.masks[valid_rows])
        expert_rows.append(task.expert_masks[valid_rows])
        value_rows.append(task.value_targets[valid_rows])

    return (
        torch.cat(obs_rows, dim=0),
        torch.cat(task_rows, dim=0),
        torch.cat(mask_rows, dim=0),
        torch.cat(expert_rows, dim=0),
        torch.cat(value_rows, dim=0),
    )


def behavior_clone(
    model: ActorCritic,
    tasks: list[MazeTask],
    config: TrainConfig,
    device: torch.device,
) -> None:
    if config.bc_epochs <= 0:
        return

    obs, task_ids, masks, expert_masks, value_targets = build_bc_dataset(tasks, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.bc_learning_rate,
        weight_decay=1e-4,
        eps=1e-5,
    )
    sample_count = obs.shape[0]

    for epoch in range(1, config.bc_epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        total_loss = 0.0

        for start in range(0, sample_count, config.bc_batch_size):
            batch_indices = permutation[start : start + config.bc_batch_size]
            batch_obs = obs[batch_indices]
            batch_task_ids = task_ids[batch_indices]
            batch_masks = masks[batch_indices]
            batch_experts = expert_masks[batch_indices]
            batch_value_targets = value_targets[batch_indices]

            logits, values = policy_logits_and_value(
                model,
                batch_obs,
                batch_task_ids,
                batch_masks,
                config,
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            expert_log_prob = torch.logsumexp(
                log_probs.masked_fill(~batch_experts, torch.finfo(log_probs.dtype).min),
                dim=-1,
            )
            policy_loss = -expert_log_prob.mean()
            value_loss = (values - batch_value_targets).pow(2).mean()
            loss = policy_loss + 0.2 * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item() * len(batch_indices)

        if epoch == 1 or epoch == config.bc_epochs or epoch % max(1, config.bc_epochs // 5) == 0:
            print(f"bc_epoch={epoch:04d} loss={total_loss / sample_count:.4f}")


def transition(
    task: MazeTask,
    state: tuple[int, int],
    action_index: int,
    visits: dict[tuple[int, int], int],
    config: TrainConfig,
) -> tuple[tuple[int, int], float, bool]:
    maze = task.maze
    _name, (dx, dy) = ACTIONS[action_index]
    x, y = state
    next_state = (x + dx, y + dy)

    if not maze.is_open(next_state) or next_state not in task.distance_map:
        return state, -1.0, False

    old_distance = task.distance_map[state]
    new_distance = task.distance_map[next_state]
    reward = config.step_penalty + config.progress_reward * (old_distance - new_distance)

    if next_state in visits:
        reward += config.repeat_penalty * min(10, visits[next_state])

    done = next_state == maze.exit
    if done:
        reward += config.finish_reward

    return next_state, reward, done


def collect_rollout(
    model: ActorCritic,
    task: MazeTask,
    config: TrainConfig,
    device: torch.device,
) -> Rollout:
    obs_rows: list[torch.Tensor] = []
    task_ids: list[int] = []
    mask_rows: list[torch.Tensor] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    episode_rewards: list[float] = []
    episode_wins: list[bool] = []
    episode_lengths: list[int] = []

    state = task.maze.start
    visits = {state: 1}
    episode_reward = 0.0
    episode_length = 0
    last_done = False

    task_id_tensor = torch.tensor([task.index], dtype=torch.long, device=device)

    for _ in range(config.rollout_steps):
        row = task.state_to_row[state]
        obs = task.obs[row]
        mask = task.masks[row]

        with torch.no_grad():
            logits, value = policy_logits_and_value(
                model,
                obs.unsqueeze(0),
                task_id_tensor,
                mask.unsqueeze(0),
                config,
            )
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze(0)

        action_index = int(action.item())
        next_state, reward, done = transition(task, state, action_index, visits, config)
        episode_length += 1

        timed_out = episode_length >= task.max_steps
        if timed_out and not done:
            reward += config.timeout_penalty

        obs_rows.append(obs)
        task_ids.append(task.index)
        mask_rows.append(mask)
        actions.append(action_index)
        rewards.append(reward)
        dones.append(done or timed_out)
        log_probs.append(log_prob.detach())
        values.append(value.squeeze(0).detach())

        state = next_state
        episode_reward += reward
        visits[state] = visits.get(state, 0) + 1
        last_done = done or timed_out

        if done or timed_out:
            episode_rewards.append(episode_reward)
            episode_wins.append(done)
            episode_lengths.append(episode_length)
            state = task.maze.start
            visits = {state: 1}
            episode_reward = 0.0
            episode_length = 0

    if last_done:
        last_value = torch.tensor(0.0, device=device)
    else:
        row = task.state_to_row[state]
        with torch.no_grad():
            _logits, bootstrap_value = model(task.obs[row].unsqueeze(0), task_id_tensor)
        last_value = bootstrap_value.squeeze(0).detach()

    return Rollout(
        obs=obs_rows,
        task_ids=task_ids,
        masks=mask_rows,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
        last_value=last_value,
        episode_rewards=episode_rewards,
        episode_wins=episode_wins,
        episode_lengths=episode_lengths,
    )


def compute_advantages(
    rollout: Rollout,
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.tensor(rollout.rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(rollout.dones, dtype=torch.float32, device=device)
    values = torch.stack(rollout.values).to(device)
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=device)

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_value = rollout.last_value
        else:
            next_value = values[step + 1]

        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + config.gamma * next_value * next_non_terminal - values[step]
        gae = delta + config.gamma * config.gae_lambda * next_non_terminal * gae
        advantages[step] = gae

    returns = advantages + values
    return advantages, returns


def build_ppo_batch(
    rollouts: list[Rollout],
    config: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    obs = torch.cat([torch.stack(rollout.obs) for rollout in rollouts], dim=0)
    masks = torch.cat([torch.stack(rollout.masks) for rollout in rollouts], dim=0)
    task_ids = torch.tensor(
        [task_id for rollout in rollouts for task_id in rollout.task_ids],
        dtype=torch.long,
        device=device,
    )
    actions = torch.tensor(
        [action for rollout in rollouts for action in rollout.actions],
        dtype=torch.long,
        device=device,
    )
    old_log_probs = torch.cat([torch.stack(rollout.log_probs) for rollout in rollouts], dim=0)
    old_values = torch.cat([torch.stack(rollout.values) for rollout in rollouts], dim=0)

    advantages_list = []
    returns_list = []
    for rollout in rollouts:
        advantages, returns = compute_advantages(rollout, config, device)
        advantages_list.append(advantages)
        returns_list.append(returns)

    advantages = torch.cat(advantages_list, dim=0)
    returns = torch.cat(returns_list, dim=0)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    return obs, task_ids, masks, actions, old_log_probs, old_values, advantages, returns


def entropy_coef_for_update(update: int, config: TrainConfig) -> float:
    if config.updates <= 1:
        return config.entropy_end
    progress = min(1.0, (update - 1) / (config.updates - 1))
    return config.entropy_start + progress * (config.entropy_end - config.entropy_start)


def update_ppo(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    rollouts: list[Rollout],
    config: TrainConfig,
    update: int,
    device: torch.device,
) -> dict[str, float]:
    (
        obs,
        task_ids,
        masks,
        actions,
        old_log_probs,
        old_values,
        advantages,
        returns,
    ) = build_ppo_batch(rollouts, config, device)

    total_steps = obs.shape[0]
    entropy_coef = entropy_coef_for_update(update, config)
    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_frac": 0.0,
        "batches": 0.0,
    }

    for _epoch in range(config.update_epochs):
        permutation = torch.randperm(total_steps, device=device)

        for start in range(0, total_steps, config.minibatch_size):
            batch_indices = permutation[start : start + config.minibatch_size]
            logits, values = policy_logits_and_value(
                model,
                obs[batch_indices],
                task_ids[batch_indices],
                masks[batch_indices],
                config,
            )
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(actions[batch_indices])
            entropy = dist.entropy().mean()
            log_ratio = new_log_probs - old_log_probs[batch_indices]
            ratio = log_ratio.exp()

            batch_advantages = advantages[batch_indices]
            unclipped_policy = ratio * batch_advantages
            clipped_policy = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * batch_advantages
            policy_loss = -torch.min(unclipped_policy, clipped_policy).mean()

            batch_returns = returns[batch_indices]
            value_unclipped = (values - batch_returns).pow(2)
            value_clipped = old_values[batch_indices] + torch.clamp(
                values - old_values[batch_indices],
                -config.clip_epsilon,
                config.clip_epsilon,
            )
            value_loss = 0.5 * torch.max(
                value_unclipped,
                (value_clipped - batch_returns).pow(2),
            ).mean()

            loss = policy_loss + config.value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().clamp_min(0.0)
                clip_frac = (
                    (ratio - 1.0).abs() > config.clip_epsilon
                ).float().mean()

            stats["policy_loss"] += float(policy_loss.item())
            stats["value_loss"] += float(value_loss.item())
            stats["entropy"] += float(entropy.item())
            stats["kl"] += float(approx_kl.item())
            stats["clip_frac"] += float(clip_frac.item())
            stats["batches"] += 1.0

        if stats["kl"] / max(1.0, stats["batches"]) > config.target_kl:
            break

    batches = max(1.0, stats.pop("batches"))
    return {key: value / batches for key, value in stats.items()}


def choose_greedy_action(
    model: ActorCritic,
    task: MazeTask,
    state: tuple[int, int],
    device: torch.device,
    config: TrainConfig,
) -> int:
    row = task.state_to_row[state]
    task_ids = torch.tensor([task.index], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _value = policy_logits_and_value(
            model,
            task.obs[row].unsqueeze(0),
            task_ids,
            task.masks[row].unsqueeze(0),
            config,
        )

    return int(torch.argmax(logits, dim=-1).item())


def extract_path(
    model: ActorCritic,
    task: MazeTask,
    device: torch.device,
    config: TrainConfig,
) -> tuple[list[tuple[int, int]], bool]:
    state = task.maze.start
    path = [state]
    seen = {state}
    looped = False

    for _ in range(task.max_steps):
        action = choose_greedy_action(model, task, state, device, config)
        next_state, _reward, done = transition(task, state, action, {}, config)
        if next_state == state:
            break
        path.append(next_state)
        state = next_state

        if done:
            break

        if state in seen:
            looped = True
            break
        seen.add(state)

    return path, looped


def evaluate(
    model: ActorCritic,
    tasks: list[MazeTask],
    device: torch.device,
    config: TrainConfig,
) -> list[EvalResult]:
    model.eval()
    results = []
    for task in tasks:
        path, looped = extract_path(model, task, device, config)
        results.append(
            EvalResult(
                task_name=task.name,
                path=path,
                reached_exit=path[-1] == task.maze.exit,
                looped=looped,
                shortest_length=task.shortest_length,
            )
        )
    model.train()
    return results


def eval_score(results: list[EvalResult]) -> tuple[int, int, int]:
    reached = sum(result.reached_exit for result in results)
    optimal = sum(result.optimal for result in results)
    penalty = 0

    for result in results:
        if result.reached_exit:
            penalty += max(0, result.path_length - result.shortest_length)
        else:
            penalty += result.shortest_length * 10 + result.path_length
            if result.looped:
                penalty += 1000

    return reached, optimal, -penalty


def print_eval_summary(prefix: str, results: list[EvalResult]) -> None:
    reached, optimal, score_tail = eval_score(results)
    avg_len = sum(result.path_length for result in results) / max(1, len(results))
    print(
        f"{prefix} solved={reached}/{len(results)} "
        f"optimal={optimal}/{len(results)} "
        f"avg_len={avg_len:.1f} "
        f"score_tail={score_tail}"
    )
    for result in results:
        status = "optimal" if result.optimal else "solved" if result.reached_exit else "failed"
        if result.looped and not result.reached_exit:
            status = "looped"
        print(
            f"  {result.task_name}: {status} "
            f"len={result.path_length} shortest={result.shortest_length}"
        )


def render_path(task: MazeTask, path: list[tuple[int, int]]) -> str:
    path_cells = set(path)
    rows = []

    for y in range(task.maze.height):
        row = ""
        for x in range(task.maze.width):
            cell = (x, y)
            if cell == task.maze.start:
                row += "S "
            elif cell == task.maze.exit:
                row += "E "
            elif cell in task.maze.walls:
                row += "##"
            elif cell in path_cells:
                row += ".."
            else:
                row += "  "
        rows.append(row.rstrip())

    return "\n".join(rows)


def train(
    tasks: list[MazeTask],
    config: TrainConfig,
    device: torch.device,
) -> tuple[ActorCritic, dict[str, torch.Tensor], list[EvalResult]]:
    input_size = tasks[0].obs.shape[1]
    model = ActorCritic(
        input_size=input_size,
        action_size=len(ACTIONS),
        task_count=len(tasks),
        hidden_size=config.hidden_size,
        task_embedding_size=config.task_embedding_size,
    ).to(device)

    print(f"device={device}")
    print(f"input_features={input_size} tasks={len(tasks)}")

    behavior_clone(model, tasks, config, device)
    best_results = evaluate(model, tasks, device, config)
    best_score = eval_score(best_results)
    best_state = clone_model_state(model)
    print_eval_summary("after_bc", best_results)

    if config.updates <= 0:
        return model, best_state, best_results

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4,
        eps=1e-5,
    )
    solved_streak = 0

    for update in range(1, config.updates + 1):
        rollouts = [collect_rollout(model, task, config, device) for task in tasks]
        stats = update_ppo(model, optimizer, rollouts, config, update, device)

        rollout_wins = [
            win for rollout in rollouts for win in rollout.episode_wins
        ]
        rollout_lengths = [
            length for rollout in rollouts for length in rollout.episode_lengths
        ]
        rollout_success = sum(rollout_wins) / max(1, len(rollout_wins))
        rollout_len = sum(rollout_lengths) / max(1, len(rollout_lengths))

        should_eval = (
            update == 1
            or update == config.updates
            or update % max(1, config.eval_every) == 0
        )

        if should_eval:
            results = evaluate(model, tasks, device, config)
            current_score = eval_score(results)
            if current_score > best_score:
                best_score = current_score
                best_state = clone_model_state(model)
                best_results = results

            reached, _optimal, _tail = current_score
            solved_streak = solved_streak + 1 if reached == len(tasks) else 0

            print(
                f"update={update:04d} "
                f"rollout_success={rollout_success:.0%} "
                f"rollout_avg_len={rollout_len:.1f} "
                f"policy_loss={stats['policy_loss']:.3f} "
                f"value_loss={stats['value_loss']:.3f} "
                f"entropy={stats['entropy']:.3f} "
                f"kl={stats['kl']:.4f}"
            )
            print_eval_summary("eval", results)

            if (
                config.early_stop_patience > 0
                and solved_streak >= config.early_stop_patience
            ):
                print(
                    f"early_stop=all_mazes_solved "
                    f"patience={config.early_stop_patience} update={update}"
                )
                break
        elif update % max(1, config.eval_every // 2) == 0:
            print(
                f"update={update:04d} "
                f"rollout_success={rollout_success:.0%} "
                f"rollout_avg_len={rollout_len:.1f} "
                f"policy_loss={stats['policy_loss']:.3f} "
                f"value_loss={stats['value_loss']:.3f} "
                f"entropy={stats['entropy']:.3f}"
            )

    model.load_state_dict(best_state)
    model.to(device)
    final_results = evaluate(model, tasks, device, config)
    print_eval_summary("best_loaded", final_results)
    return model, best_state, final_results


def save_checkpoint(
    output: Path,
    model: ActorCritic,
    tasks: list[MazeTask],
    config: TrainConfig,
    results: list[EvalResult],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": clone_model_state(model),
        "config": asdict(config),
        "actions": ACTIONS,
        "maze_files": [str(task.path) for task in tasks],
        "task_names": [task.name for task in tasks],
        "input_size": tasks[0].obs.shape[1],
        "task_count": len(tasks),
        "eval": [
            {
                "task_name": result.task_name,
                "reached_exit": result.reached_exit,
                "looped": result.looped,
                "path_length": result.path_length,
                "shortest_length": result.shortest_length,
                "optimal": result.optimal,
            }
            for result in results
        ],
    }
    torch.save(checkpoint, output)
    print(f"saved={output}")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover every maze_*.py file and train a strong masked PPO policy."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("weights/ultimate_multi_ppo_all_mazes.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--bc-epochs", type=int, default=TrainConfig.bc_epochs)
    parser.add_argument("--updates", type=int, default=TrainConfig.updates)
    parser.add_argument("--rollout-steps", type=int, default=TrainConfig.rollout_steps)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--eval-every", type=int, default=TrainConfig.eval_every)
    parser.add_argument("--early-stop-patience", type=int, default=TrainConfig.early_stop_patience)
    parser.add_argument("--distance-prior-scale", type=float, default=TrainConfig.distance_prior_scale)
    parser.add_argument("--print-paths", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny syntax/logic check instead of a real training run.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(
        seed=args.seed,
        bc_epochs=args.bc_epochs,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        hidden_size=args.hidden_size,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        distance_prior_scale=args.distance_prior_scale,
    )

    if args.smoke:
        config.bc_epochs = 3
        config.updates = 1
        config.rollout_steps = 64
        config.update_epochs = 1
        config.minibatch_size = 256
        config.eval_every = 1
        config.early_stop_patience = 0

    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    root = args.root.resolve()
    device = pick_device(args.device)
    set_reproducible(config.seed)

    sources = discover_mazes(root)
    if not sources:
        raise SystemExit(f"No maze_*.py files with the expected Maze API found under {root}")

    print(f"discovered_mazes={len(sources)}")
    for source in sources:
        maze = source.maze
        print(
            f"  {source.name}: size={maze.width}x{maze.height} "
            f"start={maze.start} exit={maze.exit}"
        )

    tasks = build_tasks(sources, root, config)
    move_tasks_to_device(tasks, device)

    for task in tasks:
        print(
            f"task={task.name} open_states={len(task.states)} "
            f"shortest={task.shortest_length} max_steps={task.max_steps}"
        )

    model, _best_state, results = train(tasks, config, device)

    if args.print_paths:
        for task, result in zip(tasks, results):
            print()
            print(
                f"path {task.name} reached={result.reached_exit} "
                f"len={result.path_length} shortest={result.shortest_length}"
            )
            print(render_path(task, result.path))

    if not args.no_save:
        output = args.output
        if not output.is_absolute():
            output = root / output
        save_checkpoint(output, model, tasks, config, results)


if __name__ == "__main__":
    main()
