from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "simple"))

from compact_original_feature_ppo import (  # noqa: E402
    ACTIONS,
    ActorCritic as CompactActorCritic,
    Config as CompactConfig,
    Task as CompactTask,
    collect as compact_collect,
    encode8,
    evaluate as compact_evaluate,
    ppo_update as compact_ppo_update,
    to_device,
)


SIZE_SPECS = {
    "small": {"width": 16, "height": 16, "sweep_rows": [1, 2, 2, 1], "branch_depth": 4},
    "medium": {"width": 25, "height": 25, "sweep_rows": [2, 3, 4, 4], "branch_depth": 6},
    "large": {"width": 35, "height": 35, "sweep_rows": [4, 5, 6, 7], "branch_depth": 8},
}


@dataclass
class GeneratedMaze:
    name: str
    size_label: str
    width: int
    height: int
    start: tuple[int, int]
    exit: tuple[int, int]
    walls: set[tuple[int, int]]
    solution_path: list[tuple[int, int]]
    seed: int

    def is_open(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.walls

    @property
    def shortest(self) -> int:
        return len(self.solution_path) - 1

    @property
    def open_count(self) -> int:
        return self.width * self.height - len(self.walls)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size_label": self.size_label,
            "width": self.width,
            "height": self.height,
            "start": list(self.start),
            "exit": list(self.exit),
            "walls": [list(cell) for cell in sorted(self.walls)],
            "solution_path": [list(cell) for cell in self.solution_path],
            "shortest": self.shortest,
            "open_count": self.open_count,
            "seed": self.seed,
        }


def neighbors(cell: tuple[int, int], width: int, height: int) -> list[tuple[int, int]]:
    x, y = cell
    out = []
    for _name, (dx, dy) in ACTIONS:
        nxt = (x + dx, y + dy)
        if 0 <= nxt[0] < width and 0 <= nxt[1] < height:
            out.append(nxt)
    return out


def build_random_tree(width: int, height: int, rng: random.Random) -> dict[tuple[int, int], list[tuple[int, int]]]:
    start = (0, 0)
    visited = {start}
    stack = [start]
    tree = {(x, y): [] for y in range(height) for x in range(width)}

    while stack:
        current = stack[-1]
        candidates = [cell for cell in neighbors(current, width, height) if cell not in visited]
        if not candidates:
            stack.pop()
            continue
        nxt = rng.choice(candidates)
        visited.add(nxt)
        tree[current].append(nxt)
        tree[nxt].append(current)
        stack.append(nxt)

    return tree


def path_in_tree(
    tree: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for nxt in tree[current]:
            if nxt not in parent:
                parent[nxt] = current
                queue.append(nxt)

    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])  # type: ignore[arg-type]
    path.reverse()
    return path


def collect_branch(
    tree: dict[tuple[int, int], list[tuple[int, int]]],
    root: tuple[int, int],
    blocked: set[tuple[int, int]],
    depth: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    opened: set[tuple[int, int]] = set()
    stack = [(root, 0)]
    seen = {root}
    while stack:
        current, current_depth = stack.pop()
        if current_depth >= depth:
            continue
        options = [cell for cell in tree[current] if cell not in seen and cell not in blocked]
        rng.shuffle(options)
        for nxt in options[: rng.randint(0, min(2, len(options)))]:
            seen.add(nxt)
            opened.add(nxt)
            stack.append((nxt, current_depth + 1))
    return opened


def generate_maze(size_label: str, maze_index: int, seed: int) -> GeneratedMaze:
    spec = SIZE_SPECS[size_label]
    width, height = int(spec["width"]), int(spec["height"])
    start, exit_cell = (0, 0), (width - 1, height - 1)
    rng = random.Random(seed)

    sweep_options = list(spec["sweep_rows"])
    sweep_rows = int(sweep_options[maze_index % len(sweep_options)])
    best_path = snake_solution_path(width, height, sweep_rows)
    solution_cells = set(best_path)
    open_cells = set(best_path)
    for cell in best_path:
        if rng.random() < 0.46:
            open_cells.update(grow_dead_end_branch(width, height, cell, open_cells, solution_cells, int(spec["branch_depth"]), rng))

    walls = {(x, y) for y in range(height) for x in range(width)} - open_cells
    return GeneratedMaze(
        name=f"{size_label}_{maze_index:02d}",
        size_label=size_label,
        width=width,
        height=height,
        start=start,
        exit=exit_cell,
        walls=walls,
        solution_path=best_path,
        seed=seed,
    )


def snake_solution_path(width: int, height: int, sweep_rows: int) -> list[tuple[int, int]]:
    path = [(0, 0)]
    x, y = 0, 0
    direction = 1
    for row in range(sweep_rows):
        row_y = min(height - 1, row * 2)
        while y < row_y:
            y += 1
            path.append((x, y))
        target_x = width - 1 if direction == 1 else 0
        while x != target_x:
            x += direction
            path.append((x, y))
        if row != sweep_rows - 1:
            direction *= -1

    while y < height - 1:
        y += 1
        path.append((x, y))
    while x < width - 1:
        x += 1
        path.append((x, y))
    while x > width - 1:
        x -= 1
        path.append((x, y))
    return path


def grow_dead_end_branch(
    width: int,
    height: int,
    root: tuple[int, int],
    open_cells: set[tuple[int, int]],
    solution_cells: set[tuple[int, int]],
    max_depth: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    branch: set[tuple[int, int]] = set()
    current = root
    for _ in range(rng.randint(1, max_depth)):
        candidates = []
        for nxt in neighbors(current, width, height):
            if nxt in open_cells or nxt in solution_cells or nxt in branch:
                continue
            adjacent_open = sum(1 for cell in neighbors(nxt, width, height) if cell in open_cells or cell in branch)
            if adjacent_open <= 1:
                candidates.append(nxt)
        if not candidates:
            break
        current = rng.choice(candidates)
        branch.add(current)
    return branch


def distance_map(maze: GeneratedMaze) -> dict[tuple[int, int], int]:
    queue = deque([maze.exit])
    dist = {maze.exit: 0}
    while queue:
        cell = queue.popleft()
        for _name, (dx, dy) in ACTIONS:
            nxt = (cell[0] + dx, cell[1] + dy)
            if maze.is_open(nxt) and nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def max_steps_for(maze: GeneratedMaze, shortest: int) -> int:
    return max(32, min(maze.width * maze.height * 4, shortest * 4 + 20))


def make_compact_task(maze: GeneratedMaze, cfg: CompactConfig) -> CompactTask:
    dist = distance_map(maze)
    states = sorted(dist, key=lambda cell: (cell[1], cell[0]))
    rows = {state: index for index, state in enumerate(states)}
    obs, masks, experts, values = [], [], [], []
    for state in states:
        obs.append(encode8(maze, state))
        mask, expert = [], []
        for _name, (dx, dy) in ACTIONS:
            nxt = (state[0] + dx, state[1] + dy)
            ok = maze.is_open(nxt) and nxt in dist
            mask.append(ok)
            expert.append(ok and dist[nxt] == dist[state] - 1)
        masks.append(mask)
        experts.append(expert)
        distance = dist[state]
        if distance <= 0:
            values.append(0.0)
        else:
            r = cfg.step_penalty + cfg.progress_reward
            values.append(r * (1 - cfg.gamma**distance) / (1 - cfg.gamma) + cfg.gamma ** (distance - 1) * cfg.finish_reward)

    return CompactTask(
        i=0,
        name=maze.name,
        path=Path(f"generated/{maze.name}"),
        maze=maze,
        dist=dist,
        states=states,
        row=rows,
        obs=torch.stack(obs),
        masks=torch.tensor(masks, dtype=torch.bool),
        experts=torch.tensor(experts, dtype=torch.bool),
        values=torch.tensor(values, dtype=torch.float32),
        shortest=dist[maze.start],
        max_steps=max_steps_for(maze, dist[maze.start]),
    )


def compact_masked_logits(model: nn.Module, obs: torch.Tensor, tids: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, values = model(obs, tids)
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min), values


def compact_bc_epoch(model: nn.Module, task: CompactTask, cfg: CompactConfig, optimizer: optim.Optimizer, device: torch.device) -> float:
    keep = task.experts.any(dim=1)
    obs = task.obs[keep]
    masks = task.masks[keep]
    experts = task.experts[keep]
    vals = task.values[keep]
    tids = torch.zeros((len(obs),), dtype=torch.long, device=device)
    permutation = torch.randperm(len(obs), device=device)
    total = 0.0
    for start in range(0, len(obs), cfg.bc_batch):
        idx = permutation[start : start + cfg.bc_batch]
        logits, values = compact_masked_logits(model, obs[idx], tids[idx], masks[idx])
        log_probs = torch.log_softmax(logits, dim=-1)
        expert_log_prob = torch.logsumexp(
            log_probs.masked_fill(~experts[idx], torch.finfo(log_probs.dtype).min),
            dim=-1,
        )
        loss = -expert_log_prob.mean() + 0.2 * (values - vals[idx]).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        total += loss.item() * len(idx)
    return total / max(1, len(obs))


def compact_eval_row(model: nn.Module, task: CompactTask, device: torch.device) -> dict[str, Any]:
    row = compact_evaluate(model, [task], device)[0]
    return {
        "solved": bool(row["reached"]),
        "optimal": bool(row["reached"] and row["len"] == row["shortest"]),
        "path_len": int(row["len"]),
        "shortest": int(row["shortest"]),
        "looped": bool(row["looped"]),
    }


class LegacyActorCritic(nn.Module):
    def __init__(self, input_size: int = 8, action_size: int = 4) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


@dataclass
class LegacyConfig:
    updates: int = 50
    rollout_steps: int = 1024
    update_epochs: int = 6
    minibatch_size: int = 128
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy: float = 0.2
    value_coef: float = 0.5
    wall_penalty: float = -5.0
    finish_reward: float = 500.0
    closer_reward: float = -0.5
    further_reward: float = -1.5
    neutral_reward: float = -1.0
    timeout_penalty: float = -50.0


def legacy_encode(maze: GeneratedMaze, state: tuple[int, int]) -> torch.Tensor:
    x, y = state
    ex, ey = maze.exit
    return torch.tensor(
        [
            x / max(1, maze.width - 1),
            y / max(1, maze.height - 1),
            (ex - x) / max(1, maze.width - 1),
            (ey - y) / max(1, maze.height - 1),
            0.0 if maze.is_open((x, y - 1)) else 1.0,
            0.0 if maze.is_open((x, y + 1)) else 1.0,
            0.0 if maze.is_open((x - 1, y)) else 1.0,
            0.0 if maze.is_open((x + 1, y)) else 1.0,
        ],
        dtype=torch.float32,
    )


def manhattan(maze: GeneratedMaze, state: tuple[int, int]) -> int:
    return abs(maze.exit[0] - state[0]) + abs(maze.exit[1] - state[1])


def legacy_step(
    maze: GeneratedMaze,
    state: tuple[int, int],
    action: int,
    cfg: LegacyConfig,
) -> tuple[tuple[int, int], float, bool]:
    _name, (dx, dy) = ACTIONS[action]
    nxt = (state[0] + dx, state[1] + dy)
    if not maze.is_open(nxt):
        return state, cfg.wall_penalty, False
    if nxt == maze.exit:
        return nxt, cfg.finish_reward, True
    old_distance = manhattan(maze, state)
    new_distance = manhattan(maze, nxt)
    if new_distance < old_distance:
        return nxt, cfg.closer_reward, False
    if new_distance > old_distance:
        return nxt, cfg.further_reward, False
    return nxt, cfg.neutral_reward, False


def legacy_collect(
    model: LegacyActorCritic,
    maze: GeneratedMaze,
    shortest: int,
    cfg: LegacyConfig,
    device: torch.device,
) -> dict[str, Any]:
    data: dict[str, Any] = {key: [] for key in ["obs", "actions", "rewards", "dones", "logps", "values", "wins", "lens"]}
    state = maze.start
    episode_len = 0
    max_steps = max_steps_for(maze, shortest)
    while len(data["obs"]) < cfg.rollout_steps:
        obs = legacy_encode(maze, state).to(device)
        with torch.no_grad():
            logits, value = model(obs.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).squeeze(0)
        nxt, reward, done = legacy_step(maze, state, int(action), cfg)
        episode_len += 1
        timed_out = episode_len >= max_steps
        if timed_out and not done:
            reward += cfg.timeout_penalty
        data["obs"].append(obs)
        data["actions"].append(int(action))
        data["rewards"].append(float(reward))
        data["dones"].append(bool(done or timed_out))
        data["logps"].append(logp.detach())
        data["values"].append(value.squeeze(0).detach())
        state = nxt
        if done or timed_out:
            data["wins"].append(bool(done))
            data["lens"].append(episode_len)
            state = maze.start
            episode_len = 0
    return data


def legacy_advantages(rollout: dict[str, Any], cfg: LegacyConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = torch.tensor(rollout["rewards"], dtype=torch.float32, device=device)
    dones = torch.tensor(rollout["dones"], dtype=torch.float32, device=device)
    values = torch.stack(rollout["values"]).to(device)
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=device)
    next_value = torch.tensor(0.0, device=device)
    for step in reversed(range(len(rewards))):
        alive = 1.0 - dones[step]
        delta = rewards[step] + cfg.gamma * next_value * alive - values[step]
        gae = delta + cfg.gamma * cfg.gae_lambda * alive * gae
        advantages[step] = gae
        next_value = values[step]
    return advantages, advantages + values


def legacy_update(
    model: LegacyActorCritic,
    optimizer: optim.Optimizer,
    rollout: dict[str, Any],
    cfg: LegacyConfig,
    device: torch.device,
) -> dict[str, float]:
    obs = torch.stack(rollout["obs"]).to(device)
    actions = torch.tensor(rollout["actions"], dtype=torch.long, device=device)
    old_logps = torch.stack(rollout["logps"]).to(device)
    old_values = torch.stack(rollout["values"]).to(device)
    advantages, returns = legacy_advantages(rollout, cfg, device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "batches": 0.0}

    for _epoch in range(cfg.update_epochs):
        permutation = torch.randperm(len(obs), device=device)
        for start in range(0, len(obs), cfg.minibatch_size):
            idx = permutation[start : start + cfg.minibatch_size]
            logits, values = model(obs[idx])
            dist = Categorical(logits=logits)
            logps = dist.log_prob(actions[idx])
            ratio = (logps - old_logps[idx]).exp()
            policy = -torch.min(
                ratio * advantages[idx],
                torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantages[idx],
            ).mean()
            clipped_values = old_values[idx] + torch.clamp(values - old_values[idx], -cfg.clip, cfg.clip)
            value = 0.5 * torch.max((values - returns[idx]).pow(2), (clipped_values - returns[idx]).pow(2)).mean()
            entropy = dist.entropy().mean()
            loss = policy + cfg.value_coef * value - cfg.entropy * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            stats["policy"] += policy.item()
            stats["value"] += value.item()
            stats["entropy"] += entropy.item()
            stats["batches"] += 1
    return {key: value / max(1.0, stats["batches"]) for key, value in stats.items() if key != "batches"}


def legacy_eval(model: LegacyActorCritic, maze: GeneratedMaze, shortest: int, cfg: LegacyConfig, device: torch.device) -> dict[str, Any]:
    state = maze.start
    path = [state]
    seen = {state}
    looped = False
    max_steps = max_steps_for(maze, shortest)
    model.eval()
    for _step in range(max_steps):
        obs = legacy_encode(maze, state).to(device)
        with torch.no_grad():
            logits, _value = model(obs.unsqueeze(0))
        action = int(logits.argmax(dim=-1))
        nxt, _reward, done = legacy_step(maze, state, action, cfg)
        if nxt == state:
            break
        path.append(nxt)
        state = nxt
        if done:
            break
        if state in seen:
            looped = True
            break
        seen.add(state)
    model.train()
    solved = path[-1] == maze.exit
    return {
        "solved": solved,
        "optimal": solved and len(path) - 1 == shortest,
        "path_len": len(path) - 1,
        "shortest": shortest,
        "looped": looped,
    }


def base_curve_row(
    model_name: str,
    maze: GeneratedMaze,
    train_seed: int,
    phase: str,
    phase_step: int,
    env_steps: int,
    expert_epochs: int,
    elapsed: float,
    eval_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "size": maze.size_label,
        "maze": maze.name,
        "maze_seed": maze.seed,
        "train_seed": train_seed,
        "phase": phase,
        "phase_step": phase_step,
        "env_steps": env_steps,
        "expert_epochs": expert_epochs,
        "elapsed_seconds": round(elapsed, 4),
        **eval_row,
    }


def run_legacy(maze: GeneratedMaze, train_seed: int, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    torch.manual_seed(train_seed)
    random.seed(train_seed)
    cfg = LegacyConfig(updates=args.legacy_updates, rollout_steps=args.legacy_rollout_steps)
    model = LegacyActorCritic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-5)
    shortest = distance_map(maze)[maze.start]
    rows = []
    started = time.perf_counter()
    rows.append(base_curve_row("legacy_simple_ppo", maze, train_seed, "ppo", 0, 0, 0, 0.0, legacy_eval(model, maze, shortest, cfg, device)))
    for update in range(1, cfg.updates + 1):
        rollout = legacy_collect(model, maze, shortest, cfg, device)
        legacy_update(model, optimizer, rollout, cfg, device)
        rows.append(
            base_curve_row(
                "legacy_simple_ppo",
                maze,
                train_seed,
                "ppo",
                update,
                update * cfg.rollout_steps,
                0,
                time.perf_counter() - started,
                legacy_eval(model, maze, shortest, cfg, device),
            )
        )
    return rows


def run_compact(maze: GeneratedMaze, train_seed: int, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    torch.manual_seed(train_seed)
    random.seed(train_seed)
    cfg = CompactConfig(
        bc_epochs=args.compact_bc_epochs,
        updates=args.compact_updates,
        rollout_steps=args.compact_rollout_steps,
        eval_every=1,
        early_stop_patience=0,
        hidden=args.compact_hidden,
    )
    task = make_compact_task(maze, cfg)
    to_device([task], device)
    model = CompactActorCritic(input_size=task.obs.shape[1], tasks=1, cfg=cfg).to(device)
    rows = []
    started = time.perf_counter()
    rows.append(base_curve_row("compact_ppo", maze, train_seed, "bc", 0, 0, 0, 0.0, compact_eval_row(model, task, device)))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.bc_lr, weight_decay=1e-4, eps=1e-5)
    for epoch in range(1, cfg.bc_epochs + 1):
        compact_bc_epoch(model, task, cfg, optimizer, device)
        if epoch == 1 or epoch % args.compact_bc_eval_every == 0 or epoch == cfg.bc_epochs:
            rows.append(
                base_curve_row(
                    "compact_ppo",
                    maze,
                    train_seed,
                    "bc",
                    epoch,
                    0,
                    epoch,
                    time.perf_counter() - started,
                    compact_eval_row(model, task, device),
                )
            )

    ppo_optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4, eps=1e-5)
    for update in range(1, cfg.updates + 1):
        rollout = compact_collect(model, task, cfg, device)
        compact_ppo_update(model, ppo_optimizer, [rollout], cfg, update, device)
        rows.append(
            base_curve_row(
                "compact_ppo",
                maze,
                train_seed,
                "ppo",
                update,
                update * cfg.rollout_steps,
                cfg.bc_epochs,
                time.perf_counter() - started,
                compact_eval_row(model, task, device),
            )
        )
    return rows


def summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = rows[-1]
    first_solved = next((row for row in rows if row["solved"]), None)
    first_optimal = next((row for row in rows if row["optimal"]), None)
    summary = {
        "model": final["model"],
        "size": final["size"],
        "maze": final["maze"],
        "maze_seed": final["maze_seed"],
        "train_seed": final["train_seed"],
        "shortest": final["shortest"],
        "final_solved": final["solved"],
        "final_optimal": final["optimal"],
        "final_path_len": final["path_len"],
        "final_looped": final["looped"],
        "first_solved_phase": first_solved["phase"] if first_solved else "",
        "first_solved_step": first_solved["phase_step"] if first_solved else "",
        "first_solved_env_steps": first_solved["env_steps"] if first_solved else "",
        "first_solved_expert_epochs": first_solved["expert_epochs"] if first_solved else "",
        "first_solved_seconds": first_solved["elapsed_seconds"] if first_solved else "",
        "first_optimal_phase": first_optimal["phase"] if first_optimal else "",
        "first_optimal_step": first_optimal["phase_step"] if first_optimal else "",
        "first_optimal_env_steps": first_optimal["env_steps"] if first_optimal else "",
        "first_optimal_expert_epochs": first_optimal["expert_epochs"] if first_optimal else "",
        "first_optimal_seconds": first_optimal["elapsed_seconds"] if first_optimal else "",
    }
    return summary


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy simple PPO and compact PPO across generated maze sizes.")
    parser.add_argument("--mazes-per-size", type=int, default=3)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[7, 17])
    parser.add_argument("--maze-seed", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--legacy-updates", type=int, default=40)
    parser.add_argument("--legacy-rollout-steps", type=int, default=1024)
    parser.add_argument("--compact-bc-epochs", type=int, default=220)
    parser.add_argument("--compact-bc-eval-every", type=int, default=10)
    parser.add_argument("--compact-updates", type=int, default=10)
    parser.add_argument("--compact-rollout-steps", type=int, default=512)
    parser.add_argument("--compact-hidden", type=int, default=192)
    parser.add_argument("--models", nargs="+", default=["legacy_simple_ppo", "compact_ppo"])
    parser.add_argument("--out", type=Path, default=EXPERIMENT_ROOT)
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    data_dir = args.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    mazes = []
    for size_label in SIZE_SPECS:
        for maze_index in range(args.mazes_per_size):
            seed = args.maze_seed + 100 * list(SIZE_SPECS).index(size_label) + maze_index
            mazes.append(generate_maze(size_label, maze_index, seed))

    with (data_dir / "generated_mazes.json").open("w", encoding="utf-8") as file:
        json.dump([maze.to_json() for maze in mazes], file, indent=2)

    all_curves: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    total_runs = len(mazes) * len(args.train_seeds) * len(args.models)
    run_index = 0
    for maze in mazes:
        for train_seed in args.train_seeds:
            for model_name in args.models:
                run_index += 1
                print(
                    f"[{run_index}/{total_runs}] model={model_name} size={maze.size_label} "
                    f"maze={maze.name} shortest={distance_map(maze)[maze.start]} train_seed={train_seed}",
                    flush=True,
                )
                if model_name == "legacy_simple_ppo":
                    rows = run_legacy(maze, train_seed, args, device)
                elif model_name == "compact_ppo":
                    rows = run_compact(maze, train_seed, args, device)
                else:
                    raise ValueError(model_name)
                all_curves.extend(rows)
                summaries.append(summarize_run(rows))
                write_csv(data_dir / "training_curves.csv", all_curves)
                write_csv(data_dir / "summary.csv", summaries)

    print(f"wrote={data_dir / 'training_curves.csv'}")
    print(f"wrote={data_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
