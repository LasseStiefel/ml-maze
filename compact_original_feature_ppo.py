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


ACTIONS = [("up", (0, -1)), ("down", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))]


@dataclass
class Config:
    seed: int = 7
    hidden: int = 192
    task_emb: int = 16
    bc_epochs: int = 300
    bc_batch: int = 512
    bc_lr: float = 1e-3
    updates: int = 160
    rollout_steps: int = 512
    update_epochs: int = 3
    minibatch: int = 1024
    lr: float = 2.5e-4
    gamma: float = 0.985
    gae_lambda: float = 0.95
    clip: float = 0.18
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
    eval_every: int = 10
    early_stop_patience: int = 3


@dataclass
class Task:
    i: int
    name: str
    path: Path
    maze: Any
    dist: dict[tuple[int, int], int]
    states: list[tuple[int, int]]
    row: dict[tuple[int, int], int]
    obs: torch.Tensor
    masks: torch.Tensor
    experts: torch.Tensor
    values: torch.Tensor
    shortest: int
    max_steps: int


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, tasks: int, cfg: Config) -> None:
        super().__init__()
        self.task_emb = nn.Embedding(tasks, cfg.task_emb)
        self.net = nn.Sequential(
            nn.Linear(input_size + cfg.task_emb, cfg.hidden),
            nn.SiLU(),
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.LayerNorm(cfg.hidden),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
        )
        self.actor = nn.Linear(cfg.hidden, len(ACTIONS))
        self.critic = nn.Linear(cfg.hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, self.task_emb(task_ids)], dim=-1)
        h = self.net(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def masked_logits(model: ActorCritic, obs: torch.Tensor, tids: torch.Tensor, masks: torch.Tensor):
    logits, values = model(obs, tids)
    return logits.masked_fill(~masks, torch.finfo(logits.dtype).min), values


def load_module(path: Path) -> Any:
    digest = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
    name = f"compact_maze_{path.stem}_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_maze(mod: Any) -> Any:
    w, h = int(mod.DEFAULT_WIDTH), int(mod.DEFAULT_HEIGHT)
    start = tuple(mod.START)
    exit_cell = tuple(mod.resolve_exit(w, h))
    return mod.Maze(w, h, start, exit_cell, mod.build_walls(w, h, start, exit_cell))


def discover(root: Path) -> list[tuple[str, Path, Any]]:
    skip = {".git", ".venv", ".vendor", "__pycache__"}
    found = []
    for path in sorted(root.rglob("maze_*.py")):
        rel = path.relative_to(root)
        if any(part in skip or part.startswith(".") for part in rel.parts):
            continue
        try:
            mod = load_module(path)
            needed = ["DEFAULT_WIDTH", "DEFAULT_HEIGHT", "START", "resolve_exit", "build_walls", "Maze"]
            if all(hasattr(mod, attr) for attr in needed):
                found.append((str(rel), rel, build_maze(mod)))
        except Exception as exc:
            print(f"skip {rel}: {exc}")
    return found


def distance_map(maze: Any) -> dict[tuple[int, int], int]:
    q = deque([maze.exit])
    dist = {maze.exit: 0}
    while q:
        x, y = q.popleft()
        for _name, (dx, dy) in ACTIONS:
            nxt = (x + dx, y + dy)
            if maze.is_open(nxt) and nxt not in dist:
                dist[nxt] = dist[(x, y)] + 1
                q.append(nxt)
    return dist


def encode8(maze: Any, state: tuple[int, int]) -> torch.Tensor:
    x, y = state
    ex, ey = maze.exit
    ws, hs = max(1, maze.width - 1), max(1, maze.height - 1)
    return torch.tensor(
        [
            x / ws,
            y / hs,
            (ex - x) / ws,
            (ey - y) / hs,
            0.0 if maze.is_open((x, y - 1)) else 1.0,
            0.0 if maze.is_open((x, y + 1)) else 1.0,
            0.0 if maze.is_open((x - 1, y)) else 1.0,
            0.0 if maze.is_open((x + 1, y)) else 1.0,
        ],
        dtype=torch.float32,
    )


def shaped_value(distance: int, cfg: Config) -> float:
    if distance <= 0:
        return 0.0
    r = cfg.step_penalty + cfg.progress_reward
    return r * (1 - cfg.gamma**distance) / (1 - cfg.gamma) + cfg.gamma ** (distance - 1) * cfg.finish_reward


def make_tasks(root: Path, cfg: Config) -> list[Task]:
    tasks = []
    for i, (name, path, maze) in enumerate(discover(root)):
        dist = distance_map(maze)
        if maze.start not in dist:
            print(f"skip {name}: no route from start to exit")
            continue
        states = sorted(dist, key=lambda p: (p[1], p[0]))
        rows = {state: j for j, state in enumerate(states)}
        obs, masks, experts, values = [], [], [], []
        for state in states:
            x, y = state
            obs.append(encode8(maze, state))
            mask, expert = [], []
            for _name, (dx, dy) in ACTIONS:
                nxt = (x + dx, y + dy)
                ok = maze.is_open(nxt) and nxt in dist
                mask.append(ok)
                expert.append(ok and dist[nxt] == dist[state] - 1)
            masks.append(mask)
            experts.append(expert)
            values.append(shaped_value(dist[state], cfg))
        shortest = dist[maze.start]
        max_steps = max(32, min(maze.width * maze.height * 4, shortest * 4 + 20))
        tasks.append(
            Task(
                i,
                name,
                path,
                maze,
                dist,
                states,
                rows,
                torch.stack(obs),
                torch.tensor(masks, dtype=torch.bool),
                torch.tensor(experts, dtype=torch.bool),
                torch.tensor(values, dtype=torch.float32),
                shortest,
                max_steps,
            )
        )
    return tasks


def to_device(tasks: list[Task], device: torch.device) -> None:
    for task in tasks:
        task.obs = task.obs.to(device)
        task.masks = task.masks.to(device)
        task.experts = task.experts.to(device)
        task.values = task.values.to(device)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def behavior_clone(model: ActorCritic, tasks: list[Task], cfg: Config, device: torch.device) -> None:
    if cfg.bc_epochs <= 0:
        return
    obs, tids, masks, experts, vals = [], [], [], [], []
    for task in tasks:
        keep = task.experts.any(dim=1)
        n = int(keep.sum())
        obs.append(task.obs[keep])
        tids.append(torch.full((n,), task.i, dtype=torch.long, device=device))
        masks.append(task.masks[keep])
        experts.append(task.experts[keep])
        vals.append(task.values[keep])
    obs, tids, masks, experts, vals = map(lambda xs: torch.cat(xs, dim=0), [obs, tids, masks, experts, vals])
    opt = optim.AdamW(model.parameters(), lr=cfg.bc_lr, weight_decay=1e-4, eps=1e-5)
    n = len(obs)
    for epoch in range(1, cfg.bc_epochs + 1):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, cfg.bc_batch):
            idx = perm[start : start + cfg.bc_batch]
            logits, values = masked_logits(model, obs[idx], tids[idx], masks[idx])
            logp = torch.log_softmax(logits, dim=-1)
            expert_logp = torch.logsumexp(logp.masked_fill(~experts[idx], torch.finfo(logp.dtype).min), dim=-1)
            loss = -expert_logp.mean() + 0.2 * (values - vals[idx]).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            total += loss.item() * len(idx)
        if epoch == 1 or epoch == cfg.bc_epochs or epoch % max(1, cfg.bc_epochs // 5) == 0:
            print(f"bc_epoch={epoch:04d} loss={total / n:.4f}")


def env_step(task: Task, state: tuple[int, int], action: int, visits: dict[tuple[int, int], int], cfg: Config):
    x, y = state
    _name, (dx, dy) = ACTIONS[action]
    nxt = (x + dx, y + dy)
    if not task.maze.is_open(nxt) or nxt not in task.dist:
        return state, -1.0, False
    reward = cfg.step_penalty + cfg.progress_reward * (task.dist[state] - task.dist[nxt])
    if nxt in visits:
        reward += cfg.repeat_penalty * min(10, visits[nxt])
    done = nxt == task.maze.exit
    return nxt, reward + (cfg.finish_reward if done else 0.0), done


def collect(model: ActorCritic, task: Task, cfg: Config, device: torch.device) -> dict[str, Any]:
    data = {k: [] for k in ["obs", "tids", "masks", "acts", "rews", "dones", "logps", "vals", "wins", "lens"]}
    state, visits, ep_len = task.maze.start, {task.maze.start: 1}, 0
    tid = torch.tensor([task.i], dtype=torch.long, device=device)
    for _ in range(cfg.rollout_steps):
        row = task.row[state]
        obs, mask = task.obs[row], task.masks[row]
        with torch.no_grad():
            logits, value = masked_logits(model, obs.unsqueeze(0), tid, mask.unsqueeze(0))
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action).squeeze(0)
        nxt, reward, done = env_step(task, state, int(action), visits, cfg)
        ep_len += 1
        timed_out = ep_len >= task.max_steps
        if timed_out and not done:
            reward += cfg.timeout_penalty
        data["obs"].append(obs)
        data["tids"].append(task.i)
        data["masks"].append(mask)
        data["acts"].append(int(action))
        data["rews"].append(reward)
        data["dones"].append(done or timed_out)
        data["logps"].append(logp.detach())
        data["vals"].append(value.squeeze(0).detach())
        state = nxt
        visits[state] = visits.get(state, 0) + 1
        if done or timed_out:
            data["wins"].append(done)
            data["lens"].append(ep_len)
            state, visits, ep_len = task.maze.start, {task.maze.start: 1}, 0
    if data["dones"][-1]:
        boot = torch.tensor(0.0, device=device)
    else:
        with torch.no_grad():
            _logits, boot = model(task.obs[task.row[state]].unsqueeze(0), tid)
        boot = boot.squeeze(0).detach()
    data["boot"] = boot
    return data


def gae(rollout: dict[str, Any], cfg: Config, device: torch.device):
    rewards = torch.tensor(rollout["rews"], dtype=torch.float32, device=device)
    dones = torch.tensor(rollout["dones"], dtype=torch.float32, device=device)
    values = torch.stack(rollout["vals"]).to(device)
    adv = torch.zeros_like(rewards)
    last = torch.tensor(0.0, device=device)
    for t in reversed(range(len(rewards))):
        next_value = rollout["boot"] if t == len(rewards) - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_value * alive - values[t]
        last = delta + cfg.gamma * cfg.gae_lambda * alive * last
        adv[t] = last
    return adv, adv + values


def ppo_update(model: ActorCritic, opt: optim.Optimizer, rollouts: list[dict[str, Any]], cfg: Config, update: int, device: torch.device):
    obs = torch.cat([torch.stack(r["obs"]) for r in rollouts])
    masks = torch.cat([torch.stack(r["masks"]) for r in rollouts])
    tids = torch.tensor([x for r in rollouts for x in r["tids"]], dtype=torch.long, device=device)
    acts = torch.tensor([x for r in rollouts for x in r["acts"]], dtype=torch.long, device=device)
    old_logps = torch.cat([torch.stack(r["logps"]) for r in rollouts])
    old_vals = torch.cat([torch.stack(r["vals"]) for r in rollouts])
    ars = [gae(r, cfg, device) for r in rollouts]
    adv, rets = torch.cat([a for a, _r in ars]), torch.cat([r for _a, r in ars])
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    entropy_coef = cfg.entropy_start + min(1.0, (update - 1) / max(1, cfg.updates - 1)) * (cfg.entropy_end - cfg.entropy_start)
    stats = dict(policy=0.0, value=0.0, entropy=0.0, kl=0.0, batches=0)
    for _epoch in range(cfg.update_epochs):
        perm = torch.randperm(len(obs), device=device)
        for start in range(0, len(obs), cfg.minibatch):
            idx = perm[start : start + cfg.minibatch]
            logits, values = masked_logits(model, obs[idx], tids[idx], masks[idx])
            dist = Categorical(logits=logits)
            logps = dist.log_prob(acts[idx])
            log_ratio = logps - old_logps[idx]
            ratio = log_ratio.exp()
            policy = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
            clipped_values = old_vals[idx] + torch.clamp(values - old_vals[idx], -cfg.clip, cfg.clip)
            value = 0.5 * torch.max((values - rets[idx]).pow(2), (clipped_values - rets[idx]).pow(2)).mean()
            entropy = dist.entropy().mean()
            loss = policy + cfg.value_coef * value - entropy_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            kl = ((ratio - 1) - log_ratio).mean().clamp_min(0.0)
            stats["policy"] += policy.item()
            stats["value"] += value.item()
            stats["entropy"] += entropy.item()
            stats["kl"] += kl.item()
            stats["batches"] += 1
        if stats["kl"] / max(1, stats["batches"]) > cfg.target_kl:
            break
    return {k: v / max(1, stats["batches"]) for k, v in stats.items() if k != "batches"}


def greedy_path(model: ActorCritic, task: Task, device: torch.device):
    state, path, seen = task.maze.start, [task.maze.start], {task.maze.start}
    tid = torch.tensor([task.i], dtype=torch.long, device=device)
    for _ in range(task.max_steps):
        row = task.row[state]
        with torch.no_grad():
            logits, _value = masked_logits(model, task.obs[row].unsqueeze(0), tid, task.masks[row].unsqueeze(0))
        action = int(logits.argmax(dim=-1))
        nxt, _reward, done = env_step(task, state, action, {}, Config())
        if nxt == state:
            break
        path.append(nxt)
        state = nxt
        if done:
            return path, False
        if state in seen:
            return path, True
        seen.add(state)
    return path, False


def evaluate(model: ActorCritic, tasks: list[Task], device: torch.device):
    model.eval()
    rows = []
    for task in tasks:
        path, looped = greedy_path(model, task, device)
        rows.append({"name": task.name, "path": path, "len": len(path) - 1, "shortest": task.shortest, "reached": path[-1] == task.maze.exit, "looped": looped})
    model.train()
    return rows


def score(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    solved = sum(r["reached"] for r in rows)
    optimal = sum(r["reached"] and r["len"] == r["shortest"] for r in rows)
    penalty = sum(max(0, r["len"] - r["shortest"]) if r["reached"] else r["shortest"] * 10 + r["len"] + 1000 * r["looped"] for r in rows)
    return solved, optimal, -penalty


def print_eval(label: str, rows: list[dict[str, Any]]) -> None:
    solved, optimal, tail = score(rows)
    print(f"{label} solved={solved}/{len(rows)} optimal={optimal}/{len(rows)} score_tail={tail}")
    for r in rows:
        status = "optimal" if r["reached"] and r["len"] == r["shortest"] else "solved" if r["reached"] else "looped" if r["looped"] else "failed"
        print(f"  {r['name']}: {status} len={r['len']} shortest={r['shortest']}")


def render(task: Task, path: list[tuple[int, int]]) -> str:
    cells, rows = set(path), []
    for y in range(task.maze.height):
        row = ""
        for x in range(task.maze.width):
            c = (x, y)
            row += "S " if c == task.maze.start else "E " if c == task.maze.exit else "##" if c in task.maze.walls else ".." if c in cells else "  "
        rows.append(row.rstrip())
    return "\n".join(rows)


def train(tasks: list[Task], cfg: Config, device: torch.device):
    model = ActorCritic(tasks[0].obs.shape[1], len(tasks), cfg).to(device)
    print(f"device={device} input_features={tasks[0].obs.shape[1]} tasks={len(tasks)}")
    behavior_clone(model, tasks, cfg, device)
    best_rows = evaluate(model, tasks, device)
    best_score, best_state = score(best_rows), clone_state(model)
    print_eval("after_bc", best_rows)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4, eps=1e-5)
    solved_streak = 0
    for update in range(1, cfg.updates + 1):
        rollouts = [collect(model, task, cfg, device) for task in tasks]
        stats = ppo_update(model, opt, rollouts, cfg, update, device)
        if update == 1 or update == cfg.updates or update % cfg.eval_every == 0:
            rows = evaluate(model, tasks, device)
            if score(rows) > best_score:
                best_rows, best_score, best_state = rows, score(rows), clone_state(model)
            wins = [w for r in rollouts for w in r["wins"]]
            lens = [l for r in rollouts for l in r["lens"]]
            print(f"update={update:04d} rollout_success={sum(wins)/max(1,len(wins)):.0%} rollout_avg_len={sum(lens)/max(1,len(lens)):.1f} policy_loss={stats['policy']:.3f} value_loss={stats['value']:.3f} entropy={stats['entropy']:.3f} kl={stats['kl']:.4f}")
            print_eval("eval", rows)
            solved_streak = solved_streak + 1 if score(rows)[0] == len(tasks) else 0
            if cfg.early_stop_patience and solved_streak >= cfg.early_stop_patience:
                print(f"early_stop=all_mazes_solved update={update}")
                break
    model.load_state_dict(best_state)
    rows = evaluate(model, tasks, device)
    print_eval("best_loaded", rows)
    return model, rows


def save(output: Path, model: ActorCritic, tasks: list[Task], cfg: Config, rows: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": clone_state(model),
            "config": asdict(cfg),
            "actions": ACTIONS,
            "maze_files": [str(t.path) for t in tasks],
            "task_names": [t.name for t in tasks],
            "input_size": tasks[0].obs.shape[1],
            "task_count": len(tasks),
            "eval": [{"task_name": r["name"], "path_length": r["len"], "shortest_length": r["shortest"], "reached_exit": r["reached"], "looped": r["looped"], "optimal": r["reached"] and r["len"] == r["shortest"]} for r in rows],
        },
        output,
    )
    print(f"saved={output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact all-maze PPO using the original 8 state features.")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, default=Path("weights/compact_original_feature_ppo.pt"))
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--bc-epochs", type=int, default=Config.bc_epochs)
    p.add_argument("--updates", type=int, default=Config.updates)
    p.add_argument("--rollout-steps", type=int, default=Config.rollout_steps)
    p.add_argument("--hidden", type=int, default=Config.hidden)
    p.add_argument("--eval-every", type=int, default=Config.eval_every)
    p.add_argument("--early-stop-patience", type=int, default=Config.early_stop_patience)
    p.add_argument("--print-paths", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(seed=args.seed, bc_epochs=args.bc_epochs, updates=args.updates, rollout_steps=args.rollout_steps, hidden=args.hidden, eval_every=args.eval_every, early_stop_patience=args.early_stop_patience)
    if args.smoke:
        cfg.bc_epochs, cfg.updates, cfg.rollout_steps, cfg.update_epochs, cfg.minibatch, cfg.eval_every, cfg.early_stop_patience = 3, 1, 64, 1, 256, 1, 0
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    root = args.root.resolve()
    tasks = make_tasks(root, cfg)
    if not tasks:
        raise SystemExit(f"No compatible maze_*.py files found under {root}")
    print(f"discovered_mazes={len(tasks)}")
    for task in tasks:
        print(f"  {task.name}: size={task.maze.width}x{task.maze.height} shortest={task.shortest} max_steps={task.max_steps}")
    to_device(tasks, device)
    model, rows = train(tasks, cfg, device)
    if args.print_paths:
        by_name = {r["name"]: r for r in rows}
        for task in tasks:
            print(f"\npath {task.name} len={by_name[task.name]['len']} shortest={task.shortest}")
            print(render(task, by_name[task.name]["path"]))
    if not args.no_save:
        output = args.output if args.output.is_absolute() else root / args.output
        save(output, model, tasks, cfg, rows)


if __name__ == "__main__":
    main()
