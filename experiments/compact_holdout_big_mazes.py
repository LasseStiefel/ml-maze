from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "simple"))

from compact_original_feature_ppo import (  # noqa: E402
    ACTIONS,
    Config,
    behavior_clone,
    evaluate,
    make_tasks,
    ppo_update,
    print_eval,
    render,
    to_device,
    collect,
    score,
)


class NoTaskActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.actor = nn.Linear(hidden, len(ACTIONS))
        self.critic = nn.Linear(hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del task_ids
        hidden = self.net(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def normalized_name(task_name: str) -> str:
    return task_name.replace("\\", "/")


def is_big_maze(task_name: str) -> bool:
    normalized = normalized_name(task_name)
    return normalized in {"maze_2/maze_2.py", "maze_3/maze_3.py"}


def is_biggest_maze(task_name: str) -> bool:
    return normalized_name(task_name) == "maze_3/maze_3.py"


def is_medium_maze(task_name: str) -> bool:
    return normalized_name(task_name) == "maze_2/maze_2.py"


def is_small_maze(task_name: str) -> bool:
    return normalized_name(task_name).startswith("maze_1/")


def split_tasks(tasks: list[Any], split: str) -> tuple[list[Any], list[Any], str]:
    if split == "small_to_big":
        train_tasks = [task for task in tasks if is_small_maze(task.name)]
        holdout_tasks = [task for task in tasks if is_big_maze(task.name)]
        label = "train_small_holdout_big"
    elif split == "biggest_to_smaller":
        train_tasks = [task for task in tasks if is_biggest_maze(task.name)]
        holdout_tasks = [task for task in tasks if not is_biggest_maze(task.name)]
        label = "train_biggest_holdout_smaller"
    elif split == "small_biggest_to_medium":
        train_tasks = [task for task in tasks if is_small_maze(task.name) or is_biggest_maze(task.name)]
        holdout_tasks = [task for task in tasks if is_medium_maze(task.name)]
        label = "train_small_biggest_holdout_medium"
    else:
        raise ValueError(f"Unknown split: {split}")
    return train_tasks, holdout_tasks, label


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_on_small_mazes(
    model: nn.Module,
    train_tasks: list[Any],
    cfg: Config,
    device: torch.device,
) -> None:
    print(f"train_tasks={len(train_tasks)} input_features={train_tasks[0].obs.shape[1]}")
    behavior_clone(model, train_tasks, cfg, device)
    best_rows = evaluate(model, train_tasks, device)
    best_score = score(best_rows)
    best_state = clone_state(model)
    print_eval("train_after_bc", best_rows)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4, eps=1e-5)
    for update in range(1, cfg.updates + 1):
        rollouts = [collect(model, task, cfg, device) for task in train_tasks]
        stats = ppo_update(model, optimizer, rollouts, cfg, update, device)
        if update == 1 or update == cfg.updates or update % cfg.eval_every == 0:
            rows = evaluate(model, train_tasks, device)
            if score(rows) > best_score:
                best_score = score(rows)
                best_state = clone_state(model)
            wins = [win for rollout in rollouts for win in rollout["wins"]]
            lens = [length for rollout in rollouts for length in rollout["lens"]]
            print(
                f"update={update:04d} rollout_success={sum(wins)/max(1, len(wins)):.0%} "
                f"rollout_avg_len={sum(lens)/max(1, len(lens)):.1f} "
                f"policy_loss={stats['policy']:.3f} value_loss={stats['value']:.3f} "
                f"entropy={stats['entropy']:.3f} kl={stats['kl']:.4f}"
            )
            print_eval("train_eval", rows)

    model.load_state_dict(best_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train compact PPO without maze 2/3 and test zero-shot on those big mazes."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument(
        "--split",
        choices=["small_to_big", "biggest_to_smaller", "small_biggest_to_medium"],
        default="small_to_big",
    )
    parser.add_argument("--print-paths", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    cfg = Config(
        bc_epochs=args.bc_epochs,
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        eval_every=args.eval_every,
        hidden=args.hidden,
        early_stop_patience=0,
    )

    tasks = make_tasks(args.root.resolve(), cfg)
    train_tasks, holdout_tasks, split_label = split_tasks(tasks, args.split)
    if not train_tasks or not holdout_tasks:
        raise SystemExit("Expected both training and holdout mazes for the selected split.")

    to_device(tasks, device)
    model = NoTaskActorCritic(train_tasks[0].obs.shape[1], cfg.hidden).to(device)

    print(f"split={split_label}")
    print("held_out=" + ", ".join(task.name for task in holdout_tasks))
    train_on_small_mazes(model, train_tasks, cfg, device)

    print()
    print_eval("train_final", evaluate(model, train_tasks, device))
    holdout_rows = evaluate(model, holdout_tasks, device)
    print_eval("holdout_zero_shot", holdout_rows)

    if args.print_paths:
        by_name = {row["name"]: row for row in holdout_rows}
        for task in holdout_tasks:
            print()
            print(f"path {task.name} len={by_name[task.name]['len']} shortest={task.shortest}")
            print(render(task, by_name[task.name]["path"]))


if __name__ == "__main__":
    main()
