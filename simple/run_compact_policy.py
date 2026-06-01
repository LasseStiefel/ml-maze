from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compact_original_feature_ppo import (
    ActorCritic,
    Config,
    evaluate,
    make_tasks,
    render,
    to_device,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT = REPO_ROOT / "simple" / "weights" / "compact_original_feature_ppo.pt"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def task_matches(name: str, selectors: list[str]) -> bool:
    normalized = name.replace("\\", "/")
    return any(selector.replace("\\", "/") in normalized for selector in selectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the saved compact PPO policy on selected mazes.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mazes",
        nargs="+",
        default=["maze_2/maze_2.py", "maze_3/maze_3.py"],
        help="Maze path fragments to evaluate.",
    )
    parser.add_argument("--print-paths", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    checkpoint = load_checkpoint(args.weight.resolve(), device)
    cfg = Config(**checkpoint["config"])

    tasks = make_tasks(args.root.resolve(), cfg)
    if not tasks:
        raise SystemExit(f"No compatible maze_*.py files found under {args.root}")
    to_device(tasks, device)

    model = ActorCritic(
        input_size=checkpoint["input_size"],
        tasks=checkpoint["task_count"],
        cfg=cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    rows = evaluate(model, tasks, device)
    selected = [(task, row) for task, row in zip(tasks, rows) if task_matches(row["name"], args.mazes)]
    if not selected:
        raise SystemExit(f"No tasks matched selectors: {', '.join(args.mazes)}")

    solved = sum(row["reached"] for _task, row in selected)
    optimal = sum(row["reached"] and row["len"] == row["shortest"] for _task, row in selected)
    print(f"compact_policy solved={solved}/{len(selected)} optimal={optimal}/{len(selected)}")
    for task, row in selected:
        status = "optimal" if row["reached"] and row["len"] == row["shortest"] else "solved" if row["reached"] else "failed"
        print(f"  {row['name']}: {status} len={row['len']} shortest={row['shortest']}")
        if args.print_paths:
            print(render(task, row["path"]))
            print()


if __name__ == "__main__":
    main()
