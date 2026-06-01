from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_ROOT / "data"
PLOTS_DIR = EXPERIMENT_ROOT / "plots"
REPORT_PATH = EXPERIMENT_ROOT / "report.md"
SIZE_ORDER = ["small", "medium", "large"]
MODEL_ORDER = ["legacy_simple_ppo", "compact_ppo"]
MODEL_LABEL = {
    "legacy_simple_ppo": "Legacy simple PPO",
    "compact_ppo": "Compact PPO",
}


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def coerce_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ["shortest", "final_path_len"]:
            item[key] = int(float(item[key]))
        for key in ["final_solved", "final_optimal", "final_looped"]:
            item[key] = parse_bool(item[key])
        for key in [
            "first_solved_step",
            "first_solved_env_steps",
            "first_solved_expert_epochs",
            "first_optimal_step",
            "first_optimal_env_steps",
            "first_optimal_expert_epochs",
        ]:
            item[key] = None if item[key] == "" else int(float(item[key]))
        for key in ["first_solved_seconds", "first_optimal_seconds"]:
            item[key] = parse_float(item[key])
        out.append(item)
    return out


def coerce_curves(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ["phase_step", "env_steps", "expert_epochs", "path_len", "shortest"]:
            item[key] = int(float(item[key]))
        item["elapsed_seconds"] = float(item["elapsed_seconds"])
        for key in ["solved", "optimal", "looped"]:
            item[key] = parse_bool(item[key])
        out.append(item)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def grouped(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def plot_final_solve_rates(summary: list[dict[str, Any]]) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(SIZE_ORDER)))
    width = 0.36
    offsets = [-width / 2, width / 2]
    for model_index, model in enumerate(MODEL_ORDER):
        rates = []
        for size in SIZE_ORDER:
            rows = [row for row in summary if row["model"] == model and row["size"] == size]
            rates.append(mean([1.0 if row["final_solved"] else 0.0 for row in rows]) * 100)
        ax.bar([value + offsets[model_index] for value in x], rates, width=width, label=MODEL_LABEL[model])
    ax.set_title("Final Greedy Solve Rate by Maze Size")
    ax.set_ylabel("Solved runs (%)")
    ax.set_xlabel("Generated maze size")
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = PLOTS_DIR / "final_solve_rate_by_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row["model"], row["size"], row["maze"], str(row["train_seed"]))


def ever_solved_lookup(curves: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, bool]]:
    lookup: dict[tuple[str, str, str, str], dict[str, bool]] = defaultdict(lambda: {"solved": False, "optimal": False})
    for row in curves:
        key = run_key(row)
        lookup[key]["solved"] = lookup[key]["solved"] or row["solved"]
        lookup[key]["optimal"] = lookup[key]["optimal"] or row["optimal"]
    return lookup


def plot_ever_solve_rates(summary: list[dict[str, Any]], curves: list[dict[str, Any]]) -> Path:
    lookup = ever_solved_lookup(curves)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(SIZE_ORDER)))
    width = 0.36
    offsets = [-width / 2, width / 2]
    for model_index, model in enumerate(MODEL_ORDER):
        rates = []
        for size in SIZE_ORDER:
            rows = [row for row in summary if row["model"] == model and row["size"] == size]
            rates.append(mean([1.0 if lookup[run_key(row)]["solved"] else 0.0 for row in rows]) * 100)
        ax.bar([value + offsets[model_index] for value in x], rates, width=width, label=MODEL_LABEL[model])
    ax.set_title("Ever-Solved Rate by Maze Size")
    ax.set_ylabel("Runs that reached a solved checkpoint (%)")
    ax.set_xlabel("Generated maze size")
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = PLOTS_DIR / "ever_solve_rate_by_size.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_learning_curves(curves: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    legacy_rows = [row for row in curves if row["model"] == "legacy_simple_ppo"]
    for size in SIZE_ORDER:
        by_step = grouped([row for row in legacy_rows if row["size"] == size], "env_steps")
        xs = sorted(step[0] for step in by_step)
        ys = [mean([1.0 if row["solved"] else 0.0 for row in by_step[(step,)]]) * 100 for step in xs]
        axes[0].plot(xs, ys, marker="o", label=size)
    axes[0].set_title("Legacy Simple PPO Learning Curve")
    axes[0].set_xlabel("Environment steps collected")
    axes[0].set_ylabel("Solved runs (%)")
    axes[0].grid(alpha=0.25)

    compact_rows = [row for row in curves if row["model"] == "compact_ppo" and row["phase"] == "bc"]
    for size in SIZE_ORDER:
        by_epoch = grouped([row for row in compact_rows if row["size"] == size], "expert_epochs")
        xs = sorted(epoch[0] for epoch in by_epoch)
        ys = [mean([1.0 if row["solved"] else 0.0 for row in by_epoch[(epoch,)]]) * 100 for epoch in xs]
        axes[1].plot(xs, ys, marker="o", label=size)
    axes[1].set_title("Compact PPO Behavior-Cloning Curve")
    axes[1].set_xlabel("BFS expert epochs")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Size")
    axes[0].legend(title="Size")
    axes[0].set_ylim(0, 105)

    path = PLOTS_DIR / "learning_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_first_solve_budget(summary: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    legacy = [row for row in summary if row["model"] == "legacy_simple_ppo"]
    compact = [row for row in summary if row["model"] == "compact_ppo"]

    for axis, rows, field, title, ylabel in [
        (axes[0], legacy, "first_solved_env_steps", "Legacy Simple PPO", "Env steps to first solve"),
        (axes[1], compact, "first_solved_expert_epochs", "Compact PPO", "BFS expert epochs to first solve"),
    ]:
        positions = range(len(SIZE_ORDER))
        medians = []
        points_x = []
        points_y = []
        unsolved_x = []
        unsolved_y = []
        for index, size in enumerate(SIZE_ORDER):
            size_rows = [row for row in rows if row["size"] == size]
            solved_values = [row[field] for row in size_rows if row[field] is not None]
            medians.append(statistics.median(solved_values) if solved_values else 0)
            for row in size_rows:
                if row[field] is None:
                    unsolved_x.append(index)
                    if field == "first_solved_env_steps":
                        unsolved_y.append(max([r["first_solved_env_steps"] or 0 for r in size_rows] + [1]))
                    else:
                        unsolved_y.append(max([r["first_solved_expert_epochs"] or 0 for r in size_rows] + [1]))
                else:
                    points_x.append(index)
                    points_y.append(row[field])
        axis.bar(positions, medians, alpha=0.55)
        axis.scatter(points_x, points_y, color="black", s=24, label="solved run")
        if unsolved_x:
            axis.scatter(unsolved_x, unsolved_y, color="crimson", marker="x", s=42, label="unsolved within budget")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(SIZE_ORDER)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    path = PLOTS_DIR / "first_solve_budget.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_path_efficiency(summary: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    data = []
    for size in SIZE_ORDER:
        for model in MODEL_ORDER:
            rows = [row for row in summary if row["size"] == size and row["model"] == model and row["final_solved"]]
            ratios = [row["final_path_len"] / max(1, row["shortest"]) for row in rows]
            model_label = MODEL_LABEL[model].replace(" ", "\n")
            labels.append(f"{size}\n{model_label}")
            data.append(ratios if ratios else [float("nan")])
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("Final Path Efficiency for Solved Runs")
    ax.set_ylabel("Path length / shortest path")
    ax.grid(axis="y", alpha=0.25)
    path = PLOTS_DIR / "path_efficiency.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(summary: list[dict[str, Any]], curves: list[dict[str, Any]], plot_paths: list[Path]) -> None:
    lookup = ever_solved_lookup(curves)
    table_rows = []
    for model in MODEL_ORDER:
        for size in SIZE_ORDER:
            rows = [row for row in summary if row["model"] == model and row["size"] == size]
            ever_solved = sum(1 for row in rows if lookup[run_key(row)]["solved"])
            final_solved = sum(1 for row in rows if row["final_solved"])
            final_optimal = sum(1 for row in rows if row["final_optimal"])
            solved_times = [row["first_solved_seconds"] for row in rows if row["first_solved_seconds"] is not None]
            solved_budget = [
                row["first_solved_env_steps"] if model == "legacy_simple_ppo" else row["first_solved_expert_epochs"]
                for row in rows
                if (row["first_solved_env_steps"] if model == "legacy_simple_ppo" else row["first_solved_expert_epochs"]) is not None
            ]
            table_rows.append(
                [
                    MODEL_LABEL[model],
                    size,
                    f"{ever_solved}/{len(rows)}",
                    f"{final_solved}/{len(rows)}",
                    f"{final_optimal}/{len(rows)}",
                    f"{statistics.median(solved_budget):.0f}" if solved_budget else "not solved",
                    f"{statistics.median(solved_times):.2f}s" if solved_times else "not solved",
                ]
            )

    report = f"""# PPO Size Learning Experiment Report

## Purpose

This experiment estimates how much training is needed for two simple PPO-style
models to solve generated mazes at the same size they train on. Each run trains
on one generated maze and evaluates greedy performance on that same maze after
training checkpoints.

## Models

- **Legacy simple PPO**: original 8-feature actor-critic style from `maze_1/multi_ppo.py`.
  It uses sampled PPO rollouts, wall penalties, Manhattan-distance shaping, and no action mask.
- **Compact PPO**: current compact implementation from `simple/compact_original_feature_ppo.py`.
  It uses the same 8 local features plus action masks, BFS expert behavior cloning, and PPO.

Important methodological note: compact PPO receives privileged BFS expert
supervision during behavior cloning. It is therefore a comparison of practical
training recipes, not a pure equal-information reinforcement-learning contest.

## Experimental Design

- Sizes: `16x16`, `25x25`, `35x35`.
- Mazes: generated as connected tree mazes with one guaranteed start-to-exit
  solution path plus randomized dead-end branches.
- Repeats: every generated maze is trained with each configured training seed.
- Success criterion: greedy policy reaches the exit before timeout.
- Optimality criterion: greedy path length equals the BFS shortest path.

## Summary

{markdown_table(["Model", "Size", "Ever solved", "Final solved", "Final optimal", "Median first-solve budget", "Median first-solve time"], table_rows)}

## Figures

"""
    for path in plot_paths:
        rel = path.relative_to(EXPERIMENT_ROOT).as_posix()
        report += f"![{path.stem}]({rel})\n\n"

    report += """## Interpretation Notes

- Solve-rate curves show reliability across generated mazes and random training
  seeds, not only a single lucky run.
- First-solve budget is intentionally model-specific: environment steps for
  legacy PPO, BFS expert epochs for compact PPO.
- Unsolved runs are retained in the CSV and plotted as failures rather than
  being discarded.
- `Ever solved` is the best-checkpoint view; `Final solved` is the last-checkpoint
  stability view. The repo's main compact trainer keeps the best checkpoint, so
  both views matter.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    summary = coerce_summary(read_csv(DATA_DIR / "summary.csv"))
    curves = coerce_curves(read_csv(DATA_DIR / "training_curves.csv"))
    plot_paths = [
        plot_final_solve_rates(summary),
        plot_ever_solve_rates(summary, curves),
        plot_learning_curves(curves),
        plot_first_solve_budget(summary),
        plot_path_efficiency(summary),
    ]
    write_report(summary, curves, plot_paths)
    print(f"wrote={REPORT_PATH}")
    for path in plot_paths:
        print(f"wrote={path}")


if __name__ == "__main__":
    main()
