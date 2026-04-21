"""
Gym-style environment wrapper for Maze 1.

The agent observes the full 16x16 grid as a flat array:
    0 = floor, 1 = wall, 2 = player, 3 = exit

Actions:
    0 = up, 1 = down, 2 = left, 3 = right

Rewards:
    +1.0  each step that reduces Manhattan distance to exit
    -1.0  each step that increases or keeps the same distance
    +50.0 reaching the exit
    -0.5  bumping into a wall (position unchanged)
"""

from __future__ import annotations

import numpy as np
from typing import Iterable


# ---------------------------------------------------------------------------
# Maze 1 layout — copied directly from maze_1.py so this file is self-contained
# ---------------------------------------------------------------------------
WIDTH = 16
HEIGHT = 16
START = (0, 0)
EXIT_CELL = (WIDTH - 1, HEIGHT - 1)   # (15, 15)

WALL_LINES = [
    (3, 1, 3, 5),
    (6, 2, 6, 6),
    (9, 1, 9, 4),
    (12, 3, 12, 7),
    (2, 8, 2, 12),
    (5, 9, 5, 14),
    (8, 7, 8, 11),
    (11, 10, 11, 14),
    (1, 4, 4, 4),
    (6, 8, 10, 8),
    (10, 12, 14, 12),
    (12, 1, 14, 1),
    (0, 10, 1, 10),
    (13, 5, 15, 5),
]

EXTRA_WALLS = {
    (1, 6),
    (4, 8),
    (7, 13),
    (10, 6),
    (14, 9),
}


def _cells_from_line(line: tuple[int, int, int, int]) -> Iterable[tuple[int, int]]:
    x1, y1, x2, y2 = line
    if x1 == x2:
        step = 1 if y2 >= y1 else -1
        for y in range(y1, y2 + step, step):
            yield (x1, y)
    elif y1 == y2:
        step = 1 if x2 >= x1 else -1
        for x in range(x1, x2 + step, step):
            yield (x, y1)
    else:
        raise ValueError(f"Wall must be horizontal or vertical: {line}")


def _build_walls() -> frozenset[tuple[int, int]]:
    walls: set[tuple[int, int]] = set(EXTRA_WALLS)
    for line in WALL_LINES:
        walls.update(_cells_from_line(line))
    walls = {c for c in walls if 0 <= c[0] < WIDTH and 0 <= c[1] < HEIGHT}
    walls.discard(START)
    walls.discard(EXIT_CELL)
    return frozenset(walls)


WALLS = _build_walls()

# Action deltas: (dx, dy)
ACTIONS = {
    0: (0, -1),   # up
    1: (0,  1),   # down
    2: (-1, 0),   # left
    3: (1,  0),   # right
}
NUM_ACTIONS = 4
OBS_SIZE = WIDTH * HEIGHT   # 256


class MazeEnv:
    """
    Minimal environment that mimics the Gymnasium API:
        obs, info  = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    def __init__(self, max_steps: int = WIDTH * HEIGHT * 4) -> None:
        self.max_steps = max_steps
        self._player: tuple[int, int] = START
        self._steps: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        self._player = START
        self._steps = 0
        return self._observe(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert 0 <= action < NUM_ACTIONS

        prev_dist = self._manhattan()
        dx, dy = ACTIONS[action]
        x, y = self._player
        next_cell = (x + dx, y + dy)

        hit_wall = not self._is_open(next_cell)
        if not hit_wall:
            self._player = next_cell

        self._steps += 1
        new_dist = self._manhattan()

        # Reward shaping
        if self._player == EXIT_CELL:
            reward = 50.0
            terminated = True
        elif hit_wall:
            reward = -0.5
            terminated = False
        elif new_dist < prev_dist:
            reward = 1.0
            terminated = False
        else:
            reward = -1.0
            terminated = False

        truncated = (not terminated) and (self._steps >= self.max_steps)
        return self._observe(), reward, terminated, truncated, {"steps": self._steps}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _observe(self) -> np.ndarray:
        """Return a flat float32 array of shape (256,) encoding the grid."""
        grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        for wx, wy in WALLS:
            grid[wy, wx] = 1.0
        ex, ey = EXIT_CELL
        grid[ey, ex] = 3.0
        px, py = self._player
        grid[py, px] = 2.0
        return grid.flatten()

    def _manhattan(self) -> int:
        px, py = self._player
        ex, ey = EXIT_CELL
        return abs(px - ex) + abs(py - ey)

    def _is_open(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < WIDTH and 0 <= y < HEIGHT and cell not in WALLS
