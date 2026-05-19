from __future__ import annotations

import random
from collections import deque

import maze_3 as maze_file


ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

EPISODES = 3000
ALPHA = 0.3
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9990
MIN_EPSILON = 0.05

# Reward policy
MAZE_FINISHED = 100
SHAPING_WEIGHT = 1.0  # reward per step closer to exit

def build_maze():
    width = maze_file.DEFAULT_WIDTH
    height = maze_file.DEFAULT_HEIGHT
    exit_cell = maze_file.resolve_exit(width, height)

    return maze_file.Maze(
        width=width,
        height=height,
        start=maze_file.START,
        exit=exit_cell,
        walls=maze_file.build_walls(width, height, maze_file.START, exit_cell),
    )

def compute_distances(maze):
    """BFS from exit — gives true shortest path distance for every reachable cell."""
    distances = {maze.exit: 0}
    queue = deque([maze.exit])
    while queue:
        state = queue.popleft()
        for dx, dy in ACTIONS.values():
            neighbor = (state[0] + dx, state[1] + dy)
            if maze.is_open(neighbor) and neighbor not in distances:
                distances[neighbor] = distances[state] + 1
                queue.append(neighbor)
    return distances

def move(maze, state, action, distances):
    dx, dy = ACTIONS[action]
    x, y = state
    next_state = (x + dx, y + dy)

    if not maze.is_open(next_state):
        return state, -5, False

    if next_state == maze.exit:
        return next_state, MAZE_FINISHED, True

    # reward shaping: BFS distance accounts for walls, no local optima
    shaping = (distances.get(state, 0) - distances.get(next_state, 0)) * SHAPING_WEIGHT
    return next_state, -1 + shaping, False

def get_q(q_table, state, action):
    return q_table.get((state, action), 0.0)

def best_action(q_table, state):
    return max(ACTIONS, key=lambda action: get_q(q_table, state, action))

def choose_action(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(list(ACTIONS))

    return best_action(q_table, state)

def train(maze):
    q_table = {}
    epsilon = EPSILON
    max_steps = maze.width * maze.height * 8
    wins = deque(maxlen=100)
    distances = compute_distances(maze)

    for episode in range(1, EPISODES + 1):
        state = maze.start
        action = choose_action(q_table, state, epsilon)
        won = False

        for _step in range(max_steps):
            next_state, reward, done = move(maze, state, action, distances)
            next_action = choose_action(q_table, next_state, epsilon)

            # SARSA update: use next_action from policy, not greedy max
            old_q = get_q(q_table, state, action)
            next_q = get_q(q_table, next_state, next_action)

            q_table[(state, action)] = old_q + ALPHA * (
                reward + GAMMA * next_q - old_q
            )

            state = next_state
            action = next_action

            if done:
                won = True
                break

        wins.append(won)
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        if episode % 500 == 0:
            recent_success = sum(wins) / len(wins)
            print(
                f"episode={episode} "
                f"recent_success={recent_success:.0%} "
                f"epsilon={epsilon:.3f}"
            )

    return q_table

def extract_path(maze, q_table):
    state = maze.start
    path = [state]
    max_steps = maze.width * maze.height * 8
    distances = compute_distances(maze)

    for _ in range(max_steps):
        action = best_action(q_table, state)
        next_state, reward, done = move(maze, state, action, distances)

        if next_state == state:
            break

        path.append(next_state)
        state = next_state

        if done:
            break

    return path

def shortest_path_length(maze):
    bfs_dist = compute_distances(maze)
    return bfs_dist.get(maze.start)

def print_path(maze, path):
    path_cells = set(path)

    for y in range(maze.height):
        row = ""
        for x in range(maze.width):
            cell = (x, y)

            if cell == maze.start:
                row += "S "
            elif cell == maze.exit:
                row += "E "
            elif cell in maze.walls:
                row += "##"
            elif cell in path_cells:
                row += ".."
            else:
                row += "  "

        print(row)

def main():
    random.seed(1)

    maze = build_maze()
    q_table = train(maze)
    path = extract_path(maze, q_table)

    print()
    print(f"learned_path_length={len(path) - 1} ")
    print(f"shortest_path_length={shortest_path_length(maze)} ")
    print(f"reached_exit={path[-1] == maze.exit}")
    print()
    print_path(maze, path)

if __name__ == "__main__":
    main()
