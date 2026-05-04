from __future__ import annotations

import random
from collections import deque

import maze_1 as maze_file


ACTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}

EPISODES = 5000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05

# Reward policy
MAZE_FINISHED = 100

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

# calculates location after next move and defines reward return
def move(maze, state, action):
    dx, dy = ACTIONS[action]
    x, y, = state
    next_state = (x + dx, y + dy)

    if not maze.is_open(next_state):
        return state, -5, False
    
    if next_state == maze.exit:
        return next_state, MAZE_FINISHED, True

    return next_state, -1, False

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
    max_steps = maze.width * maze.height * 4
    wins = deque(maxlen=100)

    for episode in range(1, EPISODES + 1):
        state = maze.start
        won = False

        for _step in range(max_steps):
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done = move(maze, state, action)

            old_q = get_q(q_table, state, action)
            next_best_q = max(get_q(q_table, next_state, a) for a in ACTIONS)

            q_table[(state, action)] = old_q + ALPHA * (
                reward + GAMMA * next_best_q - old_q
            )

            state = next_state

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
    max_steps = maze.width * maze.height * 4

    for _ in range(max_steps):
        action = best_action(q_table, state)
        next_state, reward, done = move(maze, state, action)

        if next_state == state:
            break

        path.append(next_state)
        state = next_state

        if done:
            break 

    return path

def shortest_path_length(maze):
    queue = deque([maze.start])
    distances = {maze.start: 0}

    while queue:
        state = queue.popleft()

        if state == maze.exit:
            return distances[state]
        
        for action in ACTIONS:
            next_state, _reward, _done = move(maze, state, action)

            if next_state != state and next_state not in distances:
                distances[next_state] = distances[state] + 1
                queue.append(next_state)
        
    return None

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
