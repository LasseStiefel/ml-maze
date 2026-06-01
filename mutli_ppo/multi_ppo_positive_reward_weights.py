from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import maze_1
import maze_11
import maze_13
import maze_14

MIN_MAZE_WEIGHT = 1.00
MAX_MAZE_WEIGHT = 4.00

MAZE_MODULES = [
    maze_14,
    maze_13,
    maze_11,
    maze_13,
    maze_14, 
]

ACTIONS = [
    ("up", (0, -1)),
    ("down", (0, 1)),
    ("left", (-1, 0)),
    ("right", (1, 0)),
]

EPISODES = 2000
ROLLOUT_STEPS = 4096     #Amount of game steps to collect before update
GAMMA = 0.99            #How much future awards matter
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2      #Prevent large policy updates (PPO's incremntal learning)
LEARNING_RATE = 1e-4
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 512
ENTROPY_COEF = 0.2     #Encourages exploration
VALUE_COEF = 0.5

MAZE_FINISHED = 500
CLOSER = -0.5
FURTHER = -0.5

# Builds Maze by getting dimenstions from maze.py. Uses build_walls function from maze.py1
def build_maze(maze_module):
    width = maze_module.DEFAULT_WIDTH
    height = maze_module.DEFAULT_HEIGHT
    exit_cell = maze_module.resolve_exit(width, height)

    return maze_module.Maze(
        width=width,
        height=height,
        start=maze_module.START,
        exit=exit_cell,
        walls=maze_module.build_walls(
            width,
            height, 
            maze_module.START,
            exit_cell,
        ),
    )

def build_distance_map(maze):
    queue = deque([maze.exit])
    distances = {maze.exit: 0}

    while queue:
        state = queue.popleft()
        x, y = state

        for _name, (dx, dy) in ACTIONS:
            next_state = (x + dx, y + dy)

            if maze.is_open(next_state) and next_state not in distances:
                distances[next_state] = distances[state] + 1
                queue.append(next_state)

    return distances

def move(maze, state, action_index, distance_map=None):
    _name, (dx, dy) = ACTIONS[action_index]
    x, y = state
    next_state = (x + dx, y + dy)

    if not maze.is_open(next_state):
        return state, -15, False     # hit wall -> stay in same state -> reward -5 -> episode not done

    if next_state == maze.exit:
        return next_state, MAZE_FINISHED, True      # reached exit -> reward 100 -> episode done

    #return next_state, -1, False    # valid normal step -> move to next state -> reward -1 -> episode not done
    if distance_map is None:
        return next_state, -1, False

    old_distance = distance_map.get(state)
    new_distance = distance_map.get(next_state)

    if old_distance is None or new_distance is None:
        return next_state, -1, False

    if new_distance < old_distance:
        return next_state, CLOSER, False

    if new_distance > old_distance:
        return next_state, FURTHER, False

    return next_state, -1, False

def action_mask(maze, state):
    x, y = state
    return torch.tensor(
        [
            maze.is_open((x + dx, y + dy))
            for _name, (dx, dy) in ACTIONS
        ],
        dtype=torch.bool,
    )

def apply_action_mask(logits, masks):
    return logits.masked_fill(~masks, -1e9)

def scale_positive_rewards(rewards, reward_weight):
    return [
        reward * reward_weight if reward > 0 else reward
        for reward in rewards
    ]

def build_observation_cache(maze):
    states = {}
    masks = {}

    for y in range(maze.height):
        for x in range(maze.width):
            state = (x, y)

            if maze.is_open(state):
                states[state] = encode_state(maze, state)
                masks[state] = action_mask(maze, state)

    return states, masks

# is the devision by maze size necessary or maybe a codex complication ?
def encode_state(maze, state):
    x, y = state
    ex, ey = maze.exit

    wall_up = 0 if maze.is_open((x, y - 1)) else 1
    wall_down = 0 if maze.is_open((x, y + 1)) else 1
    wall_left = 0 if maze.is_open((x - 1, y)) else 1
    wall_right = 0 if maze.is_open((x + 1, y)) else 1

    return torch.tensor(
        [
            x / max(1, maze.width - 1),
            y / max(1, maze.height - 1),
            (ex - x) / max(1, maze.width - 1),
            (ey - y) / max(1, maze.height - 1),
            wall_up,
            wall_down,
            wall_left,
            wall_right,
        ],
        dtype=torch.float32,
    )


#
class ActorCritic(nn.Module):
    def __init__(self, input_size, action_size):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, states):
        hidden = self.shared(states)
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values

    def act(self, state_tensor, mask):
        logits, value = self.forward(state_tensor.unsqueeze(0))
        masked_logits = apply_action_mask(logits, mask.unsqueeze(0))
        dist = Categorical(logits=masked_logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action).squeeze(0), value.squeeze(0)
    
def collect_rollout(maze, model, max_steps, distance_map, state_cache, mask_cache):
    rollout = Rollout([], [], [], [], [], [], [])
    episode_rewards = []
    episode_wins = []
    visits = {}

    state = maze.start
    episode_reward = 0
    episode_step = 0

    while len(rollout.states) < ROLLOUT_STEPS:
        state_tensor = state_cache[state]
        mask = mask_cache[state]
        visits[state] = visits.get(state, 0) + 1

        with torch.no_grad():
            action, log_prob, value = model.act(state_tensor, mask)

        next_state, reward, done = move(maze, state, action, distance_map)

        rollout.states.append(state_tensor)
        rollout.masks.append(mask)
        rollout.actions.append(action)
        rollout.rewards.append(reward)
        rollout.dones.append(done)
        rollout.log_probs.append(log_prob)
        rollout.values.append(value)

        state = next_state
        episode_reward += reward
        episode_step += 1

        timed_out = episode_step >= max_steps
        if timed_out and not done:
            rollout.rewards[-1] += -50
            episode_reward += -50
            rollout.dones[-1] = True


        if done or timed_out: 
            episode_rewards.append(episode_reward)
            episode_wins.append(done)

            state = maze.start
            episode_reward = 0
            episode_step = 0

    return rollout, episode_rewards, episode_wins, visits

def compute_advantages(rollout):
    rewards = rollout.rewards
    dones = rollout.dones
    values = torch.stack(rollout.values)

    advantages = []
    gae = 0.0
    next_value = 0.0

    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0

        delta = rewards[step] + GAMMA * next_value * mask - values[step].item()
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae

        advantages.append(gae)
        next_value = values[step].item()

    advantages.reverse()
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + values.detach()

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns



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

def print_heatmap(maze, visits):
    max_visit = max(visits.values(), default=1)

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
            else:
                count = visits.get(cell, 0)

                if count == 0:
                    row += "  "
                elif count < max_visit * 0.25:
                    row += ".."
                elif count < max_visit * 0.50:
                    row += "::"
                elif count < max_visit * 0.75:
                    row += "**"
                else:
                    row += "@@"

        print(row)


def shortest_path_length(maze):
    queue = deque([maze.start])
    distances = {maze.start: 0}

    while queue:
        state = queue.popleft()

        if state == maze.exit:
            return distances[state]
        
        for action_index in range(len(ACTIONS)):
            next_state, _reward, _done = move(maze, state, action_index)

            if next_state != state and next_state not in distances:
                distances[next_state] = distances[state] + 1
                queue.append(next_state)
        
    return None

def update_model(model, optimizer, rollout):
    states = torch.stack(rollout.states)
    masks = torch.stack(rollout.masks)
    actions = torch.tensor(rollout.actions, dtype=torch.long)
    old_log_probs = torch.stack(rollout.log_probs).detach()

    advantages, returns = compute_advantages(rollout)

    total_steps = len(states)
    indices = list(range(total_steps))

    for _ in range(UPDATE_EPOCHS):
        random.shuffle(indices)

        for start in range(0, total_steps, MINIBATCH_SIZE):
            batch_indices = indices[start : start + MINIBATCH_SIZE]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            batch_masks = masks[batch_indices]

            logits, values = model(batch_states)
            logits = apply_action_mask(logits, batch_masks)
            dist = Categorical(logits=logits)

            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            unclipped = ratio * batch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - CLIP_EPSILON,
                1.0 + CLIP_EPSILON,
            ) * batch_advantages

            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (batch_returns - values).pow(2).mean()

            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                - ENTROPY_COEF * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train(mazes):
    input_size = len(encode_state(mazes[0], mazes[0].start))
    action_size = len(ACTIONS)

    model = ActorCritic(input_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    distance_maps = [build_distance_map(maze) for maze in mazes]
    observation_caches = [build_observation_cache(maze) for maze in mazes]

    recent_wins = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)

    maze_recent_wins = [deque(maxlen=100) for _ in mazes]
    maze_recent_rewards = [deque(maxlen=100) for _ in mazes]

    for episode in range(1, EPISODES + 1):
        rollouts = []

        for maze_index, (maze, distance_map, observation_cache) in enumerate(
            zip(mazes, distance_maps, observation_caches)
        ):
            max_steps = maze.width * maze.height * 4
            state_cache, mask_cache = observation_cache
            maze_success_rate = (
                sum(maze_recent_wins[maze_index])
                / max(1, len(maze_recent_wins[maze_index]))
            )
            reward_weight = MIN_MAZE_WEIGHT + (
                MAX_MAZE_WEIGHT - MIN_MAZE_WEIGHT
            ) * (1.0 - maze_success_rate)

            rollout, rewards, wins, visits = collect_rollout(
                maze,
                model,
                max_steps,
                distance_map,
                state_cache,
                mask_cache,
            )

            rollout.rewards = scale_positive_rewards(rollout.rewards, reward_weight)
            rewards = scale_positive_rewards(rewards, reward_weight)

            rollouts.append(rollout)
            recent_rewards.extend(rewards)
            recent_wins.extend(wins)
            maze_recent_rewards[maze_index].extend(rewards)
            maze_recent_wins[maze_index].extend(wins)

        combined_rollout = combine_rollouts(rollouts)
        update_model(model, optimizer, combined_rollout)

        if episode % 50 == 0:
            avg_reward = sum(recent_rewards) / max(1, len(recent_rewards))
            success_rate = sum(recent_wins) / max(1, len(recent_wins))

            print(
                f"episode={episode} "
                f"avg_reward={avg_reward:.1f} "
                f"recent_success={success_rate:.0%}"
            )

            for maze_index in range(len(mazes)):
                maze_avg_reward = (
                    sum(maze_recent_rewards[maze_index])
                    / max(1, len(maze_recent_rewards[maze_index]))
                )
                maze_success_rate = (
                    sum(maze_recent_wins[maze_index])
                    / max(1, len(maze_recent_wins[maze_index]))
                )
                maze_reward_weight = MIN_MAZE_WEIGHT + (
                    MAX_MAZE_WEIGHT - MIN_MAZE_WEIGHT
                ) * (1.0 - maze_success_rate)

                print(
                    f"  maze={maze_index + 1} "
                    f"avg_reward={maze_avg_reward:.1f} "
                    f"success={maze_success_rate:.0%} "
                    f"positive_weight={maze_reward_weight:.2f}"
                )

    return model


def choose_greedy_action(model, maze, state, state_cache=None, mask_cache=None):
    if state_cache is None or mask_cache is None:
        state_tensor = encode_state(maze, state)
        mask = action_mask(maze, state)
    else:
        state_tensor = state_cache[state]
        mask = mask_cache[state]

    with torch.no_grad():
        logits, _value = model(state_tensor.unsqueeze(0))
        logits = apply_action_mask(logits, mask.unsqueeze(0))

    return torch.argmax(logits, dim=-1).item()


def extract_path(maze, model, state_cache=None, mask_cache=None):
    state = maze.start
    path = [state]
    max_steps = maze.width * maze.height * 4

    for _ in range(max_steps):
        action = choose_greedy_action(model, maze, state, state_cache, mask_cache)
        next_state, _reward, done = move(maze, state, action)

        if next_state == state:
            break

        path.append(next_state)
        state = next_state

        if done:
            break

    return path


@dataclass
class Rollout:
    states: list
    masks: list
    actions: list
    rewards: list
    dones: list
    log_probs: list
    values: list

def combine_rollouts(rollouts):
    combined = Rollout([], [], [], [], [], [], [])

    for rollout in rollouts:
        combined.states.extend(rollout.states)
        combined.masks.extend(rollout.masks)
        combined.actions.extend(rollout.actions)
        combined.rewards.extend(rollout.rewards)
        combined.dones.extend(rollout.dones)
        combined.log_probs.extend(rollout.log_probs)
        combined.values.extend(rollout.values)

    return combined

def print_training_parameters():
    print()
    print("Training parameters")
    print(f"mazes={[module.__name__ for module in MAZE_MODULES]}")
    print(f"min_positive_reward_weight={MIN_MAZE_WEIGHT}")
    print(f"max_positive_reward_weight={MAX_MAZE_WEIGHT}")
    print(f"episodes={EPISODES}")
    print(f"rollout_steps={ROLLOUT_STEPS}")
    print(f"gamma={GAMMA}")
    print(f"gae_lambda={GAE_LAMBDA}")
    print(f"clip_epsilon={CLIP_EPSILON}")
    print(f"learning_rate={LEARNING_RATE}")
    print(f"update_epochs={UPDATE_EPOCHS}")
    print(f"minibatch_size={MINIBATCH_SIZE}")
    print(f"entropy_coef={ENTROPY_COEF}")
    print(f"value_coef={VALUE_COEF}")
    print(f"maze_finished_reward={MAZE_FINISHED}")
    print(f"closer_to_exit={CLOSER}")
    print(f"further_from_exit={FURTHER}")

def main():
    random.seed(1)
    torch.manual_seed(1)

    mazes = [build_maze(module) for module in MAZE_MODULES]

    model = train(mazes)

    torch.save(model.state_dict(), "ppo_multi_maze_2000gen_0.2_0.2clip_-5onWallHit.pt")

    print_training_parameters()

    for index, maze in enumerate(mazes, start=1):
        state_cache, mask_cache = build_observation_cache(maze)
        path = extract_path(maze, model, state_cache, mask_cache)

        print()
        print(f"maze={index}")
        print(f"learned_path_length={len(path) - 1}")
        print(f"shortest_path_length={shortest_path_length(maze)}")
        print(f"reached_exit={path[-1] == maze.exit}")
        print()
        print_path(maze, path)



if __name__ == "__main__":
    main()
