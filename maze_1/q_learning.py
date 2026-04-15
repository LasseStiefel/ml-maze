import q_learning as ql

width = 16
height = 16

episodes = 5000
max_steps = width * height * 4
alpha = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05

