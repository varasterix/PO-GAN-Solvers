import torch
import torch.optim as optim
import math
import numpy
import random

from src.database import databaseTools
from src import constants
from src.DQN.dqn import DQN
from src.DQN.replay_memory import ReplayMemory


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# Get number of actions from gym action space
# n_actions = env.action_space.n

epochs = 2000
dataset = []
for i in range(2000):
    dataset.append(databaseTools.read_tsp_choco_solution_file(10, i, path="../../" +
                                                                              constants.PARAMETER_TSP_CHOCO_DATA_FILES))

n_actions = len(dataset)  # not sure about that...
n_cities = dataset[0].get_weight_matrix[0].size()  # number of cities
wm_size = dataset[0].get_weight_matrix.reshape(n_cities**2)  # weight matrix as a n_cities * n_cities input

policy_net = DQN(wm_size).to(device)
target_net = DQN(wm_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []
