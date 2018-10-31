import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import random
from gym import wrappers
from bindsnet import *
from collections import deque, namedtuple
import itertools
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--occlusionloc', dest='occlusionloc', type=int, default=0)

locals().update(vars(parser.parse_args()))

num_episodes = 100
epsilon_decay_steps = 5000
# epsilons = np.linspace(0.5, 0.0, epsilon_decay_steps)
epsilon = 0.0
noop_counter = 0


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Atari Actions: 0 (noop), 1 (fire), 2 (right) and 3 (left) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]
total_actions = len(VALID_ACTIONS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Load SpaceInvaders environment.
environment = GymEnvironment('BreakoutDeterministic-v4')

old_network = torch.load('trained models/dqn_time_difference_grayscale.pt')
network = Net().to(device)

network.load_state_dict(old_network.state_dict())

total_t = 0
episode_rewards = np.zeros(num_episodes)
episode_lengths = np.zeros(num_episodes)

experiment_dir = os.path.abspath("./ann/{}".format(environment.env.spec.id))
monitor_path = os.path.join(experiment_dir, "monitor")

# if not os.path.exists("./data/frames/"):
#     os.makedirs("./data/frames/")

# sample_number = 0
# labels = []

# def save_state(state):
#     global sample_number
#     encoded_state = torch.sum(torch.tensor([0.25, 0.5, 0.75, 1]) * state.cuda(), dim=2)
#     pickle.dump(encoded_state, open('./data/frames/' + str(sample_number) + '.frame', 'wb'))
#     sample_number += 1

environment.env = wrappers.Monitor(environment.env, directory=monitor_path, resume=True)


# states = []

def policy(q_values, eps):
    A = np.ones(4, dtype=float) * eps / 4
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


for i_episode in range(num_episodes):
    obs = environment.reset()
    state = torch.stack([obs] * 4, dim=2)

    for t in itertools.count():
        print("\rStep {} ({}) @ Episode {}/{}".format(
            t, total_t, i_episode + 1, num_episodes), end="")
        sys.stdout.flush()
        encoded_state = torch.tensor([0.25, 0.5, 0.75, 1]) * state.cuda()
        encoded_state = torch.sum(encoded_state, dim=2)
        # states.append(encoded_state)
        encoded_state[80 - 3 - occlusionloc:80 - occlusionloc, :] = 0
        q_values = network(encoded_state.view([1, -1]).cuda())[0]
        # epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
        action_probs, best_action = policy(q_values, epsilon)
        # labels.append(best_action)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        if action == 0:
            noop_counter += 1
        else:
            noop_counter = 0
        if noop_counter >= 20:
            action = np.random.choice(np.arange(len(action_probs)))
            noop_counter = 0

        next_obs, reward, done, _ = environment.step(VALID_ACTIONS[action])
        next_state = torch.clamp(next_obs - obs, min=0)
        next_state = torch.cat((state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2)
        episode_rewards[i_episode] += reward
        episode_lengths[i_episode] = t
        total_t += 1
        if done:
            print("\nEpisode Reward: {}".format(episode_rewards[i_episode]))
            break

        state = next_state
        obs = next_obs

np.savetxt('analysis/rewards_dqn_robustness_' + str(occlusionloc) + '.txt', episode_rewards)

# states = torch.stack(states, dim=0)
# torch.save(states.cpu(), 'frames.pt')
# print(states.shape)
# labels = torch.tensor(labels)
# print(labels.shape)
# torch.save(labels.cpu(), 'labels.pt')
