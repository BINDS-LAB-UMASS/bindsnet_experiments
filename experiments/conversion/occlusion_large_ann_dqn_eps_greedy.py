import os
import argparse
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from experiments import ROOT_DIR
from bindsnet.conversion import Permute
from experiments.misc.atari_wrappers import make_atari, wrap_deepmind


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'

results_path = os.path.join(ROOT_DIR, 'results', 'breakout', 'occlusion_large_ann_dqn_eps_greedy')
params_path = os.path.join(ROOT_DIR, 'params', 'breakout', 'occlusion_large_ann_dqn_eps_greedy')

for p in [results_path, params_path]:
    if not os.path.isdir(p):
        os.makedirs(p)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.pad2 = nn.ConstantPad2d((1, 2, 1, 2), value=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.pad3 = nn.ConstantPad2d((1, 1, 1, 1), value=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.perm = Permute((0, 2, 3, 1))
        self.fc1 = nn.Linear(7744, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = x / 255.0
        x = self.relu1(self.conv1(x))
        x = self.pad2(x)
        x = self.relu2(self.conv2(x))
        x = self.pad3(x)
        x = self.relu3(self.conv3(x))
        x = self.perm(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def policy(q_values, eps):
    A = np.ones(4, dtype=float) * eps / 4
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


def main(seed=0, n_episodes=100, epsilon=0.05, occlusion=0):

    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    print()
    print('Loading the trained ANN...')
    print()

    # Create and train an ANN on the MNIST dataset.
    ANN = Net()
    ANN.load_state_dict(
        torch.load(
            '../../params/pytorch_breakout_dqn.pt'
        )
    )

    environment = make_atari('BreakoutNoFrameskip-v4')
    environment = wrap_deepmind(environment, frame_stack=True, scale=False, clip_rewards=False, episode_life=False)

    print('Gathering observation data...')
    print()

    rewards = np.zeros(n_episodes)
    noop_counter = 0
    total_t = 0
    states = []

    for i in range(n_episodes):
        state = torch.tensor(environment.reset()).to(device).unsqueeze(0).permute(0, 3, 1, 2).float()

        for t in itertools.count():
            states.append(state)

            q_values = ANN(state)[0]
            probs, best_action = policy(q_values, epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            if action == 0:
                noop_counter += 1
            else:
                noop_counter = 0

            if noop_counter >= 20:
                action = np.random.choice([0, 1, 2, 3])
                noop_counter = 0

            state, reward, done, _ = environment.step(action)
            state = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2).float()
            state = state.to(device)

            rewards[i] += reward
            total_t += 1

            if done:
                print(f'Step {t} ({total_t}) @ Episode {i + 1} / {n_episodes}')
                print(f'Episode Reward: {rewards[i]}')

                break

    model_name = '_'.join([str(x) for x in [seed, n_episodes, occlusion]])
    columns = [
        'seed', 'n_episodes', 'occlusion', 'avg. reward', 'std. reward'
    ]
    data = [[
        seed, n_episodes, occlusion, np.mean(rewards), np.std(rewards)
    ]]

    path = os.path.join(results_path, 'results.csv')
    if not os.path.isfile(path):
        df = pd.DataFrame(data=data, index=[model_name], columns=columns)
    else:
        df = pd.read_csv(path, index_col=0)

        if model_name not in df.index:
            df = df.append(pd.DataFrame(data=data, index=[model_name], columns=columns))
        else:
            df.loc[model_name] = data[0]

    df.to_csv(path, index=True)

    torch.save(rewards, os.path.join(results_path, f'{model_name}_episode_rewards.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--occlusion', type=int, default=0)
    args = vars(parser.parse_args())

    main(**args)
