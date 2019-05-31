import os
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn

from bindsnet.conversion import Permute, ann_to_snn
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.nodes import IFNodes

from experiments import ROOT_DIR
from experiments.misc.atari_wrappers import make_atari, wrap_deepmind


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'


def policy(q_values, eps):
    A = np.ones(len(q_values), dtype=float) * eps / len(q_values)
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


def main(seed=0, time=50, n_episodes=25, percentile=99.9, epsilon=0.05, node_type='subtractiveIF', game='breakout',
         model=None, normalize_on_spikes=False):

    params_path = os.path.join(ROOT_DIR, 'params', game, 'large_dqn_eps_greedy')

    name = ''.join([g.capitalize() for g in game.split('_')])
    environment = make_atari(name + 'NoFrameskip-v4', max_episode_steps=18000)
    environment = wrap_deepmind(environment, frame_stack=True, scale=False, clip_rewards=False, episode_life=False)

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
            self.fc1 = nn.Linear(7744, 512)
            self.relu4 = nn.ReLU()
            self.fc2 = nn.Linear(512, environment.action_space.n)

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

    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    print()
    print('Loading the trained ANN...')
    print()

    if model is None:
        model = "pytorch_" + game + ".pt"

    ANN = Net()
    ANN.load_state_dict(
        torch.load(
            '../../params/' + model
        )
    )

    f = f'{seed}_{n_episodes}_states.pt'

    if os.path.isfile(os.path.join(params_path, f)):
        print('Loading pre-gathered observation data...')

        states = torch.load(os.path.join(params_path, f))
    else:
        print('Gathering observation data...')
        print()

        episode_rewards = np.zeros(n_episodes)
        total_t = 0
        states = []

        for i in range(n_episodes):
            state = torch.tensor(environment.reset()).to(device).unsqueeze(0).permute(0, 3, 1, 2).float()

            for t in itertools.count():
                states.append(state)

                q_values = ANN(state)[0]

                probs, best_action = policy(q_values, epsilon)
                action = np.random.choice(np.arange(len(probs)), p=probs)

                state, reward, done, _ = environment.step(action)
                state = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2).float()
                state = state.to(device)

                episode_rewards[i] += reward
                total_t += 1

                if done:
                    print(f'Step {t} ({total_t}) @ Episode {i + 1} / {n_episodes}')
                    print(f'Episode Reward: {episode_rewards[i]}')

                    break

        states = torch.cat(states, dim=0)
        torch.save(states, os.path.join(params_path, f))

    print()
    print(f'Collected {states.size(0)} Atari game frames.')
    print()
    print('Converting ANN to SNN...')

    states = states.to(device)

    # Do ANN to SNN conversion.
    if node_type == 'subtractiveIF':
        SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile,
                         normalize_on_spikes=normalize_on_spikes)
    elif node_type == 'LIF':
        SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile,
                         node_type=LIFNodes, normalize_on_spikes=normalize_on_spikes, decay=1e-2 / 13.0, rest=0.0)
    elif node_type == 'IF':
        SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile, node_type=IFNodes,
                         normalize_on_spikes=normalize_on_spikes)

    model_name = '_'.join([str(x) for x in [seed, time, n_episodes, percentile, epsilon, game, normalize_on_spikes]])
    SNN.save(os.path.join(params_path, f'{model_name}_SNN.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time', type=int, default=100)
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--percentile', type=float, default=99)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--node_type', type=str, default='subtractiveIF')
    parser.add_argument('--game', type=str, default='breakout')
    parser.add_argument('--model', type=str)
    parser.add_argument('--normalize_on_spikes', dest='normalize_on_spikes', action='store_true')
    parser.set_defaults(normalize_on_spikes=False)
    args = vars(parser.parse_args())

    main(**args)
