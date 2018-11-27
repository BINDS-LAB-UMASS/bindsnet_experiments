import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_input

from experiments import ROOT_DIR
from experiments.misc.atari_wrappers import make_atari, wrap_deepmind

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'

results_path = os.path.join(ROOT_DIR, 'results', 'breakout', 'large_dqn_eps_greedy')
params_path = os.path.join(ROOT_DIR, 'params', 'breakout', 'large_dqn_eps_greedy')

for p in [results_path, params_path]:
    if not os.path.isdir(p):
        os.makedirs(p)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(7744, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.relu1(self.conv1(x))

        # If the size is a square you can only specify a single number
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 7744)
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


def main(seed=0, time=50, n_episodes=25, n_snn_episodes=100, percentile=99.9, epsilon=0.05, plot=False):

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
    environment = wrap_deepmind(environment, frame_stack=True, scale=True)

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
            state = torch.tensor(environment.reset()).to(device).unsqueeze(0).permute(0, 3, 1, 2)

            for t in itertools.count():
                states.append(state)

                q_values = ANN(state)[0]
                probs, best_action = policy(q_values, epsilon)
                action = np.random.choice(np.arange(len(probs)), p=probs)

                state, reward, done, _ = environment.step(action)
                state = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2)
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

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states, percentile=percentile)

    for l in SNN.layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )

    spike_ims = None
    spike_axes = None
    inpt_ims = None
    inpt_axes = None

    new_life = True
    rewards = np.zeros(n_snn_episodes)
    total_t = 0
    noop_counter = 0

    print()
    print('Testing SNN on Atari Breakout game...')
    print()

    # Test SNN on Atari Breakout.
    for i in range(n_snn_episodes):
        state = torch.tensor(environment.reset()).to(device).unsqueeze(0).permute(0, 3, 1, 2)
        prev_life = 5

        for t in itertools.count():
            sys.stdout.flush()

            state = state.repeat(time, 1, 1, 1, 1)

            inpts = {'Input': state}
            SNN.run(inpts=inpts, time=time)

            spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
            voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors}
            probs, best_action = policy(voltages['9'].sum(1), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            if action == 0:
                noop_counter += 1
            else:
                noop_counter = 0

            if noop_counter >= 20:
                action = np.random.choice([0, 1, 2, 3])
                noop_counter = 0

            if new_life:
                action = 1

            next_state, reward, done, info = environment.step(action)
            next_state = torch.tensor(next_state).unsqueeze(0).permute(0, 3, 1, 2)

            if prev_life - info["ale.lives"] != 0:
                new_life = True
            else:
                new_life = False

            prev_life = info["ale.lives"]

            rewards[i] += reward
            total_t += 1

            SNN.reset_()

            if plot:
                # Get voltage recording.
                inpt = state.view(time, 4, 84, 84).sum(0).sum(0).view(84, 84)
                spike_ims, spike_axes = plot_spikes(
                    {layer: spikes[layer] for layer in spikes}, ims=spike_ims, axes=spike_axes
                )
                inpt_axes, inpt_ims = plot_input(inpt, inpt, ims=inpt_ims, axes=inpt_axes)
                plt.pause(1e-8)

            if done:
                print(f'Step {t} ({total_t}) @ Episode {i + 1} / {n_snn_episodes}')
                print(f'Episode Reward: {rewards[i]}')
                print()

                break

            state = next_state

    model_name = '_'.join([str(x) for x in [seed, time, n_episodes, n_snn_episodes, percentile, epsilon]])
    columns = [
        'seed', 'time', 'n_episodes', 'n_snn_episodes', 'percentile', 'epsilon', 'avg. reward', 'std. reward'
    ]
    data = [[
        seed, time, n_episodes, n_snn_episodes, percentile, epsilon, np.mean(rewards), np.std(rewards)
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
    parser.add_argument('--time', type=int, default=100)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--n_snn_episodes', type=int, default=100)
    parser.add_argument('--percentile', type=float, default=99)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = vars(parser.parse_args())

    main(**args)
