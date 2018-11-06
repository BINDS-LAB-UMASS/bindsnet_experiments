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
from bindsnet.environment import GymEnvironment
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_input

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'

results_path = os.path.join('..', '..', 'results', 'breakout', 'dqn_determ')
params_path = os.path.join('..', '..', 'params', 'breakout', 'dqn_determ')

for p in [results_path, params_path]:
    if not os.path.isdir(p):
        os.makedirs(p)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(6400, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def policy(q_values, eps):
    A = np.ones(4, dtype=float) * eps / 4
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action


def main(seed=0, time=50, n_episodes=25, percentile=99.9, plot=False):

    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    epsilon = 0

    print()
    print('Loading the trained ANN...')
    print()

    # Create and train an ANN on the MNIST dataset.
    ANN = Network()
    ANN.load_state_dict(
        torch.load(
            '../../params/converted_dqn_time_difference_grayscale.pt'
        )
    )

    environment = GymEnvironment('BreakoutDeterministic-v4')

    f = f'{seed}_{n_episodes}_states.pt'
    if os.path.isfile(os.path.join(params_path, f)):
        print('Loading pre-gathered observation data...')

        states = torch.load(os.path.join(params_path, f))
    else:
        print('Gathering observation data...')
        print()

        episode_rewards = np.zeros(n_episodes)
        noop_counter = 0
        total_t = 0
        states = []

        for i in range(n_episodes):
            obs = environment.reset().to(device)
            state = torch.stack([obs] * 4, dim=2)

            for t in itertools.count():
                encoded = torch.tensor([0.25, 0.5, 0.75, 1]) * state
                encoded = torch.sum(encoded, dim=2)

                states.append(encoded)

                q_values = ANN(encoded.view([1, -1]))[0]
                probs, best_action = policy(q_values, epsilon)
                action = np.random.choice(np.arange(len(probs)), p=probs)

                if action == 0:
                    noop_counter += 1
                else:
                    noop_counter = 0

                if noop_counter >= 20:
                    action = np.random.choice([0, 1, 2, 3])
                    noop_counter = 0

                next_obs, reward, done, _ = environment.step(action)
                next_obs = next_obs.to(device)

                next_state = torch.clamp(next_obs - obs, min=0)
                next_state = torch.cat(
                    (state[:, :, 1:], next_state.view(
                        [next_state.shape[0], next_state.shape[1], 1]
                    )), dim=2
                )

                episode_rewards[i] += reward
                total_t += 1

                if done:
                    print(f'Step {t} ({total_t}) @ Episode {i + 1} / {n_episodes}')
                    print(f'Episode Reward: {episode_rewards[i]}')

                    break

                state = next_state
                obs = next_obs

        states = torch.stack(states).view(-1, 6400)

        torch.save(states, os.path.join(params_path, f))

    print()
    print(f'Collected {states.size(0)} Atari game frames.')
    print()
    print('Converting ANN to SNN...')

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(ANN, input_shape=(6400,), data=states, percentile=percentile)

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
    total_t = 0
    noop_counter = 0

    print()
    print('Testing SNN on Atari Breakout game...')
    print()

    # Test SNN on Atari Breakout.
    obs = environment.reset().to(device)
    state = torch.stack([obs] * 4, dim=2)
    prev_life = 5
    total_reward = 0

    for t in itertools.count():
        sys.stdout.flush()

        encoded_state = torch.tensor([0.25, 0.5, 0.75, 1]) * state
        encoded_state = torch.sum(encoded_state, dim=2)
        encoded_state = encoded_state.view([1, -1]).repeat(time, 1)

        inpts = {'Input': encoded_state}
        SNN.run(inpts=inpts, time=time)

        spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors}
        action = torch.softmax(voltages['fc2'].sum(1), 0).argmax()

        if action == 0:
            noop_counter += 1
        else:
            noop_counter = 0

        if noop_counter >= 20:
            action = np.random.choice([0, 1, 2, 3])
            noop_counter = 0

        if new_life:
            action = 1

        next_obs, reward, done, info = environment.step(action)
        next_obs = next_obs.to(device)

        if prev_life - info["ale.lives"] != 0:
            new_life = True
        else:
            new_life = False

        prev_life = info["ale.lives"]

        next_state = torch.clamp(next_obs - obs, min=0)
        next_state = torch.cat(
            (state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2
        )

        total_reward += reward
        total_t += 1

        SNN.reset_()

        if plot:
            # Get voltage recording.
            inpt = encoded_state.view(time, 6400).sum(0).view(80, 80)
            spike_ims, spike_axes = plot_spikes(
                {layer: spikes[layer] for layer in spikes}, ims=spike_ims, axes=spike_axes
            )
            inpt_axes, inpt_ims = plot_input(state, inpt, ims=inpt_ims, axes=inpt_axes)
            plt.pause(1e-8)

        if done:
            print(f'Episode Reward: {total_reward}')
            print()

            break

        state = next_state
        obs = next_obs

    model_name = '_'.join([str(x) for x in [seed, time, n_episodes, percentile]])
    columns = [
        'seed', 'time', 'n_episodes', 'percentile', 'reward'
    ]
    data = [[
        seed, time, n_episodes, percentile, total_reward
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--n_episodes', type=int, default=25)
    parser.add_argument('--percentile', type=float, default=99)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = vars(parser.parse_args())

    main(**args)
