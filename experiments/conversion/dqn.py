import argparse
import itertools
import sys

import numpy as np
import matplotlib.pyplot as plt

from time import time as _time

import torch
import torch.nn as nn

from bindsnet.conversion import ann_to_snn
from bindsnet.environment import GymEnvironment
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_input

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


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


def main(time=50, n_episodes=10, plot=False):
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

    print('Gathering observation data...')
    print()

    episode_rewards = np.zeros(n_episodes)
    noop_counter = 0
    total_t = 0
    states = []

    for i in range(n_episodes):
        print(f'Episode progress: {i + 1} / {n_episodes}')

        obs = environment.reset()
        state = torch.stack([obs] * 4, dim=2)

        for t in itertools.count():
            encoded = torch.tensor([0.25, 0.5, 0.75, 1]) * state
            encoded = torch.sum(encoded, dim=2)

            states.append(encoded)

            q_values = ANN(encoded.view([1, -1]))[0]
            action_probs, best_action = policy(q_values, 0)

            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            if action == 0:
                noop_counter += 1
            else:
                noop_counter = 0
            if noop_counter >= 20:
                action = np.random.choice(np.arange(len(action_probs)))
                noop_counter = 0

            next_obs, reward, done, _ = environment.step(action)
            next_state = torch.clamp(next_obs - obs, min=0)
            next_state = torch.cat(
                (state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2
            )

            episode_rewards[i] += reward
            total_t += 1

            if done:
                print(f'Step {t} ({total_t}) @ Episode {i}/{n_episodes}')
                print(f'Episode Reward: {episode_rewards[i]}')
                break

            state = next_state
            obs = next_obs

    states = torch.stack(states).view(-1, 6400)

    print()
    print('Converting ANN to SNN...')

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(ANN, input_shape=(6400,), data=states)

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
    episode_rewards = np.zeros(n_episodes)
    total_t = 0
    noop_counter = 0

    print()
    print('Testing SNN on Atari Breakout game...')
    print()

    # Test SNN on Atari Breakout.
    for i in range(n_episodes):
        obs = environment.reset()
        state = torch.stack([obs] * 4, dim=2)
        prev_life = 5

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

            if prev_life - info["ale.lives"] != 0:
                new_life = True
            else:
                new_life = False

            prev_life = info["ale.lives"]

            next_state = torch.clamp(next_obs - obs, min=0)
            next_state = torch.cat(
                (state[:, :, 1:], next_state.view([next_state.shape[0], next_state.shape[1], 1])), dim=2
            )

            episode_rewards[i] += reward
            total_t += 1

            if plot:
                # Get voltage recording.
                inpt = encoded_state.view(time, 6400).sum(0).view(80, 80)
                spike_ims, spike_axes = plot_spikes(
                    {layer: spikes[layer] for layer in spikes}, ims=spike_ims, axes=spike_axes
                )
                inpt_axes, inpt_ims = plot_input(state, inpt, ims=inpt_ims, axes=inpt_axes)
                plt.pause(1e-8)

            if done:
                print(f'Step {t} ({total_t}) @ Episode {i}/{n_episodes}')
                print(f'Episode Reward: {episode_rewards[i]}')
                break

            state = next_state
            obs = next_obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = vars(parser.parse_args())

    main(**args)

