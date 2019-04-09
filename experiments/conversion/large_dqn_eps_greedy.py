import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time as t_

import torch
import torch.nn as nn

from bindsnet.network.monitors import Monitor
from bindsnet.conversion import Permute, ann_to_snn
from bindsnet.analysis.plotting import plot_spikes, plot_input, plot_voltages
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


def main(seed=0, time=50, n_episodes=25, n_snn_episodes=100, percentile=99.9, epsilon=0.05, plot=False, node_type='subtractiveIF', ann=False, game='breakout', model=None):

    results_path = os.path.join(ROOT_DIR, 'results', game, 'large_dqn_eps_greedy')
    params_path = os.path.join(ROOT_DIR, 'params', game, 'large_dqn_eps_greedy')

    for p in [results_path, params_path]:
        if not os.path.isdir(p):
            os.makedirs(p)

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
            self.relu5 = nn.ReLU()

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
            x = self.relu5(x)
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

    if os.path.isfile(os.path.join(params_path, f)) and not ann:
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
                if not ann:
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

        if ann:
            model_name = '_'.join([str(x) for x in [seed, n_episodes, percentile, epsilon, game]])
            columns = [
                'seed', 'n_episodes', 'percentile', 'epsilon', 'avg. reward', 'std. reward', 'game'
            ]

            torch.save(episode_rewards, os.path.join(results_path, f'{model_name}_ann_episode_rewards.pt'))
            return
        else:
            states = torch.cat(states, dim=0)
            torch.save(states, os.path.join(params_path, f))

    print()
    print(f'Collected {states.size(0)} Atari game frames.')
    print()
    print('Converting ANN to SNN...')

    states = states.to(device)

    # Do ANN to SNN conversion.
    if node_type == 'subtractiveIF':
        SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile)
    elif node_type == 'LIF':
        SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile, node_type=LIFNodes, decay=1e-2 / 13.0, rest=0.0)
    elif node_type == 'IF':
        SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile, node_type=IFNodes)

    if plot:
        layers = ['Input', '1', '4', '7', '10', '12']
    else:
        layers = ['12']
    for l in layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )
        else:
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s'], time=time), name=l
            )

    spike_ims = None
    spike_axes = None
    inpt_ims = None
    inpt_axes = None
    voltage_ims = None
    voltage_axes = None

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

        start = t_()
        for t in itertools.count():
            print(f'Timestep {t} (elapsed {t_() - start:.2f})')
            start = t_()

            sys.stdout.flush()

            state = state.repeat(time, 1, 1, 1, 1)

            inpts = {'Input': state.float() / 255.0}

            SNN.run(inpts=inpts, time=time)

            spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
            voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors if not layer == 'Input'}
            probs, best_action = policy(voltages['12'].sum(1), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)





            next_state, reward, done, info = environment.step(action)
            next_state = torch.tensor(next_state).unsqueeze(0).permute(0, 3, 1, 2)


            rewards[i] += reward
            total_t += 1

            SNN.reset_()

            if plot:
                # Get voltage recording.
                inpt = state.view(time, 4, 84, 84).sum(0).sum(0).view(84, 84)
                spike_ims, spike_axes = plot_spikes(
                    {layer: spikes[layer] for layer in spikes}, ims=spike_ims, axes=spike_axes
                )
                voltage_ims, voltage_axes = plot_voltages(
                    {layer: voltages[layer].view(time, -1) for layer in voltages},
                    ims=voltage_ims, axes=voltage_axes
                )
                inpt_axes, inpt_ims = plot_input(inpt, inpt, ims=inpt_ims, axes=inpt_axes)
                plt.pause(1e-8)

            if done:
                print(f'Step {t} ({total_t}) @ Episode {i + 1} / {n_snn_episodes}')
                print(f'Episode Reward: {rewards[i]}')
                print()

                break

            state = next_state

    model_name = '_'.join([str(x) for x in [seed, time, n_episodes, n_snn_episodes, percentile, epsilon, game]])
    torch.save(rewards, os.path.join(results_path, f'{model_name}_episode_rewards.pt'))
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time', type=int, default=100)
    parser.add_argument('--n_episodes', type=int, default=1)
    parser.add_argument('--n_snn_episodes', type=int, default=100)
    parser.add_argument('--percentile', type=float, default=99)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--node_type', type=str, default='subtractiveIF')
    parser.add_argument('--ann', dest='ann', action='store_true')
    parser.add_argument('--game', type=str, default='breakout')
    parser.add_argument('--model', type=str)
    parser.set_defaults(plot=False, ann=False)
    args = vars(parser.parse_args())

    main(**args)
