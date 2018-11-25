import os
import argparse
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from bindsnet.environment import GymEnvironment

from experiments import ROOT_DIR

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'

results_path = os.path.join(ROOT_DIR, 'results', 'breakout', 'ann_dqn_eps_greedy')
params_path = os.path.join(ROOT_DIR, 'params', 'breakout', 'ann_dqn_eps_greedy')

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


def main(seed=0, n_episodes=25, epsilon=0.05):

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
    ANN = Network()
    ANN.load_state_dict(
        torch.load(
            os.path.join(ROOT_DIR, 'params', 'converted_dqn_time_difference_grayscale.pt')
        )
    )

    environment = GymEnvironment('BreakoutDeterministic-v4')

    print('Gathering observation data...')
    print()

    episode_rewards = np.zeros(n_episodes)
    noop_counter = 0
    total_t = 0
    states = []
    new_life = True
    prev_life = 5

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

    model_name = '_'.join([str(x) for x in [seed, n_episodes, epsilon]])
    columns = [
        'seed', 'n_episodes', 'epsilon', 'avg. reward', 'std. reward'
    ]
    data = [[
        seed, n_episodes, epsilon, np.mean(episode_rewards), np.std(episode_rewards)
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

    torch.save(episode_rewards, os.path.join(results_path, f'{model_name}_episode_rewards.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.05)
    args = vars(parser.parse_args())

    main(**args)
