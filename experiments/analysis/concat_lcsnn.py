import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from experiments import ROOT_DIR


def main(seed=0, n_filters=100):
    path = os.path.join(
        '/', 'media', 'bigdrive2', 'djsaunde', 'crop_locally_connected',
        f'train_{seed}_12_4_{n_filters}_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    all_spikes = []
    all_labels = []
    for i in tqdm(range(1, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=torch.device('cpu'))
        summed = spikes.sum(1)
        all_spikes.append(summed)
        all_labels.append(labels)

    spikes = torch.cat(all_spikes, dim=0)
    labels = torch.cat(all_labels)
    torch.save(
        (spikes, labels),
        os.path.join(
            path, 'train_concat.pt'
        )
    )

    path = os.path.join(
        '/', 'media', 'bigdrive2', 'djsaunde', 'crop_locally_connected',
        f'test_{seed}_12_4_{n_filters}_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    all_spikes = []
    all_labels = []
    for i in tqdm(range(1, 40)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=torch.device('cpu'))
        summed = spikes.sum(1)
        all_spikes.append(summed)
        all_labels.append(labels)

    spikes = torch.cat(all_spikes, dim=0)
    labels = torch.cat(all_labels)

    torch.save(
        (spikes, labels),
        os.path.join(
            path, 'test_concat.pt'
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_filters', type=int, default=100)
    args = parser.parse_args()
    args = vars(args)
    main(**args)

