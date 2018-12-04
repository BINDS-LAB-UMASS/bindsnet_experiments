import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


filters = [100, 200, 300, 400, 500]

means = np.zeros(5)
stds = np.zeros(5)
for i, n_filters in enumerate(filters):
    spikes = torch.load(
        os.path.join(
            ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
            f'train_0_16_2_{n_filters}_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
        )
    )
    means[i] = spikes.sum(1).float().mean().item()
    stds[i] = spikes.sum(1).float().std().item()
    
plt.plot(filters, means, label='LC-SNN (train)')
plt.fill_between(filters, means + stds, means - stds, alpha=0.2)

means = np.zeros(5)
stds = np.zeros(5)
for i, n_filters in enumerate(filters):
    spikes = torch.load(
        os.path.join(
            ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
            f'test_0_16_2_{n_filters}_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
        )
    )
    means[i] = spikes.sum(1).float().mean().item()
    stds[i] = spikes.sum(1).float().std().item()

plt.plot(filters, means, label='LC-SNN (test)')
plt.fill_between(filters, means + stds, means - stds, alpha=0.2)

plt.legend()
plt.xlabel('No. filters')
plt.ylabel('Average no. spikes')
plt.savefig(os.path.join(ROOT_DIR, 'figures', 'lcsnn_dac_spikes_comp.png'))
plt.show()
