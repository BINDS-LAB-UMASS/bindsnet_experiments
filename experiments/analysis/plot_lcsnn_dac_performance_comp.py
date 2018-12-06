import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


lcsnn_filters = np.array([25, 50, 100, 250, 500])
lcsnn_synapses = np.array([32400, 64800, 129600, 576000, 1150000])
lcsnn_neurons = np.array([225, 450, 900, 2250, 4500])
lcsnn_means = np.array([84.01, 88.21, 91.68, 93.71, 94.59])
lcsnn_stds = np.array([2.28, 1.79, 1.31, 0.68, 0.61])

brian_dac_neurons = np.array([100, 400, 1600, 6400])
brian_dac_synapses = np.array([784 * n for n in brian_dac_neurons])
brian_dac_means = np.array([82.9, 87.0, 91.9, 95.0])

bindsnet_dac_neurons = np.array([100, 225, 400, 625, 900, 1225, 1600, 2025, 2500])  # , 3600, 4900, 6400])
bindsnet_dac_synapses = np.array([784 * n for n in bindsnet_dac_neurons])
bindsnet_dac_means = np.array([87.71, 90.76, 91.83, 91.51, 90.62, 89.83, 89.20, 88.88, 88.71])
bindsnet_dac_stds = np.array([0.52, 0.45, 0.46, 0.37, 0.93, 0.76, 0.33, 0.38, 0.70])

# Synapses.
plt.figure()

plt.plot(lcsnn_synapses, lcsnn_means, 'o-', label='LC-SNN')
plt.fill_between(
    lcsnn_synapses, lcsnn_means - lcsnn_stds, lcsnn_means + lcsnn_stds, alpha=0.2
)

plt.plot(brian_dac_synapses, brian_dac_means, 'o-', label='BRIAN D&C')

plt.plot(bindsnet_dac_synapses, bindsnet_dac_means, 'o-', label='LC-SNN')
plt.fill_between(
    bindsnet_dac_synapses, bindsnet_dac_means - bindsnet_dac_stds, bindsnet_dac_means + bindsnet_dac_stds, alpha=0.2
)

plt.xlabel('No. synapses')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.legend()
plt.grid()
plt.title('Performance comparison by synapses')

plt.savefig(
    os.path.join(
        ROOT_DIR, 'figures', 'lcsnn_dac_synapses.png'
    )
)

# Neurons.
plt.figure()

plt.plot(lcsnn_neurons, lcsnn_means, 'o-', label='LC-SNN')
plt.fill_between(
    lcsnn_neurons, lcsnn_means - lcsnn_stds, lcsnn_means + lcsnn_stds, alpha=0.2
)

plt.plot(brian_dac_neurons, brian_dac_means, 'o-', label='BRIAN D&C')

plt.plot(bindsnet_dac_neurons, bindsnet_dac_means, 'o-', label='LC-SNN')
plt.fill_between(
    bindsnet_dac_neurons, bindsnet_dac_means - bindsnet_dac_stds, bindsnet_dac_means + bindsnet_dac_stds, alpha=0.2
)

plt.xlabel('No. neurons')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.legend()
plt.grid()
plt.title('Performance comparison by neurons')

plt.savefig(
    os.path.join(
        ROOT_DIR, 'figures', 'lcsnn_dac_neurons.png'
    )
)

# Spikes.
plt.figure()

filters = [25, 50, 100, 250, 500]
neurons = [9 * f for f in filters]

# means = np.zeros(len(neurons))
# stds = np.zeros(len(neurons))
# for i, n_filters in enumerate(filters):
#     spikes = torch.load(
#         os.path.join(
#             ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
#             f'train_0_16_2_{n_filters}_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
#         )
#     )
#     means[i] = spikes.sum(1).float().mean().item()
#     stds[i] = spikes.sum(1).float().std().item()
#
# plt.plot(neurons, means, label='LC-SNN (train)')
# plt.fill_between(neurons, means + stds, means - stds, alpha=0.2)

means = np.zeros(len(lcsnn_filters))
stds = np.zeros(len(lcsnn_filters))
for i, n_filters in enumerate(lcsnn_filters):
    spikes = torch.load(
        os.path.join(
            ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
            f'test_0_16_2_{n_filters}_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
        )
    )
    means[i] = spikes.sum(1).float().mean().item()

plt.plot(means, lcsnn_means, 'o-', label='LC-SNN')  # (test)')
plt.fill_between(means, lcsnn_means + lcsnn_stds, lcsnn_means - lcsnn_stds, alpha=0.2)

# means = np.zeros(len(neurons))
# stds = np.zeros(len(neurons))
# for i, n_neurons in enumerate(neurons):
#     spikes = torch.load(
#         os.path.join(
#             ROOT_DIR, 'spikes', 'mnist', 'diehl_and_cook_2015',
#             f'train_0_{n_neurons}_60000_500.0_0.01_0.99_250_1_0.05_1e-07_0.5_10_250.pt'
#         )
#     )
#     means[i] = spikes.sum(1).float().mean().item()
#     stds[i] = spikes.sum(1).float().std().item()
#
# plt.plot(neurons, means, label='baseline SNN (train)')
# plt.fill_between(neurons, means + stds, means - stds, alpha=0.2)

means = np.zeros(len(bindsnet_dac_neurons))
stds = np.zeros(len(bindsnet_dac_neurons))
for i, n_neurons in enumerate(bindsnet_dac_neurons):
    spikes = torch.load(
        os.path.join(
            ROOT_DIR, 'spikes', 'mnist', 'diehl_and_cook_2015',
            f'test_0_{n_neurons}_60000_10000_500.0_0.01_0.99_250_1_0.05_1e-07_0.5_10_250.pt'
        )
    )
    means[i] = spikes.sum(1).float().mean().item()

plt.plot(means, bindsnet_dac_means, 'o-', label='baseline SNN')  # (test)')
plt.fill_between(
    means, bindsnet_dac_means + bindsnet_dac_stds, bindsnet_dac_means - bindsnet_dac_stds, alpha=0.2
)

plt.legend()
plt.xlabel('Average no. spikes')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.grid()
plt.title('Performance comparison by spikes')

plt.savefig(
    os.path.join(
        ROOT_DIR, 'figures', 'lcsnn_dac_spikes.png'
    )
)

plt.show()
