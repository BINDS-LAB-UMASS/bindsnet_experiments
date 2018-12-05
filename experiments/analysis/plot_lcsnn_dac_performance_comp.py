import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


lcsnn_filters = np.array([25, 50, 100, 250, 500])
lcsnn_synapses = np.array([32400, 64800, 129600, 576000, 1150000])
lcsnn_neurons = np.array([225, 450, 900, 2250, 4500])
lcsnn_spikes = np.array([])
lcsnn_means = np.array([84.01, 88.21, 91.68, 93.71, 94.59])
lcsnn_stds = np.array([2.28, 1.79, 1.31, 0.68, 0.61])

brian_dac_neurons = np.array([100, 400, 1600, 6400])
brian_dac_synapses = np.array([784 * n for n in brian_dac_neurons])
brian_dac_means = np.array([82.9, 87.0, 91.9, 95.0])

bindsnet_dac_neurons = np.array([100, 400, 900, 1600, 2500, 3600, 4900, 6400])
bindsnet_dac_synapses = np.array([784 * n for n in bindsnet_dac_neurons])
bindsnet_dac_spikes = np.array([])
bindsnet_dac_means = np.array([])
bindsnet_dac_stds = np.array([])

# Synapses.
plt.figure()

plt.plot(lcsnn_synapses, lcsnn_means, 'o-', label='LC-SNN')
plt.fill_between(
    lcsnn_synapses, lcsnn_means - lcsnn_stds, lcsnn_means + lcsnn_stds, alpha=0.2
)

plt.plot(brian_dac_synapses, brian_dac_means, 'o-', label='BRIAN D&C')

# plt.plot(bindsnet_dac_synapses, bindsnet_dac_means, 'o-', label='LC-SNN')
# plt.fill_between(
#     bindsnet_dac_synapses, bindsnet_dac_means - bindsnet_dac_stds, bindsnet_dac_means + bindsnet_dac_stds, alpha=0.2
# )

plt.xlabel('No. synapses')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.legend()
plt.grid()

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

# plt.plot(bindsnet_dac_neurons, bindsnet_dac_means, 'o-', label='LC-SNN')
# plt.fill_between(
#     bindsnet_dac_neurons, bindsnet_dac_means - bindsnet_dac_stds, bindsnet_dac_means + bindsnet_dac_stds, alpha=0.2
# )

plt.xlabel('No. neurons')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.legend()
plt.grid()

plt.savefig(
    os.path.join(
        ROOT_DIR, 'figures', 'lcsnn_dac_neurons.png'
    )
)

# Spikes.
# plt.figure()
#
# plt.plot(lcsnn_spikes, lcsnn_means, 'o-', label='LC-SNN')
# plt.fill_between(
#     lcsnn_spikes, lcsnn_means - lcsnn_stds, lcsnn_means + lcsnn_stds, alpha=0.2
# )
#
# # plt.plot(bindsnet_dac_spikes, bindsnet_dac_means, 'o-', label='BindsNET D&C')
# # plt.fill_between(
# #     lcsnn_spikes, lcsnn_means - lcsnn_stds, lcsnn_means + lcsnn_stds, alpha=0.2
# # )
#
# plt.xlabel('No. spikes')
# plt.ylabel('Test accuracy')
# plt.xscale('log')
# plt.legend()
# plt.grid()
#
# plt.savefig(
#     os.path.join(
#         ROOT_DIR, 'figures', 'lcsnn_dac_spikes.png'
#     )
# )

plt.show()
