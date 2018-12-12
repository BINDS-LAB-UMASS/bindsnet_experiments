import os
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


plt.rc('text', usetex=True)

probs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

synapse_means = np.array([93.38, 92.45, 91.29, 89.96, 88.41, 86.51, 83.04, 77.57, 69.74, 56.09, 9.8])
synapse_stds = np.array([0.12, 0.12, 0.23, 0.46, 0.5, 0.41, 0.49, 0.67, 1.09, 2.1, 0])

dac_synapse_means = np.array([93.46, 91.73, 89.88, 87.62, 85.47, 82.49, 78.33, 72.21, 63.17, 48.77, 9.96])
dac_synapse_stds = np.array([1.07, 1.59, 1.37, 1.73, 2.15, 2.42, 3.11, 5.51, 3.18, 5.91, 0])

plt.plot(probs, synapse_means, 'o-', color='blue', label='LC-SNN - Deleting synapses')
plt.fill_between(
    probs, synapse_means - synapse_stds, synapse_means + synapse_stds, alpha=0.2, color='blue'
)
plt.plot(probs, dac_synapse_means, 'o:', color='orange', label='baseline SNN - Deleting synapses')
plt.fill_between(
    probs, dac_synapse_means - dac_synapse_stds, dac_synapse_means + dac_synapse_stds, alpha=0.2, color='orange'
)

neuron_means = np.array([93.42, 93.12, 92.92, 92.66, 92.06, 91.59, 90.61, 88.69, 85.35, 77.04, 9.8])
neuron_stds = np.array([0.14, 0.13, 0.11, 0.08, 0.1, 0.17, 0.23, 0.66, 1.16, 2.94, 0])

dac_neuron_means = np.array([93.44, 92.98, 92.43, 91.94, 91.1, 89.79, 88.08, 84.74, 79.27, 66.77, 10.13])
dac_neuron_stds = np.array([1.1, 1.3, 1.16, 1.3, 1.66, 1.65, 1.62, 2.16, 3.93, 5.03, 0])

plt.plot(probs, neuron_means, '+-', color='blue', label='LC-SNN - Deleting neurons')
plt.fill_between(
    probs, neuron_means - dac_neuron_stds, dac_neuron_means + dac_neuron_stds, alpha=0.2, color='blue'
)
plt.plot(probs, dac_neuron_means, '+:', color='orange', label='baseline SNN - Deleting neurons')
plt.fill_between(
    probs, dac_neuron_means - dac_neuron_stds, dac_neuron_means + dac_neuron_stds, alpha=0.2, color='orange'
)

plt.legend()
plt.grid()
plt.xticks(np.linspace(0, 1, 11))
plt.yticks(range(0, 110, 10))
plt.xlabel(r'$p_\textrm{delete}$ / $p_\textrm{remove}$')
plt.ylabel('Average test accuracy')
plt.title('LC-SNN robustness test comparison')

plt.savefig(
    os.path.join(
        ROOT_DIR, 'figures', 'lcsnn_robustness.png'
    )
)

plt.show()
