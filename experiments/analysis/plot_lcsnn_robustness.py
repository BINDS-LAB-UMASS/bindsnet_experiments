import os
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


plt.rc('text', usetex=True)

probs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

synapse_means = np.array([92.56, 90.70, 88.91, 86.39, 83.67, 79.88, 72.77, 68.64, 56.34, 48.45, 10.34])
synapse_stds = np.array([0.06, 0.14, 0.23, 0.32, 0.33, 0.56, 0.78, 1.11, 1.26, 2.16, 0.3]) * 3

plt.plot(probs, synapse_means, 'o:', label='Deleting synapses')
plt.fill_between(probs, synapse_means - synapse_stds, synapse_means + synapse_stds, alpha=0.2)

neuron_means = np.array([92.56, 92.24, 92.08, 91.09, 89.36, 85.28, 82.01, 77.05, 73.55, 69.76, 10.34])
neuron_stds = np.array([0.06, 0.11, 0.18, 0.21, 0.32, 0.64, 1.07, 1.18, 1.67, 1.84, 0.3]) * 3

plt.plot(probs, neuron_means, 'o:', label='Deleting neurons')
plt.fill_between(probs, neuron_means - neuron_stds, neuron_means + neuron_stds, alpha=0.2)

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
