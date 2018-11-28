import os
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


lcsnn_params = 1e6 * np.array([
    0.0324, 0.0648, 0.0927, 0.1296, 0.1944, 0.4608, 0.576, 0.6912, 0.8064, 0.9216, 1.04, 1.15
])
lcsnn_neurons = np.array([
    25*9, 50*9, 75*9, 100*9, 150*9, 200*9, 250*9, 300*9, 350*9, 400*9, 450*9, 500*9
])
lcsnn_accuracy = np.array([
    84.01, 88.21, 90.20, 91.68, 92.71, 93.14, 93.71, 93.66, 94.05, 94.27, 94.49, 94.59
])
lcsnn_std = np.array([
    2.28, 1.79, 1.45, 1.31, 1.25, 1.13, 0.68, 0.65, 0.56, 0.69, 0.67, 0.61
])

dac_params = np.array([
    100 * 784, 400 * 784, 1600 * 784, 6400 * 784
])
dac_neurons = np.array([
    100, 400, 1600, 6400
])
dac_accuracy = np.array([
    82.90, 87.00, 91.90, 95.00
])

plt.figure()

plt.plot(lcsnn_params, lcsnn_accuracy, 'o-', color='darkorange', label='LC-SNN')
plt.fill_between(lcsnn_params, lcsnn_accuracy - lcsnn_std, lcsnn_accuracy + lcsnn_std, color='orange', alpha=0.2)
plt.plot(dac_params, dac_accuracy, 'o-', color='cornflowerblue', label='Diehl & Cook SNN')
plt.xlabel('No. parameters')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.legend()
plt.grid()
plt.savefig(os.path.join(ROOT_DIR, 'figures', 'lcsnn_dac_synapses.png'))

plt.figure()

plt.plot(lcsnn_neurons, lcsnn_accuracy, 'o-', color='darkorange', label='LC-SNN')
plt.fill_between(lcsnn_neurons, lcsnn_accuracy - lcsnn_std, lcsnn_accuracy + lcsnn_std, color='orange', alpha=0.2)
plt.plot(dac_neurons, dac_accuracy, 'o-', color='cornflowerblue', label='Diehl & Cook SNN')
plt.xlabel('No. neurons')
plt.ylabel('Test accuracy')
plt.xscale('log')
plt.legend()
plt.grid()
plt.savefig(os.path.join(ROOT_DIR, 'figures', 'lcsnn_dac_neurons.png'))

plt.show()
