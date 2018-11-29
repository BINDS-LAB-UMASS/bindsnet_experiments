import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR

plt.rcParams['figure.figsize'] = (10, 6)


def window(size=5):
    return np.ones(size) / float(size)


for n_filters in [25, 50, 100, 250, 500]:
    curve = torch.load(
        os.path.join(
            ROOT_DIR, 'curves', 'mnist', 'crop_locally_connected',
            f'train_0_12_4_{n_filters}_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
        ), map_location='cpu'
    )[0]['ngram']

    plt.plot(np.convolve(curve, window(), 'same'), label=f'LC-SNN ({n_filters} filters)')

plt.xlim([0, 235])
plt.xticks(range(0, 240, 20), range(0, 60, 5))
plt.yticks(range(0, 110, 10))
plt.xlabel('No. training examples in thousands')
plt.ylabel('Estimated test accuracy')

plt.legend()
plt.grid()
plt.show()
