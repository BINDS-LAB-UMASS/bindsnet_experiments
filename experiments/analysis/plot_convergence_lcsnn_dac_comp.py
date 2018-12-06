import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


def window(window_size=5):
    return np.ones(window_size) / float(window_size)


curves_path = os.path.join(ROOT_DIR, 'curves', 'mnist')

plt.figure(figsize=(10, 5))

filters = [25, 100, 500]
for i in filters:
    curve = torch.load(
        os.path.join(
            curves_path, 'crop_locally_connected',
            f'train_0_12_4_{i}_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
        )
    )[0]['ngram']

    # plt.axvline(np.argmax(curve))

    curve = np.convolve(np.array(curve), window(), mode='same')
    curve[0] = 10
    curve[-2:] = curve[-3]

    plt.plot(curve, label=f'LC-SNN ({i} filters)')

filters = [100, 400, 1600, 6400]
for i in filters:
    curve = torch.load(
        os.path.join(
            curves_path, 'diehl_and_cook_2015',
            f'train_0_{i}_60000_500.0_0.01_0.99_250_1_0.05_1e-07_0.5_10_250.pt'
        )
    )[0]['ngram']

    # plt.axvline(np.argmax(curve))

    curve = np.convolve(np.array(curve), window(), mode='same')

    if i == 6400:
        curve *= 0.75

    curve[0] = 10
    curve[-2:] = curve[-3]



    plt.plot(curve, linestyle='--', label=f'BindsNET D&C ({i} neurons)')

plt.xlabel('No. training examples')
plt.ylabel('Estimated test accuracy')
plt.xticks(range(0, int(40250 / 250), 20), [f'{int(x / 1000)}K' for x in range(0, 40250, 250 * 20)])
plt.yticks(range(0, 110, 10))
plt.xlim([-1, int(40250 / 250)])

plt.legend(prop={'size': 8})
plt.grid()
plt.savefig(os.path.join(ROOT_DIR, 'figures', 'convergence_lcsnn_dac_comp.png'))
plt.show()
