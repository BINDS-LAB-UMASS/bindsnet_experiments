import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from experiments import ROOT_DIR
from bindsnet.network import load

map_location = 'gpu' if torch.cuda.is_available() else 'cpu'


def main():
    params_path = os.path.join(
        ROOT_DIR, 'params', 'mnist', 'crop_locally_connected',
        '2_12_4_100_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
    )
    network = load(params_path, map_location=map_location, learning=False)
    w = network.connections['X', 'Y'].w.view(400, 100, 9)
    locations = torch.zeros(12, 12, 3, 3).long()
    for c1 in range(3):
        for c2 in range(3):
            for k1 in range(12):
                for k2 in range(12):
                    location = c1 * 4 * 20 + c2 * 4 + k1 * 20 + k2
                    locations[k1, k2, c1, c2] = location

    locations = locations.view(144, 9)

    test_spikes_path = os.path.join(
        ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
        'test_2_12_4_100_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    print(w.size())

    for i in tqdm(range(1, 40)):
        f = os.path.join(test_spikes_path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=map_location)
        for j in range(spikes.size(0)):
            s = spikes[j].sum(0).view(100, 9)
            max_indices = torch.argmax(s, dim=0)
            filters = [w[locations[:, n], index, n] for n, index in zip(range(9), max_indices)]
            x = torch.zeros(12 * 3, 12 * 3)
            for k in range(3):
                for l in range(3):
                    x[k*12: k*12 + 12, l*12: l*12 + 12] = filters[k * 3 + l].view(12, 12)

            plt.ioff()
            plt.matshow(x, cmap='hot_r')
            plt.xticks(())
            plt.yticks(())
            plt.title(f'Label: {labels[j].item()}')
            plt.show()


if __name__ == '__main__':
    main()
