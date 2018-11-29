import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.linear_model import LogisticRegression

from bindsnet.analysis.plotting import plot_spikes, plot_locally_connected_weights, plot_weights
from experiments import ROOT_DIR

from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load_network


plot = True

path = os.path.join(
    ROOT_DIR, 'params', 'mnist', 'crop_locally_connected',
    '0_12_4_150_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250.pt'
)

network = load_network(file_name=path, learning=False)

for l in network.layers:
    network.layers[l].dt = 1
    network.layers[l].lbound = None

for m in network.monitors:
    network.monitors[m].record_length = 0

network.layers['Y'].theta_plus = 0
network.layers['Y'].theta_decay = 0

del network.connections['Y', 'Y']

n_classes = 10
time = 250
n_input = 400
n_neurons = int(np.prod(network.layers['Y'].shape))
update_interval = 100
progress_interval = 10

path = os.path.join(ROOT_DIR, 'data', 'MNIST')

dataset = MNIST(path=path, download=True, shuffle=True)
images, labels = dataset.get_train()
images = images[:, 4:-4, 4:-4].contiguous()
images = images.view(-1, n_input) * 0.5
labels = labels.long()

model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
model.coef_ = np.zeros([n_classes, n_neurons])
model.intercept_ = np.zeros(n_classes)
model.classes_ = np.arange(n_classes)

full_spike_record = torch.zeros(len(images), n_neurons)
locations = network.connections['X', 'Y'].locations

spike_ims = None
spike_axes = None
weights_im = None
weights2_im = None
weights3_im = None

start = t()
for i in range(len(images)):
    if i % progress_interval == 0:
        print(f'Progress: {i} / {len(images)} ({t() - start:.4f} seconds)')
        start = t()

    image = images[i]
    inpts = {'X': poisson(datum=image, time=time, dt=1.0)}
    network.run(inpts=inpts, time=time)

    spikes = network.monitors['Y_spikes'].get('s').view(time, -1).sum(0)
    full_spike_record[i] = spikes

    if i % len(labels) == 0:
        current_labels = labels[-update_interval:]
        current_record = full_spike_record[-update_interval:]
    else:
        current_labels = labels[i % len(labels) - update_interval:i % len(labels)]
        current_record = full_spike_record[i % len(labels) - update_interval:i % len(labels)]

    if i % update_interval == 0 and i > 0:
        current_predictions = model.predict(current_record)
        current_accuracy = (current_predictions == current_labels).float().mean().item() * 100

        print()
        print(f'Accuracy after {i} examples: {current_accuracy:.2f}')
        print()

        model.fit(full_spike_record[:i], labels[:i])

    if plot:
        # Optionally plot various simulation information.
        _spikes = {
            'X': network.monitors['X_spikes'].get('s').view(n_input, time),
            'Y': network.monitors['Y_spikes'].get('s').view(n_neurons, time)
        }

        spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)

        w = network.connections[('X', 'Y')].w
        weights_im = plot_locally_connected_weights(
            w, 150, 12, (3, 3),
            locations, 20, im=weights_im
        )
        weights2_im = plot_weights(
            model.coef_, im=weights2_im
        )

        spikes = spikes.view(
            network.connections['X', 'Y'].n_filters,
            network.connections['X', 'Y'].conv_size[0] * network.connections['X', 'Y'].conv_size[1]
        )

        _, max_indices = torch.max(spikes, dim=0)

        w = w.view(400, 9 * 150)
        w = w[locations[:, max_indices % 9], max_indices]

        print(max_indices % 9)

        temp = torch.zeros(12 * 3, 12 * 3)
        for i in range(3):
            for j in range(3):
                temp[12 * j: 12 * (j + 1), 12 * i: 12 * (i + 1)] = w[:, i*3 + j].view(12, 12)

        weights3_im = plot_weights(
            temp, im=weights3_im
        )

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.


