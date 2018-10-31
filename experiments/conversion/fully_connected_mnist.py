import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

import torch
import torch.nn as nn
import torch.optim as optim

from bindsnet.datasets import MNIST
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class FullyConnectedNetwork(nn.Module):
    # language=rst
    """
    Simply fully-connected network implemented in PyTorch.
    """
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def main(n_epochs=5, batch_size=100, time=50, update_interval=50, plot=False):
    print()
    print('Creating and training the ANN...')
    print()

    # Create and train an ANN on the MNIST dataset.
    ANN = FullyConnectedNetwork()

    # Get the MNIST data.
    images, labels = MNIST('../../data/MNIST', download=True).get_train()
    images /= images.max()  # Standardizing to [0, 1].
    images = images.view(-1, 784)
    labels = labels.long()

    # Specify optimizer and loss function.
    optimizer = optim.Adam(params=ANN.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the ANN.
    batches_per_epoch = int(images.size(0) / batch_size)
    for i in range(n_epochs):
        losses = []
        accuracies = []
        for j in range(batches_per_epoch):
            batch_idxs = torch.from_numpy(
                np.random.choice(np.arange(images.size(0)), size=batch_size, replace=False)
            )
            im_batch = images[batch_idxs]
            label_batch = labels[batch_idxs]

            outputs = ANN.forward(im_batch)
            loss = criterion(outputs, label_batch)
            predictions = torch.max(outputs, 1)[1]
            correct = (label_batch == predictions).sum().float() / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accuracies.append(correct.item())

        print(f'Epoch: {i+1} / {n_epochs}; Loss: {np.mean(losses):.4f}; Accuracy: {np.mean(accuracies) * 100:.4f}')

    print()
    print('Converting ANN to SNN...')

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(ANN, input_shape=(784,), data=images)

    for l in SNN.layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )

    spike_ims = None
    spike_axes = None
    correct = []

    print()
    print('Testing SNN on MNIST data...')
    print()

    # Test SNN on MNIST data.
    start = t()
    for i in range(images.size(0)):
        if i > 0 and i % update_interval == 0:
            print(
                f'Progress: {i} / {images.size(0)}; Elapsed: {t() - start:.4f}; Accuracy: {np.mean(correct) * 100:.4f}'
            )
            start = t()

        SNN.run(inpts={'Input': images[i].repeat(time, 1, 1)}, time=time)

        spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors}
        prediction = torch.softmax(voltages['fc3'].sum(1), 0).argmax()
        correct.append((prediction == labels[i]).item())

        SNN.reset_()

        if plot:
            spikes = {k: spikes[k].cpu() for k in spikes}
            spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
            plt.pause(1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=50)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = vars(parser.parse_args())

    main(**args)
