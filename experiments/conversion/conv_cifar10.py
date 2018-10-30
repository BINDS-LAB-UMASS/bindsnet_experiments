import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

import torch
import torch.nn as nn
import torch.optim as optim

from bindsnet.datasets import CIFAR10
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

params_path = os.path.join('..', '..', 'params', 'cifar10_conversion')
if not os.path.isdir(params_path):
    os.makedirs(params_path)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.mp1(out)
        out = self.relu2(self.conv2(out))
        out = self.mp2(out)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)

        return out


def main(n_epochs=1, batch_size=100, time=50, update_interval=50, n_examples=1000, plot=False, save=True):
    print()
    print('Loading CIFAR-10 data...')

    # Get the CIFAR-10 data.
    images, labels = CIFAR10('../../data/CIFAR10', download=True).get_train()
    images /= images.max()  # Standardizing to [0, 1].
    images = images.permute(0, 3, 1, 2)
    labels = labels.long()

    model_name = '_'.join([
        str(x) for x in [n_epochs, batch_size, time, update_interval, n_examples]
    ])

    ANN = LeNet()

    if save and os.path.isfile(os.path.join(params_path, model_name + '.pt')):
        print()
        print('Loading trained ANN from disk...')
        ANN.load_state_dict(torch.load(os.path.join(params_path, model_name + '.pt')))
    else:
        print()
        print('Creating and training the ANN...')
        print()

        # Specify optimizer and loss function.
        optimizer = optim.Adam(params=ANN.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        batches_per_epoch = int(images.size(0) / batch_size)

        # Train the ANN.
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

        if save:
            torch.save(ANN.state_dict(), os.path.join(params_path, model_name + '.pt'))

    print()
    print('Converting ANN to SNN...')

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(ANN, input_shape=(1, 3, 32, 32), data=images[:n_examples])

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
                f'Progress: {i} / {images.size(0)}; Elapsed: {t() - start:.4f}; Accuracy: {np.mean(correct) * 100:.4f}')
            start = t()

        print(images[i].repeat(time, 1, 1, 1).shape, SNN.layers['Input'].shape)

        SNN.run(inpts={'Input': images[i].repeat(time, 1, 1, 1)}, time=time)

        spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors}
        prediction = torch.softmax(voltages['fc3'].sum(1), 0).argmax()
        correct.append(prediction == labels[i])

        SNN.reset_()

        if plot:
            spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
            plt.pause(1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=50)
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(plot=False, save=True)
    args = vars(parser.parse_args())

    main(**args)
