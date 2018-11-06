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

params_path = os.path.join('..', '..', 'params', 'cifar10_fully_conv')
if not os.path.isdir(params_path):
    os.makedirs(params_path)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(in_features=9216, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        return self.f(x)


def main(seed=0, n_epochs=1, batch_size=100, time=50, update_interval=50,
         n_examples=1000, percentile=100, plot=False, save=True):

    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    print()
    print('Loading CIFAR-10 data...')

    # Get the CIFAR-10 data.
    images, labels = CIFAR10('../../data/CIFAR10', download=True).get_train()
    images /= images.max()  # Standardizing to [0, 1].
    images = images.permute(0, 3, 1, 2)
    labels = labels.long()

    test_images, test_labels = CIFAR10('../../data/CIFAR10', download=True).get_test()
    test_images /= test_images.max()  # Standardizing to [0, 1].
    test_images = test_images.permute(0, 3, 1, 2)
    test_labels = test_labels.long()

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()

    model_name = '_'.join([
        str(x) for x in [seed, n_epochs, batch_size, time, update_interval, n_examples]
    ])

    ANN = LeNet()

    # Specify loss function.
    criterion = nn.CrossEntropyLoss()
    if save and os.path.isfile(os.path.join(params_path, model_name + '.pt')):
        print()
        print('Loading trained ANN from disk...')
        ANN.load_state_dict(torch.load(os.path.join(params_path, model_name + '.pt')))

        if torch.cuda.is_available():
            ANN = ANN.cuda()
    else:
        print()
        print('Creating and training the ANN...')
        print()

        # Specify optimizer.
        optimizer = optim.Adam(params=ANN.parameters(), lr=1e-3, weight_decay=1e-4)

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
                accuracies.append(correct.item() * 100)

            outputs = ANN.forward(test_images)
            test_loss = criterion(outputs, test_labels).item()
            predictions = torch.max(outputs, 1)[1]
            test_accuracy = ((test_labels == predictions).sum().float() / test_labels.numel()).item() * 100

            train_loss = np.mean(losses)
            train_acc = np.mean(accuracies)

            print(f'Epoch: {i+1} / {n_epochs}; Train Loss: {train_loss:.4f}; Train Accuracy: {train_acc:.4f}')
            print(f'\tTest Loss: {test_loss:.4f}; Test Accuracy: {test_accuracy:.4f}')

        if save:
            torch.save(ANN.state_dict(), os.path.join(params_path, model_name + '.pt'))

    outputs = ANN.forward(images)
    loss = criterion(outputs, labels)
    predictions = torch.max(outputs, 1)[1]
    accuracy = ((labels == predictions).sum().float() / labels.numel()).item() * 100

    print()
    print(f'(Post training) Training Loss: {loss:.4f}; Training Accuracy: {accuracy:.4f}')

    print()
    print('Converting ANN to SNN...')

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(
        ANN, input_shape=(1, 3, 32, 32), data=images[:n_examples], percentile=percentile
    )

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

        inpts = {'Input': images[i].repeat(time, 1, 1, 1, 1)}
        SNN.run(inpts=inpts, time=time)

        spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors}
        prediction = torch.softmax(voltages['9'].sum(1), 0).argmax()
        correct.append((prediction == labels[i]).item())

        SNN.reset_()

        if plot:
            inpts = {'Input': inpts['Input'].view(time, -1).t()}
            spikes = {**inpts, **spikes}
            spike_ims, spike_axes = plot_spikes(
                {k: spikes[k].cpu() for k in spikes}, ims=spike_ims, axes=spike_axes
            )
            plt.pause(1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=50)
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--percentile', type=float, default=99)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(plot=False, save=True)
    args = vars(parser.parse_args())

    main(**args)
