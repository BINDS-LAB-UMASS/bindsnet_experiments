import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.analysis.plotting import plot_spikes, plot_input, plot_weights
from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import Network
from bindsnet.learning import Hebbian
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes
from bindsnet.utils import get_square_weights

from experiments import ROOT_DIR


def main(n_hidden=100, time=100, lr=5e-2, plot=False, gpu=False):
    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    network = Network()

    input_layer = Input(n=784, traces=True)
    hidden_layer = DiehlAndCookNodes(n=n_hidden, rest=0, reset=0, thresh=1, traces=True)
    output_layer = LIFNodes(n=784, rest=0, reset=0, thresh=1, traces=True)
    input_hidden_connection = Connection(
        input_layer, hidden_layer, wmin=0, wmax=1, norm=75, update_rule=Hebbian, nu=[0, lr]
    )
    hidden_hidden_connection = Connection(
        hidden_layer, hidden_layer, wmin=-500, wmax=0,
        w=-500 * torch.zeros(n_hidden, n_hidden) - torch.diag(torch.ones(n_hidden))
    )
    hidden_output_connection = Connection(
        hidden_layer, input_layer, wmin=0, wmax=1, norm=15, update_rule=Hebbian, nu=[lr, 0]
    )

    network.add_layer(input_layer, name='X')
    network.add_layer(hidden_layer, name='H')
    network.add_layer(output_layer, name='Y')
    network.add_connection(input_hidden_connection, source='X', target='H')
    network.add_connection(hidden_hidden_connection, source='H', target='H')
    network.add_connection(hidden_output_connection, source='H', target='Y')

    for layer in network.layers:
        monitor = Monitor(
            obj=network.layers[layer], state_vars=('s',), time=time
        )
        network.add_monitor(monitor, name=layer)

    dataset = MNIST(
        path=os.path.join(ROOT_DIR, 'data', 'MNIST'), shuffle=True, download=True
    )

    images, labels = dataset.get_train()
    images = images.view(-1, 784)
    images /= 4
    labels = labels.long()

    spikes_ims = None
    spikes_axes = None
    weights1_im = None
    weights2_im = None
    inpt_ims = None
    inpt_axes = None

    for image, label in zip(images, labels):
        spikes = poisson(image, time=time, dt=network.dt)
        inpts = {'X': spikes}
        clamp = {'Y': spikes}
        unclamp = {'Y': ~spikes}

        network.run(
            inpts=inpts, time=time, clamp=clamp, unclamp=unclamp
        )

        if plot:
            spikes = {
                l: network.monitors[l].get('s') for l in network.layers
            }
            spikes_ims, spikes_axes = plot_spikes(
                spikes, ims=spikes_ims, axes=spikes_axes
            )

            inpt = spikes['X'].float().mean(1).view(28, 28)
            rcstn = spikes['Y'].float().mean(1).view(28, 28)

            inpt_axes, inpt_ims = plot_input(
                inpt, rcstn, label=label, axes=inpt_axes, ims=inpt_ims
            )

            w1 = get_square_weights(
                network.connections['X', 'H'].w.view(784, n_hidden), int(np.ceil(np.sqrt(n_hidden))), 28
            )
            w2 = get_square_weights(
                network.connections['H', 'Y'].w.view(n_hidden, 784).t(), int(np.ceil(np.sqrt(n_hidden))), 28
            )

            weights1_im = plot_weights(
                w1, wmin=0, wmax=1, im=weights1_im
            )
            weights2_im = plot_weights(
                w2, wmin=0, wmax=1, im=weights2_im
            )

            plt.pause(0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_hidden', type=int, default=100)
    parser.add_argument('--time', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-2)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(plot=False, gpu=False)
    args = parser.parse_args()
    args = vars(args)

    main(**args)
