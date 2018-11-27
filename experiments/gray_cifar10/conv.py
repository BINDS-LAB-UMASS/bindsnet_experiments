import os
import torch
import argparse
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.analysis.plotting import plot_conv2d_weights, plot_spikes, plot_input, plot_voltages
from bindsnet.datasets import CIFAR10
from bindsnet.encoding import poisson_loader
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.network.topology import Conv2dConnection, Connection

print()


def main(seed=0, n_train=60000, n_test=10000, kernel_size=16, stride=4, n_filters=25, padding=0, inhib=500, lr=0.01,
         lr_decay=0.99, time=50, dt=1, intensity=1, progress_interval=10, update_interval=250, train=True, plot=False,
         gpu=False):

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    if not train:
        update_interval = n_test

    if kernel_size == 32:
        conv_size = 1
    else:
        conv_size = int((32 - kernel_size + 2 * padding) / stride) + 1

    per_class = int((n_filters * conv_size * conv_size) / 10)

    # Build network.
    network = Network()
    input_layer = Input(n=1024, shape=(1, 1, 32, 32), traces=True)

    conv_layer = DiehlAndCookNodes(
        n=n_filters * conv_size * conv_size, shape=(1, n_filters, conv_size, conv_size), traces=True
    )

    conv_conn = Conv2dConnection(
        input_layer, conv_layer, kernel_size=kernel_size, stride=stride, update_rule=PostPre,
        norm=0.4 * kernel_size ** 2, nu=[0, lr], wmin=0, wmax=1
    )

    w = -inhib * torch.ones(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
    for f in range(n_filters):
        for i in range(conv_size):
            for j in range(conv_size):
                w[f, i, j, f, i, j] = 0

    w = w.view(n_filters * conv_size ** 2, n_filters * conv_size ** 2)
    recurrent_conn = Connection(conv_layer, conv_layer, w=w)

    network.add_layer(input_layer, name='X')
    network.add_layer(conv_layer, name='Y')
    network.add_connection(conv_conn, source='X', target='Y')
    network.add_connection(recurrent_conn, source='Y', target='Y')

    # Voltage recording for excitatory and inhibitory layers.
    voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
    network.add_monitor(voltage_monitor, name='output_voltage')

    # Load CIFAR-10 data.
    dataset = CIFAR10(path=os.path.join('..', '..', 'data', 'CIFAR10'), download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images *= intensity
    images = images.mean(-1)

    # Lazily encode data as Poisson spike trains.
    data_loader = poisson_loader(data=images, time=time, dt=dt)

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    voltages = {}
    for layer in set(network.layers) - {'X'}:
        voltages[layer] = Monitor(network.layers[layer], state_vars=['v'], time=time)
        network.add_monitor(voltages[layer], name='%s_voltages' % layer)

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    voltage_ims = None
    voltage_axes = None

    # Train the network.
    print('Begin training.\n')
    start = t()

    for i in range(n_train):
        if i % progress_interval == 0:
            print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start))
            start = t()

            if train and i > 0:
                network.connections['X', 'Y'].nu[1] *= lr_decay

        # Get next input sample.
        sample = next(data_loader).unsqueeze(1).unsqueeze(1)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        # Optionally plot various simulation information.
        if plot:
            # inpt = inpts['X'].view(time, 1024).sum(0).view(32, 32)

            weights1 = conv_conn.w
            _spikes = {
                'X': spikes['X'].get('s').view(32 ** 2, time),
                'Y': spikes['Y'].get('s').view(n_filters * conv_size ** 2, time)
            }
            _voltages = {'Y': voltages['Y'].get('v').view(n_filters * conv_size ** 2, time)}

            # inpt_axes, inpt_ims = plot_input(
            #     images[i].view(32, 32), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims
            # )
            # voltage_ims, voltage_axes = plot_voltages(_voltages, ims=voltage_ims, axes=voltage_axes)

            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_conv2d_weights(weights1, im=weights_im)

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
    print('Training complete.\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_train', type=int, default=60000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--kernel_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--n_filters', type=int, default=25)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--inhib', type=float, default=500)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--dt', type=int, default=1.0)
    parser.add_argument('--intensity', type=float, default=0.5)
    parser.add_argument('--progress_interval', type=int, default=10)
    parser.add_argument('--update_interval', type=int, default=250)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()
    args = vars(args)

    main(**args)