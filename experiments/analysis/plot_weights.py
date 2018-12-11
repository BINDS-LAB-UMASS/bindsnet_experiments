import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import plot_weights, plot_locally_connected_weights, plot_conv2d_weights

from experiments import ROOT_DIR
from experiments.analysis import download_params


def main(model='diehl_and_cook_2015', data='mnist', param_string=None, cmap='hot_r', p_destroy=0, p_delete=0):
    assert param_string is not None, 'Pass "--param_string" argument on command line or main method.'

    f = os.path.join(ROOT_DIR, 'params', data, model, f'{param_string}.pt')
    if not os.path.isfile(f):
        print('File not found locally. Attempting download from swarm2 cluster.')
        download_params.main(model=model, data=data, param_string=param_string)

    network = torch.load(open(f, 'rb'))

    if data in ['mnist', 'letters', 'fashion_mnist']:
        if model in ['diehl_and_cook_2015', 'two_level_inhibition', 'real_dac', 'unclamp_dac']:
            params = param_string.split('_')
            n_sqrt = int(np.ceil(np.sqrt(int(params[1]))))
            side = int(np.sqrt(network.layers['X'].n))

            w = network.connections['X', 'Y'].w
            w = get_square_weights(w, n_sqrt, side)
            plot_weights(w, cmap=cmap)

        elif model in ['conv']:
            w = network.connections['X', 'Y'].w
            plot_conv2d_weights(w, wmax=network.connections['X', 'Y'].wmax, cmap=cmap)

        elif model in ['fully_conv', 'locally_connected', 'crop_locally_connected', 'bern_crop_locally_connected']:
            params = param_string.split('_')
            kernel_size = int(params[1])
            stride = int(params[2])
            n_filters = int(params[3])

            if model in ['crop_locally_connected', 'bern_crop_locally_connected']:
                crop = int(params[4])
                side_length = 28 - crop * 2
            else:
                side_length = 28

            if kernel_size == side_length:
                conv_size = 1
            else:
                conv_size = int((side_length - kernel_size) / stride) + 1

            locations = torch.zeros(kernel_size, kernel_size, conv_size, conv_size).long()
            for c1 in range(conv_size):
                for c2 in range(conv_size):
                    for k1 in range(kernel_size):
                        for k2 in range(kernel_size):
                            location = c1 * stride * side_length + c2 * stride + k1 * side_length + k2
                            locations[k1, k2, c1, c2] = location

            locations = locations.view(kernel_size ** 2, conv_size ** 2)

            w = network.connections[('X', 'Y')].w

            mask = torch.bernoulli(p_destroy * torch.ones(w.size())).byte()
            w[mask] = 0

            mask = torch.bernoulli(p_delete * torch.ones(w.size(1))).byte()
            w[:, mask] = 0

            plot_locally_connected_weights(
                w, n_filters, kernel_size, conv_size, locations, side_length, wmin=w.min(), wmax=w.max(), cmap=cmap
            )

        elif model in ['backprop']:
            w = network.connections['X', 'Y'].w
            weights = [
                w[:, i].view(28, 28) for i in range(10)
            ]
            w = torch.zeros(5 * 28, 2 * 28)
            for i in range(5):
                for j in range(2):
                    w[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = weights[i + j * 5]

            plot_weights(w, wmin=-1, wmax=1, cmap=cmap)

        elif model in ['two_layer_backprop']:
            params = param_string.split('_')
            sqrt = int(np.ceil(np.sqrt(int(params[1]))))

            w = network.connections['Y', 'Z'].w
            weights = [
                w[:, i].view(sqrt, sqrt) for i in range(10)
            ]
            w = torch.zeros(5 * sqrt, 2 * sqrt)
            for i in range(5):
                for j in range(2):
                    w[i * sqrt: (i + 1) * sqrt, j * sqrt: (j + 1) * sqrt] = weights[i + j * 5]

            plot_weights(w, wmin=-1, wmax=1, cmap=cmap)

            w = network.connections['X', 'Y'].w
            square_weights = get_square_weights(w, sqrt, 28)
            plot_weights(square_weights, wmin=-1, wmax=1, cmap=cmap)

        else:
            raise NotImplementedError('Weight plotting not implemented for this data, model combination.')

    elif data in ['breakout']:
        if model in ['crop', 'rebalance', 'two_level']:
            params = param_string.split('_')
            n_sqrt = int(np.ceil(np.sqrt(int(params[1]))))
            side = (50, 72)

            if model in ['crop', 'rebalance']:
                w = network.connections[('X', 'Ae')].w
            else:
                w = network.connections[('X', 'Y')].w

            w = get_square_weights(w, n_sqrt, side)
            plot_weights(w, cmap=cmap)

        elif model in ['backprop']:
            w = network.connections['X', 'Y'].w
            weights = [
                w[:, i].view(50, 72) for i in range(4)
            ]
            w = torch.zeros(2 * 50, 2 * 72)
            for j in range(2):
                for k in range(2):
                    w[j * 50: (j + 1) * 50, k * 72: (k + 1) * 72] = weights[j + k * 2]

            plot_weights(w, cmap=cmap)

        else:
            raise NotImplementedError('Weight plotting not implemented for this data, model combination.')

    if data in ['cifar10']:
        if model in ['backprop']:
            wmin = network.connections['X', 'Y'].wmin
            wmax = network.connections['X', 'Y'].wmax

            w = network.connections['X', 'Y'].w
            weights = [w[:, i].view(32, 32, 3).mean(2) for i in range(10)]
            w = torch.zeros(5 * 32, 2 * 32)
            for i in range(5):
                for j in range(2):
                    w[i * 32: (i + 1) * 32, j * 32: (j + 1) * 32] = weights[i + j * 5]

            plot_weights(w, wmin=wmin, wmax=wmax, cmap=cmap)

        else:
            raise NotImplementedError('Weight plotting not implemented for this data, model combination.')

    path = os.path.join(ROOT_DIR, 'plots', data, model, 'weights')
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, f'{param_string}.png'))

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--param_string', type=str, default=None)
    parser.add_argument('--cmap', type=str, default='hot_r')
    parser.add_argument('--p_destroy', type=float, default=0)
    parser.add_argument('--p_delete', type=float, default=0)
    args = vars(parser.parse_args())

    main(**args)
