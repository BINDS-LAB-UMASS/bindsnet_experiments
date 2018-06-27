import os
import sys
import torch
import argparse
import numpy as np
import pickle as p
import pandas as pd

from bindsnet.utils import *
from bindsnet.analysis.plotting import *

import download_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--param_string', type=str, default=None)
    args = parser.parse_args()

    model = args.model
    data = args.data
    param_string = args.param_string

    assert param_string is not None, 'Pass "--param_string" argument on command line.'

    f = os.path.join('..', 'params', f'{model}_{data}', f'{param_string}.p')
    if not os.path.isfile(f):
        print('File not found. Downloading from swarm2 cluster.')
        download_params.main(model=model, data=data, param_string=param_string)

    network = p.load(open(f, 'rb'))

    if model in ['diehl_and_cook_2015']:
        params = param_string.split('_')
        n_sqrt = int(np.ceil(np.sqrt(int(params[1]))))
        side = int(np.sqrt(network.layers['X'].n))

        w = network.connections[('X', 'Ae')].w
        w = get_square_weights(w, n_sqrt, side)
        plot_weights(w)

    elif model in ['conv']:
        raise NotImplementedError('Automated plotting not yet implemented for "conv" network model.')
    elif model in ['fully_conv', 'locally_connected']:
        params = param_string.split('_')
        kernel_size = int(params[1])
        stride = int(params[2])
        n_filters = int(params[3])
        
        input_sqrt = int(np.sqrt(network.layers['X'].n))

        if kernel_size == input_sqrt:
            conv_size = 1
        else:
            conv_size = int((input_sqrt - kernel_size) / stride) + 1

        locations = torch.zeros(kernel_size, kernel_size, conv_size ** 2).long()
        for c in range(conv_size ** 2):
            for k1 in range(kernel_size):
                for k2 in range(kernel_size):
                    locations[k1, k2, c] = (c % conv_size) * stride * 28 + (c // conv_size) * stride + k1 * 28 + k2

        locations = locations.view(kernel_size ** 2, conv_size ** 2)

        w = network.connections[('X', 'Y')].w
        plot_locally_connected_weights(w, n_filters, kernel_size, conv_size, locations, input_sqrt)

    path = os.path.join('..', 'plots', f'{data}', f'{model}', 'weights')
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, f'{param_string}.png'))