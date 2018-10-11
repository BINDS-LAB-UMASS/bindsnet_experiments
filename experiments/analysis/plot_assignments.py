import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.utils import get_square_assignments
from bindsnet.analysis.plotting import plot_assignments

from experiments import ROOT_DIR
from experiments.analysis import download_params


def main(model='diehl_and_cook_2015', data='mnist', param_string=None):
    assert param_string is not None, 'Pass "--param_string" argument on command line or main method.'

    f = os.path.join(ROOT_DIR, 'params', data, model, f'auxiliary_{param_string}.pt')
    if not os.path.isfile(f):
        print('File not found locally. Attempting download from swarm2 cluster.')
        download_params.main(model=model, data=data, param_string=param_string)

    auxiliary = torch.load(open(f, 'rb'))

    if data in ['breakout']:
        assignments = auxiliary[0]
        assignments = get_square_assignments(assignments=assignments, n_sqrt=int(np.sqrt(assignments.numel())))
        plot_assignments(assignments=assignments, classes=['no-op', 'fire', 'right', 'left'])

    path = os.path.join(ROOT_DIR, 'plots', data, model, 'assignments')
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, f'{param_string}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--param_string', type=str, default=None)
    args = parser.parse_args()

    model = args.model
    data = args.data
    param_string = args.param_string

    main(model, data, param_string)