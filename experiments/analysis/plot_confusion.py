import os
import torch
import argparse
import matplotlib.pyplot as plt

from experiments import ROOT_DIR
from experiments.analysis import download_confusion


def main(model='diehl_and_cook_2015', data='mnist', train=False, cluster='swarm2', param_string=None, match=None):
    assert param_string is not None, 'Pass "--param_string" argument on command line or main method.'

    mode = 'train' if train else 'test'
    f = os.path.join(ROOT_DIR, 'confusion', data, model, f'{mode}_{param_string}.pt')
    if not os.path.isfile(f):
        print('File not found locally. Attempting download from swarm2 cluster.')
        download_confusion.main(
            model=model, data=data, train=train, cluster=cluster, param_string=param_string, match=match
        )

    f = os.path.join('..', 'confusion', data, model, f'{mode}_{param_string}.pt')
    confusions = torch.load(open(f, 'rb'))

    if data in ['mnist', 'cifar10']:
        labels = range(10)
    elif data == 'breakout':
        labels = ['no-op', 'fire', 'right', 'left']

    for scheme in confusions:
        confusion = confusions[scheme]

        normed = confusion / confusion.sum(1)

        path = os.path.join(ROOT_DIR, 'plots', data, model, 'confusion_matrices', param_string)
        if not os.path.isdir(path):
            os.makedirs(path)

        plt.matshow(normed)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.colorbar()
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels, rotation='vertical')
        plt.tight_layout()

        plt.savefig(os.path.join(path, f'{scheme}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--cluster', type=str, default='swarm2')
    parser.add_argument('--param_string', type=str, default=None)
    parser.add_argument('--match', type=str, default=None)
    parser.set_defaults(train=False)
    args = parser.parse_args()

    model = args.model
    data = args.data
    train = args.train
    cluster = args.cluster
    param_string = args.param_string
    match = args.match

    main(model, data, train, cluster, param_string, match)
