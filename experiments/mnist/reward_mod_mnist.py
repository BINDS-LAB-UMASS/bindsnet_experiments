import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.learning import NoOp, MSTDP
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.network.topology import Connection
from bindsnet.network import load_network, Network
from bindsnet.network.nodes import IFNodes, RealInput, DiehlAndCookNodes
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance

from experiments import ROOT_DIR

model = 'reward_mod_mnist'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_neurons=100, n_train=60000, n_test=10000, inhib=100, lr=0.01, lr_decay=1, time=350, dt=1,
         theta_plus=0.05, theta_decay=1e-7, progress_interval=10, update_interval=250, plot=False,
         train=True, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                            'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, inhib, lr_decay, time, dt,
        theta_plus, theta_decay, progress_interval, update_interval
    ]

    test_params = [
        seed, n_neurons, n_train, n_test, inhib, lr_decay, time, dt,
        theta_plus, theta_decay, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    n_examples = n_train if train else n_test
    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    n_classes = 10

    # Build network.
    if train:
        network = Network(dt=dt)

        input_layer = RealInput(n=784, traces=True, trace_tc=5e-2)
        network.add_layer(input_layer, name='X')

        output_layer = DiehlAndCookNodes(
            n=n_neurons, traces=True, rest=0, reset=1, thresh=1, refrac=0,
            decay=1e-2, trace_tc=5e-2, theta_plus=theta_plus, theta_decay=theta_decay
        )
        network.add_layer(output_layer, name='Y')

        readout = IFNodes(n=n_classes, reset=0, thresh=1)
        network.add_layer(readout, name='Z')

        w = torch.rand(784, n_neurons)
        input_connection = Connection(
            source=input_layer, target=output_layer, w=w,
            update_rule=MSTDP, nu=lr, wmin=0, wmax=1, norm=78.4
        )
        network.add_connection(input_connection, source='X', target='Y')

        w = -inhib * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
        recurrent_connection = Connection(
            source=output_layer, target=output_layer, w=w, wmin=-inhib, wmax=0
        )
        network.add_connection(recurrent_connection, source='Y', target='Y')

        readout_connection = Connection(
            source=network.layers['Y'], target=readout, w=torch.rand(n_neurons, n_classes), norm=10
        )
        network.add_connection(readout_connection, source='Y', target='Z')

    else:
        network = load_network(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].theta_decay = 0
        network.layers['Y'].theta_plus = 0

    # Load MNIST data.
    dataset = MNIST(path=data_path, download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images = images.view(-1, 784)
    labels = labels.long()

    spikes = {}
    for layer in set(network.layers) - {'X'}:
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    weights2_im = None
    assigns_im = None
    perf_ax = None

    predictions = torch.zeros(update_interval).long()

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

            if i > 0 and train:
                network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

        # Get next input sample.
        image = images[i % len(images)]

        # Run the network on the input.
        for j in range(time):
            readout = network.layers['Z'].s

            if readout[labels[i % len(labels)]]:
                network.run(inpts={'X': image.unsqueeze(0)}, time=1, reward=1, a_minus=0, a_plus=1)
            else:
                network.run(inpts={'X': image.unsqueeze(0)}, time=1, reward=0)

        label = spikes['Z'].get('s').sum(1).argmax()
        predictions[i % update_interval] = label.long()

        if i > 0 and i % update_interval == 0:
            if i % len(labels) == 0:
                current_labels = labels[-update_interval:]
            else:
                current_labels = labels[i % len(images) - update_interval:i % len(images)]

            accuracy = 100 * (predictions == current_labels).float().mean().item()
            print(f'Accuracy over last {update_interval} examples: {accuracy}')

        # Optionally plot various simulation information.
        if plot:
            _spikes = {layer: spikes[layer].get('s') for layer in spikes}
            input_exc_weights = network.connections['X', 'Y'].w
            square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
            exc_readout_weights = network.connections['Y', 'Z'].w

            # _input = image.view(28, 28)
            # reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
            # square_assignments = get_square_assignments(assignments, n_sqrt)

            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            weights2_im = plot_weights(exc_readout_weights, im=weights2_im)

            # inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(curves, ax=perf_ax)

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_neurons', type=int, default=100, help='no. of output layer neurons')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='no. of test samples')
    parser.add_argument('--inhib', type=float, default=100.0, help='inhibition connection strength')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=350, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1, help='simulation integreation timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--theta_decay', type=float, default=1e-7, help='adaptive threshold decay time constant')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
