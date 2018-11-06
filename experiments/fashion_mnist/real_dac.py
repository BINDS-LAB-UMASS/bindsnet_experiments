import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import FashionMNIST
from bindsnet.learning import PostPre, NoOp
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network import load_network, Network
from bindsnet.network.nodes import RealInput, DiehlAndCookNodes
from bindsnet.evaluation import assign_labels, update_ngram_scores
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights

from experiments.utils import update_curves, print_results

model = 'real_dac'
data = 'fashion_mnist'


def main(seed=0, n_neurons=100, n_train=60000, n_test=10000, inhib=250, time=50, lr=1e-2, lr_decay=0.99, dt=1,
         theta_plus=0.05, theta_decay=1e-7, progress_interval=10, update_interval=250, train=True, plot=False,
         gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                            'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, inhib, time, lr, lr_decay,
        theta_plus, theta_decay, progress_interval, update_interval
    ]

    test_params = [
        seed, n_neurons, n_train, n_test, inhib, time, lr, lr_decay,
        theta_plus, theta_decay, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    if train:
        n_examples = n_train
    else:
        n_examples = n_test

    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    n_classes = 10

    # Build network.
    if train:
        network = Network(dt=dt)

        input_layer = RealInput(n=784, traces=True, trace_tc=5e-2)
        network.add_layer(input_layer, name='X')

        output_layer = DiehlAndCookNodes(
            n=n_neurons, traces=True, rest=0, reset=0, thresh=1, refrac=0,
            decay=1e-2, trace_tc=5e-2, theta_plus=theta_plus, theta_decay=theta_decay
        )
        network.add_layer(output_layer, name='Y')

        w = 0.3 * torch.rand(784, n_neurons)
        input_connection = Connection(
            source=network.layers['X'], target=network.layers['Y'], w=w, update_rule=PostPre,
            nu=[0, lr], wmin=0, wmax=1, norm=78.4
        )
        network.add_connection(input_connection, source='X', target='Y')

        w = -inhib * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
        recurrent_connection = Connection(
            source=network.layers['Y'], target=network.layers['Y'], w=w, wmin=-inhib, wmax=0
        )
        network.add_connection(recurrent_connection, source='Y', target='Y')

    else:
        path = os.path.join('..', '..', 'params', data, model)
        network = load_network(os.path.join(path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].theta_decay = 0
        network.layers['Y'].theta_plus = 0

    # Load Fashion-MNIST data.
    dataset = FashionMNIST(path=os.path.join('..', '..', 'data', 'FashionMNIST'), download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images = images.view(-1, 784)
    images = images / 255

    # if train:
    #     for i in range(n_neurons):
    #         network.connections['X', 'Y'].w[:, i] = images[i] + images[i].mean() * torch.randn(784)

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, time, n_neurons)

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones_like(torch.Tensor(n_neurons))
        proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        ngram_scores = {}
    else:
        path = os.path.join('..', '..', 'params', data, model)
        path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}

    if train:
        best_accuracy = 0

    spikes = {}

    for layer in set(network.layers):
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
    assigns_im = None
    perf_ax = None

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0 and train:
            network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

        if i % update_interval == 0 and i > 0:
            if i % len(labels) == 0:
                current_labels = labels[-update_interval:]
            else:
                current_labels = labels[i % len(images) - update_interval:i % len(images)]

            # Update and print accuracy evaluations.
            curves, predictions = update_curves(
                curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                proportions=proportions, ngram_scores=ngram_scores, n=2
            )
            print_results(curves)

            if train:
                if any([x[-1] > best_accuracy for x in curves.values()]):
                    print('New best accuracy! Saving network parameters to disk.')

                    # Save network to disk.
                    path = os.path.join('..', '..', 'params', data, model)
                    if not os.path.isdir(path):
                        os.makedirs(path)

                    network.save(os.path.join(path, model_name + '.pt'))
                    path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.pt')
                    torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

                    best_accuracy = max([x[-1] for x in curves.values()])

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

                # Compute ngram scores.
                ngram_scores = update_ngram_scores(spike_record, current_labels, n_classes, 2, ngram_scores)

            print()

        # Get next input sample.
        image = images[i % n_examples].repeat([time, 1])
        inpts = {'X': image}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Y'].get('s').sum() < 5 and retries < 3:
            retries += 1
            image *= 2
            inpts = {'X': image}
            network.run(inpts=inpts, time=time)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Y'].get('s').t()

        # Optionally plot various simulation information.
        if plot:
            _input = images[i % n_examples].view(28, 28)
            reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
            _spikes = {layer: spikes[layer].get('s') for layer in spikes}
            input_exc_weights = network.connections['X', 'Y'].w
            square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
            square_assignments = get_square_assignments(assignments, n_sqrt)

            # inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im, wmax=0.25)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(curves, ax=perf_ax)

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    i += 1

    if i % len(labels) == 0:
        current_labels = labels[-update_interval:]
    else:
        current_labels = labels[i % len(images) - update_interval:i % len(images)]

    # Update and print accuracy evaluations.
    curves, predictions = update_curves(
        curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
        proportions=proportions, ngram_scores=ngram_scores, n=2
    )
    print_results(curves)

    if train:
        if any([x[-1] > best_accuracy for x in curves.values()]):
            print('New best accuracy! Saving network parameters to disk.')

            # Save network to disk.
            if train:
                path = os.path.join('..', '..', 'params', data, model)
                if not os.path.isdir(path):
                    os.makedirs(path)

                network.save(os.path.join(path, model_name + '.pt'))
                path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.pt')
                torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')

    print('Average accuracies:\n')
    for scheme in curves.keys():
        print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    path = os.path.join('..', '..', 'curves', data, model)
    if not os.path.isdir(path):
        os.makedirs(path)

    if train:
        to_write = ['train'] + params
    else:
        to_write = ['test'] + params

    to_write = [str(x) for x in to_write]
    f = '_'.join(to_write) + '.pt'

    torch.save((curves, update_interval, n_examples), open(os.path.join(path, f), 'wb'))

    # Save results to disk.
    path = os.path.join('..', '..', 'results', data, model)
    if not os.path.isdir(path):
        os.makedirs(path)

    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    ]

    if train:
        to_write = params + results
    else:
        to_write = test_params + results

    to_write = [str(x) for x in to_write]

    if train:
        name = 'train.csv'
    else:
        name = 'test.csv'

    if not os.path.isfile(os.path.join(path, name)):
        with open(os.path.join(path, name), 'w') as f:
            if train:
                f.write('random_seed,n_neurons,n_train,inhib,time,lr,lr_decay,theta_plus,theta_decay,'
                        'progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                        'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n')
            else:
                f.write('random_seed,n_neurons,n_train,n_test,inhib,time,lr,lr_decay,theta_plus,theta_decay,'
                        'progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                        'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n')

    with open(os.path.join(path, name), 'a') as f:
        f.write(','.join(to_write) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_train', type=int, default=60000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--inhib', type=float, default=250)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--dt', type=int, default=1)
    parser.add_argument('--theta_plus', type=float, default=0.05)
    parser.add_argument('--theta_decay', type=float, default=1e-7)
    parser.add_argument('--progress_interval', type=int, default=10)
    parser.add_argument('--update_interval', type=int, default=250)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
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
