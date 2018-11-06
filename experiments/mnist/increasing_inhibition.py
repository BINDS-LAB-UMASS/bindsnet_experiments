import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import euclidean

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.encoding import poisson
from bindsnet.network import load_network
from bindsnet.learning import PostPre, NoOp
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.evaluation import assign_labels, update_ngram_scores
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_spikes, plot_weights, plot_input, plot_assignments, plot_performance

from experiments import ROOT_DIR
from experiments.utils import print_results, update_curves

model = 'increasing_inhibition'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_neurons=100, n_train=60000, n_test=10000, c_low=1, c_high=25, p_low=0.5, time=250, dt=1,
         theta_plus=0.05, theta_decay=1e-7, intensity=1, progress_interval=10,
         update_interval=250, plot=False, train=True, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0,\
        'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, c_low, c_high, p_low, time, dt,
        theta_plus, theta_decay, intensity, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    if not train:
        test_params = [
            seed, n_neurons, n_train, n_test, c_low, c_high, p_low, time, dt, theta_plus,
            theta_decay, intensity, progress_interval, update_interval
        ]

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
        input_layer = Input(n=784, traces=True)
        exc_layer = DiehlAndCookNodes(n=n_neurons, traces=True)

        w = torch.rand(input_layer.n, exc_layer.n)
        input_exc_conn = Connection(
            input_layer, exc_layer, w=w, update_rule=PostPre, norm=78.4, nu=(1e-4, 1e-2), wmax=1.0
        )

        w = torch.zeros(exc_layer.n, exc_layer.n)
        for k1 in range(n_neurons):
            for k2 in range(n_neurons):
                if k1 != k2:
                    x1, y1 = k1 // np.sqrt(n_neurons), k1 % np.sqrt(n_neurons)
                    x2, y2 = k2 // np.sqrt(n_neurons), k2 % np.sqrt(n_neurons)

                    w[k1, k2] = max(-c_high, -c_low * np.sqrt(euclidean([x1, y1], [x2, y2])))

        recurrent_conn = Connection(exc_layer, exc_layer, w=w)

        network.add_layer(input_layer, name='X')
        network.add_layer(exc_layer, name='Y')
        network.add_connection(input_exc_conn, source='X', target='Y')
        network.add_connection(recurrent_conn, source='Y', target='Y')
    else:
        network = load_network(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].theta_decay = 0
        network.layers['Y'].theta_plus = 0

    # Load MNIST data.
    dataset = MNIST(data_path, download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images = images.view(-1, 784)
    images *= intensity

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, int(time / dt), n_neurons)

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones_like(torch.Tensor(n_neurons))
        proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
        rates = torch.zeros_like(torch.Tensor(n_neurons, 10))
        ngram_scores = {}
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

    if train:
        best_accuracy = 0

    spikes = {}
    for layer in set(network.layers) - {'X'}:
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=int(time / dt))
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

    # Calculate linear increase every update interval.
    if train:
        n_increase = int(p_low * n_examples) / update_interval
        increase = (c_high - c_low) / n_increase
        increases = 0
        inhib = c_low

    start = t()
    for i in range(n_examples):
        if train and i % update_interval == 0 and i > 0 and increases < n_increase:
            inhib = inhib + increase

            print(f'\nIncreasing inhibition to {inhib}.\n')

            w = torch.zeros(n_neurons, n_neurons)
            for k1 in range(n_neurons):
                for k2 in range(n_neurons):
                    if k1 != k2:
                        x1, y1 = k1 // np.sqrt(n_neurons), k1 % np.sqrt(n_neurons)
                        x2, y2 = k2 // np.sqrt(n_neurons), k2 % np.sqrt(n_neurons)

                        w[k1, k2] = max(-c_high, -inhib * np.sqrt(euclidean([x1, y1], [x2, y2])))

            network.connections['Y', 'Y'].w = w

            increases += 1

        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

        if i % update_interval == 0 and i > 0:
            if i % len(labels) == 0:
                current_labels = labels[-update_interval:]
            else:
                current_labels = labels[i % len(images) - update_interval:i % len(images)]

            # Update and print accuracy evaluations.
            curves, preds = update_curves(
                curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                proportions=proportions, ngram_scores=ngram_scores, n=2
            )
            print_results(curves)

            for scheme in preds:
                predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

            # Save accuracy curves to disk.
            to_write = ['train'] + params if train else ['test'] + params
            f = '_'.join([str(x) for x in to_write]) + '.pt'
            torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

            if train:
                if any([x[-1] > best_accuracy for x in curves.values()]):
                    print('New best accuracy! Saving network parameters to disk.')

                    # Save network to disk.
                    network.save(os.path.join(params_path, model_name + '.pt'))
                    path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
                    torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

                    best_accuracy = max([x[-1] for x in curves.values()])

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)

                # Compute ngram scores.
                ngram_scores = update_ngram_scores(spike_record, labels[i - update_interval:i], 10, 2, ngram_scores)

            print()

        # Get next input sample.
        image = images[i]
        sample = poisson(datum=image, time=int(time / dt))
        inpts = {'X' : sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Y'].get('s').sum() < 5 and retries < 3:
            retries += 1
            image *= 2
            sample = poisson(datum=image, time=int(time / dt))
            inpts = {'X' : sample}
            network.run(inpts=inpts, time=time)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Y'].get('s').t()

        # Optionally plot various simulation information.
        if plot:
            inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
            _spikes = {layer: spikes[layer].get('s') for layer in spikes}
            input_exc_weights = network.connections['X', 'Y'].w
            square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
            square_assignments = get_square_assignments(assignments, n_sqrt)

            # inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
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
    curves, preds = update_curves(
        curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
        proportions=proportions, ngram_scores=ngram_scores, n=2
    )
    print_results(curves)

    for scheme in preds:
        predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

    if train:
        if any([x[-1] > best_accuracy for x in curves.values()]):
            print('New best accuracy! Saving network parameters to disk.')

            # Save network to disk.
            network.save(os.path.join(params_path, model_name + '.pt'))
            path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
            torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')

    print('Average accuracies:\n')
    for scheme in curves.keys():
        print(f'\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    to_write = ['train'] + params if train else ['test'] + params
    to_write = [str(x) for x in to_write]
    f = '_'.join(to_write) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    ]

    to_write = params + results if train else test_params + results
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            if train:
                f.write('random_seed,n_neurons,n_train,excite,c_low,c_high,p_low,time,timestep,theta_plus,theta_decay,'
                        'intensity,progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                        'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n')
            else:
                f.write('random_seed,n_neurons,n_train,n_test,excite,c_low,c_high,p_low,time,timestep,theta_plus,theta_decay,'
                        'intensity,progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                        'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n')

    with open(os.path.join(results_path, name), 'a') as f:
        f.write(','.join(to_write) + '\n')

    if labels.numel() > n_examples:
        labels = labels[:n_examples]
    else:
        while labels.numel() < n_examples:
            if 2 * labels.numel() > n_examples:
                labels = torch.cat([labels, labels[:n_examples - labels.numel()]])
            else:
                labels = torch.cat([labels, labels])

    # Compute confusion matrices and save them to disk.
    confusions = {}
    for scheme in predictions:
        confusions[scheme] = confusion_matrix(labels, predictions[scheme])

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save(confusions, os.path.join(confusion_path, f))

    print()


if __name__ == '__main__':
    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_neurons', type=int, default=100)
    parser.add_argument('--n_train', type=int, default=60000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--c_low', type=float, default=1)
    parser.add_argument('--c_high', type=float, default=25)
    parser.add_argument('--p_low', type=float, default=0.5)
    parser.add_argument('--time', type=int, default=350)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--theta_plus', type=float, default=0.05)
    parser.add_argument('--theta_decay', type=float, default=1e-7)
    parser.add_argument('--intensity', type=float, default=0.5)
    parser.add_argument('--progress_interval', type=int, default=10)
    parser.add_argument('--update_interval', type=int, default=250)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    seed = args.seed
    n_neurons = args.n_neurons
    n_train = args.n_train
    n_test = args.n_test
    c_low = args.c_low
    c_high = args.c_high
    p_low = args.p_low
    time = args.time
    dt = args.dt
    theta_plus = args.theta_plus
    theta_decay = args.theta_decay
    intensity = args.intensity
    progress_interval = args.progress_interval
    update_interval = args.update_interval
    plot = args.plot
    train = args.train
    gpu = args.gpu

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(seed=seed, n_neurons=n_neurons, n_train=n_train, n_test=n_test, c_low=c_low, c_high=c_high, p_low=p_low,
         time=time, dt=dt, theta_plus=theta_plus, theta_decay=theta_decay, intensity=intensity,
         progress_interval=progress_interval, update_interval=update_interval, plot=plot, train=train, gpu=gpu)

    print()
