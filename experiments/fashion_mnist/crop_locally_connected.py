import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix

from bindsnet.learning import NoOp
from bindsnet.encoding import poisson
from bindsnet.network import load_network
from bindsnet.datasets import FashionMNIST
from bindsnet.network.monitors import Monitor
from bindsnet.models import LocallyConnectedNetwork
from bindsnet.evaluation import assign_labels, update_ngram_scores
from bindsnet.analysis.plotting import plot_locally_connected_weights, plot_spikes

from experiments.utils import update_curves, print_results

model = 'crop_locally_connected'
data = 'fashion_mnist'

top_level = os.path.join('..', '..')
data_path = os.path.join(top_level, 'data', 'FashionMNIST')
params_path = os.path.join(top_level, 'params', data, model)
curves_path = os.path.join(top_level, 'curves', data, model)
results_path = os.path.join(top_level, 'results', data, model)
confusion_path = os.path.join(top_level, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_train=60000, n_test=10000, inhib=250, kernel_size=(16,), stride=(2,), time=50, n_filters=25, crop=0,
         lr=1e-2, lr_decay=0.99, dt=1, theta_plus=0.05, theta_decay=1e-7, norm=0.2, progress_interval=10,
         update_interval=250, train=True, relabel=False, plot=False, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0 or relabel, \
        'No. examples must be divisible by update_interval'

    params = [
        seed, kernel_size, stride, n_filters, crop, lr, lr_decay, n_train, inhib,
        time, dt, theta_plus, theta_decay, norm, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    if not train:
        test_params = [
            seed, kernel_size, stride, n_filters, crop, lr, lr_decay, n_train, n_test, inhib,
            time, dt, theta_plus, theta_decay, norm, progress_interval, update_interval
        ]

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    side_length = 28 - crop * 2
    n_inpt = side_length ** 2
    n_examples = n_train if train else n_test
    n_classes = 10

    # Build network.
    if train:
        network = LocallyConnectedNetwork(
            n_inpt=n_inpt, input_shape=[side_length, side_length], kernel_size=kernel_size, stride=stride,
            n_filters=n_filters, inh=inhib, dt=dt, nu=[.1 * lr, lr], theta_plus=theta_plus,
            theta_decay=theta_decay, wmin=0, wmax=1.0, norm=norm
        )
        network.layers['Y'].thresh = 1
        network.layers['Y'].reset = 0
        network.layers['Y'].rest = 0

    else:
        network = load_network(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].theta_decay = 0
        network.layers['Y'].theta_plus = 0

    conv_size = network.connections['X', 'Y'].conv_size
    locations = network.connections['X', 'Y'].locations
    conv_prod = int(np.prod(conv_size))
    n_neurons = n_filters * conv_prod

    # Voltage recording for excitatory and inhibitory layers.
    voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
    network.add_monitor(voltage_monitor, name='output_voltage')

    # Load Fashion-MNIST data.
    dataset = FashionMNIST(path=data_path, download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    if crop != 0:
        images = images[:, crop:-crop, crop:-crop]

    # Record spikes during the simulation.
    if not train:
        update_interval = n_examples

    spike_record = torch.zeros(update_interval, time, n_neurons)

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones_like(torch.Tensor(n_neurons))
        proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
        rates = torch.zeros_like(torch.Tensor(n_neurons, 10))
        ngram_scores = {}
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    if train:
        best_accuracy = 0

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name=f'{layer}_spikes')

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    spike_ims = None
    spike_axes = None
    weights_im = None

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
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

                # Compute ngram scores.
                ngram_scores = update_ngram_scores(spike_record, current_labels, n_classes, 2, ngram_scores)

            print()

        # Get next input sample.
        image = images[i % len(images)].contiguous().view(-1)
        sample = poisson(datum=image, time=time, dt=dt)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Y'].get('s').sum() < 5 and retries < 3:
            retries += 1
            image *= 2
            sample = poisson(datum=image, time=time, dt=dt)
            inpts = {'X': sample}
            network.run(inpts=inpts, time=time)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Y'].get('s').t()

        # Optionally plot various simulation information.
        if plot:
            _spikes = {'X': spikes['X'].get('s').view(side_length ** 2, time),
                       'Y': spikes['Y'].get('s').view(n_filters * conv_prod, time)}

            spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_locally_connected_weights(
                network.connections['X', 'Y'].w, n_filters, kernel_size, conv_size,
                locations, side_length, im=weights_im, wmin=0, wmax=1
            )

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    i += 1

    if i % len(labels) == 0:
        current_labels = labels[-update_interval:]
    else:
        current_labels = labels[i % len(images) - update_interval:i % len(images)]

    if not train and relabel:
        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

        # Compute ngram scores.
        ngram_scores = update_ngram_scores(spike_record, current_labels, n_classes, 2, ngram_scores)

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
        print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    to_write = ['train'] + params if train else ['test'] + params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

    # Save results to disk.
    path = os.path.join('..', '..', 'results', data, model)
    if not os.path.isdir(path):
        os.makedirs(path)

    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    ]

    to_write = params + results if train else test_params + results
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(path, name), 'w') as f:
            if train:
                f.write(
                    'random_seed,kernel_size,stride,n_filters,crop,n_train,inhib,time,lr,lr_decay,timestep,theta_plus,'
                    'theta_decay,norm,progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                    'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
                )
            else:
                f.write(
                    'random_seed,kernel_size,stride,n_filters,crop,n_train,n_test,inhib,time,lr,lr_decay,timestep,'
                    'theta_plus,theta_decay,norm,progress_interval,update_interval,mean_all_activity,'
                    'mean_proportion_weighting,mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
                )

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


if __name__ == '__main__':
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_train', type=int, default=60000)
    parser.add_argument('--n_test', type=int, default=10000)
    parser.add_argument('--inhib', type=float, default=250)
    parser.add_argument('--kernel_size', nargs='+', type=int, default=[16])
    parser.add_argument('--stride', nargs='+', type=int, default=[4])
    parser.add_argument('--n_filters', type=int, default=16)
    parser.add_argument('--crop', type=int, default=0)
    parser.add_argument('--time', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--theta_plus', type=float, default=0.05)
    parser.add_argument('--theta_decay', type=float, default=1e-7)
    parser.add_argument('--norm', type=float, default=0.1)
    parser.add_argument('--progress_interval', type=int, default=10)
    parser.add_argument('--update_interval', type=int, default=250)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--relabel', dest='relabel', action='store_true')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.set_defaults(plot=False, gpu=False, train=True, relabel=False)

    args = parser.parse_args()

    kernel_size = args.kernel_size
    stride = args.stride

    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    if len(stride) == 1:
        stride = stride[0]

    args = vars(args)
    args['kernel_size'] = kernel_size
    args['stride'] = stride

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
