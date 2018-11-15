import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from bindsnet.datasets import MNIST
from bindsnet.network import Network, load_network
from bindsnet.learning import PostPre, NoOp
from bindsnet.encoding import bernoulli
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import assign_labels, logreg_fit
from bindsnet.network.topology import Connection, Conv2dConnection
from bindsnet.network.nodes import Input, DiehlAndCookNodes, LIFNodes
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_conv2d_weights, plot_weights, plot_assignments
from bindsnet.utils import get_square_assignments

from experiments import ROOT_DIR
from experiments.utils import print_results, update_curves

model = 'conv'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_train=60000, n_test=10000, kernel_size=(8,), stride=(4,), n_filters=25, n_full=100, padding=0,
         inhib=100, time=100, lr=1e-3, lr_decay=0.99, dt=1, intensity=1, progress_interval=10,
         update_interval=250, plot=False, train=True, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
        'No. examples must be divisible by update_interval'

    params = [
        seed, n_train, kernel_size, stride, n_filters, n_full,
        padding, inhib, time, lr, lr_decay, dt, intensity, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    if not train:
        test_params = [
            seed, n_train, n_test, kernel_size, stride, n_filters, n_full,
            padding, inhib, time, lr, lr_decay, dt, intensity, update_interval
        ]

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    n_examples = n_train if train else n_test
    input_shape = [28, 28]

    if kernel_size == input_shape:
        conv_size = [1, 1]
    else:
        conv_size = (
            int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
            int((input_shape[1] - kernel_size[1]) / stride[1]) + 1
        )

    n_classes = 10
    total_kernel_size = int(np.prod(kernel_size))
    total_conv_size = int(np.prod(conv_size))
    n_neurons = n_filters * total_conv_size
    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

    # Build network.
    if train:
        network = Network()
        input_layer = Input(n=784, shape=(1, 1, 28, 28), traces=True)
        conv_layer = DiehlAndCookNodes(
            n=n_filters * total_conv_size, shape=(1, n_filters, *conv_size), thresh=-64.0,
            traces=True, theta_plus=0.05, refrac=0
        )
        conv_layer_prime = LIFNodes(
            n=n_filters * total_conv_size, shape=(1, n_filters, *conv_size), refrac=0, traces=True
        )
        conv_conn = Conv2dConnection(
            input_layer, conv_layer, kernel_size=kernel_size, stride=stride, update_rule=PostPre,
            norm=0.5 * int(np.sqrt(total_kernel_size)), nu=[0, lr], wmax=2.0
        )
        conv_conn_prime = Conv2dConnection(
            input_layer, conv_layer_prime, w=conv_conn.w,
            kernel_size=kernel_size, stride=stride, nu=[0, 0], wmax=2.0
        )

        w = -inhib * torch.ones(
            n_filters, conv_size[0], conv_size[1], n_filters, conv_size[0], conv_size[1]
        )
        for f in range(n_filters):
            for i in range(conv_size[0]):
                for j in range(conv_size[1]):
                    w[f, i, j, f, i, j] = 0

        w = w.view(n_filters * conv_size[0] * conv_size[1], n_filters * conv_size[0] * conv_size[1])
        recurrent_conn = Connection(conv_layer, conv_layer, w=w)

        full_layer = DiehlAndCookNodes(
            n=n_full, thresh=-52.0, traces=True, theta_plus=0.05, refrac=0
        )
        full_layer_prime = LIFNodes(
            n=n_full, refrac=0
        )
        full_conn = Connection(
            conv_layer_prime, full_layer, update_rule=PostPre, norm=0.2 * n_neurons, nu=[0, 10 * lr], wmax=1
        )
        full_conn_prime = Connection(
            conv_layer_prime, full_layer_prime, 0, wmax=1
        )

        w = -inhib * (torch.ones(n_full, n_full) - torch.diag(torch.ones(n_full)))
        recurrent_conn2 = Connection(full_layer, full_layer, w=w)

        network.add_layer(input_layer, name='X')
        network.add_layer(conv_layer, name='Y')
        network.add_layer(conv_layer_prime, name='Y_')
        network.add_layer(full_layer, name='Z')
        network.add_layer(full_layer_prime, name='Z_')

        network.add_connection(conv_conn, source='X', target='Y')
        network.add_connection(conv_conn_prime, source='X', target='Y_')
        network.add_connection(recurrent_conn, source='Y', target='Y')
        network.add_connection(full_conn, source='Y_', target='Z')
        network.add_connection(full_conn_prime, source='Y_', target='Z_')
        network.add_connection(recurrent_conn2, source='Z', target='Z')

        # Voltage recording for excitatory and inhibitory layers.
        voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
        network.add_monitor(voltage_monitor, name='output_voltage')
    else:
        network = load_network(os.path.join(params_path, model_name + '.pt'))

        for connection in network.connections.values():
            connection.update_rule = NoOp(connection, connection.nu)
            connection.theta_decay = 0
            connection.theta_plus = 0

    # Load MNIST data.
    dataset = MNIST(data_path, download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images *= intensity

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, time, n_full)

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones_like(torch.Tensor(n_full))
        proportions = torch.zeros_like(torch.Tensor(n_full, n_classes))
        rates = torch.zeros_like(torch.Tensor(n_full, n_classes))
        logreg_model = LogisticRegression(warm_start=True, n_jobs=-1, solver='lbfgs')
        logreg_model.coef_ = np.zeros([n_classes, n_full])
        logreg_model.intercept_ = np.zeros(n_classes)
        logreg_model.classes_ = np.arange(n_classes)
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, logreg_coef, logreg_intercept = torch.load(open(path, 'rb'))
        logreg_model = LogisticRegression(warm_start=True, n_jobs=-1, solver='lbfgs')
        logreg_model.coef_ = logreg_coef
        logreg_model.intercept_ = logreg_intercept
        logreg_model.classes_ = np.arange(n_classes)

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'logreg': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

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

    inpt_ims = None
    inpt_axes = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    weights_im2 = None
    assigns_im = None

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print('Progress: %d / %d (%.4f seconds)' % (i, n_examples, t() - start))
            start = t()

        if i % update_interval == 0 and i > 0:
            if train:
                network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

            if i % len(labels) == 0:
                current_labels = labels[-update_interval:]
            else:
                current_labels = labels[i % len(images) - update_interval:i % len(images)]

            # Update and print accuracy evaluations.
            curves, preds = update_curves(
                curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                proportions=proportions, logreg=logreg_model
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
                    torch.save(
                        (
                            assignments, proportions, rates, logreg_model.coef_, logreg_model.intercept_
                        ), open(path, 'wb')
                    )
                    best_accuracy = max([x[-1] for x in curves.values()])

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

                # Refit logistic regression model.
                logreg_model = logreg_fit(spike_record, current_labels, logreg_model)

            print()

        # Get next input sample.
        image = images[i % len(images)]
        sample = bernoulli(datum=image, time=time, dt=dt, max_prob=0.5).unsqueeze(1).unsqueeze(1)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Z'].get('s').sum() < 5 and retries < 3:
            retries += 1
            sample = bernoulli(datum=image, time=time, dt=dt, max_prob=0.5 + retries * 0.15).unsqueeze(1).unsqueeze(1)
            inpts = {'X': sample}
            network.run(inpts=inpts, time=time)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Z'].get('s').view(time, -1)

        # Optionally plot various simulation information.
        if plot:
            _input = inpts['X'].view(time, 784).sum(0).view(28, 28)
            w = network.connections['X', 'Y'].w
            w2 = network.connections['Y_', 'Z'].w
            _spikes = {
                'X': spikes['X'].get('s').view(28 ** 2, time),
                'Y': spikes['Y'].get('s').view(n_neurons, time),
                'Y_': spikes['Y_'].get('s').view(n_neurons, time),
                'Z': spikes['Z'].get('s').view(n_full, time),
                'Z_': spikes['Z_'].get('s').view(n_full, time)
            }
            square_assignments = get_square_assignments(assignments, n_sqrt)

            inpt_axes, inpt_ims = plot_input(
                image.view(28, 28), _input, label=labels[i], ims=inpt_ims, axes=inpt_axes
            )
            spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_conv2d_weights(w, im=weights_im, wmax=0.2)
            weights_im2 = plot_weights(w2, im=weights_im2, wmax=1)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)

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
        proportions=proportions, logreg=logreg_model
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
            torch.save(
                (
                    assignments, proportions, rates, logreg_model.coef_, logreg_model.intercept_
                ), open(path, 'wb')
            )

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')

    print('Average accuracies:\n')
    for scheme in curves.keys():
        print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    to_write = ['train'] + params if train else ['test'] + params
    to_write = [str(x) for x in to_write]
    f = '_'.join(to_write) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

    # Save results to disk.
    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['logreg']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['logreg'])
    ]

    to_write = params + results if train else test_params + results
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            if train:
                columns = [
                    'seed', 'n_train', 'kernel_size', 'stride', 'n_filters', 'padding', 'inhib', 'time',
                    'lr', 'lr_decay', 'dt', 'intensity', 'update_interval', 'mean_all_activity',
                    'mean_proportion_weighting', 'mean_logreg', 'max_all_activity', 'max_proportion_weighting',
                    'max_logreg'
                ]

                header = ','.join(columns) + '\n'
                f.write(header)
            else:
                columns = [
                    'seed', 'n_train', 'n_test', 'kernel_size', 'stride', 'n_filters', 'padding', 'inhib', 'time',
                    'lr', 'lr_decay', 'dt', 'intensity', 'update_interval', 'mean_all_activity',
                    'mean_proportion_weighting', 'mean_logreg', 'max_all_activity', 'max_proportion_weighting',
                    'max_logreg'
                ]

                header = ','.join(columns) + '\n'
                f.write(header)

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

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='no. of test samples')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[8], help='one or two kernel side lengths')
    parser.add_argument('--stride', type=int, nargs='+', default=[4], help='one or two horizontal stride lengths')
    parser.add_argument('--n_filters', type=int, default=25, help='no. of convolutional filters')
    parser.add_argument('--n_full', type=int, default=100, help='no. of fully-connected neurons')
    parser.add_argument('--padding', type=int, default=0, help='horizontal, vertical padding size')
    parser.add_argument('--inhib', type=float, default=250, help='inhibition connection strength')
    parser.add_argument('--time', default=100, type=int, help='simulation time')
    parser.add_argument('--lr', type=float, default=1e-3, help='post-synaptic learning rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='rate at which to decay learning rate')
    parser.add_argument('--dt', type=float, default=1.0, help='simulation integreation timestep')
    parser.add_argument('--intensity', type=float, default=1, help='constant to multiple input data by')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    kernel_size = args.kernel_size
    stride = args.stride

    if len(kernel_size) == 1:
        kernel_size = (kernel_size[0], kernel_size[0])
    else:
        kernel_size = tuple(kernel_size)

    if len(stride) == 1:
        stride = (stride[0], stride[0])
    else:
        stride = tuple(stride)

    args = vars(args)
    args.update(
        {'kernel_size': kernel_size, 'stride': stride}
    )

    print('\nCommand-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
