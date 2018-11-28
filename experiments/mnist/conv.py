import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from bindsnet.datasets import MNIST
from bindsnet.encoding import bernoulli
from bindsnet.network.monitors import Monitor
from bindsnet.network import Network, load_network
from bindsnet.evaluation import assign_labels, logreg_fit
from bindsnet.learning import WeightDependentPostPre, NoOp
from bindsnet.network.topology import Connection, Conv2dConnection
from bindsnet.network.nodes import Input, DiehlAndCookNodes, LIFNodes
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_conv2d_weights

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


def main(seed=0, n_train=60000, n_test=10000, kernel_size=(16,), stride=(4,), n_filters=25, padding=0, inhib=100,
         time=25, lr=1e-3, lr_decay=0.99, dt=1, intensity=1, progress_interval=10, update_interval=250, plot=False,
         train=True, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
        'No. examples must be divisible by update_interval'

    params = [
        seed, n_train, kernel_size, stride, n_filters, padding,
        inhib, time, lr, lr_decay, dt, intensity, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    if not train:
        test_params = [
            seed, n_train, n_test, kernel_size, stride, n_filters, padding,
            inhib, time, lr, lr_decay, dt, intensity, update_interval
        ]

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    n_examples = n_train if train else n_test
    input_shape = [20, 20]

    if kernel_size == input_shape:
        conv_size = [1, 1]
    else:
        conv_size = (int((input_shape[0] - kernel_size[0]) / stride[0]) + 1,
                     int((input_shape[1] - kernel_size[1]) / stride[1]) + 1)

    n_classes = 10
    n_neurons = n_filters * np.prod(conv_size)
    total_kernel_size = int(np.prod(kernel_size))
    total_conv_size = int(np.prod(conv_size))

    # Build network.
    if train:
        network = Network()
        input_layer = Input(n=400, shape=(1, 1, 20, 20), traces=True)
        conv_layer = DiehlAndCookNodes(
            n=n_filters * total_conv_size, shape=(1, n_filters, *conv_size), thresh=-64.0,
            traces=True, theta_plus=0.05 * (kernel_size[0] / 20), refrac=0
        )
        conv_layer2 = LIFNodes(n=n_filters * total_conv_size, shape=(1, n_filters, *conv_size), refrac=0)
        conv_conn = Conv2dConnection(
            input_layer, conv_layer, kernel_size=kernel_size, stride=stride, update_rule=WeightDependentPostPre,
            norm=0.05 * total_kernel_size, nu=[0, lr], wmin=0, wmax=0.25
        )
        conv_conn2 = Conv2dConnection(
            input_layer, conv_layer2, w=conv_conn.w, kernel_size=kernel_size,
            stride=stride, update_rule=None, wmax=0.25
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

        network.add_layer(input_layer, name='X')
        network.add_layer(conv_layer, name='Y')
        network.add_layer(conv_layer2, name='Y_')
        network.add_connection(conv_conn, source='X', target='Y')
        network.add_connection(conv_conn2, source='X', target='Y_')
        network.add_connection(recurrent_conn, source='Y', target='Y')

        # Voltage recording for excitatory and inhibitory layers.
        voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
        network.add_monitor(voltage_monitor, name='output_voltage')
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

    images *= intensity
    images = images[:, 4:-4, 4:-4].contiguous()

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, time, n_neurons)
    full_spike_record = torch.zeros(n_examples, n_neurons)

    # Neuron assignments and spike proportions.
    if train:
        logreg_model = LogisticRegression(warm_start=True, n_jobs=-1, solver='lbfgs')
        logreg_model.coef_ = np.zeros([n_classes, n_neurons])
        logreg_model.intercept_ = np.zeros(n_classes)
        logreg_model.classes_ = np.arange(n_classes)
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        logreg_coef, logreg_intercept = torch.load(open(path, 'rb'))
        logreg_model = LogisticRegression(warm_start=True, n_jobs=-1, solver='lbfgs')
        logreg_model.coef_ = logreg_coef
        logreg_model.intercept_ = logreg_intercept
        logreg_model.classes_ = np.arange(n_classes)

    # Sequence of accuracy estimates.
    curves = {'logreg': []}
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
                current_record = full_spike_record[-update_interval:]
            else:
                current_labels = labels[i % len(labels) - update_interval:i % len(labels)]
                current_record = full_spike_record[i % len(labels) - update_interval:i % len(labels)]

            # Update and print accuracy evaluations.
            curves, preds = update_curves(
                curves, current_labels, n_classes, full_spike_record=current_record, logreg=logreg_model
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
                        (logreg_model.coef_, logreg_model.intercept_), open(path, 'wb')
                    )
                    best_accuracy = max([x[-1] for x in curves.values()])

                # Refit logistic regression model.
                logreg_model = logreg_fit(full_spike_record[:i], labels[:i], logreg_model)

            print()

        # Get next input sample.
        image = images[i % len(images)]
        sample = bernoulli(datum=image, time=time, dt=dt, max_prob=1).unsqueeze(1).unsqueeze(1)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        network.connections['X', 'Y_'].w = network.connections['X', 'Y'].w

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Y_'].get('s').view(time, -1)
        full_spike_record[i] = spikes['Y_'].get('s').view(time, -1).sum(0)

        # Optionally plot various simulation information.
        if plot:
            _input = inpts['X'].view(time, 400).sum(0).view(20, 20)
            w = network.connections['X', 'Y'].w

            _spikes = {
                'X': spikes['X'].get('s').view(400, time),
                'Y': spikes['Y'].get('s').view(n_filters * total_conv_size, time),
                'Y_': spikes['Y_'].get('s').view(n_filters * total_conv_size, time)
            }

            inpt_axes, inpt_ims = plot_input(
                image.view(20, 20), _input, label=labels[i % len(labels)], ims=inpt_ims, axes=inpt_axes
            )
            spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_conv2d_weights(w, im=weights_im, wmax=network.connections['X', 'Y'].wmax)

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    i += 1

    if i % len(labels) == 0:
        current_labels = labels[-update_interval:]
        current_record = full_spike_record[-update_interval:]
    else:
        current_labels = labels[i % len(labels) - update_interval:i % len(labels)]
        current_record = full_spike_record[i % len(labels) - update_interval:i % len(labels)]

    # Update and print accuracy evaluations.
    curves, preds = update_curves(
        curves, current_labels, n_classes, full_spike_record=current_record, logreg=logreg_model
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
                (logreg_model.coef_, logreg_model.intercept_), open(path, 'wb')
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
        np.mean(curves['logreg']), np.std(curves['logreg'])
    ]

    to_write = params + results if train else test_params + results
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            if train:
                columns = [
                    'seed', 'n_train', 'kernel_size', 'stride', 'n_filters', 'padding', 'inhib', 'time',
                    'lr', 'lr_decay', 'dt', 'intensity', 'update_interval', 'mean_logreg', 'std_logreg'
                ]

                header = ','.join(columns) + '\n'
                f.write(header)
            else:
                columns = [
                    'seed', 'n_train', 'n_test', 'kernel_size', 'stride', 'n_filters', 'padding', 'inhib', 'time',
                    'lr', 'lr_decay', 'dt', 'intensity', 'update_interval', 'mean_logreg', 'std_logreg'
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
    parser.add_argument('--padding', type=int, default=0, help='horizontal, vertical padding size')
    parser.add_argument('--inhib', type=float, default=250, help='inhibition connection strength')
    parser.add_argument('--time', default=25, type=int, help='simulation time')
    parser.add_argument('--lr', type=float, default=1e-3, help='post-synaptic learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='rate at which to decay learning rate')
    parser.add_argument('--dt', type=float, default=1.0, help='simulation integreation timestep')
    parser.add_argument('--intensity', type=float, default=0.5, help='constant to multiple input data by')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    seed = args.seed
    n_train = args.n_train
    n_test = args.n_test
    kernel_size = args.kernel_size
    stride = args.stride
    n_filters = args.n_filters
    padding = args.padding
    inhib = args.inhib
    time = args.time
    lr = args.lr
    lr_decay = args.lr_decay
    dt = args.dt
    intensity = args.intensity
    progress_interval = args.progress_interval
    update_interval = args.update_interval
    train = args.train
    plot = args.plot
    gpu = args.gpu

    if len(kernel_size) == 1:
        kernel_size = (kernel_size[0], kernel_size[0])
    else:
        kernel_size = tuple(kernel_size)

    if len(stride) == 1:
        stride = (stride[0], stride[0])
    else:
        stride = tuple(stride)

    args = vars(args)

    print('\nCommand-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(seed=seed, n_train=n_train, n_test=n_test, kernel_size=kernel_size, stride=stride, n_filters=n_filters,
         padding=padding, inhib=inhib, time=time, lr=lr, lr_decay=lr_decay, dt=dt, intensity=intensity,
         progress_interval=progress_interval, update_interval=update_interval, plot=plot, train=train, gpu=gpu)

    print()
