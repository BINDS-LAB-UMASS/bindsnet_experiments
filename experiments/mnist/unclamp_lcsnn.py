import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from sklearn.metrics import confusion_matrix

from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load_network
from bindsnet.utils import get_square_weights
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.models import LocallyConnectedNetwork
from bindsnet.network.nodes import DiehlAndCookNodes
from bindsnet.learning import NoOp, WeightDependentPostPre
from bindsnet.analysis.plotting import plot_locally_connected_weights, plot_spikes, plot_weights

from experiments import ROOT_DIR

model = 'unclamp_lcsnn'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_train=60000, n_test=10000, inhib=250, kernel_size=(16,), stride=(2,), n_filters=25, n_output=100,
         time=100, crop=0, lr=1e-2, lr_decay=0.99, dt=1, theta_plus=0.05, theta_decay=1e-7, intensity=1, norm=0.2,
         progress_interval=10, update_interval=250, train=True, test=False, relabel=False, plot=False, gpu=False):

    assert train + test + relabel == 1
    assert n_train % update_interval == 0 and n_test % update_interval == 0 or relabel, \
        'No. examples must be divisible by update_interval'

    params = [
        seed, kernel_size, stride, n_filters, crop, lr, lr_decay, n_train, inhib, time, dt,
        theta_plus, theta_decay, intensity, norm, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    if test or relabel:
        test_params = [
            seed, kernel_size, stride, n_filters, crop, lr, lr_decay, n_train, n_test, inhib, time, dt,
            theta_plus, theta_decay, intensity, norm, progress_interval, update_interval
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
            n_filters=n_filters, inh=inhib, dt=dt, nu_pre=0, nu_post=lr, theta_plus=theta_plus,
            theta_decay=theta_decay, wmin=0.0, wmax=1.0, norm=norm
        )

        output_layer = DiehlAndCookNodes(
            n=n_output, traces=True, rest=0, reset=0, thresh=1, refrac=0,
            decay=1e-2, trace_tc=5e-2, theta_plus=theta_plus, theta_decay=theta_decay
        )

        conv_size = network.connections['X', 'Y'].conv_size
        conv_prod = int(np.prod(conv_size))
        n_neurons = n_filters * conv_prod

        hidden_output_connection = Connection(
            network.layers['Y'], output_layer, nu=[0, 5 * lr],
            update_rule=WeightDependentPostPre, wmin=0,
            wmax=1, norm=0.75 * norm * n_neurons
        )

        w = -inhib * (torch.ones(n_output, n_output) - torch.diag(torch.ones(n_output)))
        output_recurrent_connection = Connection(
            output_layer, output_layer, w=w, update_rule=NoOp, wmin=-inhib, wmax=0
        )

        network.add_layer(output_layer, name='Z')
        network.add_connection(hidden_output_connection, source='Y', target='Z')
        network.add_connection(output_recurrent_connection, source='Z', target='Z')
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

    # Load MNIST data.
    dataset = MNIST(path=data_path, download=True)

    if train:
        images, labels = dataset.get_train()
    elif test:
        images, labels = dataset.get_test()

    images *= intensity
    images = images[:, crop:-crop, crop:-crop].contiguous().view(-1, side_length ** 2)

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name=f'{layer}_spikes')

    # Train the network.
    if train:
        print('\nBegin training.\n')
    elif test:
        print('\nBegin test.\n')

    spike_ims = None
    spike_axes = None
    weights_im = None
    weights2_im = None

    unclamps = {}
    per_class = int(n_output / n_classes)
    for label in range(n_classes):
        unclamp = torch.ones(n_output).byte()
        unclamp[label * per_class: (label + 1) * per_class] = 0
        unclamps[label] = unclamp

    predictions = torch.zeros(n_examples)
    corrects = torch.zeros(n_examples)

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

        if i % update_interval == 0 and i > 0:
            network.save(os.path.join(params_path, model_name + '.pt'))

            if train:
                network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

        # Get next input sample.
        image = images[i % len(images)]
        label = labels[i % len(images)].item()
        sample = poisson(datum=image, time=time, dt=dt)
        inpts = {'X': sample}

        # Run the network on the input.
        if train:
            network.run(inpts=inpts, time=time, unclamp={'Z': unclamps[label]})
        else:
            network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Z'].get('s').sum() < 5 and retries < 3:
            retries += 1
            image *= 2
            sample = poisson(datum=image, time=time, dt=dt)
            inpts = {'X': sample}

            if train:
                network.run(inpts=inpts, time=time, unclamp={'Z': unclamps[label]})
            else:
                network.run(inpts=inpts, time=time)

        output = spikes['Z'].get('s')
        summed_neurons = output.sum(dim=1).view(per_class, n_classes)
        summed_classes = summed_neurons.sum(dim=1)
        prediction = torch.argmax(summed_classes).item()
        correct = prediction == label

        predictions[i] = prediction
        corrects[i] = int(correct)

        # Optionally plot various simulation information.
        if plot:
            _spikes = {
                'X': spikes['X'].get('s').view(side_length ** 2, time),
                'Y': spikes['Y'].get('s').view(n_neurons, time),
                'Z': spikes['Z'].get('s').view(n_output, time)
            }

            spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_locally_connected_weights(
                network.connections['X', 'Y'].w, n_filters, kernel_size,
                conv_size, locations, side_length, im=weights_im
            )

            n_sqrt = int(np.ceil(np.sqrt(n_output)))
            w = network.connections['Y', 'Z'].w
            w = get_square_weights(w, n_sqrt=n_sqrt, side=15)

            weights2_im = plot_weights(
                w, im=weights2_im
            )

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    network.save(os.path.join(params_path, model_name + '.pt'))

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')

    accuracy = torch.mean(corrects).item() * 100

    print(f'\nAccuracy: {accuracy}\n')

    to_write = params + [accuracy] if train else test_params + [accuracy]
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            if train:
                f.write(
                    'random_seed,kernel_size,stride,n_filters,crop,lr,lr_decay,n_train,inhib,time,timestep,theta_plus,'
                    'theta_decay,intensity,norm,progress_interval,accuracy\n'
                )
            else:
                f.write(
                    'random_seed,kernel_size,stride,n_filters,crop,lr,lr_decay,n_train,n_test,inhib,time,timestep,'
                    'theta_plus,theta_decay,intensity,norm,progress_interval,update_interval,accuracy\n'
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
    confusion = confusion_matrix(labels, predictions)

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save(confusion, os.path.join(confusion_path, f))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='no. of test samples')
    parser.add_argument('--inhib', type=float, default=250, help='inhibition connection strength')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[16], help='one or two kernel side lengths')
    parser.add_argument('--stride', type=int, nargs='+', default=[2], help='one or two horizontal stride lengths')
    parser.add_argument('--n_filters', type=int, default=25, help='no. of locally connected filters')
    parser.add_argument('--n_output', type=int, default=100, help='no. of output neurons')
    parser.add_argument('--crop', type=int, default=4, help='amount to crop images at borders')
    parser.add_argument('--lr', type=float, default=0.01, help='post-synaptic learning rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=100, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1.0, help='simulation integreation timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--theta_decay', type=float, default=1e-7, help='adaptive threshold decay time constant')
    parser.add_argument('--intensity', type=float, default=1, help='constant to multiple input data by')
    parser.add_argument('--norm', type=float, default=0.2, help='plastic synaptic weight normalization constant')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--no-train', dest='train', action='store_false', help='train phase')
    parser.add_argument('--test', dest='test', action='store_true', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True, test=False, relabel=False)
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
