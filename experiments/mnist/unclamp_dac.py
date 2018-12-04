import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from sklearn.metrics import confusion_matrix

from bindsnet.learning import NoOp
from bindsnet.datasets import MNIST
from bindsnet.encoding import bernoulli
from bindsnet.network import load_network, Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from bindsnet.learning import WeightDependentPostPre
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_voltages

from experiments import ROOT_DIR

model = 'unclamp_dac'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_neurons=100, n_train=60000, n_test=10000, inhib=250, lr=1e-2, lr_decay=1, time=100, dt=1,
         theta_plus=0.05, theta_decay=1e-7, intensity=1, progress_interval=10, update_interval=100, plot=False,
         train=True, gpu=False, no_inhib=False, no_theta=False):

    assert n_train % update_interval == 0, 'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, inhib, lr, lr_decay, time, dt, theta_plus,
        theta_decay, intensity, progress_interval, update_interval
    ]

    test_params = [
        seed, n_neurons, n_train, n_test, inhib, lr, lr_decay, time, dt,
        theta_plus, theta_decay, intensity, progress_interval, update_interval
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
    per_class = int(n_neurons / n_classes)

    # Build network.
    if train:
        network = Network()

        input_layer = Input(n=784, traces=True, trace_tc=5e-2)
        network.add_layer(input_layer, name='X')

        output_layer = DiehlAndCookNodes(
            n=n_neurons, traces=True, rest=0, reset=0, thresh=5, refrac=0,
            decay=1e-2, trace_tc=5e-2, theta_plus=theta_plus, theta_decay=theta_decay
        )
        network.add_layer(output_layer, name='Y')

        w = 0.3 * torch.rand(784, n_neurons)
        input_connection = Connection(
            source=network.layers['X'], target=network.layers['Y'], w=w, update_rule=WeightDependentPostPre,
            nu=[0, lr], wmin=0, wmax=1, norm=78.4
        )
        network.add_connection(input_connection, source='X', target='Y')

    else:
        network = load_network(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].theta_decay = 0
        network.layers['Y'].theta_plus = 0

        if no_inhib:
            del network.connections['Y', 'Y']

        if no_theta:
            network.layers['Y'].theta = 0

    # Load MNIST data.
    dataset = MNIST(path=data_path, download=True, shuffle=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images = images.view(-1, 784)
    images *= intensity
    labels = labels.long()

    monitors = {}
    for layer in set(network.layers):
        if 'v' in network.layers[layer].__dict__:
            monitors[layer] = Monitor(network.layers[layer], state_vars=['s', 'v'], time=time)
        else:
            monitors[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)

        network.add_monitor(monitors[layer], name=layer)

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    voltage_ims = None
    voltage_axes = None
    weights_im = None
    theta_im = None

    unclamps = {}
    for label in range(n_classes):
        unclamp = torch.ones(n_neurons).byte()
        unclamp[label * per_class: (label + 1) * per_class] = 0
        unclamps[label] = unclamp

    predictions = torch.zeros(n_examples)
    corrects = torch.zeros(n_examples)
    spike_record = torch.zeros(n_examples, n_neurons)

    flag = False

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

        if i % update_interval == 0 and i > 0 and train:
            network.save(os.path.join(params_path, model_name + '.pt'))
            network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

            if not flag:
                w = -inhib * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
                recurrent_connection = Connection(
                    source=network.layers['Y'], target=network.layers['Y'], w=w, wmin=-inhib, wmax=0
                )
                network.add_connection(recurrent_connection, source='Y', target='Y')

            flag = True

        # Get next input sample.
        image = images[i % len(images)]
        label = labels[i % len(images)].item()
        sample = bernoulli(datum=image, time=time, dt=dt, max_prob=1)
        inpts = {'X': sample}

        # Run the network on the input.
        if train:
            network.run(inpts=inpts, time=time, unclamp={'Y': unclamps[label]})
        else:
            network.run(inpts=inpts, time=time)

        output = monitors['Y'].get('s')
        summed_neurons = output.sum(dim=1).view(n_classes, per_class)
        summed_classes = summed_neurons.sum(dim=1).long()
        prediction = torch.argmax(summed_classes).item()
        correct = prediction == label

        predictions[i] = prediction
        corrects[i] = int(correct)

        spike_record[i] = output.float().sum(dim=1)

        # Optionally plot various simulation information.
        if plot and i % update_interval == 0:
            # _input = image.view(28, 28)
            # reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
            # v = {'Y': monitors['Y'].get('v')}

            s = {layer: monitors[layer].get('s') for layer in monitors}
            input_exc_weights = network.connections['X', 'Y'].w
            square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
            theta = network.layers['Y'].theta.view(per_class, per_class)

            # inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            # voltage_ims, voltage_axes = plot_voltages(v, ims=voltage_ims, axes=voltage_axes)

            spike_ims, spike_axes = plot_spikes(s, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)

            if theta_im is None:
                theta_im = plt.matshow(theta)
                cax = plt.colorbar()
            else:
                theta_im.set_data(theta)
                cax.set_clim(theta.min(), theta.max())

            plt.pause(1e-1)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    if train:
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
                    'random_seed,n_neurons,n_train,inhib,lr,lr_decay,time,timestep,theta_plus,'
                    'theta_decay,intensity,progress_interval,update_interval,accuracy\n'
                )
            else:
                f.write(
                    'random_seed,n_neurons,n_train,n_test,inhib,lr,lr_decay,time,timestep,'
                    'theta_plus,theta_decay,intensity,progress_interval,update_interval,accuracy\n'
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

    if plot:
        plt.ioff()
        plt.matshow(confusion)
        plt.show()

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save(confusion, os.path.join(confusion_path, f))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_neurons', type=int, default=100, help='no. of output layer neurons')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='no. of test samples')
    parser.add_argument('--inhib', type=float, default=250, help='inhibition connection strength')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=50, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1, help='simulation integreation timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--theta_decay', type=float, default=1e-7, help='adaptive threshold decay time constant')
    parser.add_argument('--intensity', type=float, default=0.5, help='constant to multiple input data by')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=100, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.add_argument('--no-inhib', dest='no_inhib', action='store_true', help='whether to remove inhibition on test.')
    parser.add_argument('--no-theta', dest='no_theta', action='store_true', help='whether to remove inhibition on test.')
    parser.set_defaults(plot=False, gpu=False, train=True, no_inhib=False, no_theta=False)
    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
