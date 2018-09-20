import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.learning import NoOp
from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load_network
from bindsnet.network.monitors import Monitor
from bindsnet.models import LocallyConnectedNetwork
from bindsnet.analysis.plotting import plot_locally_connected_weights, plot_spikes
from bindsnet.evaluation import all_activity, proportion_weighting, ngram, assign_labels, update_ngram_scores

sys.path.append('..')

from utils import *

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--inhib', type=float, default=250.0)
parser.add_argument('--kernel_size', nargs='+', type=int, default=[16])
parser.add_argument('--stride', nargs='+', type=int, default=[4])
parser.add_argument('--n_filters', type=int, default=16)
parser.add_argument('--time', type=int, default=250)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--theta_plus', type=float, default=0.05)
parser.add_argument('--theta_decay', type=float, default=1e-7)
parser.add_argument('--intensity', type=float, default=1)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_train = args.n_train
n_test = args.n_test
inhib = args.inhib
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
time = args.time
dt = args.dt
theta_plus = args.theta_plus
theta_decay = args.theta_decay
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

if len(kernel_size) == 1:
    kernel_size = kernel_size[0]
if len(stride) == 1:
    stride = stride[0]

args = vars(args)

print()
print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

model = 'locally_connected'
data = 'mnist'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [
    seed, kernel_size, stride, n_filters, n_train, inhib, time, dt,
    theta_plus, theta_decay, intensity, progress_interval, update_interval
]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [
        seed, kernel_size, stride, n_filters, n_train, n_test, inhib, time, dt,
        theta_plus, theta_decay, intensity, progress_interval, update_interval
    ]

np.random.seed(seed)

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

n_examples = n_train if train else n_test
start_intensity = intensity
n_classes = 10
per_class = int(n_examples / n_classes)

# Build network.
if train:
    network = LocallyConnectedNetwork(
        n_inpt=784, input_shape=[28, 28], kernel_size=kernel_size, stride=stride, n_filters=n_filters, inh=inhib,
        dt=dt, nu_pre=1e-4, nu_post=1e-2, theta_plus=theta_plus, theta_decay=theta_decay, wmin=0.0, wmax=1.0, norm=0.2
    )
else:
    path = os.path.join('..', '..', 'params', data, model)
    network = load_network(os.path.join(path, model_name + '.pt'))
    network.connections[('X', 'Y')].update_rule = NoOp(
        connection=network.connections[('X', 'Y')], nu=network.connections[('X', 'Y')].nu
    )

conv_size = network.connections[('X', 'Y')].conv_size
locations = network.connections[('X', 'Y')].locations
conv_prod = int(np.prod(conv_size))
n_neurons = n_filters * conv_prod

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

# Load MNIST data.
dataset = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'),
                download=True)

if train:
    images, labels = dataset.get_train()
else:
    images, labels = dataset.get_test()

images *= intensity

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
if train:
    assignments = -torch.ones_like(torch.Tensor(n_neurons))
    proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
    rates = torch.zeros_like(torch.Tensor(n_neurons, 10))
    ngram_scores = {}
else:
    path = os.path.join('..', '..', 'params', data, model)
    path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.pt')
    assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

# Accuracy curves recording.
curves = {'all': [], 'proportion': [], 'ngram': []}

if train:
    best_accuracy = 0

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
    if i % progress_interval == 0:
        print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
        start = t()

    if i % update_interval == 0 and i > 0:
        if i % len(labels) == 0:
            current_labels = labels[-update_interval:]
        else:
            current_labels = labels[i - update_interval:i]

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
            assignments, proportions, rates = assign_labels(spike_record, current_labels, 10, rates)

            # Compute ngram scores.
            ngram_scores = update_ngram_scores(spike_record, current_labels, 10, 2, ngram_scores)

        print()

    # Get next input sample.
    image = images[i].view(-1)
    sample = poisson(datum=image, time=time)
    inpts = {'X': sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)

    retries = 0
    while spikes['Y'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = poisson(datum=image, time=time)
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Y'].get('s').t()
    
    # Optionally plot various simulation information.
    if plot:
        inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
        _spikes = {'X': spikes['X'].get('s').view(28 ** 2, time),
                   'Y': spikes['Y'].get('s').view(n_filters * conv_prod, time)}

        spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
        weights_im = plot_locally_connected_weights(
            network.connections[('X', 'Y')].w, n_filters, kernel_size, conv_size, locations, 28, im=weights_im
        )

        plt.pause(1e-8)
    
    network.reset_()  # Reset state variables.

print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

i += 1

if i % len(labels) == 0:
    current_labels = labels[-update_interval:]
else:
    current_labels = labels[i - update_interval:i]

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

            network.save(os.path.join(path, model_name + '.p'))
            path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.p')
            torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

        best_accuracy = max([x[-1] for x in curves.values()])

if train:
    print('\nTraining complete.\n')
else:
    print('\nTest complete.\n')

print('Average accuracies:\n')
for scheme in curves.keys():
    print('\t%s: %.2f' % (scheme, np.mean(curves[scheme])))

# Save accuracy curves to disk.
path = os.path.join('..', '..', 'curves', data, model)
if not os.path.isdir(path):
    os.makedirs(path)

if train:
    to_write = ['train'] + params
else:
    to_write = ['test'] + params

to_write = [str(x) for x in to_write]
f = '_'.join(to_write) + '.p'

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
            f.write('random_seed,kernel_size,stride,n_filters,'
                    'n_train,inhib,time,timestep,'
                    'theta_plus,theta_decay,intensity,'
                    'progress_interval,update_interval,mean_all_activity,'
                    'mean_proportion_weighting,mean_ngram,max_all_activity,'
                    'max_proportion_weighting,max_ngram\n')
        else:
            f.write('random_seed,kernel_size,stride,n_filters,'
                    'n_train,n_test,inhib,time,timestep,'
                    'theta_plus,theta_decay,intensity,'
                    'progress_interval,update_interval,mean_all_activity,'
                    'mean_proportion_weighting,mean_ngram,max_all_activity,'
                    'max_proportion_weighting,max_ngram\n')

with open(os.path.join(path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

print()
