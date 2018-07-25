import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.learning import post_pre
from bindsnet.encoding import bernoulli
from bindsnet.analysis.plotting import *
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.network.topology import Connection, Conv2dConnection

sys.path.append('..')

from utils import *

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--n_filters', type=int, default=25)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--inhib', type=float, default=100.0)
parser.add_argument('--time', type=int, default=25)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--intensity', type=float, default=1)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)

args = vars(parser.parse_args())
locals().update(args)

print('\nCommand-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

model = 'conv'
data = 'mnist'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
    'No. examples must be divisible by update_interval'

params = [seed, n_train, kernel_size, stride, n_filters,
          padding, inhib, time, dt, intensity, update_interval]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [seed, n_train, n_test, kernel_size, stride, n_filters,
                   padding, inhib, time, dt, intensity, update_interval]

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

if kernel_size == 28:
	conv_size = 1
else:
	conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1

n_classes = 10
n_neurons = n_filters * conv_size ** 2
per_class = int(n_neurons / n_classes)

# Build network.
network = Network()
input_layer = Input(n=784, shape=(1, 1, 28, 28), traces=True)
conv_layer = DiehlAndCookNodes(n=n_filters * conv_size * conv_size, shape=(1, n_filters, conv_size, conv_size),
                               thresh=-64.0, traces=True, theta_plus=0.05 * (kernel_size / 28), refrac=0)
conv_layer2 = DiehlAndCookNodes(n=n_filters * conv_size * conv_size, shape=(1, n_filters, conv_size, conv_size),
                                refrac=0, thresh=-60.0)
conv_conn = Conv2dConnection(input_layer, conv_layer, kernel_size=kernel_size, stride=stride, update_rule=post_pre,
                             norm=kernel_size ** 2, nu_pre=0, nu_post=1e-2, wmax=2.0)
conv_conn2 = Conv2dConnection(input_layer, conv_layer2, w=conv_conn.w, kernel_size=kernel_size, stride=stride,
                              update_rule=None, norm=kernel_size ** 2, nu_pre=0, nu_post=1e-2, wmax=2.0)

w = torch.zeros(1, n_filters, conv_size, conv_size, 1, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        # for i1 in range(conv_size):
        #     for j1 in range(conv_size):
        #         for i2 in range(conv_size):
        #             for j2 in range(conv_size):
        #                 if not (i1 == i2 and j1 == j2):
        #                     w[0, fltr1, i1, j1, 0, fltr2, i2, j2] = -inhib

        # if fltr1 != fltr2:
        #     for i in range(conv_size):
        #         for j in range(conv_size):
        #             w[0, fltr1, i, j, 0, fltr2, i, j] = -inhib
        
        for i1 in range(conv_size):
            for j1 in range(conv_size):
                for i2 in range(conv_size):
                    for j2 in range(conv_size):
                        if not (fltr1 == fltr2 and i1 == i2 and j1 == j2):
                            w[0, fltr1, i1, j1, 0, fltr2, i2, j2] = -inhib

        # if fltr1 != fltr2:
        #     for i1 in range(conv_size):
        #         for j1 in range(conv_size):
        #             for i2 in range(conv_size):
        #                 for j2 in range(conv_size):
        #                     w[0, fltr1, i1, j1, 0, fltr2, i2, j2] = -inhib
                    
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

# Load MNIST data.
dataset = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'), download=True)

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
    proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
    rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
    ngram_scores = {}
else:
    path = os.path.join('..', '..', 'params', data, model)
    path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.pt')
    assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

# Sequence of accuracy estimates.
curves = {'all' : [], 'proportion' : [], 'ngram' : []}

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

start = t()
for i in range(n_examples):
    conv_conn2.w = conv_conn.w

    if i % progress_interval == 0:
        print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start))
        start = t()

    if i % update_interval == 0 and i > 0:
        if i % len(labels) == 0:
            current_labels = labels[-update_interval:]
        else:
            current_labels = labels[i - update_interval:i]

        # Update and print accuracy evaluations.
        curves = update_curves(curves, current_labels, n_classes, spike_record=spike_record,
                               assignments=assignments, proportions=proportions,
                               ngram_scores=ngram_scores, n=2)
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

                best_accuracy = max([x[-1] for x in curves.values()])

            # Assign labels to excitatory layer neurons.
            if i % len(labels) == 0:
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)
            else:
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

            # Compute ngram scores.
            ngram_scores = update_ngram_scores(spike_record, labels[i - update_interval:i], n_classes, 2, ngram_scores)

        print()

     # Get next input sample.
    image = images[i]
    sample = bernoulli(datum=image, time=time, max_prob=0.5).unsqueeze(1).unsqueeze(1)
    inpts = {'X' : sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)

    retries = 0
    while spikes['Y_'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = bernoulli(datum=image, time=time, max_prob=0.5).unsqueeze(1).unsqueeze(1)
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Y_'].get('s').view(time, -1)

    # Optionally plot various simulation information.
    if plot:
        _input = inpts['X'].view(time, 784).sum(0).view(28, 28)
        w = conv_conn2.w
        _spikes = {'X': spikes['X'].get('s').view(28 ** 2, time),
                   'Y': spikes['Y'].get('s').view(n_filters * conv_size ** 2, time),
                   'Y_': spikes['Y_'].get('s').view(n_filters * conv_size ** 2, time)}
        
        if i == 0:
            inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), _input, label=labels[i])
            spike_ims, spike_axes = plot_spikes(spikes=_spikes)
            weights_im = plot_conv2d_weights(w, wmax=conv_conn.wmax)
            
        else:
            inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), _input, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_conv2d_weights(w, im=weights_im)
        
        plt.pause(1e-8)
    
    network.reset_()  # Reset state variables.

print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

i += 1

if i % len(labels) == 0:
    current_labels = labels[-update_interval:]
else:
    current_labels = labels[i - update_interval:i]

# Update and print accuracy evaluations.
curves = update_curves(curves, current_labels, n_classes, spike_record=spike_record,
                       assignments=assignments, proportions=proportions,
                       ngram_scores=ngram_scores, n=2)
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
f = '_'.join(to_write) + '.pt'

torch.save((curves, update_interval, n_examples), open(os.path.join(path, f), 'wb'))

# Save results to disk.
path = os.path.join('..', '..', 'results', data, model)
if not os.path.isdir(path):
    os.makedirs(path)

results = [np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
           np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])]

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
            columns = ['seed', 'n_train', 'kernel_size', 'stride', 'n_filters', 'padding', 'inhib', 'time', 'dt',
                       'intensity',  'update_interval', 'mean_all_activity', 'mean_proportion_weighting',
                       'mean_ngram', 'max_all_activity', 'max_proportion_weighting', 'max_ngram']

            header = ','.join(columns) + '\n'
            f.write(header)
        else:
            columns = ['seed', 'n_train', 'n_test', 'kernel_size', 'stride', 'n_filters', 'padding', 'inhib', 'time',
                       'dt', 'intensity',  'update_interval', 'mean_all_activity', 'mean_proportion_weighting',
                       'mean_ngram', 'max_all_activity', 'max_proportion_weighting', 'max_ngram']

            header = ','.join(columns) + '\n'
            f.write(header)

with open(os.path.join(path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

print()
