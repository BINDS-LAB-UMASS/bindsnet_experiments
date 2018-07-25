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
from bindsnet.analysis.plotting import *
from bindsnet.encoding import poisson_loader
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.network.topology import Connection, Conv2dConnection

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--n_filters', type=int, default=25)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--time', type=int, default=50)
parser.add_argument('--dt', type=int, default=1.0)
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

params = [seed, n_neurons, n_train, inhib, time, dt, theta_plus, theta_decay,
          intensity, progress_interval, update_interval, X_Ae_decay]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [seed, n_neurons, n_train, n_test, inhib, time, dt, theta_plus, theta_decay,
                   intensity, progress_interval, update_interval, X_Ae_decay]

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

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

if kernel_size == 28:
	conv_size = 1
else:
	conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1

per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(n=784, shape=(1, 1, 28, 28), traces=True)
conv_layer = DiehlAndCookNodes(n=n_filters * conv_size * conv_size, shape=(1, n_filters, conv_size, conv_size),
                               traces=True)
conv_conn = Conv2dConnection(input_layer, conv_layer, kernel_size=kernel_size, stride=stride, update_rule=post_pre,
                             norm=0.4 * kernel_size ** 2, nu_pre=1e-4, nu_post=1e-2, wmax=1.0)

w = torch.zeros(1, n_filters, conv_size, conv_size, 1, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[0, fltr1, i, j, 0, fltr2, i, j] = -100.0
                    
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer, name='Y')
network.add_connection(conv_conn, source='X', target='Y')
network.add_connection(recurrent_conn, source='Y', target='Y')

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'), download=True)

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
    path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.p')
    assignments, proportions, rates, ngram_scores = p.load(open(path, 'rb'))

# Sequence of accuracy estimates.
curves = {'all' : [], 'proportion' : [], 'ngram' : []}

if train:
    best_accuracy = 0

# Lazily encode data as Poisson spike trains.
data_loader = poisson_loader(data=images, time=time)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

# Train the network.
if train:
    print('\nBegin training.\n')
else:
    print('\nBegin test.\n')

for i in range(n_examples):    
    if i % progress_interval == 0:
        print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start)); start = t()

    if i % update_interval == 0 and i > 0:
        # Get network predictions.
        for scheme in curves:
            if scheme == 'all':
                prediction = all_activity(spike_record, assignments, 10)
            elif scheme == 'proportion':
                prediction = proportion_weighting(spike_record, assignments, proportions, 10)
            elif scheme == 'ngram_pred':
                prediction = ngram(spike_record, ngram_scores, 10, 2)

            if i % n_examples == 0:
            curves[scheme] = 100 * torch.sum(labels[i - update_interval:i].long() == prediction) / update_interval

        # Compute network accuracy according to chosen classification strategies.
        all_acc = 100 * torch.sum(labels[i - update_interval:i].long() == all_pred) / update_interval
        proportion_acc = 100 * torch.sum(labels[i - update_interval:i].long() == prop_pred) / update_interval
        ngram_acc = 100 * torch.sum(labels[i - update_interval:i].long() == ngram_pred) / update_interval

        curves['all'].append(all_acc)
        curves['proportion'].append(proportion_acc)
        curves['ngram'].append(ngram_acc)

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
                    p.dump((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

                best_accuracy = max([x[-1] for x in curves.values()])

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)

            # Compute ngram scores.
            ngram_scores = update_ngram_scores(spike_record, labels[i - update_interval:i], 10, 2, ngram_scores)

        print()

     # Get next input sample.
    image = images[i]
    sample = poisson(datum=image, time=time).unsqueeze(1).unsqueeze(1)
    inpts = {'X' : sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)

    retries = 0
    while spikes['Ae'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = poisson(datum=image, time=time)
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Ae'].get('s').t()
    
    # Optionally plot various simulation information.
    if plot:
        _input = inpts['X'].view(time, 784).sum(0).view(28, 28)
        w = conv_conn.w
        _spikes = {'X' : spikes['X'].get('s').view(28 ** 2, time),
                   'Y' : spikes['Y'].get('s').view(n_filters * conv_size ** 2, time)}
        _voltages = {'Y' : voltages['Y'].get('v').view(n_filters * conv_size ** 2, time)}
        
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

# Get network predictions.
all_activity_pred = all_activity(spike_record, assignments, 10)
proportion_pred = proportion_weighting(spike_record, assignments, proportions, 10)
ngram_pred = ngram(spike_record, ngram_scores, 10, 2)

# Compute network accuracy according to available classification strategies.
curves['all'].append(100 * torch.sum(labels[i - update_interval:i].long() == all_activity_pred) / update_interval)
curves['proportion'].append(100 * torch.sum(labels[i - update_interval:i].long() == proportion_pred) / update_interval)
curves['ngram'].append(100 * torch.sum(labels[i - update_interval:i].long() == ngram_pred) / update_interval)

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
            p.dump((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

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

p.dump((curves, update_interval, n_examples), open(os.path.join(path, f), 'wb'))

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
            f.write('random_seed,n_neurons,n_train,' + \
                    'inhib,time,timestep,theta_plus,theta_decay,' + \
                    'intensity,progress_interval,update_interval,' + \
                    'X_Ae_decay,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')
        else:
            f.write('random_seed,n_neurons,n_train,n_test,' + \
                    'inhib,time,timestep,theta_plus,theta_decay,' + \
                    'intensity,progress_interval,update_interval,' + \
                    'X_Ae_decay,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')

with open(os.path.join(path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

print()
