import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from bindsnet import *
from time     import time as t

sys.path.append('..')

from utils import *

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--inhib', type=float, default=250.0)
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--n_filters', type=int, default=16)
parser.add_argument('--time', type=int, default=250)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--theta_plus', type=float, default=0.05)
parser.add_argument('--theta_decay', type=float, default=1e-7)
parser.add_argument('--intensity', type=float, default=0.25)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)

args = vars(parser.parse_args())
locals().update(args)

print(); print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

model = 'locally_connected'
data = 'cifar10'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [seed, kernel_size, stride, n_filters, n_train,
          inhib, time, dt, theta_plus, theta_decay,
          intensity, progress_interval, update_interval]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [seed, kernel_size, stride, n_filters, n_train,
                   n_test, inhib, time, dt, theta_plus,
                   theta_decay, intensity, progress_interval,
                   update_interval]

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

start_intensity = intensity

if kernel_size == 32:
	conv_size = 1
else:
	conv_size = int((32 - kernel_size) / stride) + 1

n_neurons = n_filters * conv_size ** 2

locations = torch.zeros(kernel_size, kernel_size, conv_size ** 2).long()
for c in range(conv_size ** 2):
    for k1 in range(kernel_size):
        for k2 in range(kernel_size):
            locations[k1, k2, c] = (c % conv_size) * stride * 32 + (c // conv_size) * stride + k1 * 32 + k2

locations = locations.view(kernel_size ** 2, conv_size ** 2)

# Build network.
if train:
    network = Network()
    input_layer = Input(n=32*32*3,
                        traces=True)

    conv_layer = DiehlAndCookNodes(n=n_filters * conv_size * conv_size,
                                   traces=True)

    w = torch.zeros(32 * 32, 3, conv_layer.n)
    for f in range(n_filters):
        for c in range(conv_size ** 2):
            for k in range(kernel_size ** 2):
                w[locations[k, c], 0, f * (conv_size ** 2) + c] = np.random.rand()
                w[locations[k, c], 1, f * (conv_size ** 2) + c] = np.random.rand()
                w[locations[k, c], 2, f * (conv_size ** 2) + c] = np.random.rand()

    w = w.view(input_layer.n, conv_layer.n)

    conv_conn = Connection(input_layer,
                           conv_layer,
                           w=w,
                           update_rule=post_pre,
                           norm=0.6 * kernel_size ** 2,
                           nu_pre=0,
                           nu_post=2e-4,
                           wmax=1.0)

    w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
    for fltr1 in range(n_filters):
        for fltr2 in range(n_filters):
            if fltr1 != fltr2:
                for i in range(conv_size):
                    for j in range(conv_size):
                        w[fltr1, i, j, fltr2, i, j] = -inhib

    recurrent_conn = Connection(conv_layer,
                                conv_layer,
                                w=w)

    network.add_layer(input_layer, name='X')
    network.add_layer(conv_layer, name='Y')
    network.add_connection(conv_conn, source='X', target='Y')
    network.add_connection(recurrent_conn, source='Y', target='Y')
else:
    path = os.path.join('..', '..', 'params', data, model)
    network = load_network(os.path.join(path, model_name + '.p'))
    network.connections[('X', 'Y')].update_rule = None

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

mask = network.connections[('X', 'Y')].w == 0

# Load CIFAR-10 data.
dataset = CIFAR10(path=os.path.join('..', '..', 'data', 'CIFAR10'),
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
    path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.p')
    assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

# Accuracy curves recording.
curves = {'all' : [], 'proportion' : [], 'ngram' : []}

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

start = t()
for i in range(n_examples):
    if i % progress_interval == 0:
        print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
        start = t()

    if i % update_interval == 0 and i > 0:
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
                    torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

                best_accuracy = max([x[-1] for x in curves.values()])

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)

            # Compute ngram scores.
            ngram_scores = update_ngram_scores(spike_record, labels[i - update_interval:i], 10, 2, ngram_scores)

        print()

    # Get next input sample.
    image = images[i].view(-1)
    sample = poisson(datum=image, time=time)
    inpts = {'X' : sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)
    network.connections[('X', 'Y')].w.masked_fill_(mask, 0)

    retries = 0
    while spikes['Y'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = poisson(datum=image, time=time)
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)
        network.connections[('X', 'Y')].w.masked_fill_(mask, 0)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Y'].get('s').t()

    if plot:
        image = image.view(32, 32, 3) / intensity
        image /= image.max()
        inpt = 255 - sample.view(time, 3*32*32).sum(0).view(32, 32, 3).sum(2).float()
        weights = conv_conn.w.view(32, 32, 3, -1).mean(2).view(32 * 32, -1)
        _spikes = {'X' : spikes['X'].get('s').view(input_layer.n, time),
                   'Y' : spikes['Y'].get('s').view(n_filters * conv_size ** 2, time)}
            
        if i == 0:
            spike_ims, spike_axes = plot_spikes(_spikes)
            # inpt_axes, inpt_ims = plot_input(image, inpt, label=labels[i])
            # assigns_im = plot_assignments(square_assignments, classes=classes)
            # perf_ax = plot_performance(curves)
            weights_im = plot_locally_connected_weights(weights, n_filters, kernel_size,
                                                        conv_size, locations, 32,
                                                        wmax=conv_conn.wmax)
            weights_im2 = plot_weights(conv_conn.w)
        else:
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            # inpt_axes, inpt_ims = plot_input(image, inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(curves, ax=perf_ax)
            weights_im = plot_locally_connected_weights(weights, n_filters, kernel_size,
                                                        conv_size, locations, 32,
                                                        im=weights_im)
            weights_im2 = plot_weights(conv_conn.w, im=weights_im2)

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

results = [np.mean(curves['all']),
           np.mean(curves['proportion']),
           np.mean(curves['ngram']),
           np.max(curves['all']),
           np.max(curves['proportion']),
           np.max(curves['ngram'])]

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
            f.write('random_seed,kernel_size,stride,n_filters,' + \
                    'n_train,inhib,time,timestep,' + \
                    'theta_plus,theta_decay,intensity,' + \
                    'progress_interval,update_interval,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')
        else:
            f.write('random_seed,kernel_size,stride,n_filters,' + \
                    'n_train,n_test,inhib,time,timestep,' + \
                    'theta_plus,theta_decay,intensity,' + \
                    'progress_interval,update_interval,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')

with open(os.path.join(path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

print()
