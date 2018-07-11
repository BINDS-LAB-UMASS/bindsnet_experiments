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
parser.add_argument('--n_neurons', type=int, default=100)
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

args = vars(parser.parse_args())
locals().update(args)

print(); print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

model = 'two_layer_locally_connected'
data = 'mnist'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [seed, kernel_size, stride, n_filters, n_neurons,
          n_train, inhib, time, dt, theta_plus, theta_decay,
          intensity, progress_interval, update_interval]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [seed, kernel_size, stride, n_filters, n_neurons,
                   n_train, n_test, inhib, time, dt, theta_plus,
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

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

if kernel_size == 28:
	conv_size = 1
else:
	conv_size = int((28 - kernel_size) / stride) + 1

n_conv_neurons = n_filters * conv_size ** 2
n_conv_sqrt = int(np.ceil(np.sqrt(n_conv_neurons)))

locations = torch.zeros(kernel_size, kernel_size, conv_size ** 2).long()
for c in range(conv_size ** 2):
    for k1 in range(kernel_size):
        for k2 in range(kernel_size):
            locations[k1, k2, c] = (c % conv_size) * stride * 28 + (c // conv_size) * stride + k1 * 28 + k2

locations = locations.view(kernel_size ** 2, conv_size ** 2)

# Build network.
if train:
    network = Network()
    input_layer = Input(n=784,
                        traces=True)

    conv_layer = DiehlAndCookNodes(n=n_filters * conv_size * conv_size,
                                   traces=True)

    output_layer = DiehlAndCookNodes(n=n_neurons,
                                     traces=True)

    w = torch.zeros(input_layer.n, conv_layer.n)
    for f in range(n_filters):
        for c in range(conv_size ** 2):
            for k in range(kernel_size ** 2):
                w[locations[k, c], f * (conv_size ** 2) + c] = np.random.rand()

    conv_conn = Connection(input_layer,
                           conv_layer,
                           w=w,
                           update_rule=post_pre,
                           norm=0.2 * kernel_size ** 2,
                           nu_pre=1e-4,
                           nu_post=1e-2,
                           wmax=1.0)

    w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
    for fltr1 in range(n_filters):
        for fltr2 in range(n_filters):
            if fltr1 != fltr2:
                for i in range(conv_size):
                    for j in range(conv_size):
                        w[fltr1, i, j, fltr2, i, j] = -inhib

    conv_recurrent_conn = Connection(conv_layer,
                                conv_layer,
                                w=w)

    output_conn = Connection(conv_layer,
                             output_layer,
                             update_rule=post_pre,
                             norm=0.5 * n_conv_neurons,
                             nu_pre=1e-3,
                             nu_post=1e-1,
                             wmax=2.0)

    w = -inhib * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
    output_recurrent_conn = Connection(output_layer,
                                       output_layer,
                                       w=w)

    network.add_layer(input_layer, name='X')
    network.add_layer(conv_layer, name='Y')
    network.add_layer(output_layer, name='Z')
    network.add_connection(conv_conn, source='X', target='Y')
    network.add_connection(conv_recurrent_conn, source='Y', target='Y')
    network.add_connection(output_conn, source='Y', target='Z')
    # network.add_connection(output_recurrent_conn, source='Z', target='Z')
else:
    path = os.path.join('..', '..', 'params', data, model)
    network = load_network(os.path.join(path, model_name + '.p'))
    
    for c in network.connections:
        network.connections[c].update_rule = None

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

mask = network.connections[('X', 'Y')].w == 0

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
    path = os.path.join(path, '_'.join(['auxiliary', model_name]) + '.p')
    assignments, proportions, rates, ngram_scores = p.load(open(path, 'rb'))

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
                    p.dump((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

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
    while spikes['Z'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = poisson(datum=image, time=time)
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)
        network.connections[('X', 'Y')].w.masked_fill_(mask, 0)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Z'].get('s').t()
    
    # Optionally plot various simulation information.
    if plot and i % progress_interval == 0:
        inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
        _spikes = {'X' : spikes['X'].get('s').view(28 ** 2, time),
                   'Y' : spikes['Y'].get('s').view(n_filters * conv_size ** 2, time),
                   'Z' : spikes['Z'].get('s').view(n_neurons, time)}
        conv_weights = get_square_weights(output_conn.w.view(n_conv_neurons, n_neurons), n_sqrt, n_conv_sqrt)

        if i == 0:
            spike_ims, spike_axes = plot_spikes(_spikes)
            conv_weights_im = plot_locally_connected_weights(conv_conn.w, n_filters, kernel_size,
                                                             conv_size, locations, 28,
                                                             wmax=conv_conn.wmax)
            output_weights_im = plot_weights(conv_weights, wmax=2.0)
        else:
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            conv_weights_im = plot_locally_connected_weights(conv_conn.w, n_filters, kernel_size,
                                                             conv_size, locations, 28,
                                                             im=conv_weights_im)
            output_weights_im = plot_weights(conv_weights, wmax=2.0, im=output_weights_im)
        
        plt.pause(1e-8)
    
    network._reset()  # Reset state variables.

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
                    'n_neurons,n_train,inhib,time,timestep,' + \
                    'theta_plus,theta_decay,intensity,' + \
                    'progress_interval,update_interval,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')
        else:
            f.write('random_seed,kernel_size,stride,n_filters,' + \
                    'n_neurons,n_train,n_test,inhib,time,timestep,' + \
                    'theta_plus,theta_decay,intensity,' + \
                    'progress_interval,update_interval,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')

with open(os.path.join(path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

print()
