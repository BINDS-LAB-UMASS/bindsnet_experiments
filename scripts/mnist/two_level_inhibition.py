import os
import sys
import torch
import argparse
import numpy as np
import pickle as p
import matplotlib.pyplot as plt

from bindsnet import *
from time import time as t
from scipy.spatial.distance import euclidean

sys.path.append('..')

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--c_low', type=float, default=5.0)
parser.add_argument('--c_high', type=float, default=300.0)
parser.add_argument('--p_low', type=float, default=0.1)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--theta_plus', type=float, default=0.05)
parser.add_argument('--theta_decay', type=float, default=1e-7)
parser.add_argument('--intensity', type=float, default=0.5)
parser.add_argument('--X_Ae_decay', type=float, default=0.5)
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

model = 'two_level_inhibition'
data = 'mnist'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [seed, n_neurons, n_train, c_low, c_high,
          p_low, time, dt, theta_plus, theta_decay,
          intensity, progress_interval,
          update_interval, X_Ae_decay]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [seed, n_neurons, n_train, n_test, c_low,
                   c_high, p_low, time, dt, theta_plus,
                   theta_decay, intensity, progress_interval,
                   update_interval, X_Ae_decay]

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

if train:
    iter_increase = int(n_train * p_low)
    print(f'Iteration to increase from c_low to c_high: {iter_increase}\n')

# Build network.
if train:
    network = Network(dt=dt)
    input_layer = Input(n=784,
                        traces=True)

    exc_layer = DiehlAndCookNodes(n=n_neurons,
                                  traces=True)

    w = torch.rand(input_layer.n, exc_layer.n)

    input_exc_conn = Connection(input_layer,
                                exc_layer,
                                w=w,
                                update_rule=post_pre,
                                norm=78.4,
                                nu_pre=1e-4,
                                nu_post=1e-2,
                                wmax=1.0)

    w = torch.zeros(exc_layer.n, exc_layer.n)
    for k1 in range(n_neurons):
        for k2 in range(n_neurons):
            if k1 != k2:
                x1, y1 = k1 // np.sqrt(n_neurons), k1 % np.sqrt(n_neurons)
                x2, y2 = k2 // np.sqrt(n_neurons), k2 % np.sqrt(n_neurons)

                w[k1, k2] = max(-c_high, -c_low * np.sqrt(euclidean([x1, y1], [x2, y2])))

    recurrent_conn = Connection(exc_layer,
                                exc_layer,
                                w=w)

    network.add_layer(input_layer, name='X')
    network.add_layer(exc_layer, name='Y')
    network.add_connection(input_exc_conn, source='X', target='Y')
    network.add_connection(recurrent_conn, source='Y', target='Y')
else:
    path = os.path.join('..', '..', 'params', data, model)
    network = load_network(os.path.join(path, model_name + '.p'))
    network.connections[('X', 'Y')].update_rule = None

# Load MNIST data.
dataset = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'),
                download=True)

if train:
    images, labels = dataset.get_train()
else:
    images, labels = dataset.get_test()

images = images.view(-1, 784)
images *= intensity

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, int(time / dt), n_neurons)

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

spikes = {}
for layer in set(network.layers) - {'X'}:
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=int(time / dt))
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

# Train the network.
if train:
    print('\nBegin training.\n')
else:
    print('\nBegin test.\n')

start = t()
for i in range(n_examples):
    if train and i == iter_increase:
        w = -c_high * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
        network.connections[('Y', 'Y')].w = w

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
    image = images[i]
    sample = poisson(datum=image, time=int(time / dt))
    inpts = {'X' : sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)

    retries = 0
    while spikes['Y'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = poisson(datum=image, time=int(time / dt))
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Y'].get('s').t()

    # Optionally plot various simulation information.
    if plot:
        # inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
        input_exc_weights = network.connections[('X', 'Y')].w
        square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
        # square_assignments = get_square_assignments(assignments, n_sqrt)

        if i == 0:
            # inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i])
            spike_ims, spike_axes = plot_spikes({layer : spikes[layer].get('s') for layer in spikes})
            weights_im = plot_weights(square_weights)
            # assigns_im = plot_assignments(square_assignments)
            # perf_ax = plot_performance(curves)

        else:
            # inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes({layer : spikes[layer].get('s') for layer in spikes},
                                                ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(curves, ax=perf_ax)

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
            f.write('random_seed,n_neurons,n_train,excite,' + \
                    'inhib,time,timestep,theta_plus,theta_decay,' + \
                    'intensity,progress_interval,update_interval,' + \
                    'X_Ae_decay,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')
        else:
            f.write('random_seed,n_neurons,n_train,n_test,excite,' + \
                    'inhib,time,timestep,theta_plus,theta_decay,' + \
                    'intensity,progress_interval,update_interval,' + \
                    'X_Ae_decay,mean_all_activity,' + \
                    'mean_proportion_weighting,mean_ngram,max_all_activity,' + \
                    'max_proportion_weighting,max_ngram\n')

with open(os.path.join(path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

print()
