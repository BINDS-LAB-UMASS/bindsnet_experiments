import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import CIFAR10
from bindsnet.encoding import poisson
from bindsnet.network import load_network
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import ngram, all_activity, proportion_weighting, assign_labels, update_ngram_scores
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance

sys.path.append('..')

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--excite', type=float, default=22.5)
parser.add_argument('--inhib', type=float, default=50.0)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--dt', type=int, default=1.0)
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

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train
n_test = args.n_test
excite = args.excite
inhib = args.inhib
time = args.time
dt = args.dt
theta_plus = args.theta_plus
theta_decay = args.theta_decay
intensity = args.intensity
X_Ae_decay = args.X_Ae_decay
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

args = vars(args)

print(); print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

model = 'diehl_and_cook_2015'
data = 'cifar10'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [
    seed, n_neurons, n_train, excite, inhib, time, dt, theta_plus,
    theta_decay, intensity, progress_interval, update_interval, X_Ae_decay
]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [
        seed, n_neurons, n_train, n_test, excite, inhib, time, dt, theta_plus,
        theta_decay, intensity, progress_interval, update_interval, X_Ae_decay
    ]

np.random.seed(seed)

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

n_examples = n_train if train else n_test
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
if train:
    network = DiehlAndCook2015(
        n_inpt=32*32*3, n_neurons=n_neurons, exc=excite, inh=inhib, dt=dt, norm=307.2, theta_plus=0.05
    )

else:
    path = os.path.join('..', '..', 'params', data, model)
    network = load_network(os.path.join(path, model_name + '.pt'))
    network.connections[('X', 'Ae')].update_rule = None

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers['Ae'], ['v'], time=time)
inh_voltage_monitor = Monitor(network.layers['Ai'], ['v'], time=time)
network.add_monitor(exc_voltage_monitor, name='exc_voltage')
network.add_monitor(inh_voltage_monitor, name='inh_voltage')

# Load CIFAR-10 data.
dataset = CIFAR10(
    path=os.path.join('..', '..', 'data', 'CIFAR10'), download=True
)

images, labels = dataset.get_train() if train else dataset.get_test()
images = images.view(-1, 3072)
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

# Sequence of accuracy estimates.
curves = {'all': [], 'proportion': [], 'ngram': []}

# Image categories.
classes = ['none', 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if train:
    best_accuracy = 0

spikes = {}

for layer in set(network.layers) - {'X'}:
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

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
        curves['all'].append(100 * torch.sum(labels[i - update_interval:i].long() \
                                                == all_activity_pred) / update_interval)
        curves['proportion'].append(100 * torch.sum(labels[i - update_interval:i].long() \
                                                        == proportion_pred) / update_interval)
        curves['ngram'].append(100 * torch.sum(labels[i - update_interval:i].long() \
                                                        == ngram_pred) / update_interval)

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
            assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)

            # Compute ngram scores.
            ngram_scores = update_ngram_scores(spike_record, labels[i - update_interval:i], 10, 2, ngram_scores)

        print()

    # Get next input sample.
    image = images[i]
    sample = poisson(datum=image, time=time, dt=dt)
    inpts = {'X' : sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)

    retries = 0
    while spikes['Ae'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = poisson(datum=image, time=time, dt=dt)
        inpts = {'X' : sample}
        network.run(inpts=inpts, time=time)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get('v')
    inh_voltages = inh_voltage_monitor.get('v')

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Ae'].get('s').t()

    # Optionally plot various simulation information.
    if plot:
        image = image.view(32, 32, 3) / intensity
        image /= image.max()
        inpt = 255 - sample.view(time, 3*32*32).sum(0).view(32, 32, 3).sum(2).float()
        weights = network.connections[('X', 'Ae')].w.view(32, 32, 3, n_neurons)
        weights = weights.sum(2).view(32 * 32, n_neurons)
            
        # square_assignments = get_square_assignments(assignments, n_sqrt)
        square_weights = get_square_weights(weights, n_sqrt, 32)
        
        if i == 0:
            inpt_axes, inpt_ims = plot_input(image, inpt, label=labels[i])
            # assigns_im = plot_assignments(square_assignments, classes=classes)
            # perf_ax = plot_performance(curves)
            weights_ax = plot_weights(square_weights)
        else:
            inpt_axes, inpt_ims = plot_input(image, inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(curves, ax=perf_ax)
            weights_im = plot_weights(square_weights, im=weights_ax)

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.

print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

i += 1

# Get network predictions.
all_activity_pred = all_activity(spike_record, assignments, 10)
proportion_pred = proportion_weighting(spike_record, assignments, proportions, 10)
ngram_pred = ngram(spike_record, ngram_scores, 10, 2)

# Compute network accuracy according to available classification strategies.
curves['all'].append(100 * torch.sum(labels[i - update_interval:i].long() \
                                        == all_activity_pred) / update_interval)
curves['proportion'].append(100 * torch.sum(labels[i - update_interval:i].long() \
                                                == proportion_pred) / update_interval)
curves['ngram'].append(100 * torch.sum(labels[i - update_interval:i].long() \
                                                        == ngram_pred) / update_interval)

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
