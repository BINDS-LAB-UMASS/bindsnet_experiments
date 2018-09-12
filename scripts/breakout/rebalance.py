import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix

from bindsnet.learning import NoOp
from bindsnet.encoding import bernoulli
from bindsnet.network import load_network
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.evaluation import update_ngram_scores, assign_labels
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_spikes, plot_performance, plot_assignments, plot_weights, plot_input

sys.path.append('..')

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=10000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--inhib', type=float, default=100.0)
parser.add_argument('--time', type=int, default=300)
parser.add_argument('--dt', type=int, default=1.0)
parser.add_argument('--theta_plus', type=float, default=0.5)
parser.add_argument('--theta_decay', type=float, default=1e-7)
parser.add_argument('--intensity', type=float, default=0.5)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=100)
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
inhib = args.inhib
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

args = vars(args)

print()
print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

model = 'rebalance'
data = 'breakout'

top_level = os.path.join('..', '..')
data_path = os.path.join(top_level, 'data', 'Breakout')
params_path = os.path.join(top_level, 'params', data, model)
curves_path = os.path.join(top_level, 'curves', data, model)
results_path = os.path.join(top_level, 'results', data, model)
confusion_path = os.path.join(top_level, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [
    seed, n_neurons, n_train, inhib, time, dt, theta_plus, theta_decay, intensity, progress_interval, update_interval
]

test_params = [
    seed, n_neurons, n_train, n_test, inhib, time, dt, theta_plus,
    theta_decay, intensity, progress_interval, update_interval
]

model_name = '_'.join([str(x) for x in params])

np.random.seed(seed)

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

device = 'cuda' if gpu else 'cpu'

n_examples = n_train if train else n_test
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
n_classes = 4
per_class = int(n_examples / n_classes)

# Build network.
if train:
    network = DiehlAndCook2015(
        n_inpt=50*72, n_neurons=n_neurons, exc=25.0, inh=inhib,
        dt=dt, norm=64, theta_plus=theta_plus, theta_decay=theta_decay
    )
else:
    network = load_network(os.path.join(params_path, model_name + '.pt'))
    network.connections[('X', 'Ae')].update_rule = NoOp(connection=network.connections[('X', 'Ae')])

# Load Breakout data.
images = torch.load(os.path.join(data_path, 'frames.pt'), map_location=torch.device(device))
labels = torch.load(os.path.join(data_path, 'labels.pt'), map_location=torch.device(device))
images = images[:, 30:, 4:-4].contiguous().view(-1, 50*72)  # Crop out the borders of the frames.

# Randomly sample n_examples examples, with n_examples / 4 per class.
_images = torch.Tensor().float()
_labels = torch.Tensor().long()
for i in range(4):
    indices = np.where(labels == i)[0]
    indices = np.random.choice(indices, size=per_class, replace=True)
    _images = torch.cat([_images, images[indices]])
    _labels = torch.cat([_labels, labels[indices]])

images = _images
labels = _labels

# Randomly permute the data.
permutation = torch.randperm(images.size(0))
images = images[permutation]
labels = labels[permutation]

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
if train:
    assignments = -torch.ones_like(torch.Tensor(n_neurons))
    proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
    rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
    ngram_scores = {}
else:
    path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
    assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

# Sequence of accuracy estimates.
curves = {'all': [], 'proportion': [], 'ngram': []}
predictions = {'all': torch.Tensor().long(), 'proportion': torch.Tensor().long(), 'ngram': torch.Tensor().long()}

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

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights_im = None
assigns_im = None
perf_ax = None

start = t()
for i in range(n_examples):
    if i % progress_interval == 0:
        elapsed = t() - start
        print(f'Progress: {i} / {n_examples} ({elapsed:.4f} seconds)')
        start = t()

    if i % update_interval == 0 and i > 0:
        if i % len(labels) == 0:
            current_labels = labels[-update_interval:]
        else:
            current_labels = labels[i % len(labels) - update_interval:i % len(labels)]

        # Update and print accuracy evaluations.
        curves, preds = update_curves(
            curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
            proportions=proportions, ngram_scores=ngram_scores, n=2
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
                torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

                best_accuracy = max([x[-1] for x in curves.values()])

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

            # Compute ngram scores.
            ngram_scores = update_ngram_scores(spike_record, current_labels, n_classes, 2, ngram_scores)

        print()

    # Get next input sample.
    image = images[i % len(images)]
    sample = bernoulli(datum=image, time=time)
    inpts = {'X': sample}

    # Run the network on the input.
    network.run(inpts=inpts, time=time)

    retries = 0
    while spikes['Ae'].get('s').sum() < 5 and retries < 3:
        retries += 1
        image *= 2
        sample = bernoulli(datum=image, time=time)
        inpts = {'X': sample}
        network.run(inpts=inpts, time=time)

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Ae'].get('s').t()

    # Optionally plot various simulation information.
    if plot:
        inpt = images[i % len(images)].view(50, 72)
        reconstruction = inpts['X'].view(time, 50*72).sum(0).view(50, 72)
        _spikes = {layer: spikes[layer].get('s') for layer in spikes}
        input_exc_weights = network.connections[('X', 'Ae')].w
        square_weights = get_square_weights(input_exc_weights.view(50*72, n_neurons), n_sqrt, side=(50, 72))
        square_assignments = get_square_assignments(assignments, n_sqrt)

        inpt_axes, inpt_ims = plot_input(inpt, reconstruction, label=labels[i], axes=inpt_axes, ims=inpt_ims)
        spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
        weights_im = plot_weights(square_weights, im=weights_im)
        assigns_im = plot_assignments(square_assignments, im=assigns_im)
        perf_ax = plot_performance(curves, ax=perf_ax)

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.

print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

i += 1

if i % len(labels) == 0:
    current_labels = labels[-update_interval:]
else:
    current_labels = labels[i % len(labels) - update_interval:i % len(labels)]

# Update and print accuracy evaluations.
curves, preds = update_curves(
    curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
    proportions=proportions, ngram_scores=ngram_scores, n=2
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
        torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

        best_accuracy = max([x[-1] for x in curves.values()])

    print('\nTraining complete.\n')
else:
    print('\nTest complete.\n')

print('Average accuracies:\n')
for scheme in curves.keys():
    print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

# Save accuracy curves to disk.
to_write = ['train'] + params if train else ['test'] + test_params
f = '_'.join([str(x) for x in to_write]) + '.pt'
torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

# Save results to disk.
results = [
    np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
    np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
]

to_write = params + results if train else test_params + results
to_write = [str(x) for x in to_write]

name = 'train.csv' if train else 'test.csv'

if not os.path.isfile(os.path.join(results_path, name)):
    with open(os.path.join(results_path, name), 'w') as f:
        if train:
            f.write(
                'random_seed,n_neurons,n_train,inhib,time,timestep,theta_plus,theta_decay,intensity,progress_interval,'
                'update_interval,mean_all_activity,mean_proportion_weighting,mean_ngram,max_all_activity,'
                'max_proportion_weighting,max_ngram\n'
            )
        else:
            f.write(
                'random_seed,n_neurons,n_train,n_test,inhib,time,timestep,theta_plus,theta_decay,intensity,'
                'progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,mean_ngram,'
                'max_all_activity,max_proportion_weighting,max_ngram\n'
            )

with open(os.path.join(results_path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

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

print()
