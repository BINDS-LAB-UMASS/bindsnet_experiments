import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix

from bindsnet.network import Network, load_network
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import RealInput, IFNodes
from bindsnet.analysis.plotting import plot_spikes, plot_weights


# Parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=10000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--time', default=10, type=int)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--lr_decay', default=0.95, type=float)
parser.add_argument('--wmin', default=-1, type=float)
parser.add_argument('--wmax', default=1, type=float)
parser.add_argument('--norm', default=500.0, type=float)
parser.add_argument('--update_interval', default=100, type=int)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, train=True, gpu=False)
args = parser.parse_args()

seed = args.seed  # random seed
n_train = args.n_train  # no. of training samples
n_test = args.n_test  # no. of test samples
time = args.time  # simulation time
lr = args.lr  # learning rate
lr_decay = args.lr_decay  # learning rate decay
wmin = args.wmin  # minimum connection strength
wmax = args.wmax  # maximum connection strength
norm = args.norm  # total synaptic weight per output layer neuron
update_interval = args.update_interval  # no. examples between evaluation
plot = args.plot  # visualize spikes + connection weights
train = args.train  # train or test mode
gpu = args.gpu  # whether to use gpu or cpu tensors

args = vars(args)

print()
print('Command-line argument values:')
for key, value in args.items():
    print('-', key, ':', value)

print()

data = 'breakout'
model = 'backprop'

assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                        'No. examples must be divisible by update_interval'

params = [
    seed, n_train, time, lr, lr_decay, wmin, wmax, norm, update_interval
]

model_name = '_'.join([str(x) for x in params])

if not train:
    test_params = [
        seed, n_train, n_test, time, lr, lr_decay, wmin, wmax, norm, update_interval
    ]

np.random.seed(seed)

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

device = 'cuda' if gpu else 'cpu'

# Paths.
top_level = os.path.join('..', '..')
data_path = os.path.join(top_level, 'data', 'Breakout')
params_path = os.path.join(top_level, 'params', data, model)
curves_path = os.path.join(top_level, 'curves', data, model)
results_path = os.path.join(top_level, 'results', data, model)
confusion_path = os.path.join(top_level, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

criterion = torch.nn.CrossEntropyLoss()  # Loss function on output firing rates.
n_examples = n_train if train else n_test

if train:
    # Network building.
    network = Network()

    # Groups of neurons.
    input_layer = RealInput(n=50*72, sum_input=True)
    output_layer = IFNodes(n=4, sum_input=True)
    bias = RealInput(n=1, sum_input=True)
    network.add_layer(input_layer, name='X')
    network.add_layer(output_layer, name='Y')
    network.add_layer(bias, name='Y_b')

    # Connections between groups of neurons.
    input_connection = Connection(source=input_layer, target=output_layer, norm=norm, wmin=wmin, wmax=wmax)
    bias_connection = Connection(source=bias, target=output_layer)
    network.add_connection(input_connection, source='X', target='Y')
    network.add_connection(bias_connection, source='Y_b', target='Y')

    # State variable monitoring.
    for l in network.layers:
        m = Monitor(network.layers[l], state_vars=['s'], time=time)
        network.add_monitor(m, name=l)
else:
    network = load_network(os.path.join(params_path, model_name + '.pt'))

# Load Breakout data.
images = torch.load(os.path.join(data_path, 'frames.pt'), map_location=torch.device(device))
labels = torch.load(os.path.join(data_path, 'labels.pt'), map_location=torch.device(device))
images = images[:, 30:, 4:-4].contiguous().view(-1, 50*72)  # Crop out the borders of the frames.

# Randomly permute the data.
permutation = torch.randperm(images.size(0))
images = images[permutation]
labels = labels[permutation]


_images = torch.Tensor()
_labels = torch.Tensor().long()

if train:
    while _labels.numel() < n_train:
        _images = torch.cat([_images, images[:min(n_train - _labels.numel(), 5500)]])
        _labels = torch.cat([_labels, labels[:min(n_train - _labels.numel(), 5500)]])
else:
    while _labels.numel() < n_test:
        _images = torch.cat([_images, images[:min(n_test - _labels.numel(), 5500)]])
        _labels = torch.cat([_labels, labels[:min(n_test - _labels.numel(), 5500)]])

images, labels = _images, _labels

grads = {}
accuracies = []
predictions = []
ground_truth = []
best = -np.inf
spike_ims, spike_axes, weights_im = None, None, None
losses = torch.zeros(update_interval)
correct = torch.zeros(update_interval)

# Run training.
start = beginning = t()
for i, (image, label) in enumerate(zip(images, labels)):
    label = torch.Tensor([label]).long()

    # Run simulation for single datum.
    inpts = {
        'X': image.repeat(time, 1), 'Y_b': torch.ones(time, 1)
    }
    network.run(inpts=inpts, time=time)

    # Retrieve spikes and summed inputs from both layers.
    spikes = {l: network.monitors[l].get('s') for l in network.layers}
    summed_inputs = {l: network.layers[l].summed for l in network.layers}

    # Compute softmax of output spiking activity and get predicted label.
    output = summed_inputs['Y'].softmax(0).view(1, -1)
    predicted = output.argmax(1).item()
    correct[i % update_interval] = int(predicted == label[0].item())
    predictions.append(predicted)
    ground_truth.append(label)

    # Compute cross-entropy loss between output and true label.
    losses[i % update_interval] = criterion(output, label)

    if train:
        # Compute gradient of the loss WRT average firing rates.
        grads['dl/df'] = summed_inputs['Y'].softmax(0)
        grads['dl/df'][label] -= 1

        # Compute gradient of the summed voltages WRT connection weights.
        # This is an approximation; the summed voltages are not a
        # smooth function of the connection weights.
        grads['dl/dw'] = torch.ger(summed_inputs['X'], grads['dl/df'])
        grads['dl/db'] = grads['dl/df']

        # Do stochastic gradient descent calculation.
        network.connections['X', 'Y'].w -= lr * grads['dl/dw']
        network.connections['Y_b', 'Y'].w -= lr * grads['dl/db']

    if i > 0 and i % update_interval == 0:
        accuracies.append(correct.mean() * 100)

        if train:
            if accuracies[-1] > best:
                print()
                print('New best accuracy! Saving network parameters to disk.')

                # Save network to disk.
                network.save(os.path.join(params_path, model_name + '.pt'))
                best = accuracies[-1]

        print()
        print(f'Progress: {i} / {n_examples} ({t() - start:.3f} seconds)')
        print(f'Average cross-entropy loss: {losses.mean():.3f}')
        print(f'Last accuracy: {accuracies[-1]:.3f}')
        print(f'Average accuracy: {np.mean(accuracies):.3f}')

        # Decay learning rate.
        lr *= lr_decay

        if train:
            print(f'Best accuracy: {best:.3f}')
            print(f'Current learning rate: {lr:.3f}')

        start = t()

    if plot:
        w = network.connections['X', 'Y'].w
        weights = [
            w[:, i].view(50, 72) for i in range(4)
        ]
        w = torch.zeros(2*50, 2*72)
        for j in range(2):
            for k in range(2):
                w[j*50: (j+1)*50, k*72: (k+1)*72] = weights[j + k * 2]

        spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
        weights_im = plot_weights(w, im=weights_im, wmin=wmin, wmax=wmax)

        plt.pause(1e-1)

    network.reset_()  # Reset state variables.

accuracies.append(correct.mean() * 100)

if train:
    lr *= lr_decay
    for c in network.connections:
        network.connections[c].update_rule.weight_decay *= lr_decay

    if accuracies[-1] > best:
        print()
        print('New best accuracy! Saving network parameters to disk.')

        # Save network to disk.
        network.save(os.path.join(params_path, model_name + '.pt'))
        best = accuracies[-1]

print()
print(f'Progress: {n_examples} / {n_examples} ({t() - start:.3f} seconds)')
print(f'Average cross-entropy loss: {losses.mean():.3f}')
print(f'Last accuracy: {accuracies[-1]:.3f}')
print(f'Average accuracy: {np.mean(accuracies):.3f}')

if train:
    print(f'Best accuracy: {best:.3f}')

if train:
    print('\nTraining complete.\n')
else:
    print('\nTest complete.\n')

print(f'Average accuracy: {np.mean(accuracies):.3f}')

# Save accuracy curves to disk.
to_write = ['train'] + params if train else ['test'] + params
f = '_'.join([str(x) for x in to_write]) + '.pt'
torch.save((accuracies, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

results = [np.mean(accuracies), np.max(accuracies)]
to_write = params + results if train else test_params + results
to_write = [str(x) for x in to_write]
name = 'train.csv' if train else 'test.csv'

if not os.path.isfile(os.path.join(results_path, name)):
    with open(os.path.join(results_path, name), 'w') as f:
        if train:
            f.write(
                'seed,n_train,time,lr,lr_decay,update_interval,wmin,wmax,norm,mean_accuracy,max_accuracy\n'
            )
        else:
            f.write(
                'seed,n_train,n_test,time,lr,lr_decay,update_interval,wmin,wmax,norm,mean_accuracy,max_accuracy\n'
            )

with open(os.path.join(results_path, name), 'a') as f:
    f.write(','.join(to_write) + '\n')

# Compute confusion matrices and save them to disk.
confusion = confusion_matrix(ground_truth, predictions)

to_write = ['train'] + params if train else ['test'] + test_params
f = '_'.join([str(x) for x in to_write]) + '.pt'
torch.save(confusion, os.path.join(confusion_path, f))

print()
