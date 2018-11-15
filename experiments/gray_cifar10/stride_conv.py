import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from bindsnet import *
from time     import time as t

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--inhib', type=float, default=250.0)
parser.add_argument('--kernel_size', type=int, default=14)
parser.add_argument('--stride', type=int, default=7)
parser.add_argument('--n_neurons', type=int, default=100)
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

locals().update(vars(parser.parse_args()))

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

if kernel_size == 32:
	conv_size = 1
else:
	conv_size = int((32 - kernel_size + 2 * padding) / stride) + 1

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
n_input = kernel_size ** 2

# Build network.
network = Network()
input_layer = Input(n=n_input, traces=True)
conv_layer = DiehlAndCookNodes(n=n_neurons, traces=True)
conv_conn = Connection(input_layer, conv_layer, update_rule=post_pre,
                       norm=0.5*n_neurons, nu_pre=1e-4, nu_post=1e-2, wmax=5.0)

w = -inhib * (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons)))
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer, name='Y')
network.add_connection(conv_conn, source='X', target='Y')
network.add_connection(recurrent_conn, source='Y', target='Y')

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

locations = torch.zeros(kernel_size, kernel_size, conv_size ** 2).long()
for c in range(conv_size ** 2):
    for k1 in range(kernel_size):
        for k2 in range(kernel_size):
            locations[k1, k2, -c] = (c // conv_size) * stride * 32 + (c % conv_size) * stride + k1 * 32 + k2

locations = locations.view(kernel_size ** 2, conv_size ** 2)

# Load CIFAR-10 data.
dataset = CIFAR10(path=os.path.join('..', '..', 'data', 'CIFAR10'), download=True)

if train:
    images, labels = dataset.get_train()
else:
    images, labels = dataset.get_test()

images *= intensity
images = images.mean(-1)

# Lazily encode data as Poisson spike trains.
data_loader = poisson_loader(data=images, time=time, dt=dt)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

voltages = {}
for layer in set(network.layers) - {'X'}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=['v'], time=time)
    network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Train the network.
print('Begin training.\n'); start = t()

first = True
for i in range(n_train):    
    if i % progress_interval == 0:
        print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start)); start = t()
    
    # Get next input sample.
    sample = next(data_loader).view(time, -1)
    
    # Run the network on the input.
    for c in range(conv_size ** 2):
        s = sample[:, locations[:, c]]
        inpts = {'X' : sample[:, locations[:, c]]}
        network.run(inpts=inpts, time=time)
    
        # Optionally plot various simulation information.
        if plot:
            _input = s.view(time, kernel_size, kernel_size).sum(0)
            sub_image = images[i].view(-1)[locations[:, c]].view(kernel_size, kernel_size)
            w = get_square_weights(conv_conn.w, n_sqrt=n_sqrt, side=kernel_size)
            _spikes = {'X' : spikes['X'].get('s').view(n_input, time),
                       'Y' : spikes['Y'].get('s').view(n_neurons, time)}
            _voltages = {'Y' : voltages['Y'].get('v').view(n_neurons, time)}
            
            if first:
                inpt_axes, inpt_ims = plot_input(sub_image, _input, label=labels[i])
                spike_ims, spike_axes = plot_spikes(spikes=_spikes)
                weights_im = plot_weights(w, wmax=conv_conn.wmax)
                voltage_ims, voltage_axes = plot_voltages(_voltages)

                first = False
                
            else:
                inpt_axes, inpt_ims = plot_input(sub_image, _input, label=labels[i], axes=inpt_axes, ims=inpt_ims)
                spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
                weights_im = plot_weights(w, im=weights_im)
                voltage_ims, voltage_axes = plot_voltages(_voltages, ims=voltage_ims, axes=voltage_axes)
        
            plt.pause(1e-8)
    
        network.reset_()  # Reset state variables.

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
print('Training complete.\n')
