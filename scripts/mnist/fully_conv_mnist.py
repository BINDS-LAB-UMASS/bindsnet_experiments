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
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--n_filters', type=int, default=16)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--time', type=int, default=300)
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

if not train:
    update_interval = n_test

if kernel_size == 28:
	conv_size = 1
else:
	conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1

locations = torch.zeros(kernel_size, kernel_size, conv_size ** 2).long()
for c in range(conv_size ** 2):
    for k1 in range(kernel_size):
        for k2 in range(kernel_size):
            locations[k1, k2, c] = (c % conv_size) * stride + (c // conv_size) * stride * 28 + k1 * 28 + k2

locations = locations.view(kernel_size ** 2, conv_size ** 2)

# Build network.
network = Network()
input_layer = Input(n=784,
                    traces=True)

conv_layer = DiehlAndCookNodes(n=n_filters * conv_size * conv_size,
                               traces=True)

w = torch.zeros(input_layer.n, conv_layer.n)
for f in range(n_filters):
    for c in range(conv_size ** 2):
        for k in range(kernel_size ** 2):
            w[locations[k, c], f * (conv_size ** 2) + c] = np.random.rand()

mask = w == 0

conv_conn = Connection(input_layer,
                       conv_layer,
                       w=w,
                       kernel_size=kernel_size,
                       stride=stride,
                       update_rule=post_pre,
                       norm=50.0,
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

recurrent_conn = Connection(conv_layer,
                            conv_layer,
                            w=w)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer, name='Y')
network.add_connection(conv_conn, source='X', target='Y')
network.add_connection(recurrent_conn, source='Y', target='Y')

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'),
                       download=True).get_train()
images *= intensity

# Lazily encode data as Poisson spike trains.
data_loader = poisson_loader(data=images, time=time)

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

for i in range(n_train):    
    if i % progress_interval == 0:
        print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start)); start = t()
    
    # Get next input sample.
    sample = next(data_loader).view(time, -1)
    inpts = {'X' : sample}
    
    # Run the network on the input.
    network.run(inpts=inpts, time=time)
    network.connections[('X', 'Y')].w.masked_fill_(mask, 0)
    
    # Optionally plot various simulation information.
    if plot:
        inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
        _spikes = {'X' : spikes['X'].get('s').view(28 ** 2, time),
                   'Y' : spikes['Y'].get('s').view(n_filters * conv_size ** 2, time)}
        _voltages = {'Y' : voltages['Y'].get('v').view(n_filters * conv_size ** 2, time)}

        if i == 0:
            spike_ims, spike_axes = plot_spikes(_spikes)
            weights_im = plot_fully_conv_weights(conv_conn.w, n_filters, kernel_size, conv_size, locations, 28, wmax=conv_conn.wmax)
            # inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i])
            # voltage_ims, voltage_axes = plot_voltages(_voltages)
            
        else:
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_fully_conv_weights(conv_conn.w, n_filters, kernel_size, conv_size, locations, 28, im=weights_im)
            # inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            # voltage_ims, voltage_axes = plot_voltages(_voltages, ims=voltage_ims, axes=voltage_axes)
        
        plt.pause(1e-8)
    
    network._reset()  # Reset state variables.

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
print('Training complete.\n')