import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.utils import get_square_weights
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import bernoulli_loader
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, IFNodes
from bindsnet.analysis.plotting import plot_spikes, plot_weights


# Paths.
data_path = os.path.join('..', '..', 'data', 'MNIST')

# Parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--n_hidden', default=100, type=int)
parser.add_argument('--time', default=100, type=int)
parser.add_argument('--lr', default=1.0, type=float)
parser.add_argument('--lr_decay', default=1.0, type=float)
parser.add_argument('--update_interval', default=100, type=int)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.set_defaults(plot=False)
args = parser.parse_args()

n_hidden = args.n_hidden  # no. of hidden layer neurons
time = args.time  # simulation time
lr = args.lr  # learning rate
lr_decay = args.lr_decay  # learning rate decay
update_interval = args.update_interval  # no. examples between evaluation.
plot = args.plot  # visualize spikes + connection weights

criterion = torch.nn.CrossEntropyLoss()  # Loss function on output firing rates.
sqrt = int(np.ceil(np.sqrt(n_hidden)))

# Network building.
network = Network()

# Groups of neurons.
input_layer = Input(n=784)
hidden_layer = IFNodes(n=n_hidden)
output_layer = IFNodes(n=10)
network.add_layer(input_layer, name='X')
network.add_layer(hidden_layer, name='Y')
network.add_layer(output_layer, name='Z')

# Connections between groups of neurons.
input_connection = Connection(source=input_layer, target=hidden_layer, norm=100, wmin=-1.0, wmax=1.0)
hidden_connection = Connection(source=hidden_layer, target=output_layer, norm=10, wmin=-1.0, wmax=1.0)
network.add_connection(input_connection, source='X', target='Y')
network.add_connection(hidden_connection, source='Y', target='Z')

# State variable monitoring.
for l in network.layers:
    m = Monitor(network.layers[l], state_vars=['s'], time=time)
    network.add_monitor(m, name=l)

# MNIST data loading.
images, labels = MNIST(path=data_path, download=True).get_train()
images = bernoulli_loader(images.view(-1, 784), time=time, max_prob=0.1)
labels = iter(labels)

spike_ims, spike_axes, weights1_im, weights2_im = None, None, None, None
losses = torch.zeros(update_interval)
correct = torch.zeros(update_interval)

# Run training.
grads = {}
for i, (image, label) in enumerate(zip(images, labels)):
    label = torch.Tensor([label]).long()

    # Run simulation for single datum.
    network.run(inpts={'X': image}, time=time)

    # Calculate time-averaged firing rates of input and output neurons.
    spikes = {l: network.monitors[l].get('s') for l in network.layers}
    firing_rates = {l: spikes[l].mean(-1) for l in network.layers}

    # Compute softmax of output spiking activity and get predicted label.
    output = firing_rates['Z'].softmax(0).view(1, -1)
    predicted = output.argmax(1).item()
    correct[i % update_interval] = int(predicted == label[0].item())

    # Compute cross-entropy loss between output and true label.
    losses[i % update_interval] = criterion(output, label)

    # Compute gradient of the loss WRT average firing rates.
    grads['dl/df2'] = firing_rates['Z'].softmax(0)
    grads['dl/df2'][label] -= 1

    # Compute gradient of the firing rates WRT connection weights.
    # This is an approximation; the firing rates are not a
    # smooth function of the connection weights.
    grads['dl/dw2'] = torch.ger(firing_rates['Y'], grads['dl/df2'])
    grads['dl/db2'] = grads['dl/df2']
    grads['dl/dw1'] = torch.ger(firing_rates['X'], network.connections['Y', 'Z'].w @ grads['dl/df2'])
    grads['dl/db1'] = network.connections['Y', 'Z'].w @ grads['dl/df2']

    # Do stochastic gradient descent calculation.
    network.connections['X', 'Y'].w -= lr * grads['dl/dw1']
    network.layers['Y'].thresh = torch.max(
        network.layers['Y'].thresh - lr * grads['dl/db1'], -64.0 * torch.ones(n_hidden)
    )
    network.connections['Y', 'Z'].w -= lr * grads['dl/dw2']
    network.layers['Z'].thresh = torch.max(
        network.layers['Z'].thresh - lr * grads['dl/db2'], -64.0 * torch.ones(10)
    )

    if i > 0 and i % update_interval == 0:
        print()
        print(f'Average cross-entropy loss: {losses.mean():.3f}')
        print(f'Average accuracy: {correct.mean():.3f}')

        lr *= lr_decay

    if plot:
        w = network.connections['Y', 'Z'].w
        weights = [
            w[:, i].view(sqrt, sqrt) for i in range(10)
        ]
        w = torch.zeros(5*sqrt, 2*sqrt)
        for i in range(5):
            for j in range(2):
                w[i*sqrt: (i+1)*sqrt, j*sqrt: (j+1)*sqrt] = weights[i + j * 5]

        spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
        weights1_im = plot_weights(w, im=weights1_im, wmin=-1, wmax=1)

        w = network.connections['X', 'Y'].w
        square_weights = get_square_weights(w, sqrt, 28)
        weights2_im = plot_weights(square_weights, im=weights2_im, wmin=-1, wmax=1)

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.
