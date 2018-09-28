import os
import torch
import matplotlib.pyplot as plt

from bindsnet.analysis.plotting import plot_spikes, plot_weights
from bindsnet.datasets import MNIST
from bindsnet.encoding import bernoulli_loader
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes


# Paths.
data_path = os.path.join('..', '..', 'data', 'MNIST')

# Parameters.
time = 100  # simulation time
lr = 1.0  # learning rate
lr_decay = 0.9  # learning rate decay
plot = True  # visualize spikes + connection weights
criterion = torch.nn.CrossEntropyLoss()  # Loss function on output firing rates.
update_interval = 100  # no. examples between evaluation.

# Network building.
network = Network()

# Groups of neurons.
input_layer = Input(n=784)
output_layer = LIFNodes(n=10)
network.add_layer(input_layer, name='X')
network.add_layer(output_layer, name='Y')

# Connections between groups of neurons.
connection = Connection(
    source=input_layer, target=output_layer, wmin=-1.0, wmax=1.0, norm=150
)
network.add_connection(connection, source='X', target='Y')

# State variable monitoring.
for l in network.layers:
    m = Monitor(network.layers[l], state_vars=['s'], time=time)
    network.add_monitor(m, name=l)

# MNIST data loading.
images, labels = MNIST(path=data_path, download=True).get_train()
images = bernoulli_loader(images.view(-1, 784), time=time, max_prob=0.25)
labels = iter(labels)

spike_ims, spike_axes, weights_im = None, None, None
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
    output = firing_rates['Y'].softmax(0).view(1, -1)
    predicted = output.argmax(1).item()
    correct[i % update_interval] = int(predicted == label[0].item())

    # Compute cross-entropy loss between output and true label.
    losses[i % update_interval] = criterion(output, label)

    # Compute gradient of the loss WRT average firing rates.
    grads['dl/df'] = firing_rates['Y'].softmax(0)
    grads['dl/df'][label] -= 1

    # Compute gradient of the firing rates WRT connection weights.
    # This is an approximation; the firing rates are not a
    # smooth function of the connection weights.
    grads['df/dw'] = firing_rates['X']
    grads['dl/dw'] = grads['df/dw'].view(-1, 1) @ grads['dl/df'].view(1, -1)
    grads['df/db'] = 1
    grads['dl/db'] = grads['df/db'] * grads['dl/df']

    # Do stochastic gradient descent calculation.
    network.connections['X', 'Y'].w -= lr * grads['dl/dw']
    network.layers['Y'].thresh = torch.max(
        network.layers['Y'].thresh - lr * grads['dl/db'], -64.0 * torch.ones(10)
    )

    if i > 0 and i % update_interval == 0:
        print()
        print(f'Average cross-entropy loss: {losses.mean():.3f}')
        print(f'Average accuracy: {correct.mean():.3f}')

        lr *= lr_decay

    if plot:
        w = network.connections['X', 'Y'].w
        weights = [
            w[:, i].view(28, 28) for i in range(10)
        ]
        w = torch.zeros(5*28, 2*28)
        for i in range(5):
            for j in range(2):
                w[i*28: (i+1)*28, j*28: (j+1)*28] = weights[i + j * 5]

        spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
        weights_im = plot_weights(w, im=weights_im, wmin=-1, wmax=1)

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.