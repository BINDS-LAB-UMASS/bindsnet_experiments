import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.datasets import FashionMNIST
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import RealInput, IFNodes
from bindsnet.analysis.plotting import plot_spikes, plot_weights

# Network building.
network = Network()

input_layer = RealInput(n=784, sum_input=True)
output_layer = IFNodes(n=10, sum_input=True)
bias = RealInput(n=1, sum_input=True)
network.add_layer(input_layer, name='X')
network.add_layer(output_layer, name='Y')
network.add_layer(bias, name='Y_b')

input_connection = Connection(source=input_layer, target=output_layer, norm=150, wmin=-1, wmax=1)
bias_connection = Connection(source=bias, target=output_layer)
network.add_connection(input_connection, source='X', target='Y')
network.add_connection(bias_connection, source='Y_b', target='Y')

# State variable monitoring.
time = 25
for l in network.layers:
    m = Monitor(network.layers[l], state_vars=['s'], time=time)
    network.add_monitor(m, name=l)

# Load Fashion-MNIST data.
images, labels = FashionMNIST(path='../../data/FashionMNIST', download=True).get_train()

# Run training.
grads = {}
lr, lr_decay = 1e-2, 0.95
criterion = torch.nn.CrossEntropyLoss()
spike_ims, spike_axes, weights_im = None, None, None
for i, (image, label) in enumerate(zip(images.view(-1, 784) / 255, labels)):
    label = torch.Tensor([label]).long()

    # Run simulation for single datum.
    inpts = {'X': image.repeat(time, 1), 'Y_b': torch.ones(time, 1)}
    network.run(inpts=inpts, time=time)

    # Retrieve spikes and summed inputs from both layers.
    spikes = {l: network.monitors[l].get('s') for l in network.layers if '_b' not in l}
    summed_inputs = {l: network.layers[l].summed for l in network.layers}

    # Compute softmax of output activity, get predicted label.
    output = summed_inputs['Y'].softmax(0).view(1, -1)
    predicted = output.argmax(1).item()

    # Compute gradient of loss and do SGD update.
    grads['dl/df'] = summed_inputs['Y'].softmax(0)
    grads['dl/df'][label] -= 1
    grads['dl/dw'] = torch.ger(summed_inputs['X'], grads['dl/df'])
    grads['dl/db'] = grads['dl/df']
    network.connections['X', 'Y'].w -= lr * grads['dl/dw']
    network.connections['Y_b', 'Y'].w -= lr * grads['dl/db']

    # Decay learning rate.
    if i > 0 and i % 500 == 0:
        lr *= lr_decay

    w = network.connections['X', 'Y'].w
    weights = torch.zeros(5*28, 2*28)
    for j in range(5):
        for k in range(2):
            weights[j*28: (j+1)*28, k*28: (k+1)*28] = w[:, j + k * 5].view(28, 28)

    spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
    weights_im = plot_weights(weights, im=weights_im, wmin=-1, wmax=1)
    plt.pause(1e-1)

    network.reset_()
