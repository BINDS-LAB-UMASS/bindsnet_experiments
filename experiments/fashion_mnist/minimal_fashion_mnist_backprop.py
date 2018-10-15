import torch
from bindsnet.network import Network
from bindsnet.datasets import FashionMNIST
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import RealInput, IFNodes

# Network building.
network = Network()

input_layer = RealInput(n=784, sum_input=True)
output_layer = IFNodes(n=10, sum_input=True)
network.add_layer(input_layer, name='X')
network.add_layer(output_layer, name='Y')

input_connection = Connection(input_layer, output_layer, norm=150, wmin=-1, wmax=1)
network.add_connection(input_connection, source='X', target='Y')

# State variable monitoring.
time = 25  # No. of simulation time steps per example.
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
    # Run simulation for single datum.
    inpts = {'X': image.repeat(time, 1), 'Y_b': torch.ones(time, 1)}
    network.run(inpts=inpts, time=time)

    # Retrieve spikes and summed inputs from both layers.
    label = torch.tensor(label).long()
    spikes = {l: network.monitors[l].get('s') for l in network.layers}
    summed_inputs = {l: network.layers[l].summed for l in network.layers}

    # Compute softmax of output activity, get predicted label.
    output = spikes['Y'].sum(-1).softmax(0).view(1, -1)
    predicted = output.argmax(1).item()

    # Compute gradient of loss and do SGD update.
    grads['dl/df'] = summed_inputs['Y'].softmax(0)
    grads['dl/df'][label] -= 1
    grads['dl/dw'] = torch.ger(summed_inputs['X'], grads['dl/df'])
    network.connections['X', 'Y'].w -= lr * grads['dl/dw']

    # Decay learning rate.
    if i > 0 and i % 500 == 0:
        lr *= lr_decay

    network.reset_()
