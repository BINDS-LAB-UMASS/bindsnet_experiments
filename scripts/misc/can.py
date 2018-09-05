import torch
import argparse
import matplotlib.pyplot as plt

from bindsnet.network import *
from bindsnet.analysis.plotting import plot_spikes


plt.ioff()

parser = argparse.ArgumentParser()
parser.add_argument('--n_cans', type=int, default=5)
parser.add_argument('--input_size', type=int, default=5)
parser.add_argument('--n_neurons', type=int, default=1000)
parser.add_argument('--excitation', type=float, default=0.0)
parser.add_argument('--time', type=int, default=500)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.set_defaults(plot=False)
args = parser.parse_args()

n_cans = args.n_cans
input_size = args.input_size
n_neurons = args.n_neurons
excitation = args.excitation
time = args.time
plot = args.plot

network = Network(dt=1.0)  # Instantiates network.
In = nodes.Input(n=input_size)
network.add_layer(layer=In, name='X')
Mi = monitors.Monitor(obj=In, state_vars=['s'], time=time)
network.add_monitor(monitor=Mi, name='input monitor')
M = []


for i in range(n_cans):
    # add CAN column
    # CAN = nodes.IzhikevichNodes(n=n_neurons, excitatory=excitation)
    CAN = nodes.LIFNodes(n=n_neurons)

    # create random weights for internal connection of CAN
    n_ex = int(excitation * 10000)
    w = torch.rand(n_neurons, n_neurons)
    w[:n_ex] *= 0.5
    w[n_ex:] *= -1

    # create weights between input to output.
    wi = torch.zeros(input_size, n_neurons)
    wi[i] = 1

    # connect the internal connection in CAN
    C = topology.Connection(source=CAN, target=CAN, w=w)
    # connect the CAN to the input layer
    Ci = topology.Connection(source=In, target=CAN, w=wi)

    M.append(monitors.Monitor(obj=CAN, state_vars=['s', 'v'], time=time))

    temp_CAN_name = 'CAN' + str(i)
    network.add_layer(layer=CAN, name=temp_CAN_name)
    network.add_monitor(monitor=M[i], name=temp_CAN_name)
    network.add_connection(connection=C, source=temp_CAN_name, target=temp_CAN_name)
    network.add_connection(connection=Ci, source='Input Layer', target=temp_CAN_name)


# Create Poisson-distributed spike train inputs.
#data = 15 * torch.rand(1, input_size)  # Generate random Poisson rates for 100 input neurons.
#train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.
train = torch.bernoulli(torch.rand(time, input_size))

# # Simulate network on generated spike trains.
network.run(inpts={'X': train}, time=time)  # Run network simulation.

if plot:
    # Plot spikes of input and output layers.
    temp_spikes = torch.zeros(n_neurons * n_cans, time)
    voltages = torch.zeros(n_neurons * n_cans, time)
    for i in range(n_cans):
        temp = i * n_neurons
        temp_spikes[temp:temp + n_neurons, :] = M[i].get('s')
        voltages[temp:temp + n_neurons, :] = M[i].get('v')

    voltages = voltages.numpy()
    spikes = {'X': Mi.get('s'), 'Y': temp_spikes}

    plot_spikes(spikes)
    # figure = plt.figure()
    # plt.plot(voltages.T[:, ::100])
    plt.tight_layout()
    plt.show()

