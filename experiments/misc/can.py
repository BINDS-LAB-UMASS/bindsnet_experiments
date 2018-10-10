import torch
import argparse
import matplotlib.pyplot as plt

from bindsnet.models import CANs
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

network = CANs(n_inpt=input_size, n_neurons=n_neurons, n_cans=n_cans, dt=1.0)

Mi = monitors.Monitor(obj=network.layers['X'], state_vars=['s'], time=time)
network.add_monitor(monitor=Mi, name='input monitor')
M = []

for i in range(n_cans):
    M.append(monitors.Monitor(obj=network.layers[f'CAN-{i}'], state_vars=['s', 'v'], time=time))

    name = f'CAN-{i}'
    network.add_monitor(monitor=M[i], name=name)


# Create data.
train = torch.rand(time, input_size)

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

