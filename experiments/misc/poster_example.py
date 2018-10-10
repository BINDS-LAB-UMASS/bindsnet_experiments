import torch
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network import nodes, topology, monitors
from bindsnet.analysis.plotting import plot_spikes


# Parameters.
n_input = 100
n_output = 100
time = 1000

# Create network object.
network = Network()

# Create input and output groups of neurons.
input_group = nodes.Input(n=n_input)  # 100 input nodes.
output_group = nodes.LIFNodes(n=n_output)  # 500 output nodes.

network.add_layer(input_group, name='input')
network.add_layer(output_group, name='output')

# Input -> output connection.
# Unit Gaussian feed-forward weights.
w = torch.randn(n_input, n_output)
forward_conn = topology.Connection(input_group, output_group, w=w)

# Output -> output connection.
# Random, inhibitory recurrent weights.
w = torch.bernoulli(torch.rand(n_output, n_output)) - torch.diag(torch.ones(n_output))
recurrent_conn = topology.Connection(output_group, output_group, w=w)

network.add_connection(forward_conn, source='input', target='output')
network.add_connection(recurrent_conn, source='output', target='output')

# Monitor input and output spikes during the simulation.
for l in network.layers:
    monitor = monitors.Monitor(network.layers[l], state_vars=['s'], time=time)
    network.add_monitor(monitor, name=l)

# Create input ~ Bernoulli(0.1) for 1,000 timesteps.
inpts = {'input': torch.bernoulli(0.05 * torch.rand(time, n_input))}

# Run network simulation for 1,000 timesteps and retrieve spikes.
network.run(inpts=inpts, time=time)
spikes = {l: network.monitors[l].get('s') for l in network.layers}

# Plot spikes from simulation.
plt.ioff()
plot_spikes(spikes)
plt.show()