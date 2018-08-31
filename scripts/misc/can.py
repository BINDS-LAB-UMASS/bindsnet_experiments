import matplotlib.pyplot as plt
import torch
from bindsnet.network import *


plt.ioff()

# param
Nu_of_CANs = 5
input_size = 1600
Nu_Neurons_in_CAN = 1000
Excitation_percent_in_CAN = 1
running_time = 5000

network = Network(dt=1.0)  # Instantiates network.
In = nodes.Input(n=input_size)
network.add_layer(layer=In, name='X')
Mi = monitors.Monitor(obj=In, state_vars=['s'], time=running_time)
M = []


for i in range(Nu_of_CANs):
    # add CAN column
    CAN = nodes.IzhikevichNodes(n=Nu_Neurons_in_CAN, excitatory=Excitation_percent_in_CAN)

    # create random weights for internal connection of CAN
    w = (torch.rand(Nu_Neurons_in_CAN, Nu_Neurons_in_CAN))
    # make the weights of inhibitory neurons negative
    temp = CAN.excitatory == False

    if torch.sum(temp) > 0:
        w = torch.where(temp, 0.5 * w, -w)

    # create weights between input to output.
    wi = torch.zeros(input_size, Nu_Neurons_in_CAN)
    wi[i, :] = 1.

    # connect the internal connection in CAN
    C = topology.Connection(source=CAN, target=CAN, w=w)
    # connect the CAN to the input layer
    Ci = topology.Connection(source=In, target=CAN, w=wi)

    M.append(monitors.Monitor(obj=CAN, state_vars=['s'], time=running_time))

    temp_CAN_name = 'CAN' + str(i)
    network.add_layer(layer=CAN, name=temp_CAN_name)
    network.add_monitor(monitor=Mi, name='input monitor')
    network.add_monitor(monitor=M[i], name=temp_CAN_name)
    network.add_connection(connection=C, source=temp_CAN_name, target= temp_CAN_name)
    network.add_connection(connection=Ci, source='Input Layer', target=temp_CAN_name)


# Create Poisson-distributed spike train inputs.
#data = 15 * torch.rand(1, input_size)  # Generate random Poisson rates for 100 input neurons.
#train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.
train = 3.5 * torch.rand(running_time, input_size)

# # Simulate network on generated spike trains.
network.run(inpts={'X': train}, time=running_time)  # Run network simulation.

# Plot spikes of input and output layers.
temp_spikes = torch.zeros(Nu_Neurons_in_CAN * Nu_of_CANs, running_time)
for i in range(Nu_of_CANs):
    temp = i*Nu_Neurons_in_CAN
    temp_spikes[temp:temp + Nu_Neurons_in_CAN, :] = M[i].get('s')

spikes = {'X': Mi.get('s'), 'Y': temp_spikes}

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for i, layer in enumerate(spikes):
    axes[i].matshow(spikes[layer], cmap='binary')
    axes[i].set_title('%s spikes' % layer)
    axes[i].set_xlabel('Time');
    axes[i].set_ylabel('Index of neuron')
    axes[i].set_xticks(());
    axes[i].set_yticks(())
    axes[i].set_aspect('auto')

plt.tight_layout()
plt.show()

