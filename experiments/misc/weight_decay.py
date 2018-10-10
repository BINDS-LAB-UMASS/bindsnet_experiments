import torch
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.analysis.plotting import plot_weights


plt.ioff()

time = 1000
weight_decay = 1e-3

network = Network()

input_layer = Input(n=50)
output_layer = LIFNodes(n=250)

connection = Connection(source=input_layer, target=output_layer, weight_decay=weight_decay)

w0 = connection.w.clone()

network.add_layer(layer=input_layer, name='X')
network.add_layer(layer=output_layer, name='Y')
network.add_connection(connection=connection, source='X', target='Y')

inpts = {'X': torch.bernoulli(torch.rand(time, input_layer.n))}

network.run(inpts=inpts, time=time)

w1 = connection.w

plot_weights(w0)
plot_weights(w1)

plt.show()
