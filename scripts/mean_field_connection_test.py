import torch
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MeanFieldConnection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_weights


network = Network()

X = Input(n=100)
Y = LIFNodes(n=100)

C = MeanFieldConnection(source=X, target=Y, norm=100.0)

M_X = Monitor(X, state_vars=['s'])
M_Y = Monitor(Y, state_vars=['s', 'v'])
M_C = Monitor(C, state_vars=['w'])

network.add_layer(X, name='X')
network.add_layer(Y, name='Y')
network.add_connection(C, source='X', target='Y')
network.add_monitor(M_X, 'M_X')
network.add_monitor(M_Y, 'M_Y')
network.add_monitor(M_C, 'M_C')

spikes = torch.bernoulli(torch.rand(1000, 100))
inpts = {'X': spikes}

network.run(inpts=inpts, time=1000)

spikes = {'X': M_X.get('s'), 'Y': M_Y.get('s')}
weights = M_C.get('w')

plt.ioff()
plot_spikes(spikes)
plot_weights(weights)
plt.show()

