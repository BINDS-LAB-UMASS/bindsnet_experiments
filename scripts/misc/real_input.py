import torch
import argparse
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.network.nodes import RealInput, LIFNodes


plt.ioff()

parser = argparse.ArgumentParser()
parser.add_argument('--n_input', type=int, default=100)
parser.add_argument('--n_output', type=int, default=500)
parser.add_argument('--time', type=int, default=1000)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.set_defaults(plot=False)
args = parser.parse_args()

n_input = args.n_input
n_output = args.n_output
time = args.time
plot = args.plot

network = Network()

input_layer = RealInput(n=n_input)
output_layer = LIFNodes(n=500)

connection = Connection(
    source=input_layer, target=output_layer, w=torch.rand(input_layer.n, output_layer.n)
)

network.add_layer(layer=input_layer, name='X')
network.add_layer(layer=output_layer, name='Y')
network.add_connection(connection=connection, source='X', target='Y')
network.add_monitor(monitor=Monitor(obj=input_layer, state_vars=['s']), name='X')
network.add_monitor(monitor=Monitor(obj=output_layer, state_vars=['s', 'v']), name='Y')

input_data = {'X': torch.randn(time, n_input)}

network.run(inpts=input_data, time=time)

if plot:
    # Plot spikes of input and output layers.
    spikes = {'X': network.monitors['X'].get('s'), 'Y': network.monitors['Y'].get('s')}

    plot_spikes(spikes)
    plt.tight_layout()
    plt.show()