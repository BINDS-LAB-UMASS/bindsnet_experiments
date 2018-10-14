import torch
import argparse
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.nodes import RealInput, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

plt.ioff()


def main(n_input=1, n_output=10, time=1000):
    # Network building.
    network = Network(dt=1.0)
    input_layer = RealInput(n=n_input)
    output_layer = LIFNodes(n=n_output)
    connection = Connection(source=input_layer, target=output_layer)
    monitor = Monitor(obj=output_layer, state_vars=('v',), time=time)

    # Adding network components.
    network.add_layer(input_layer, name='X')
    network.add_layer(output_layer, name='Y')
    network.add_connection(connection, source='X', target='Y')
    network.add_monitor(monitor, name='X_monitor')

    # Creating real-valued inputs and running simulation.
    inpts = {'X': torch.ones(time, n_input)}
    network.run(inpts=inpts, time=time)

    # Plot voltage activity.
    plt.plot(monitor.get('v').numpy().T)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=100)
    parser.add_argument('--n_input', type=int, default=1)
    parser.add_argument('--n_output', type=int, default=10)
    args = parser.parse_args()

    main(time=args.time, n_input=args.n_input, n_output=args.n_output)
