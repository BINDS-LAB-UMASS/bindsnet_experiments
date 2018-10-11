import os
from time import time

import pandas as pd
import torch

from experiments import ROOT_DIR

benchmark_path = os.path.join(ROOT_DIR, 'benchmark')
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)


def bindsnet_cpu(n_neurons, time):
    torch.set_default_tensor_type('torch.FloatTensor')

    from bindsnet.network import Network
    from bindsnet.network.topology import Connection
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.encoding import poisson

    network = Network()
    network.add_layer(Input(n=n_neurons), name='X')
    network.add_layer(LIFNodes(n=n_neurons), name='Y')
    network.add_connection(
        Connection(source=network.layers['X'], target=network.layers['Y']), source='X', target='Y'
    )

    data = {'X': poisson(datum=torch.rand(n_neurons), time=time)}
    network.run(inpts=data, time=time)


def bindsnet_gpu(n_neurons, time):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        from bindsnet.network import Network
        from bindsnet.network.topology import Connection
        from bindsnet.network.nodes import Input, LIFNodes
        from bindsnet.encoding import poisson

        network = Network()
        network.add_layer(Input(n=n_neurons), name='X')
        network.add_layer(LIFNodes(n=n_neurons), name='Y')
        network.add_connection(
            Connection(source=network.layers['X'], target=network.layers['Y']), source='X', target='Y'
        )

        data = {'X': poisson(datum=torch.rand(n_neurons), time=time)}
        network.run(inpts=data, time=time)


def neuron(n_neurons, time):
    pass


def brian2(n_neurons, time):
    pass


def nest(n_neurons, time):
    pass


def main():
    f = os.path.join(benchmark_path, 'benchmark.csv')
    if os.path.isfile(f):
        os.remove(f)

    times = {
        'bindsnet_cpu': [], 'bindsnet_gpu': [], 'brian2': [], 'neuron': [], 'nest': []
    }
    for n_neurons in range(100, 10100, 100):
        for framework in times.keys():
            print(f'\nRunning {framework} with {n_neurons} neurons...', end=' ')

            start = time()

            f = globals()[framework]
            f(n_neurons=n_neurons, time=1000)

            elapsed = time() - start
            times[framework].append(elapsed)

            print(f'(elapsed: {elapsed:.4f})')

    df = pd.DataFrame.from_dict(times)

    print(df)
    print()


if __name__ == '__main__':
    main()