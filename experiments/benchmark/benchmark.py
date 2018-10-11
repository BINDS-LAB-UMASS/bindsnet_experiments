import os
import torch
import argparse
import pandas as pd

from time import time as t
# from brian2 import PoissonGroup, ms, mV, Hz, NeuronGroup, Synapses, StateMonitor, SpikeMonitor, run, second
from brian2 import *

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


def brian(n_neurons, time):
    taum = 10 * ms
    taupre = 20 * ms
    taupost = taupre
    Ee = 0 * mV
    vt = -54 * mV
    vr = -60 * mV
    El = -74 * mV
    taue = 5 * ms
    F = 15 * Hz
    gmax = .01
    dApre = .01
    dApost = -dApre * taupre / taupost * 1.05
    dApost *= gmax
    dApre *= gmax

    eqs_neurons = '''
    dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
    dge/dt = -ge / taue : 1
    '''

    input = PoissonGroup(n_neurons, rates=F)
    neurons = NeuronGroup(
        n_neurons, eqs_neurons, threshold='v>vt', reset='v = vr', method='exact'
    )
    S = Synapses(
        input, neurons, '''w : 1'''
    )
    S.connect()
    S.w = 'rand() * gmax'

    run(time * ms)


def neuron(n_neurons, time):
    pass


def nest(n_neurons, time):
    pass


def main(start=100, stop=1000, step=100, time=1000):
    f = os.path.join(benchmark_path, f'benchmark_{start}_{stop}_{step}_{time}.csv')
    if os.path.isfile(f):
        os.remove(f)

    times = {
        'bindsnet_cpu': [], 'bindsnet_gpu': [], 'brian': [], 'neuron': [], 'nest': []
    }

    for n_neurons in range(start, stop + step, step):
        print(f'\nRunning benchmark with {n_neurons} neurons.')
        for framework in times.keys():
            print(f'- {framework}:', end=' ')

            t1 = t()

            fn = globals()[framework]
            fn(n_neurons=n_neurons, time=time)

            elapsed = t() - t1
            times[framework].append(elapsed)

            print(f'(elapsed: {elapsed:.4f})')

    df = pd.DataFrame.from_dict(times)
    df.index = list(range(start, stop + step, step))

    print(df)
    print()

    df.to_csv(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--stop', type=int, default=1000)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--time', type=int, default=1000)
    args = parser.parse_args()

    main(start=args.start, stop=args.stop, step=args.step, time=args.time)
