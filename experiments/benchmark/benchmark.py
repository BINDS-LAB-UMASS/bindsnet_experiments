import os
import torch
import argparse
import pandas as pd

from brian2 import *
from nest import *
from time import time as t
from experiments import ROOT_DIR

from bindsnet.network import Network
from bindsnet.encoding import poisson
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes

benchmark_path = os.path.join(ROOT_DIR, 'benchmark')
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)

# "Warm up" the GPU.
torch.set_default_tensor_type('torch.cuda.FloatTensor')
x = torch.rand(1000)
del x


def BindsNET_cpu(n_neurons, time):
    torch.set_default_tensor_type('torch.FloatTensor')

    network = Network()
    network.add_layer(Input(n=n_neurons), name='X')
    network.add_layer(LIFNodes(n=n_neurons), name='Y')
    network.add_connection(
        Connection(source=network.layers['X'], target=network.layers['Y']), source='X', target='Y'
    )

    data = {'X': poisson(datum=torch.rand(n_neurons), time=time)}
    network.run(inpts=data, time=time)


def BindsNET_gpu(n_neurons, time):
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        network = Network()
        network.add_layer(Input(n=n_neurons), name='X')
        network.add_layer(LIFNodes(n=n_neurons), name='Y')
        network.add_connection(
            Connection(source=network.layers['X'], target=network.layers['Y']), source='X', target='Y'
        )

        data = {'X': poisson(datum=torch.rand(n_neurons), time=time)}
        network.run(inpts=data, time=time)


def BRIAN2(n_neurons, time):
    eqs_neurons = '''
        dv/dt = (ge * (-60 * mV) + (-74 * mV) - v) / (10 * ms) : volt
        dge/dt = -ge / (5 * ms) : 1
    '''

    input = PoissonGroup(n_neurons, rates=15 * Hz)
    neurons = NeuronGroup(
        n_neurons, eqs_neurons, threshold='v > (-54 * mV)', reset='v = -60 * mV', method='exact'
    )
    S = Synapses(input, neurons, '''w: 1''')
    S.connect()
    S.w = 'rand() * 0.01'

    run(time * ms)


def PyNEST(n_neurons, time):
    ResetKernel()
    SetKernelStatus({"local_num_threads": 8, "resolution": 1.0})

    print(NumProcesses())

    r_ex = 60.0  # [Hz] rate of exc. neurons

    neuron = Create("iaf_psc_alpha", n_neurons)
    noise = Create("poisson_generator", n_neurons)

    SetStatus(noise, [{"rate": r_ex}])
    Connect(noise, neuron)    

    Simulate(time)


def main(start=100, stop=1000, step=100, time=1000):
    f = os.path.join(benchmark_path, f'benchmark_{start}_{stop}_{step}_{time}.csv')
    if os.path.isfile(f):
        os.remove(f)

    times = {
        'BindsNET_cpu': [], 'BindsNET_gpu': [], 'BRIAN2': [], 'PyNEST': []
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
