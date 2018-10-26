import os
from brian2 import *
import brian2genn
from time import time as t

n_neurons = 5000000000

t0 = t()

set_device('genn', build_on_run=False)
defaultclock = 1.0 * ms
device.build()

print(f'Time to build: {t() - t0:.4f}')
t1 = t()

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
S.w = 'rand()'

run(1000 * ms)

print(f'Time to simulate: {t() - t1:.4f}')

