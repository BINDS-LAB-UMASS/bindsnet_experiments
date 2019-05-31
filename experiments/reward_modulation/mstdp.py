import datetime

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from bindsnet.analysis.plotting import plot_voltages, plot_spikes
from bindsnet.learning import MSTDP, MSTDPET
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

seed = 3

## Plot settings
plt.ion()
plot_volt = False
demonstrate_input = False

## Parameters
# General (Section 4.1)
dt = 1.0  # ms

# LIF neuron (Section 4.1)
rest_lif = -70.0  # mV
thresh_lif = -54.0  # mV
reset_lif = rest_lif
tau_lif = 20.0  # ms

# TODO: find refractory period
refrac_lif = 0  # ms

# Learning rules (Section 4.1)
tau_plus = 20.0  # ms
tau_minus = 20.0  # ms
tau_z = 25.0  # ms
a_plus = 1.0
a_minus = -1.0

# Learning rules (Section 4.3)
gamma_mstdp = 0.01  # mV
gamma_mstdpet = 0.25  # mV

# Network (Section 4.3)
n_in = 2
n_hidden = 20
n_out = 1
w_min_1 = -10.0  # mV
w_max_1 = 10.0  # mV
w_min_2 = 0.0  # mV
w_max_2 = 10.0  # mV

# Reward (Section 4.2)
reward = 1
punish = -1

# Input patterns (Section 4.3)
n_spike = 50
time_pattern = 500.0  # ms
steps_pattern = int(time_pattern / dt)

# Run configuration (Section 4.3)
epochs = 200

## Build network
# MSTDP
torch.manual_seed(seed)
np.random.seed(seed)

inpt_mstdp = Input(n_in)
hiddn_mstdp = LIFNodes(
    n_hidden, thresh=thresh_lif, rest=rest_lif, reset=reset_lif, tc_decay=tau_lif, refrac=refrac_lif
)
outpt_mstdp = LIFNodes(
    n_out, thresh=thresh_lif, rest=rest_lif, reset=reset_lif, tc_decay=tau_lif, refrac=refrac_lif
)

mstdp_1 = Connection(
    source=inpt_mstdp, target=hiddn_mstdp, update_rule=MSTDP, wmin=w_min_1, wmax=w_max_1, nu=gamma_mstdp
)
mstdp_2 = Connection(
    source=hiddn_mstdp, target=outpt_mstdp, update_rule=MSTDP, wmin=w_min_2, wmax=w_max_2, nu=gamma_mstdp
)

network_mstdp = Network(dt=dt)
network_mstdp.add_layer(name='Input', layer=inpt_mstdp)
network_mstdp.add_layer(name='Hidden', layer=hiddn_mstdp)
network_mstdp.add_layer(name='Output', layer=outpt_mstdp)
network_mstdp.add_connection(source='Input', target='Hidden', connection=mstdp_1)
network_mstdp.add_connection(source='Hidden', target='Output', connection=mstdp_2)
network_mstdp.add_monitor(name='In', monitor=Monitor(obj=inpt_mstdp, state_vars=['s'], time=100))
network_mstdp.add_monitor(name='Hid', monitor=Monitor(obj=hiddn_mstdp, state_vars=['s', 'v'], time=100))

# MSTDPET
torch.manual_seed(seed)
np.random.seed(seed)

inpt_mstdpet = Input(n_in)
hiddn_mstdpet = LIFNodes(
    n_hidden, thresh=thresh_lif, rest=rest_lif, reset=reset_lif, tc_decay=tau_lif, refrac=refrac_lif
)
outpt_mstdpet = LIFNodes(
    n_out, thresh=thresh_lif, rest=rest_lif, reset=reset_lif, tc_decay=tau_lif, refrac=refrac_lif
)

mstdpet_1 = Connection(
    source=inpt_mstdpet, target=hiddn_mstdpet, update_rule=MSTDPET, wmin=w_min_1, wmax=w_max_1, nu=gamma_mstdpet
)
mstdpet_2 = Connection(
    source=hiddn_mstdpet, target=outpt_mstdpet, update_rule=MSTDPET, wmin=w_min_2, wmax=w_max_2, nu=gamma_mstdpet
)

network_mstdpet = Network(dt=dt)
network_mstdpet.add_layer(name='Input', layer=inpt_mstdpet)
network_mstdpet.add_layer(name='Hidden', layer=hiddn_mstdpet)
network_mstdpet.add_layer(name='Output', layer=outpt_mstdpet)
network_mstdpet.add_connection(source='Input', target='Hidden', connection=mstdpet_1)
network_mstdpet.add_connection(source='Hidden', target='Output', connection=mstdpet_2)

## Check if initialized identically
assert torch.equal(mstdp_1.w, mstdpet_1.w)

## Saving variables
rewards_mstdp = []
rewards_mstdpet = []
fig, axes, lines = None, None, None

## Run
for e in range(epochs):
    ## Create input spike trains
    one = torch.zeros(size=(steps_pattern, n_in // 2))
    zero = torch.zeros(size=(steps_pattern, n_in // 2))
    one[torch.randint(0, steps_pattern, size=(n_spike,)), :] = 1
    zero[torch.randint(0, steps_pattern, size=(n_spike,)), :] = 1

    spikes = torch.zeros(size=(steps_pattern, n_in, 4))
    labels = torch.zeros(size=(steps_pattern, 1, 4))

    # {0, 0}
    spikes[:, :, 0] = torch.cat((zero, zero), 1)
    labels[:, :, 0] = 0

    # {0, 1}
    spikes[:, :, 1] = torch.cat((zero, one), 1)
    labels[:, :, 1] = 1

    # {1, 0}
    spikes[:, :, 2] = torch.cat((one, zero), 1)
    labels[:, :, 2] = 1

    # {1, 1}
    spikes[:, :, 3] = torch.cat((one, one), 1)
    labels[:, :, 3] = 0

    # Demonstrate patterns
    if e == 0 and demonstrate_input:
        fig, axes = plt.subplots(1, 4)
        for i, ax in enumerate(axes):
            ax.imshow(spikes[:, :, i].repeat(1, 40))
            ax.set_title(f'Label: {int(labels[0, 0, i])}')

        plt.tight_layout()
        plt.show()

    # Shuffle and concatenate
    idx = torch.randperm(4)
    spikes = spikes[:, :, idx].permute(2, 0, 1).contiguous().view(-1, n_in)
    labels = labels[:, :, idx].permute(2, 0, 1).contiguous().view(-1, 1)

    # Demonstrate final
    if e == 0 and demonstrate_input:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(spikes.repeat(1, 40))
        axes[1].imshow(labels.repeat(1, 20))
        plt.tight_layout()
        plt.show()

    ## Epoch variables
    reward_mstdp = 0
    reward_mstdpet = 0
    fig_volt, ax_volt = None, None
    fig_spik, ax_spik = None, None

    ## Run through networks
    for i in range(steps_pattern * 4):
        # Get reward (MSTDP)
        if labels[i, 0] == 0 and network_mstdp.layers['Output'].s.sum() == 1:
            r_mstdp = punish
        elif labels[i, 0] == 1 and network_mstdp.layers['Output'].s.sum() == 1:
            r_mstdp = reward
        else:
            r_mstdp = 0

        # Get reward (MSTDPET)
        if labels[i, 0] == 0 and network_mstdpet.layers['Output'].s.sum() == 1:
            r_mstdpet = punish
        elif labels[i, 0] == 1 and network_mstdpet.layers['Output'].s.sum() == 1:
            r_mstdpet = reward
        else:
            r_mstdpet = 0

        # Run networks
        # None to add extra dimension
        network_mstdp.run(inpts={'Input': spikes[i, None, :]}, time=1, reward=r_mstdp, a_plus=a_plus, a_minus=a_minus,
                          tc_plus=tau_plus, tc_minus=tau_minus)
        network_mstdpet.run(inpts={'Input': spikes[i, None, :]}, time=1, reward=r_mstdpet, a_plus=a_plus,
                            a_minus=a_minus, tc_plus=tau_plus, tc_minus=tau_minus, tc_z=tau_z)

        # Monitor
        if plot_volt:
            fig_volt, ax_volt = plot_voltages(
                {'Hidden': network_mstdp.monitors['Hid'].get('v')}, ims=fig_volt, axes=ax_volt)
            fig_spik, ax_spik = plot_spikes(
                {'Input': network_mstdp.monitors['In'].get('s'), 'Hidden': network_mstdp.monitors['Hid'].get('s')},
                ims=fig_spik, axes=ax_spik)
            plt.pause(0.0001)

        # Increment rewards
        reward_mstdp += r_mstdp
        reward_mstdpet += r_mstdpet

    ## On episode ends
    rewards_mstdp.append(reward_mstdp)
    rewards_mstdpet.append(reward_mstdpet)
    network_mstdp.reset_()
    network_mstdpet.reset_()

    ## Plot rewards
    # Create figure on first epoch
    if e == 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey='all')
        lines = [ax.plot([0], [0])[0] for ax in axes]
        axes[0].set_ylabel('Reward')
        axes[0].set_xlabel('Epoch')
        axes[0].set_title('MSTDP')
        axes[1].set_xlabel('Epoch')
        axes[1].set_title('MSTDPET')

    # Replot afterwards
    lines[0].set_data(range(e + 1), rewards_mstdp)
    lines[1].set_data(range(e + 1), rewards_mstdpet)

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

## Save reward plot
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
fig.savefig(f'reward_temp_{seed}+{timestamp}.png')
