import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time as t_

import torch
import torch.nn as nn
import torch.nn.functional as F


from bindsnet.network.monitors import Monitor
from bindsnet.conversion import Permute, ann_to_snn
from bindsnet.analysis.plotting import plot_spikes, plot_input, plot_voltages
from bindsnet.network.nodes import LIFNodes
from bindsnet.network import Network
import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology
from typing import Union, Sequence, Optional, Tuple, Dict


from experiments import ROOT_DIR
from experiments.misc.atari_wrappers import make_atari, wrap_deepmind


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = 'cuda'
else:
    device = 'cpu'

results_path = os.path.join(ROOT_DIR, 'bash', 'breakout', 'pso')
params_path = os.path.join(ROOT_DIR, 'params', 'breakout', 'large_dqn_eps_greedy')

for p in [results_path, params_path]:
    if not os.path.isdir(p):
        os.makedirs(p)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.pad2 = nn.ConstantPad2d((1, 2, 1, 2), value=0)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.pad3 = nn.ConstantPad2d((1, 1, 1, 1), value=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.perm = Permute((0, 2, 3, 1))
        self.fc1 = nn.Linear(7744, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = x / 255.0
        x = self.relu1(self.conv1(x))
        x = self.pad2(x)
        x = self.relu2(self.conv2(x))
        x = self.pad3(x)
        x = self.relu3(self.conv3(x))
        x = self.perm(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def policy(q_values, eps):
    A = np.ones(4, dtype=float) * eps / 4
    best_action = torch.argmax(q_values)
    A[best_action] += (1.0 - eps)
    return A, best_action

class Permute(nn.Module):
    # language=rst
    """
    PyTorch module for the explicit permutation of a tensor's dimensions in a parent
    module's ``forward`` pass (as opposed to ``torch.permute``).
    """

    def __init__(self, dims):
        # language=rst
        """
        Constructor for ``Permute`` module.

        :param dims: Ordering of dimensions for permutation.
        """
        super(Permute, self).__init__()

        self.dims = dims

    def forward(self, x):
        # language=rst
        """
        Forward pass of permutation module.

        :param x: Input tensor to permute.
        :return: Permuted input tensor.
        """
        return x.permute(*self.dims).contiguous()

class PermuteConnection(topology.AbstractConnection):
    # language=rst
    """
    Special-purpose connection for emulating the custom ``Permute`` module in spiking neural networks.
    """

    def __init__(self, source: nodes.Nodes, target: nodes.Nodes, dims: Sequence,
                 nu: Optional[Union[float, Sequence[float]]] = None, weight_decay: float = 0.0, **kwargs) -> None:

        # language=rst
        """
        Constructor for ``PermuteConnection``.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param dims: Order of dimensions to permute.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.dims = dims

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Permute input.

        :param s: Input.
        :return: Permuted input.
        """
        return s.permute(self.dims).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``update``.
        """
        pass

    def normalize(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``normalize``.
        """
        pass

    def reset_(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``reset_``.
        """
        pass

class PassThroughNodes(nodes.Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_ with using reset by
    subtraction.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Sequence[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        self.v = torch.zeros(self.shape)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        self.s = x

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        self.s = torch.zeros(self.shape)

class ConstantPad2dConnection(topology.AbstractConnection):
    # language=rst
    """
    Special-purpose connection for emulating the ``ConstantPad2d`` PyTorch module in spiking neural networks.
    """

    def __init__(self, source: nodes.Nodes, target: nodes.Nodes, padding: Tuple,
                 nu: Optional[Union[float, Sequence[float]]] = None, weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``ConstantPad2dConnection``.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param padding: Padding of input tensors; passed to ``torch.nn.functional.pad``.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """

        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.padding = padding

    def compute(self, s: torch.Tensor):
        # language=rst
        """
        Pad input.

        :param s: Input.
        :return: Padding input.
        """
        return F.pad(s, self.padding).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``update``.
        """
        pass

    def normalize(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``normalize``.
        """
        pass

    def reset_(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``reset_``.
        """
        pass


def _ann_to_snn_helper(prev, current, scale):
    # language=rst
    """
    Helper function for main ``ann_to_snn`` method.

    :param prev: Previous PyTorch module in artificial neural network.
    :param current: Current PyTorch module in artificial neural network.
    :return: Spiking neural network layer and connection corresponding to ``prev`` and ``current`` PyTorch modules.
    """
    if isinstance(current, nn.Linear):
        layer = LIFNodes(n=current.out_features, refrac=0, traces=True, thresh=-52, rest=-65.0, decay=1e-2)
        connection = topology.Connection(
            source=prev, target=layer, w=current.weight.t() * scale
        )

    elif isinstance(current, nn.Conv2d):
        input_height, input_width = prev.shape[2], prev.shape[3]
        out_channels, output_height, output_width = current.out_channels, prev.shape[2], prev.shape[3]

        width = (input_height - current.kernel_size[0] + 2 * current.padding[0]) / current.stride[0] + 1
        height = (input_width - current.kernel_size[1] + 2 * current.padding[1]) / current.stride[1] + 1
        shape = (1, out_channels, int(width), int(height))

        layer = LIFNodes(shape=shape, refrac=0, traces=True, thresh=-52, rest=-65.0, decay=1e-2,)
        connection = topology.Conv2dConnection(
            source=prev, target=layer, kernel_size=current.kernel_size, stride=current.stride,
            padding=current.padding, dilation=current.dilation, w=current.weight * scale
        )

    elif isinstance(current, Permute):
        layer = PassThroughNodes(
            shape=[
                prev.shape[current.dims[0]], prev.shape[current.dims[1]],
                prev.shape[current.dims[2]], prev.shape[current.dims[3]]
            ]
        )

        connection = PermuteConnection(
            source=prev, target=layer, dims=current.dims
        )

    elif isinstance(current, nn.ConstantPad2d):
        layer = PassThroughNodes(
            shape=[
                prev.shape[0], prev.shape[1],
                current.padding[0] + current.padding[1] + prev.shape[2],
                current.padding[2] + current.padding[3] + prev.shape[3]
            ]
        )

        connection = ConstantPad2dConnection(
            source=prev, target=layer, padding=current.padding
        )

    else:
        return None, None

    return layer, connection


def main(seed=0, time=250, n_snn_episodes=1, epsilon=0.05, plot=False, parameter1=1.0,
         parameter2=1.0, parameter3=1.0, parameter4=1.0, parameter5=1.0):

    np.random.seed(seed)

    parameters = [parameter1, parameter2, parameter3, parameter4, parameter5]

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    print()
    print('Loading the trained ANN...')
    print()

    ANN = Net()
    ANN.load_state_dict(
        torch.load(
            '../../params/pytorch_breakout_dqn.pt'
        )
    )

    environment = make_atari('BreakoutNoFrameskip-v4')
    environment = wrap_deepmind(environment, frame_stack=True, scale=False, clip_rewards=False, episode_life=False)

    print('Converting ANN to SNN...')
    # Do ANN to SNN conversion.

    # SNN = ann_to_snn(ANN, input_shape=(1, 4, 84, 84), data=states / 255.0, percentile=percentile, node_type=LIFNodes, decay=1e-2 / 13.0, rest=0.0)

    SNN = Network()

    input_layer = nodes.RealInput(shape=(1, 4, 84, 84))
    SNN.add_layer(input_layer, name='Input')

    children = []
    for c in ANN.children():
        if isinstance(c, nn.Sequential):
            for c2 in list(c.children()):
                children.append(c2)
        else:
            children.append(c)

    i = 0
    prev = input_layer
    scale_index = 0
    while i < len(children) - 1:
        current, nxt = children[i:i + 2]
        layer, connection = _ann_to_snn_helper(prev, current, scale=parameters[scale_index])

        i += 1

        if layer is None or connection is None:
            continue

        SNN.add_layer(layer, name=str(i))
        SNN.add_connection(connection, source=str(i - 1), target=str(i))

        prev = layer

        if isinstance(current, nn.Linear) or isinstance(current, nn.Conv2d):
            scale_index += 1

    current = children[-1]
    layer, connection = _ann_to_snn_helper(prev, current, scale=parameters[scale_index])

    i += 1

    if layer is not None or connection is not None:
        SNN.add_layer(layer, name=str(i))
        SNN.add_connection(connection, source=str(i - 1), target=str(i))

    for l in SNN.layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )
        else:
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s'], time=time), name=l
            )

    spike_ims = None
    spike_axes = None
    inpt_ims = None
    inpt_axes = None
    voltage_ims = None
    voltage_axes = None

    rewards = np.zeros(n_snn_episodes)
    total_t = 0

    print()
    print('Testing SNN on Atari Breakout game...')
    print()

    # Test SNN on Atari Breakout.
    for i in range(n_snn_episodes):
        state = torch.tensor(environment.reset()).to(device).unsqueeze(0).permute(0, 3, 1, 2)

        start = t_()
        for t in itertools.count():
            print(f'Timestep {t} (elapsed {t_() - start:.2f})')
            start = t_()

            sys.stdout.flush()

            state = state.repeat(time, 1, 1, 1, 1)

            inpts = {'Input': state.float() / 255.0}

            SNN.run(inpts=inpts, time=time)

            spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
            voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors if not layer == 'Input'}
            probs, best_action = policy(spikes['12'].sum(1), epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)

            next_state, reward, done, info = environment.step(action)
            next_state = torch.tensor(next_state).unsqueeze(0).permute(0, 3, 1, 2)

            rewards[i] += reward
            total_t += 1

            SNN.reset_()

            if plot:
                # Get voltage recording.
                inpt = state.view(time, 4, 84, 84).sum(0).sum(0).view(84, 84)
                spike_ims, spike_axes = plot_spikes(
                    {layer: spikes[layer] for layer in spikes}, ims=spike_ims, axes=spike_axes
                )
                voltage_ims, voltage_axes = plot_voltages(
                    {layer: voltages[layer].view(time, -1) for layer in voltages},
                    ims=voltage_ims, axes=voltage_axes
                )
                inpt_axes, inpt_ims = plot_input(inpt, inpt, ims=inpt_ims, axes=inpt_axes)
                plt.pause(1e-8)

            if done:
                print(f'Step {t} ({total_t}) @ Episode {i + 1} / {n_snn_episodes}')
                print(f'Episode Reward: {rewards[i]}')
                print()

                break

            state = next_state

    model_name = '_'.join([str(x) for x in [seed, parameter1, parameter2, parameter3, parameter4, parameter5]])
    columns = [
        'seed', 'time', 'n_snn_episodes', 'avg. reward', 'parameter1', 'parameter2',
        'parameter3', 'parameter4', 'parameter5'
    ]
    data = [[
        seed, time, n_snn_episodes, np.mean(rewards), parameter1, parameter2, parameter3, parameter4, parameter5
    ]]

    path = os.path.join(results_path, 'results.csv')
    if not os.path.isfile(path):
        df = pd.DataFrame(data=data, index=[model_name], columns=columns)
    else:
        df = pd.read_csv(path, index_col=0)

        if model_name not in df.index:
            df = df.append(pd.DataFrame(data=data, index=[model_name], columns=columns))
        else:
            df.loc[model_name] = data[0]

    df.to_csv(path, index=True)

    torch.save(rewards, os.path.join(results_path, f'{model_name}_episode_rewards.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='@')
    parser.add_argument('@@seed', type=int, default=0)
    parser.add_argument('@@time', type=int, default=250)
    parser.add_argument('@@n_snn_episodes', type=int, default=1)
    parser.add_argument('@@parameter1', type=float, default=1.0)
    parser.add_argument('@@parameter2', type=float, default=1.0)
    parser.add_argument('@@parameter3', type=float, default=1.0)
    parser.add_argument('@@parameter4', type=float, default=1.0)
    parser.add_argument('@@:parameter5', type=float, default=1.0)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    args = vars(parser.parse_args())

    main(**args)
