import os

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from gym import Space
from time import time as t
from typing import Tuple, Dict

from bindsnet.encoding import repeat
from bindsnet.pipeline import Pipeline
from bindsnet.learning import NoOp, MSTDPET
from bindsnet.datasets import MNIST, Dataset
from bindsnet.environment import Environment
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.network.topology import Connection
from bindsnet.network import load_network, Network
from bindsnet.network.nodes import RealInput, DiehlAndCookNodes
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance, \
    plot_voltages

from experiments import ROOT_DIR

model = 'reward_mod_mnist'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


class MNISTEnvironment(Environment):
    # language=rst
    """
    A wrapper around any object from the ``datasets`` module to pass to the ``Pipeline`` object.
    """

    def __init__(self, dataset: Dataset = MNIST, train: bool = True, time: int = 350, **kwargs):
        # language=rst
        """
        Initializes the environment wrapper around the dataset.

        :param dataset: Object from datasets module.
        :param train: Whether to use train or test dataset.
        :param time: Length of spike train per example.
        :param kwargs: Raw data is multiplied by this value.
        """
        self.dataset = dataset
        self.train = train
        self.time = time

        # Keyword arguments.
        self.intensity = kwargs.get('intensity', 1)
        self.max_prob = kwargs.get('max_prob', 1)

        assert 0 < self.max_prob <= 1, 'Maximum spiking probability must be in (0, 1].'

        self.obs = None

        if train:
            self.data, self.labels = self.dataset.get_train()
        else:
            self.data, self.labels = self.dataset.get_test()

        self.data = iter(self.data)
        self.labels = iter(self.labels)
        self.datum = next(self.data)
        self.label = next(self.labels)

        self.obs = self.datum
        self.preprocess()

        self.action_space = Space(shape=(10,), dtype=int)
        self.action_space.n = 10

    def step(self, a: int = None) -> Tuple[torch.Tensor, int, bool, Dict[str, int]]:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``step()`` function.

        :param a: Index of spiking neuron in output layer.
        :return: Observation, reward (fixed to 0), done (fixed to False), and information dictionary.
        """
        # Info dictionary contains label of MNIST digit.
        label = self.label.item()
        info = {'label': label}

        if a == label:
            reward = 1
        # elif a != label and a >= 0:
        #     reward = 0
        else:
            reward = 0

        return self.obs, reward, False, info

    def reset(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``reset()`` function.
        """
        try:
            # Attempt to fetch the next observation.
            self.datum = next(self.data)
            self.label = next(self.labels)
        except StopIteration:
            # If out of samples, reload data and label generators.
            self.data = iter(self.data)
            self.labels = iter(self.labels)
            self.datum = next(self.data)
            self.label = next(self.labels)

        self.obs = self.datum
        self.preprocess()

        # import matplotlib.pyplot as plt
        #
        # plt.ioff()
        # plt.matshow(self.obs.view(28, 28))
        # plt.title(str(self.label))
        # plt.show()

    def render(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``render()`` function.
        """
        pass

    def close(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``close()`` function.
        """
        pass

    def preprocess(self) -> None:
        # language=rst
        """
        Preprocessing step for a state specific to dataset objects.
        """
        self.obs = self.obs.view(-1)
        self.obs /= self.obs.max()

    def reshape(self) -> torch.Tensor:
        # language=rst
        """
        Get reshaped observation for plotting purposes.

        :return: Reshaped observation to plot in ``plt.imshow()`` call.
        """
        return self.obs.view(28, 28)


def select_spiked(pipeline: Pipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action probabilistically based on spiking activity from a network layer.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action sampled from multinomial over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('select_spiked() requires an "output" layer argument.')

    output = pipeline.network.layers[output]
    action_space = pipeline.env.action_space

    assert output.n == action_space.n, 'Output layer size not divisible by size of action space.'

    spikes = output.s
    _sum = spikes.sum().float()

    # Choose action based on population's spiking.
    if _sum == 0:
        action = -1
    else:
        action = torch.argmax(spikes)

    return action


def main(seed=0, n_neurons=100, n_train=60000, n_test=10000, inhib=100, lr=0.01, lr_decay=1, time=350, dt=1,
         theta_plus=0.05, theta_decay=1e-7, progress_interval=10, update_interval=250, plot=False,
         train=True, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                            'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, inhib, lr_decay, time, dt,
        theta_plus, theta_decay, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    n_examples = n_train if train else n_test
    n_classes = 10

    # Build network.
    if train:
        network = Network(dt=dt)

        input_layer = RealInput(n=784, traces=True, trace_tc=5e-2)
        network.add_layer(input_layer, name='X')

        output_layer = DiehlAndCookNodes(
            n=n_classes, rest=0, reset=1, thresh=1, decay=1e-2,
            theta_plus=theta_plus, theta_decay=theta_decay, traces=True, trace_tc=5e-2
        )
        network.add_layer(output_layer, name='Y')

        w = torch.rand(784, n_classes)
        input_connection = Connection(
            source=input_layer, target=output_layer, w=w,
            update_rule=MSTDPET, nu=lr, wmin=0, wmax=1,
            norm=78.4, tc_e_trace=0.1
        )
        network.add_connection(input_connection, source='X', target='Y')

    else:
        network = load_network(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].theta_decay = 0
        network.layers['Y'].theta_plus = 0

    # Load MNIST data.
    environment = MNISTEnvironment(
        dataset=MNIST(path=data_path, download=True), train=train, time=time
    )

    # Create pipeline.
    pipeline = Pipeline(
        network=network, environment=environment, encoding=repeat,
        action_function=select_spiked, output='Y', reward_delay=None
    )

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=('s',), time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    network.add_monitor(Monitor(
            network.connections['X', 'Y'].update_rule, state_vars=('e_trace',), time=time
        ), 'X_Y_e_trace')

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    spike_ims = None
    spike_axes = None
    weights_im = None
    elig_axes = None
    elig_ims = None

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

            if i > 0 and train:
                network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

        # Run the network on the input.
        for j in range(time):
            pipeline.step(a_plus=1, a_minus=0)

        if plot:
            _spikes = {layer: spikes[layer].get('s') for layer in spikes}
            w = network.connections['X', 'Y'].w
            square_weights = get_square_weights(w.view(784, n_classes), 4, 28)

            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            elig_ims, elig_axes = plot_voltages(
                {'Y': network.monitors['X_Y_e_trace'].get('e_trace').view(-1, time)[1500:2000]},
                plot_type='line', ims=elig_ims, axes=elig_axes
            )

            plt.pause(1e-8)

        pipeline.reset_()  # Reset state variables.
        network.connections['X', 'Y'].update_rule.e_trace = torch.zeros(784, n_classes)

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_neurons', type=int, default=100, help='no. of output layer neurons')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='no. of test samples')
    parser.add_argument('--inhib', type=float, default=100.0, help='inhibition connection strength')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=25, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1, help='simulation integreation timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--theta_decay', type=float, default=1e-7, help='adaptive threshold decay time constant')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
