import os
import torch
import argparse
import matplotlib.pyplot as plt

from experiments import ROOT_DIR
from bindsnet.datasets import MNIST

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')


def rank_order(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a rank order coding-like representation. One spike per neuron, temporally ordered by decreasing
    intensity. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Create spike times in order of decreasing intensity.
    datum /= datum.max()
    times = torch.zeros(size)
    times[datum != 0] = 1 / datum[datum != 0]
    times *= time / times.max()  # Extended through simulation time.
    times = torch.ceil(times).long()

    print(times.min(), times.max())

    # Create spike times tensor.
    spikes = torch.zeros(time, size).byte()
    for i in range(size):
        if times[i] != 0:
            spikes[times[i] - 1, i] = 1

    return spikes.reshape(time, *shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=1)
    args = parser.parse_args()

    dataset = MNIST(path=data_path, download=True, shuffle=True)
    images, _ = dataset.get_train()
    image = images[0]
    spikes = rank_order(datum=image, time=args.time, dt=args.dt)

    plt.matshow(spikes.view(-1, 784).t())
    plt.matshow(spikes.sum(dim=0).view(28, 28))
    plt.colorbar()
    plt.matshow(image)
    plt.show()


if __name__ == '__main__':
    main()
