import os
import torch
import argparse
import matplotlib.pyplot as plt

from experiments import ROOT_DIR
from bindsnet.datasets import MNIST

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be non-negative. Inter-spike
    intervals (ISIs) for non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time]))
    intervals[:, datum != 0] += 1

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time, size).byte()
    spikes[times, torch.arange(size)] = 1
    return spikes.view(time, *shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--intensity', type=float, default=0.35)
    parser.add_argument('--time', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=1)
    args = parser.parse_args()

    dataset = MNIST(path=data_path, download=True, shuffle=True)
    images, _ = dataset.get_train()
    image = images[0] * args.intensity
    spikes = poisson(datum=image, time=args.time, dt=args.dt)

    plt.matshow(spikes.view(-1, 784).t())
    plt.matshow(spikes.sum(dim=0).view(28, 28))
    plt.colorbar()
    plt.matshow(image)
    plt.show()


if __name__ == '__main__':
    main()
