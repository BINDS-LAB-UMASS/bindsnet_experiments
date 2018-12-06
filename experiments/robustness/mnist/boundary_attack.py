import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.evaluation import ngram
from bindsnet.network import load_network
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_locally_connected_weights, plot_spikes, plot_input

from experiments import ROOT_DIR

model = 'crop_locally_connected'
data = 'mnist'
experiment = 'black_box_adversarial'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, experiment)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, experiment)

for path in [params_path, results_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def get_diff(sample_1, sample_2):
    return torch.norm(sample_1 - sample_2)


def orthogonal_perturbation(delta, image, target):
    # Generate perturbation.
    perturb = torch.randn(image.size())
    perturb /= get_diff(perturb, torch.zeros_like(perturb))
    perturb *= delta * torch.mean(get_diff(target, image))

    # Project perturbation onto sphere around target.
    diff = (target - image)
    diff /= get_diff(target, image)
    perturb -= perturb.dot(diff) * diff

    # Check overflow and underflow.
    overflow = image + perturb - 255
    perturb -= overflow * (overflow > 0).float()

    underflow = image + perturb
    perturb += underflow * (underflow > 0).float()

    perturb = perturb.abs()

    return perturb


def forward_perturbation(epsilon, image, target):
    perturb = (target - image)
    perturb /= get_diff(target, image)
    perturb *= epsilon

    return perturb


def main(seed=0, n_examples=100, gpu=False, plot=False):

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    model_name = '0_12_4_150_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'

    network = load_network(
        os.path.join(
            params_path, f'{model_name}.pt'
        )
    )

    for l in network.layers:
        network.layers[l].dt = network.dt

    for c in network.connections:
        network.connections[c].dt = network.dt

    network.layers['Y'].one_spike = True
    network.layers['Y'].lbound = None

    kernel_size = 12
    side_length = 20
    n_filters = 150
    time = 250
    intensity = 0.5
    crop = 4
    conv_size = network.connections['X', 'Y'].conv_size
    locations = network.connections['X', 'Y'].locations
    conv_prod = int(np.prod(conv_size))
    n_neurons = n_filters * conv_prod
    n_classes = 10

    # Voltage recording for excitatory and inhibitory layers.
    voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
    network.add_monitor(voltage_monitor, name='output_voltage')

    # Load MNIST data.
    dataset = MNIST(path=data_path, download=True)

    images, labels = dataset.get_test()
    images *= intensity
    images = images[:, crop:-crop, crop:-crop]

    # Neuron assignments and spike proportions.
    path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
    assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name=f'{layer}_spikes')

    # Train the network.
    print('\nBegin black box adversarial attack.\n')

    spike_ims = None
    spike_axes = None
    weights_im = None
    inpt_ims = None
    inpt_axes = None

    max_iters = 25
    delta = 0.1
    epsilon = 0.1

    for i in range(n_examples):
        # Get next input sample.
        original = images[i % len(images)].contiguous().view(-1)
        label = labels[i % len(images)]

        # Check if the image is correctly classified.
        sample = poisson(datum=original, time=time)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        # Check for incorrect classification.
        s = spikes['Y'].get('s').view(1, n_neurons, time)
        prediction = ngram(spikes=s, ngram_scores=ngram_scores, n_labels=10, n=2).item()

        if prediction != label:
            continue

        # Create adversarial example.
        adversarial = False
        while not adversarial:
            adv_example = 255 * torch.rand(original.size())
            sample = poisson(datum=adv_example, time=time)
            inpts = {'X': sample}

            # Run the network on the input.
            network.run(inpts=inpts, time=time)

            # Check for incorrect classification.
            s = spikes['Y'].get('s').view(1, n_neurons, time)
            prediction = ngram(spikes=s, ngram_scores=ngram_scores, n_labels=n_classes, n=2).item()

            if prediction == label:
                adversarial = True

        j = 0
        current = original.clone()
        while j < max_iters:
            # Orthogonal perturbation.
            # perturb = orthogonal_perturbation(delta=delta, image=adv_example, target=original)
            # temp = adv_example + perturb

            # # Forward perturbation.
            # temp = temp.clone() + forward_perturbation(epsilon * get_diff(temp, original), temp, adv_example)

            # print(temp)

            perturbation = torch.randn(original.size())

            unnormed_source_direction = original - perturbation
            source_norm = torch.norm(unnormed_source_direction)
            source_direction = unnormed_source_direction / source_norm

            dot = torch.dot(perturbation, source_direction)
            perturbation -= dot * source_direction
            perturbation *= epsilon * source_norm / torch.norm(perturbation)

            D = 1 / np.sqrt(epsilon ** 2 + 1)
            direction = perturbation - unnormed_source_direction
            spherical_candidate = current + D * direction

            spherical_candidate = torch.clamp(spherical_candidate, 0, 255)

            new_source_direction = original - spherical_candidate
            new_source_direction_norm = torch.norm(new_source_direction)

            # length if spherical_candidate would be exactly on the sphere
            length = delta * source_norm

            # length including correction for deviation from sphere
            deviation = new_source_direction_norm - source_norm
            length += deviation

            # make sure the step size is positive
            length = max(0, length)

            # normalize the length
            length = length / new_source_direction_norm

            candidate = spherical_candidate + length * new_source_direction
            candidate = torch.clamp(candidate, 0, 255)

            sample = poisson(datum=candidate, time=time)
            inpts = {'X': sample}

            # Run the network on the input.
            network.run(inpts=inpts, time=time)

            # Check for incorrect classification.
            s = spikes['Y'].get('s').view(1, n_neurons, time)
            prediction = ngram(spikes=s, ngram_scores=ngram_scores, n_labels=10, n=2).item()

            # Optionally plot various simulation information.
            if plot:
                _input = original.view(side_length, side_length)
                reconstruction = candidate.view(side_length, side_length)
                _spikes = {
                    'X': spikes['X'].get('s').view(side_length ** 2, time),
                    'Y': spikes['Y'].get('s').view(n_neurons, time)
                }
                w = network.connections['X', 'Y'].w

                spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
                weights_im = plot_locally_connected_weights(
                    w, n_filters, kernel_size, conv_size, locations, side_length, im=weights_im
                )
                inpt_axes, inpt_ims = plot_input(
                    _input, reconstruction, label=labels[i], ims=inpt_ims, axes=inpt_axes
                )

                plt.pause(1e-8)

            if prediction == label:
                print('Attack failed.')
            else:
                print('Attack succeeded.')
                adv_example = candidate

            j += 1

        network.reset_()  # Reset state variables.

    print('\nAdversarial attack complete.\n')


if __name__ == '__main__':
    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_examples', type=int, default=100)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(gpu=False, plot=False)
    args = parser.parse_args()
    args = vars(args)

    print()
    main(**args)
    print()
