import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix

from bindsnet.learning import NoOp
from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load_network
from bindsnet.network.monitors import Monitor
from bindsnet.models import LocallyConnectedNetwork
from bindsnet.evaluation import assign_labels, update_ngram_scores
from bindsnet.analysis.plotting import plot_locally_connected_weights, plot_spikes

from experiments import ROOT_DIR
from experiments.utils import update_curves, print_results

model = 'crop_locally_connected'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_train=60000, inhib=250, kernel_size=(16,), stride=(2,), n_filters=25, crop=4, lr=0.01,
         lr_decay=1, time=100, dt=1, theta_plus=0.05, theta_decay=1e-7, intensity=1, norm=0.2, progress_interval=10,
         update_interval=250, p_remove=0.1, plot=False, gpu=False):

    assert n_train % update_interval == 0, 'No. examples must be divisible by update_interval'

    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    if len(stride) == 1:
        stride = stride[0]

    params = [
        seed, kernel_size, stride, n_filters, crop, lr, lr_decay, n_train, inhib, time, dt,
        theta_plus, theta_decay, intensity, norm, progress_interval, update_interval
    ]

    test_params = [
        seed, kernel_size, stride, n_filters, crop, lr, lr_decay, n_train, inhib, time, dt,
        theta_plus, theta_decay, intensity, norm, progress_interval, update_interval, p_remove
    ]

    model_name = '_'.join([str(x) for x in params])

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    side_length = 28 - crop * 2
    n_examples = 10000
    n_classes = 10

    # Load network.
    network = load_network(os.path.join(params_path, model_name + '.pt'))
    network.connections['X', 'Y'].update_rule = NoOp(
        connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
    )
    network.layers['Y'].theta_decay = 0
    network.layers['Y'].theta_plus = 0
    network.connections['X', 'Y'].norm = None

    # Remove `p_remove` percentage of neurons (set outgoing synapses to 0).
    mask = torch.bernoulli(p_remove * torch.ones(network.layers['Y'].shape)).byte()
    network.connections['X', 'Y'].w[:, mask] = 0

    conv_size = network.connections['X', 'Y'].conv_size
    locations = network.connections['X', 'Y'].locations
    conv_prod = int(np.prod(conv_size))
    n_neurons = n_filters * conv_prod

    # Voltage recording for excitatory and inhibitory layers.
    voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
    network.add_monitor(voltage_monitor, name='output_voltage')

    # Load MNIST data.
    dataset = MNIST(path=data_path, download=True)

    images, labels = dataset.get_test()
    images *= intensity
    images = images[:, crop:-crop, crop:-crop]

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, time, n_neurons)

    # Neuron assignments and spike proportions.
    path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
    assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name=f'{layer}_spikes')

    spike_ims = None
    spike_axes = None
    weights_im = None

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

        if i % update_interval == 0 and i > 0:
            if i % len(labels) == 0:
                current_labels = labels[-update_interval:]
            else:
                current_labels = labels[i % len(images) - update_interval:i % len(images)]

            # Update and print accuracy evaluations.
            curves, preds = update_curves(
                curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                proportions=proportions, ngram_scores=ngram_scores, n=2
            )
            print_results(curves)

            for scheme in preds:
                predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

            # Save accuracy curves to disk.
            to_write = ['neuron_robustness'] + test_params
            f = '_'.join([str(x) for x in to_write]) + '.pt'
            torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

            print()

        # Get next input sample.
        image = images[i % len(images)].contiguous().view(-1)
        sample = poisson(datum=image, time=time)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Y'].get('s').sum() < 5 and retries < 3:
            retries += 1
            image *= 2
            sample = poisson(datum=image, time=time)
            inpts = {'X': sample}
            network.run(inpts=inpts, time=time)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Y'].get('s').t()

        # Optionally plot various simulation information.
        if plot:
            _spikes = {
                'X': spikes['X'].get('s').view(side_length ** 2, time),
                'Y': spikes['Y'].get('s').view(n_filters * conv_prod, time)
            }

            spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_locally_connected_weights(
                network.connections[('X', 'Y')].w, n_filters, kernel_size,
                conv_size, locations, side_length, im=weights_im
            )

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    i += 1

    if i % len(labels) == 0:
        current_labels = labels[-update_interval:]
    else:
        current_labels = labels[i % len(images) - update_interval:i % len(images)]

    # Update and print accuracy evaluations.
    curves, preds = update_curves(
        curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
        proportions=proportions, ngram_scores=ngram_scores, n=2
    )
    print_results(curves)

    for scheme in preds:
        predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

    print('Average accuracies:\n')
    for scheme in curves.keys():
        print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    f = '_'.join([str(x) for x in ['neuron_robustness'] + test_params]) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

    # Save results to disk.
    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    ]

    to_write = [str(x) for x in test_params + results]
    name = 'neuron_robustness.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            f.write(
                'random_seed,kernel_size,stride,n_filters,crop,lr,lr_decay,n_train,inhib,time,timestep,'
                'theta_plus,theta_decay,intensity,norm,progress_interval,update_interval,p_remove,mean_all_activity,'
                'mean_proportion_weighting,mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
            )

    with open(os.path.join(results_path, name), 'a') as f:
        f.write(','.join(to_write) + '\n')

    if labels.numel() > n_examples:
        labels = labels[:n_examples]
    else:
        while labels.numel() < n_examples:
            if 2 * labels.numel() > n_examples:
                labels = torch.cat([labels, labels[:n_examples - labels.numel()]])
            else:
                labels = torch.cat([labels, labels])

    # Compute confusion matrices and save them to disk.
    confusions = {}
    for scheme in predictions:
        confusions[scheme] = confusion_matrix(labels, predictions[scheme])

    f = '_'.join([str(x) for x in ['neuron_robustness'] + test_params]) + '.pt'
    torch.save(confusions, os.path.join(confusion_path, f))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--inhib', type=float, default=250, help='inhibition connection strength')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[16], help='one or two kernel side lengths')
    parser.add_argument('--stride', type=int, nargs='+', default=[2], help='one or two horizontal stride lengths')
    parser.add_argument('--n_filters', type=int, default=25, help='no. of convolutional filters')
    parser.add_argument('--crop', type=int, default=4, help='amount to crop images at borders')
    parser.add_argument('--lr', type=float, default=0.01, help='post-synaptic learning rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=25, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1.0, help='simulation integreation timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--theta_decay', type=float, default=1e-7, help='adaptive threshold decay time constant')
    parser.add_argument('--intensity', type=float, default=1, help='constant to multiple input data by')
    parser.add_argument('--norm', type=float, default=0.2, help='plastic synaptic weight normalization constant')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--p_remove', default=0.1, type=float, help='percent neurons to remove')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False)
    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
