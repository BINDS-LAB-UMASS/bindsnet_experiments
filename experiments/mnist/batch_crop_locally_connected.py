import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from sklearn.metrics import confusion_matrix
from torchvision.transforms import transforms
from tqdm import tqdm

from bindsnet.learning import NoOp
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import poisson, PoissonEncoder
from bindsnet.network import load
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
spikes_path = os.path.join(ROOT_DIR, 'spikes', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, spikes_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_epochs=1, batch_size=32, update_steps=256, inhib=250, kernel_size=(16,), stride=(2,), time=100,
         n_filters=25, crop=0, lr_pre=1e-4 ,lr_post=1e-2, lr_decay=1, dt=1, theta_plus=0.05, tc_theta_decay=1e-7,
         intensity=5, norm=0.2, progress_interval=10, train=True, plot=False, gpu=False):

    update_interval = update_steps * batch_size

    params = [
        seed, kernel_size, stride, n_filters, crop, lr_pre, lr_post, lr_decay, n_epochs, batch_size,
        inhib, time, dt, theta_plus, tc_theta_decay, intensity, norm, progress_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    if not train:
        test_params = [
            seed, kernel_size, stride, n_filters, crop, lr_pre, lr_post, lr_decay, n_epochs, batch_size, inhib,
            time, dt, theta_plus, tc_theta_decay, intensity, norm, progress_interval, update_interval
        ]

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    side_length = 28 - crop * 2
    n_inpt = side_length ** 2
    n_classes = 10

    # Build network.
    if train:
        network = LocallyConnectedNetwork(
            n_inpt=n_inpt, input_shape=[side_length, side_length], kernel_size=kernel_size, stride=stride,
            n_filters=n_filters, inh=inhib, dt=dt, nu=[lr_pre, lr_post], theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay, wmin=0.0, wmax=1.0, norm=norm
        )
    else:
        network = load(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].tc_theta_decay = 0
        network.layers['Y'].theta_plus = 0

    conv_size = network.connections['X', 'Y'].conv_size
    locations = network.connections['X', 'Y'].locations
    conv_prod = int(np.prod(conv_size))
    n_neurons = n_filters * conv_prod

    # Voltage recording for excitatory and inhibitory layers.
    voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
    network.add_monitor(voltage_monitor, name='output_voltage')

    # Load MNIST data.
    dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x * intensity),
             transforms.Lambda(lambda x: x[:, crop:-crop, crop:-crop].contiguous().view(-1))]
        ),
    )

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, time, n_neurons)

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones(n_neurons)
        proportions = torch.zeros(n_neurons, n_classes)
        rates = torch.zeros(n_neurons, n_classes)
        ngram_scores = {}
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    if train:
        best_accuracy = 0

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name=f'{layer}_spikes')

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    spike_ims = None
    spike_axes = None
    weights_im = None

    for i in range(n_epochs):
        labels = []

        # Create a dataloader to iterate and batch data
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=gpu,
        )

        for step, batch in enumerate(tqdm(dataloader)):
            # Get next input sample.
            inpts = {"X": batch["encoded_image"]}
            if gpu:
                inpts = {k: v.cuda() for k, v in inpts.items()}

            if step % update_steps == 0 and step > 0:
                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels)

                # Update and print accuracy evaluations.
                curves, preds = update_curves(
                    curves, label_tensor, n_classes, spike_record=spike_record, assignments=assignments,
                    proportions=proportions, ngram_scores=ngram_scores, n=2
                )
                print_results(curves)

                for scheme in preds:
                    predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

                if train:
                    # Decay learning rate.
                    network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

                    print(network.connections['X', 'Y'].update_rule.nu[1])

                    # Save accuracy curves to disk.
                    to_write = ['train'] + params
                    f = '_'.join([str(x) for x in to_write]) + '.pt'
                    torch.save((curves, update_interval, n_epochs), open(os.path.join(curves_path, f), 'wb'))

                    if any([x[-1] > best_accuracy for x in curves.values()]):
                        print('New best accuracy! Saving network parameters to disk.')

                        # Save network to disk.
                        network.save(os.path.join(params_path, model_name + '.pt'))
                        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
                        torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

                        best_accuracy = max([x[-1] for x in curves.values()])

                    # Assign labels to excitatory layer neurons.
                    assignments, proportions, rates = assign_labels(spike_record, label_tensor, n_classes, rates)

                    # Compute ngram scores.
                    ngram_scores = update_ngram_scores(spike_record, label_tensor, n_classes, 2, ngram_scores)

                print()

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

            labels.extend(batch["label"].tolist())

            # Run the network on the input.
            network.run(inpts=inpts, time=time)

            # Add to spikes recording.
            s = spikes["Y"].get("s").permute((1, 0, 2))
            spike_record[
            (step * batch_size)
            % update_interval: (step * batch_size % update_interval)
                               + s.size(0)
            ] = s

            # Optionally plot various simulation information.
            if plot:
                _spikes = {
                    'X': spikes['X'].get('s').view(batch_size, time, side_length ** 2)[0],
                    'Y': spikes['Y'].get('s').view(batch_size, time, n_filters * conv_prod)[0]
                }

                spike_ims, spike_axes = plot_spikes(spikes=_spikes, ims=spike_ims, axes=spike_axes)
                weights_im = plot_locally_connected_weights(
                    network.connections['X', 'Y'].w, n_filters, kernel_size,
                    conv_size, locations, side_length, im=weights_im
                )

                plt.pause(1e-8)

            network.reset_()  # Reset state variables.

    # if i % len(labels) == 0:
    #     current_labels = labels[-update_interval:]
    # else:
    #     current_labels = labels[i % len(images) - update_interval:i % len(images)]
    #
    # if labels.numel() > n_examples:
    #     labels = labels[:n_examples]
    # else:
    #     while labels.numel() < n_examples:
    #         if 2 * labels.numel() > n_examples:
    #             labels = torch.cat([labels, labels[:n_examples - labels.numel()]])
    #         else:
    #             labels = torch.cat([labels, labels])
    #
    # # Update and print accuracy evaluations.
    # curves, preds = update_curves(
    #     curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
    #     proportions=proportions, ngram_scores=ngram_scores, n=2
    # )
    # print_results(curves)
    #
    # for scheme in preds:
    #     predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)
    #
    # if train:
    #     if any([x[-1] > best_accuracy for x in curves.values()]):
    #         print('New best accuracy! Saving network parameters to disk.')
    #
    #         # Save network to disk.
    #         network.save(os.path.join(params_path, model_name + '.pt'))
    #         path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
    #         torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

    # if train:
    #     print('\nTraining complete.\n')
    # else:
    #     print('\nTest complete.\n')
    #
    # print('Average accuracies:\n')
    # for scheme in curves.keys():
    #     print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))
    #
    # # Save accuracy curves to disk.
    # if train:
    #     to_write = ['train'] + params
    #     f = '_'.join([str(x) for x in to_write]) + '.pt'
    #     torch.save((curves, update_interval, n_epochs), open(os.path.join(curves_path, f), 'wb'))
    #
    # # Save results to disk.
    # results = [
    #     np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
    #     np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    # ]
    #
    # to_write = params + results if train else test_params + results
    # to_write = [str(x) for x in to_write]
    #
    # if train:
    #     name = 'train.csv'
    # else:
    #     name = 'test.csv'
    #
    # if not os.path.isfile(os.path.join(results_path, name)):
    #     with open(os.path.join(results_path, name), 'w') as f:
    #         if train:
    #             f.write(
    #                 'random_seed,kernel_size,stride,n_filters,crop,lr,lr_decay,n_train,inhib,time,timestep,theta_plus,'
    #                 'tc_theta_decay,intensity,norm,progress_interval,update_interval,mean_all_activity,'
    #                 'mean_proportion_weighting,mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
    #             )
    #         else:
    #             f.write(
    #                 'random_seed,kernel_size,stride,n_filters,crop,lr,lr_decay,n_train,n_test,inhib,time,timestep,'
    #                 'theta_plus,tc_theta_decay,intensity,norm,progress_interval,update_interval,mean_all_activity,'
    #                 'mean_proportion_weighting,mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
    #             )
    #
    # with open(os.path.join(results_path, name), 'a') as f:
    #     f.write(','.join(to_write) + '\n')
    #
    # # Compute confusion matrices and save them to disk.
    # confusions = {}
    # for scheme in predictions:
    #     confusions[scheme] = confusion_matrix(labels, predictions[scheme])
    #
    # to_write = ['train'] + params if train else ['test'] + test_params
    # f = '_'.join([str(x) for x in to_write]) + '.pt'
    # torch.save(confusions, os.path.join(confusion_path, f))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_epochs', type=int, default=1, help='no. of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
    parser.add_argument('--inhib', type=float, default=250, help='inhibition connection strength')
    parser.add_argument('--kernel_size', type=int, nargs='+', default=[16], help='one or two kernel side lengths')
    parser.add_argument('--stride', type=int, nargs='+', default=[2], help='one or two horizontal stride lengths')
    parser.add_argument('--n_filters', type=int, default=25, help='no. of convolutional filters')
    parser.add_argument('--crop', type=int, default=4, help='amount to crop images at borders')
    parser.add_argument('--lr_pre', type=float, default=0.0001, help='pre-synaptic learning rate')
    parser.add_argument('--lr_post', type=float, default=0.01, help='post-synaptic learning rate')
    parser.add_argument('--lr_decay', type=float, default=1, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=100, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1.0, help='simulation integration timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--tc_theta_decay', type=float, default=1e-7, help='adaptive threshold decay time constant')
    parser.add_argument('--intensity', type=float, default=128, help='constant to multiple input data by')
    parser.add_argument('--norm', type=float, default=0.2, help='plastic synaptic weight normalization constant')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='train phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    kernel_size = args.kernel_size
    stride = args.stride

    if len(kernel_size) == 1:
        kernel_size = kernel_size[0]
    if len(stride) == 1:
        stride = stride[0]

    args = vars(args)
    args['kernel_size'] = kernel_size
    args['stride'] = stride

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
