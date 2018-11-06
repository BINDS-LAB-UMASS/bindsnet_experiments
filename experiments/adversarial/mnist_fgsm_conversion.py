import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from time import time as t

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf

from bindsnet.datasets import MNIST
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

data = 'mnist'
model = 'fgsm'

params_path = os.path.join('..', '..', 'params', data, model)
if not os.path.isdir(params_path):
    os.makedirs(params_path)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FullyConnectedNetwork(nn.Module):
    # language=rst
    """
    Simply fully-connected network implemented in PyTorch.
    """
    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def main(seed=0, n_epochs=5, batch_size=100, time=50, update_interval=50, plot=False, save=True):

    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    print()
    print('Loading MNIST data...')
    print()

    # Get the CIFAR-10 data.
    images, labels = MNIST('../../data/MNIST', download=True).get_train()
    images /= images.max()  # Standardizing to [0, 1].
    images = images.view(-1, 784)
    labels = labels.long()

    test_images, test_labels = MNIST('../../data/MNIST', download=True).get_test()
    test_images /= test_images.max()  # Standardizing to [0, 1].
    test_images = test_images.view(-1, 784)
    test_labels = test_labels.long()

    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()

    ANN = FullyConnectedNetwork()

    model_name = '_'.join([
        str(x) for x in [seed, n_epochs, batch_size, time, update_interval]
    ])

    # Specify loss function.
    criterion = nn.CrossEntropyLoss()
    if save and os.path.isfile(os.path.join(params_path, model_name + '.pt')):
        print()
        print('Loading trained ANN from disk...')
        ANN.load_state_dict(torch.load(os.path.join(params_path, model_name + '.pt')))

        if torch.cuda.is_available():
            ANN = ANN.cuda()
    else:
        print()
        print('Creating and training the ANN...')
        print()

        # Specify optimizer.
        optimizer = optim.Adam(params=ANN.parameters(), lr=1e-3, weight_decay=1e-4)

        batches_per_epoch = int(images.size(0) / batch_size)

        # Train the ANN.
        for i in range(n_epochs):
            losses = []
            accuracies = []
            for j in range(batches_per_epoch):
                batch_idxs = torch.from_numpy(
                    np.random.choice(np.arange(images.size(0)), size=batch_size, replace=False)
                )
                im_batch = images[batch_idxs]
                label_batch = labels[batch_idxs]

                outputs = ANN.forward(im_batch)
                loss = criterion(outputs, label_batch)
                predictions = torch.max(outputs, 1)[1]
                correct = (label_batch == predictions).sum().float() / batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                accuracies.append(correct.item() * 100)

            outputs = ANN.forward(test_images)
            loss = criterion(outputs, test_labels).item()
            predictions = torch.max(outputs, 1)[1]
            test_accuracy = ((test_labels == predictions).sum().float() / test_labels.numel()).item() * 100

            avg_loss = np.mean(losses)
            avg_acc = np.mean(accuracies)

            print(f'Epoch: {i+1} / {n_epochs}; Train Loss: {avg_loss:.4f}; Train Accuracy: {avg_acc:.4f}')
            print(f'\tTest Loss: {loss:.4f}; Test Accuracy: {test_accuracy:.4f}')

        if save:
            torch.save(ANN.state_dict(), os.path.join(params_path, model_name + '.pt'))

    outputs = ANN.forward(test_images)
    loss = criterion(outputs, test_labels)
    predictions = torch.max(outputs, 1)[1]
    accuracy = ((test_labels == predictions).sum().float() / test_labels.numel()).item() * 100

    print()
    print(f'(Post training) Test Loss: {loss:.4f}; Test Accuracy: {accuracy:.4f}')

    print()
    print('Evaluating ANN on adversarial examples from FSGM method...')

    # Convert pytorch model to a tf_model and wrap it in cleverhans.
    tf_model_fn = convert_pytorch_model_to_tf(ANN)
    cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

    sess = tf.Session()
    x_op = tf.placeholder(tf.float32, shape=(None, 784,))

    # Create an FGSM attack.
    fgsm_op = FastGradientMethod(cleverhans_model, sess=sess)
    fgsm_params = {
        'eps': 0.2, 'clip_min': 0.0, 'clip_max': 1.0
    }
    adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
    adv_preds_op = tf_model_fn(adv_x_op)

    # Run an evaluation of our model against FGSM white-box attack.
    total = 0
    correct = 0
    adv_preds = sess.run(adv_preds_op, feed_dict={x_op: test_images})
    correct += (np.argmax(adv_preds, axis=1) == test_labels).sum()
    total += len(test_images)
    accuracy = float(correct) / total

    print()
    print('Adversarial accuracy: {:.3f}'.format(accuracy * 100))

    print()
    print('Converting ANN to SNN...')

    with sess.as_default():
        test_images = adv_x_op.eval(feed_dict={x_op: test_images})

    test_images = torch.tensor(test_images)

    # Do ANN to SNN conversion.
    SNN = ann_to_snn(ANN, input_shape=(784,), data=test_images, percentile=100)

    for l in SNN.layers:
        if l != 'Input':
            SNN.add_monitor(
                Monitor(SNN.layers[l], state_vars=['s', 'v'], time=time), name=l
            )

    print()
    print('Testing SNN on FGSM-modified MNIST data...')
    print()

    # Test SNN on MNIST data.
    spike_ims = None
    spike_axes = None
    correct = []

    n_images = test_images.size(0)

    start = t()
    for i in range(n_images):
        if i > 0 and i % update_interval == 0:
            accuracy = np.mean(correct) * 100
            print(f'Progress: {i} / {n_images}; Elapsed: {t() - start:.4f}; Accuracy: {accuracy:.4f}')
            start = t()

        SNN.run(inpts={'Input': test_images[i].repeat(time, 1, 1)}, time=time)

        spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in SNN.monitors}
        prediction = torch.softmax(voltages['fc3'].sum(1), 0).argmax()
        correct.append((prediction == test_labels[i]).item())

        SNN.reset_()

        if plot:
            spikes = {k: spikes[k].cpu() for k in spikes}
            spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
            plt.pause(1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--time', type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=50)
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(plot=False, save=True)
    args = vars(parser.parse_args())

    main(**args)
