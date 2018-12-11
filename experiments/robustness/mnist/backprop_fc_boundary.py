import os
import foolbox
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from experiments import ROOT_DIR
from bindsnet.datasets import MNIST
from foolbox.models import PyTorchModel

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


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


def main(seed=0, n_epochs=5, batch_size=100):

    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    print()
    print('Creating and training the ANN...')
    print()

    # Create and train an ANN on the MNIST dataset.
    ANN = FullyConnectedNetwork()

    # Get the MNIST data.
    images, labels = MNIST(os.path.join(
        ROOT_DIR, 'data', 'MNIST'
    ), download=True).get_train()

    images /= images.max()  # Standardizing to [0, 1].
    images = images.view(-1, 784)
    labels = labels.long()

    # Specify optimizer and loss function.
    optimizer = optim.Adam(params=ANN.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the ANN.
    batches_per_epoch = int(images.size(0) / batch_size)
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
            accuracies.append(correct.item())

        print(f'Epoch: {i+1} / {n_epochs}; Loss: {np.mean(losses):.4f}; Accuracy: {np.mean(accuracies) * 100:.4f}')

    ANN = ANN.eval()

    fmodel = PyTorchModel(
        ANN, bounds=(0, 1), num_classes=10
    )

    # apply attack on source image
    for i in range(10000):
        image = images[i].cpu().numpy()
        label = labels[i].long().item()

        attack = foolbox.attacks.BoundaryAttack(fmodel)
        try:
            adversarial = attack(image, label, verbose=True, iterations=1000) * 1.001
        except AssertionError:
            continue

        print(f'{i}: adversarial')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    args = vars(parser.parse_args())

    main(**args)
