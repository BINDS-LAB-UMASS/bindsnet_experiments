import numpy as np
import matplotlib.pyplot as plt

from time import time as t

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from bindsnet.datasets import MNIST
from bindsnet.conversion import ann_to_snn
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

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
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


print()
print('Creating and training the ANN...')
print()

# Create and train an ANN on the MNIST dataset.
ANN = FullyConnectedNetwork()

# Get the MNIST data.
images, labels = MNIST('../../data/MNIST', download=True).get_train()
images /= images.max()  # Standardizing to [0, 1].
images = images.view(-1, 784)
labels = labels.long()

# Specify optimizer and loss function.
optimizer = optim.Adam(params=ANN.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train the ANN.
n_epochs = 10
batch_size = 100
batches_per_epoch = int(images.size(0) / batch_size)
for i in range(n_epochs):
    losses = []
    accuracies = []
    for j in range(batches_per_epoch):
        batch_idxs = torch.from_numpy(np.random.choice(np.arange(images.size(0)), size=batch_size, replace=False))
        im_batch = images[batch_idxs]
        label_batch = labels[batch_idxs]

        outputs = ANN.forward(im_batch)
        loss = criterion(outputs, label_batch)
        predictions = torch.max(outputs, 1)[1]
        correct = (label_batch == predictions).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(correct)

    print(f'Epoch: {i+1} / {n_epochs}; Loss: {np.mean(losses)}; Accuracy: {np.mean(accuracies)}')

print()
print('Converting ANN to SNN...')

# Do ANN to SNN conversion.
SNN = ann_to_snn(ANN, input_shape=(784,))

time = 100
update_interval = 50
plot = False

for l in SNN.layers:
    SNN.add_monitor(
        Monitor(SNN.layers[l], state_vars=['s'], time=time), name=l
    )

spike_ims = None
spike_axes = None
correct = []

print()
print('Testing SNN on MNIST data...')
print()

# Test SNN on MNIST data.
start = t()
for i in range(images.size(0)):
    if i > 0 and i % update_interval == 0:
        print(f'Progress: {i} / {images.size(0)}; Elapsed: {t() - start:.4f}; Accuracy: {np.mean(correct) * 100:.4f}')
        start = t()

    SNN.run(inpts={'Input': images[i].repeat(time, 1, 1)}, time=time)

    spikes = {layer: SNN.monitors[layer].get('s') for layer in SNN.monitors}
    prediction = torch.max(spikes['fc3'].sum(1), 0)[1].item()
    correct.append(prediction == labels[i])

    SNN.reset_()

    if plot:
        spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
        plt.pause(1e-3)
