import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import EMNIST

from experiments import ROOT_DIR


dataset = EMNIST(
    os.path.join(
        ROOT_DIR, 'data', 'EMNIST'
    ), split='letters', train=True, download=True
)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = x.view(-1)

        print(x.shape)


model = CNN()
