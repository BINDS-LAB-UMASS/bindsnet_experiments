import os
import torch

from experiments import ROOT_DIR

from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load_network

network = load_network(
    os.path.join(
        ROOT_DIR, 'params', 'mnist', 'diehl_and_cook_2015',
        '2_400_60000_500.0_0.01_0.99_250_1_0.05_1e-07_0.5_10_250.pt'
    )
)

auxiliary = torch.load(
    os.path.join(
        ROOT_DIR, 'params', 'mnist', 'diehl_and_cook_2015',
        'auxiliary_2_400_60000_500.0_0.01_0.99_250_1_0.05_1e-07_0.5_10_250.pt'
    )
)

images, labels = MNIST(
    path=os.path.join(
        ROOT_DIR, 'data', 'MNIST'
    ), download=True, shuffle=True
).get_train()

spikes = poisson(
    datum=images[0].view(-1), time=250, dt=1
)

network.run(
    inpts={'X': spikes}, time=250
)
