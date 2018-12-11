import os
import torch
import foolbox

from bindsnet.datasets import MNIST
from bindsnet.network import load_network

from experiments import ROOT_DIR
from experiments.robustness.mnist import BindsNETModel


intensity = 0.5
crop = 4

# Load network.
model_name = '0_12_4_150_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
network = load_network(
    os.path.join(
        ROOT_DIR, 'params', 'mnist', 'crop_locally_connected', f'{model_name}.pt'
    )
)

network.layers['Y'].theta_plus = 0
network.layers['Y'].theta_decay = 0

# Neuron assignments and spike proportions.
path = os.path.join(
    ROOT_DIR, 'params', 'mnist', 'crop_locally_connected', '_'.join(['auxiliary', model_name]) + '.pt'
)
_, _, _, ngram_scores = torch.load(open(path, 'rb'))

# Load MNIST data.
dataset = MNIST(
    path=os.path.join(
        ROOT_DIR, 'data', 'MNIST'
    ), download=True
)

images, labels = dataset.get_test()
images *= intensity
images = images[:, crop:-crop, crop:-crop].contiguous()
images = images.view(10000, -1)

for l in network.layers:
    network.layers[l].dt = network.dt

for c in network.connections:
    network.connections[c].dt = network.dt

for m in network.monitors:
    network.monitors[m].record_length = 0

network.layers['Y'].one_spike = True
network.layers['Y'].lbound = None

preprocessing = (0, 1)
fmodel = BindsNETModel(
    network, bounds=(0, 255), preprocessing=preprocessing, ngram_scores=ngram_scores
)

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
# pdb.set_trace()
for i in range(10000):
    image = images[i].numpy()
    label = labels[i].long().item()

    attack = foolbox.attacks.BoundaryAttack(fmodel)
    adversarial = attack(image, label, verbose=True)
# if the attack fails, adversarial will be None and a warning will be printed