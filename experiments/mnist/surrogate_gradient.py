from typing import Optional, Union

import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import datasets, transforms


def fast_sigmoid(h):
    return h / (1 + torch.abs(h))

def grad_fast_sigmoid(h):
    return 1 / ((1 + torch.abs(h)) ** 2)


class LIFFunction(Function):

    @staticmethod
    def forward(ctx, x, v, decay, rest, thresh):
        # Decay voltage
        v = decay * (v - rest) + rest + x
        ctx.save_for_backward(x, v, thresh)

        # Check for spiking neurons.
        s = v >= thresh

        # Voltage reset.
        v.masked_fill_(s, rest)

        return s

    @staticmethod
    def backward(ctx, grad_output):
        x, v, thresh = ctx.saved_tensors
        print(grad_output.float().sum().item())
        return grad_output.float() * fast_sigmoid(v - thresh), None, None, None, None
        # return -(v - thresh) * (1 - grad_output.float()), None, None, None, None


class LIFNodes(torch.nn.Module):

    def __init__(
        self,
        n: Optional[int] = None,
        thresh: Union[float, torch.Tensor] = 0.,
        rest: Union[float, torch.Tensor] = 1.,
        tc_decay: Union[float, torch.Tensor] = 100.0,
    ) -> None:
        super(LIFNodes, self).__init__()

        self.n = n  # No. of neurons provided.
        self.shape = [self.n]  # Shape is equal to the size of the layer.

        self.register_buffer("s", torch.ByteTensor())  # Spike occurrences.
        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer("tc_decay", torch.tensor(tc_decay))  # Time constant of neuron voltage decay.
        self.register_buffer("decay", torch.zeros(*self.shape))  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.

    def forward(self, x: torch.Tensor) -> None:
        return LIFFunction.apply(x, self.v, self.decay, self.rest, self.thresh)

    def compute_decays(self, dt) -> None:
        self.dt = dt
        self.decay = torch.exp(-self.dt / self.tc_decay)  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        self.batch_size = batch_size
        self.s = torch.zeros(batch_size, *self.shape, requires_grad=True)
        self.v = self.rest * torch.ones(batch_size, *self.shape, requires_grad=True)


class LIFNetwork(torch.nn.Module):

    def __init__(self):
        super(LIFNetwork, self).__init__()

        self.dense = torch.nn.Linear(784, 10)
        self.lif_nodes = LIFNodes(n=10)
        self.lif_nodes.compute_decays(1)
        self.lif_nodes.set_batch_size(1)

    def forward(self, input):
        input = input.view(-1, 784)
        synaptic_input = self.dense(input)
        spikes = self.lif_nodes(synaptic_input)
        logits = torch.nn.functional.log_softmax(spikes.float())
        return logits


def train(model, device, train_loader, optimizer, epoch):
    import matplotlib.pyplot as plt
    plt.ion()

    fig, ax = plt.subplots()

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        for _ in range(50):
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward(retain_graph=True)

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            weights = list(network.dense.parameters())[0]
            print(weights.shape)
            weights = [
                weights[i, :].view(28, 28) for i in range(10)
            ]
            w = torch.zeros(5 * 28, 2 * 28)
            for i in range(5):
                for j in range(2):
                    w[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = weights[i + j * 5]

            ax.matshow(w.detach().numpy(), cmap='hot_r')
            plt.pause(0.1)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


network = LIFNetwork()
print(network)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda') if use_cuda else torch.device('cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Get MNIST data.
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1, shuffle=True, **kwargs)


optimizer = optim.Adam(network.parameters(), lr=1e-3)

for epoch in range(10):
    train(network, device, train_loader, optimizer, epoch + 1)
    test(network, device, test_loader)
