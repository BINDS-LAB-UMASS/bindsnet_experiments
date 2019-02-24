import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from experiments import ROOT_DIR

location = 'gpu' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)

        self.batch_size = x.size(1)

        rnn_out, (self.hidden, cell) = self.lstm(x)
        out = self.fc(self.hidden)

        return out.view(-1, self.output_size)


def main():
    path = os.path.join(
        ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
        'train_2_12_4_100_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    all_spikes = []
    all_labels = []
    for i in tqdm(range(220, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        all_spikes.append(spikes)
        all_labels.append(labels)

    spikes = torch.cat(all_spikes, dim=0)
    labels = torch.cat(all_labels).long()

    n_input = 100 * 9
    n_hidden = 64
    n_categories = 10
    train_epochs = 100
    batch_size = 128

    model = RNN(batch_size, n_input, n_hidden, n_categories)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_accuracy(logit, target, batch_size):
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()

    for i in range(train_epochs):
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()

        for j in range(spikes.size(0) // batch_size):
            optimizer.zero_grad()

            batch_indices = np.random.choice(np.arange(spikes.size(0)), size=batch_size, replace=False)
            batch_spikes, batch_labels = spikes[batch_indices], labels[batch_indices]

            outputs = model(batch_spikes)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, batch_labels, batch_size)

        model.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' % (i, train_running_loss / j, train_acc / j))

    # path = os.path.join(
    #     ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
    #     'test_2_12_4_100_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    # )


if __name__ == '__main__':
    main()
