import os
import torch

from tqdm import tqdm
from experiments import ROOT_DIR
from sklearn.linear_model import LogisticRegression

location = 'gpu' if torch.cuda.is_available() else 'cpu'


def main():
    path = os.path.join(
        ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
        'train_2_12_4_100_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    all_spikes = []
    all_labels = []
    for i in tqdm(range(1, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        all_spikes.append(spikes.sum(1))
        all_labels.append(labels)

    spikes = torch.cat(all_spikes, dim=0)
    labels = torch.cat(all_labels)

    model = LogisticRegression()
    model.fit(spikes, labels)
    accuracy = (model.predict(spikes) == labels.numpy()).mean() * 100
    print(f'Training accuracy: {accuracy:.2f}')

    path = os.path.join(
        ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
        'test_2_12_4_100_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    all_spikes = []
    all_labels = []
    for i in tqdm(range(1, 40)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        all_spikes.append(spikes.sum(1))
        all_labels.append(labels)

    spikes = torch.cat(all_spikes, dim=0)
    labels = torch.cat(all_labels)

    accuracy = (model.predict(spikes) == labels.numpy()).mean() * 100
    print(f'Test accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    main()
