import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from experiments import ROOT_DIR
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

def main(seed=0, n_filters=100):
    path = os.path.join(
        '/', 'media', 'bigdrive2', 'djsaunde', 'crop_locally_connected',
        f'train_{seed}_12_4_{n_filters}_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    model = SGDClassifier(loss='log')

    # all_spikes = []
    all_labels = []
    predictions = []
    for i in tqdm(range(1, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=torch.device('cpu'))
        summed = spikes.sum(1)
        model.partial_fit(summed, labels, classes=np.arange(10))
        predictions.append(model.predict(summed))

        # all_spikes.append(spikes.sum(1))
        all_labels.append(labels)

    # spikes = torch.cat(all_spikes, dim=0)
    predictions = np.concatenate(predictions)
    labels = torch.cat(all_labels)

    # model = LogisticRegression()
    # model.fit(spikes, labels)
    
    # accuracy = (model.predict(spikes) == labels.cpu().numpy()).mean() * 100
    train_accuracy = (predictions == labels.cpu().numpy()).mean() * 100
    print(f'Training accuracy: {train_accuracy:.2f}')

    path = os.path.join(
        '/', 'media', 'bigdrive2', 'djsaunde', 'crop_locally_connected',
        f'test_{seed}_12_4_{n_filters}_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    # all_spikes = []
    all_labels = []
    predictions = []
    for i in tqdm(range(1, 40)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=torch.device('cpu'))
        predictions.append(model.predict(spikes.sum(1)))
        all_labels.append(labels)

    # spikes = torch.cat(all_spikes, dim=0)
    predictions = np.concatenate(predictions)
    labels = torch.cat(all_labels)

    # accuracy = (model.predict(spikes) == labels.cpu().numpy()).mean() * 100
    test_accuracy = (predictions == labels.cpu().numpy()).mean() * 100
    print(f'Test accuracy: {test_accuracy:.2f}')

    results_path = os.path.join(
        ROOT_DIR, 'results', 'lcsnn_logreg'
    )
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    if not os.path.isfile(os.path.join(results_path, 'results.csv')):
        with open(os.path.join(results_path, 'results.csv'), 'w') as f:
            f.write('seed,n_filters,train_accuracy,test_accuracy\n')

    with open(os.path.join(results_path, 'results.csv'), 'a') as f:
        f.write(f'{seed},{n_filters},{train_accuracy},{test_accuracy}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_filters', type=int, default=100)
    args = parser.parse_args()
    args = vars(args)
    main(**args)

