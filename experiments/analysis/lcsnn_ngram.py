import os
import torch

from tqdm import tqdm
from experiments import ROOT_DIR
from bindsnet.evaluation import ngram, update_ngram_scores

location = 'gpu' if torch.cuda.is_available() else 'cpu'


def main():
    path = os.path.join(
        ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
        'train_2_12_4_100_4_0.01_0.99_60000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    ngram_scores = {}
    for i in tqdm(range(200, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        ngram_scores = update_ngram_scores(
            spikes=spikes, labels=labels, n_labels=10, n=2, ngram_scores=ngram_scores
        )

    all_labels = torch.LongTensor()
    all_predictions = torch.LongTensor()
    for i in tqdm(range(200, 240)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        predictions = ngram(
            spikes=spikes, ngram_scores=ngram_scores, n_labels=10, n=2
        )
        all_labels = torch.cat([all_labels, labels.long()])
        all_predictions = torch.cat([all_predictions, predictions.long()])

    accuracy = (all_labels == all_predictions).float().mean() * 100
    print(f'Training accuracy: {accuracy:.2f}')

    path = os.path.join(
        ROOT_DIR, 'spikes', 'mnist', 'crop_locally_connected',
        'test_2_12_4_100_4_0.01_0.99_60000_10000_250.0_250_1.0_0.05_1e-07_0.5_0.2_10_250'
    )

    all_labels = torch.LongTensor()
    all_predictions = torch.LongTensor()
    for i in tqdm(range(1, 40)):
        f = os.path.join(path, f'{i}.pt')
        spikes, labels = torch.load(f, map_location=location)
        predictions = ngram(
            spikes=spikes, ngram_scores=ngram_scores, n_labels=10, n=2
        )
        all_labels = torch.cat([all_labels, labels.long()])
        all_predictions = torch.cat([all_predictions, predictions.long()])

    accuracy = (all_labels == all_predictions).float().mean() * 100
    print(f'Test accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    main()
