import os
import torch

from tqdm import tqdm
from itertools import product
from typing import Dict, Tuple
from experiments import ROOT_DIR

location = 'gpu' if torch.cuda.is_available() else 'cpu'


def ngram(spikes: torch.Tensor, ngram_scores: Dict[Tuple[int, ...], torch.Tensor], n_labels: int,
          n: int) -> torch.Tensor:
    # language=rst
    """
    Predicts between ``n_labels`` using ``ngram_scores``.

    :param spikes: Spikes of shape ``(n_examples, time, n_neurons)``.
    :param ngram_scores: Previously recorded scores to update.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :return: Predictions per example.
    """
    predictions = []
    for activity in spikes:
        score = torch.zeros(n_labels)

        # Aggregate all of the firing neurons' indices
        fire_order = []
        for t in range(activity.size()[0]):
            ordering = torch.nonzero(activity[t].view(-1))
            if ordering.numel() > 0:
                fire_order += ordering[:, 0].tolist()

        # Consider all n-gram sequences.
        for j in range(len(fire_order) - n):
            if tuple(fire_order[j:j + n]) in ngram_scores:
                score += ngram_scores[tuple(fire_order[j:j + n])]

        predictions.append(torch.argmax(score))

    return torch.Tensor(predictions).long()


def update_ngram_scores(spikes: torch.Tensor, labels: torch.Tensor, n_labels: int, n: int,
                        ngram_scores: Dict[Tuple[int, ...], torch.Tensor]) -> Dict[Tuple[int, ...], torch.Tensor]:
    # language=rst
    """
    Updates ngram scores by adding the count of each spike sequence of length n from the past ``n_examples``.

    :param spikes: Spikes of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param ngram_scores: Previously recorded scores to update.
    :return: Dictionary mapping n-grams to vectors of per-class spike counts.
    """
    for i, activity in enumerate(spikes):
        # Obtain firing order for spiking activity.
        fire_order = []

        # Aggregate all of the firing neurons' indices.
        for t in range(spikes.size(1)):
            # Gets the indices of the neurons which fired on this timestep.
            ordering = torch.nonzero(activity[t]).view(-1)
            if ordering.numel() > 0:  # If there was more than one spike...
                # Add the indices of spiked neurons to the fire ordering.
                ordering = ordering.tolist()
                fire_order.append(ordering)

        # Check every sequence of length n.
        for order in zip(*(fire_order[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence not in ngram_scores:
                    ngram_scores[sequence] = torch.zeros(n_labels)

                ngram_scores[sequence][int(labels[i])] += 1

    return ngram_scores


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
