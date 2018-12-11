import torch
import numpy as np

from typing import Dict, Tuple

from bindsnet.evaluation import all_activity, proportion_weighting, ngram, logreg_predict


def print_results(results: Dict[str, list]) -> None:
    # language=rst
    """
    Prints out latest, mean, and maximum results per classification scheme.

    :param results: Mapping from name of classification scheme to list of accuracy evaluations.
    """
    print()
    for s in results:
        last = results[s][-1]
        mean = torch.mean(torch.tensor(results[s])).item()
        std = torch.std(torch.tensor(results[s])).item()
        best = torch.max(torch.tensor(results[s])).item()

        print(f'Results for scheme "{s}": {last:.2f} (last), {mean:.2f} (mean), {std:.2f} (std.), {best:.2f} (best)')


def update_curves(curves: Dict[str, list], labels: torch.Tensor,
                  n_classes: int, **kwargs) -> Tuple[Dict[str, list], Dict[str, torch.Tensor]]:
    # language=rst
    """
    Updates accuracy curves for each classification scheme.

    :param curves: Mapping from name of classification scheme to list of accuracy evaluations.
    :param labels: One-dimensional ``torch.Tensor`` of integer data labels.
    :param n_classes: Number of data categories.
    :param kwargs: Additional keyword arguments for classification scheme evaluation functions.
    :return: Updated accuracy curves and predictions.
    """
    predictions = {}
    for scheme in curves:
        # Branch based on name of classification scheme
        if scheme == 'all':
            spike_record = kwargs['spike_record']
            assignments = kwargs['assignments']

            prediction = all_activity(spike_record, assignments, n_classes)
        elif scheme == 'proportion':
            spike_record = kwargs['spike_record']
            assignments = kwargs['assignments']
            proportions = kwargs['proportions']

            prediction = proportion_weighting(spike_record, assignments, proportions, n_classes)
        elif scheme == 'ngram':
            spike_record = kwargs['spike_record']
            ngram_scores = kwargs['ngram_scores']
            n = kwargs['n']

            prediction = ngram(spike_record, ngram_scores, n_classes, n)
        elif scheme == 'logreg':
            full_spike_record = kwargs['full_spike_record']
            logreg = kwargs['logreg']

            prediction = logreg_predict(spikes=full_spike_record, logreg=logreg)

        else:
            raise NotImplementedError

        # Compute accuracy with current classification scheme.
        predictions[scheme] = prediction
        accuracy = torch.sum(labels.long() == prediction).float() / len(labels)
        curves[scheme].append(100 * accuracy)

    return curves, predictions


def bit_flip(x: torch.Tensor, p: float) -> torch.Tensor:
    # language=rst
    """
    Takes a binary tensor and flips each entry with probability p.

    :param x: Arbitrarily shaped binary tensor.
    :param p: Bit flip probability.
    """
    x = x.float()
    i = torch.bernoulli(p * torch.ones_like(x)).byte()
    x = x.byte()
    x[i] = ~x[i]
    return x.float()