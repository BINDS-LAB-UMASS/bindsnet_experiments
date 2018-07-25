import torch
import numpy as np

from typing import Dict

from bindsnet.evaluation import *


def print_results(results: Dict[str, list]) -> None:
    """
    Prints out latest, mean, and maximum results per classification scheme.

    :param results: Mapping from name of classification scheme to list of accuracy evaluations.
    """
    print()
    for s in results:
        print(f'Results for scheme "{s}": {results[s][-1]:.2f} (last), ' \
              f'{np.mean(results[s]):.2f} (average), {np.max(results[s]):.2f} (best)')


def update_curves(curves: Dict[str, list], labels: torch.Tensor, n_classes: int, **kwargs) -> Dict[str, list]:
    """
    Updates accuracy curves for each classification scheme.

    :param curves: Mapping from name of classification scheme to list of accuracy evaluations.
    :param labels: One-dimensional ``torch.Tensor`` of integer data labels.
    :param n_classes: Number of data categories.
    :param Dict[str, Any] kwargs: Additional keyword arguments for classification scheme evaluation functions.
    :return: Updated accuracy curves.
    """
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

        # Compute accuracy with current classification scheme.
        accuracy = torch.sum(labels.long() == prediction).float() / len(labels)
        curves[scheme].append(100 * accuracy)

    return curves