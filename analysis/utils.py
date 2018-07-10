import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product

sns.set()


def sort_best_params(model='diehl_and_cook_2015',
				     data='mnist',
				     train=True,
                     fix=None,
                     vary=None,
				     metric='mean_all_activity',
                     top=None):
    """
    Gets the train or test results from a particular data / model combo's
    experiments, fix certain parameters to values, and return the results sorted
    in descending order of some metric.
    """
    name = '_'.join([model, data])
    f = 'train.csv' if train else 'test.csv'
    path = os.path.join('..', 'results', name, f)

    df = pd.read_csv(path)

    for v in df.columns:
        df[v] = df[v].apply(str)

    for v in fix:
        df = df[df[v] == fix[v]]

    unique = {}
    for v in vary:
        unique[v] = df[v].unique()

    df.drop(labels=[c for c in df.columns if 'max' in c],
            axis='columns',
            inplace=True)

    groupby_columns = list(set(df.columns) - {'random_seed'} - \
                           set([c for c in df.columns if 'mean' in c]))

    average_columns = [c for c in df.columns if 'mean' in c]
    for c in average_columns:
        df[c] = df[c].astype(float)

    dfs = {}
    for p in product(*unique.values()):
        temp = df.copy()
        for i, v in enumerate(vary):
            temp = temp[temp[v] == p[i]]
        dfs[p] = temp

    averaged = {}
    for key, df in dfs.items():
        df = df.groupby(by=groupby_columns)[average_columns].agg({'mean' : np.mean, 'std' : np.std})
        df = df.sort_values(by=('mean', metric), ascending=False)

        if df.size != 0:
            if top is None:
                averaged[key] = df
            else:
                averaged[key] = df.head(1)

    return averaged


def compute_stats(df,
                  stats=['mean', 'min', 'max', 'std'],
                  metrics=['mean_all_activity', 'mean_proportion_weighting']):
    """
    Computes stats over metrics for a particular results dataframe, over all
    parameters besides random seeds.
    """
    for s in stats:
        for m in metrics:
            comp = df.apply(s)
            print(s, m, comp)

