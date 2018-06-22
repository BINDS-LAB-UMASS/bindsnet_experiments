import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def sort_best_params(model='diehl_and_cook_2015',
				     data='mnist',
				     train=True,
                     fix=None,
				     metric='mean_all_activity',
                     top=None):
    '''
    Gets the train or test results from a particular data / model combo's
    experiments, fix certain parameters to values, and return the results sorted
    in descending order of some metric.
    '''

    name = '_'.join([model, data])
    f = 'train.csv' if train else 'test.csv'
    path = os.path.join('..', 'results', name, f)

    df = pd.read_csv(path)

    for v in df.columns:
        df[v] = df[v].apply(str)

    for v in fix:
        df = df[df[v] == fix[v]]

    df = df.sort_values(by=metric, ascending=False)

    if top is None:
        return df

    return df.head(top)

def compute_stats(df,
                  stats=['mean', 'min', 'max', 'std'],
                  metrics=['mean_all_activity', 'mean_proportion_weighting']):
    '''
    Computes stats over metrics for a particular results dataframe, over all
    parameters besides random seeds.
    '''
    for s in stats:
        for m in metrics:
            comp = df.apply(s)
            print(s, m, comp)

