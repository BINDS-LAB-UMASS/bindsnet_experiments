import os
import argparse
import numpy as np
import pandas as pd

from itertools import product


def main(model='diehl_and_cook_2015', data='mnist', train=True, fix={}, vary=[], metric='mean_all_activity', top=None):
    # language=rst
    """
    Gets the train or test results from a particular data / model combo's
    experiments, fix certain parameters to values, and return the results sorted
    in descending order of some metric.
    """
    f = 'train.csv' if train else 'test.csv'
    path = os.path.join('..', 'results', data, model, f)

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
        df = df.groupby(by=groupby_columns)[average_columns].agg({'mean': np.mean, 'std': np.std})
        df = df.sort_values(by=('mean', metric), ascending=False)

        if df.size != 0:
            df.index = df.index.ravel()
            temp = np.copy(df.index)
            for i in range(len(temp)):
                temp[i] = '_'.join(df.index[i])

            df.index = temp

            if top is None:
                averaged[key] = df
            else:
                averaged[key] = df.head(top)

    if len(vary) == 0:
        return averaged[key]

    return averaged


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diehl_and_cook_2015')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='train', action='store_false')
    parser.add_argument('--fix', default=[], nargs='*')
    parser.add_argument('--vary', default=[], nargs='*')
    parser.add_argument('--metric', type=str, default='mean_all_activity')
    parser.add_argument('--top', type=int, default=None)
    parser.set_defaults(train=True)
    args = parser.parse_args()

    model = args.model
    data = args.data
    train = args.train
    vary = args.vary
    metric = args.metric
    top = args.top

    assert(len(args.fix) % 2 == 0)

    fix = {}
    for i in range(int(len(args.fix) / 2)):
        fix[args.fix[2 * i]] = args.fix[2 * i + 1]

    df = main(model, data, train, fix, vary, metric, top)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
