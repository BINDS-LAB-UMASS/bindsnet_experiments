import os
import argparse
import pandas as pd

from utils import sort_best_params


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

    df = sort_best_params(model, data, train, fix, vary, metric, top)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
