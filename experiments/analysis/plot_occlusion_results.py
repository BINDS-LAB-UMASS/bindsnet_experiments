import os
import pandas as pd
import matplotlib.pyplot as plt

from experiments import ROOT_DIR

results_path = os.path.join(ROOT_DIR, 'results', 'breakout')
ann_path = os.path.join(results_path, 'occlusion_test_ann_eps_greedy')
snn_path = os.path.join(results_path, 'occlusion_dqn_eps_greedy')

ann_results = pd.read_csv(os.path.join(ann_path, 'results.csv'))
snn_results = pd.read_csv(os.path.join(snn_path, 'results.csv'))

stats = snn_results[snn_results.time == 100].groupby(['percentile', 'occlusion'])['avg. reward'].agg(['mean', 'std'])
for percentile in [98, 98.5, 99, 99.5]:
    stats.xs(percentile)['mean'].plot(yerr=stats.xs(percentile)['std'], label=f'SNN ({percentile} normalization)')

ann_results['avg. reward'].plot(yerr=ann_results['std. reward'], label='ANN')

plt.legend()
plt.title('Occlusion robustness test')
plt.ylabel('Reward')
plt.xlabel('Horizontal occlusion location')
plt.show()
