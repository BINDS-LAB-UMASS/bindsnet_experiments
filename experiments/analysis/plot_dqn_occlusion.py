import os
import pandas as pd
import matplotlib.pyplot as plt

from experiments import ROOT_DIR


results_path = os.path.join(ROOT_DIR, 'results', 'breakout')
snn_df = pd.read_csv(os.path.join(results_path, 'occlusion_dqn_eps_greedy', 'results.csv'))
ann_df = pd.read_csv(os.path.join(results_path, 'occlusion_test_ann_eps_greedy', 'results.csv'))

for percentile in [98.5]:
    sub_df = snn_df[snn_df.percentile == percentile]
    sub_df = sub_df.sort_values(by='occlusion')

    plt.plot(
        sub_df['occlusion'], sub_df['avg. reward'], label=f'SNN (p={percentile})'
    )
    plt.fill_between(
        sub_df['occlusion'],
        sub_df['avg. reward'] - sub_df['std. rewards'],
        sub_df['avg. reward'] + sub_df['std. rewards'],
        alpha=0.15
    )

plt.plot(
    ann_df['occlusion'], ann_df['avg. reward'], label=f'ANN'
)
plt.fill_between(
    ann_df['occlusion'],
    ann_df['avg. reward'] - ann_df['std. reward'],
    ann_df['avg. reward'] + ann_df['std. reward'],
    alpha=0.15
)

plt.xlabel('Occlusion location')
plt.ylabel('Average reward (100 episodes)')
plt.legend()
plt.savefig(os.path.join(ROOT_DIR, 'figures', 'ann_snn_dqn_occlusion_comp.png'))
plt.show()
