import os
import numpy as np


def print_results(results):
	print()

	for s in results:
		print(f'Results for scheme "{s}": {results[s][-1]:.2f} (last), ' \
		      f'{np.mean(results[s]):.2f} (average), {np.max(results[s]):.2f} (best)')

