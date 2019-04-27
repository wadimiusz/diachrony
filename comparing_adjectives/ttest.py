# python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind as test

file0 = sys.argv[1]
file1 = sys.argv[2]

dataset_eval = pd.read_csv(file0)
dataset_fillers = pd.read_csv(file1)

print('Comparing', file0, 'and', file1)

comparison_names = \
    ['Procrustes distances', 'Global Anchors distances',
     'Sum of Procrustes deltas', 'Sum of Global Anchors deltas']

eval_frequencies = dataset_eval['frequency'].values
filler_frequencies = dataset_fillers['frequency'].values
freq_welch = test(eval_frequencies, filler_frequencies, equal_var=False)
print('Average normalized frequencies: %.5f (evaluative), %.5f (fillers)' %
      (np.average(eval_frequencies), np.average(filler_frequencies)))
print('Difference: %.5f' % (np.average(eval_frequencies) - np.average(filler_frequencies)))
print('Welch T-test:', freq_welch)


for name in dataset_eval.columns.values[3:]:  # We don't need first columns
    print('=====')
    print(name)
    evals = dataset_eval[name].values
    fillers = dataset_fillers[name].values
    print('Averages: %.3f \t %.3f' % (np.average(evals), np.average(fillers)))
    print('T-test (difference and p-value): %.5f, %.5f' % test(evals, fillers, equal_var=False))



