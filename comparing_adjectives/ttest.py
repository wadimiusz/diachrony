# python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind as test

file0 = sys.argv[1]
file1 = sys.argv[2]

distances0 = []
distances1 = []

distances0_2 = []
distances1_2 = []

sums0 = []
sums1 = []

sums0_2 = []
sums1_2 = []


for idx, values in pd.read_csv(file0).iterrows():
    distance = values["mean_dist_procrustes"]
    if distance:
        distances0.append(distance)
    distance0_2 = values["mean_dist_globalanchors"]
    if distance0_2:
        distances0_2.append(distance0_2)
    sum0 = values["sum_deltas_procrustes"]
    if sum0:
        sums0.append(sum0)
    sum0_2 = values["sum_deltas_globalanchors"]
    if sum0_2:
        sums0_2.append(sum0_2)

for idx, values in pd.read_csv(file1).iterrows():
    distance = values["mean_dist_procrustes"]
    if distance:
        distances1.append(distance)
    distance1_2 = values["mean_dist_globalanchors"]
    if distance1_2:
        distances1_2.append(distance1_2)
    sum1 = values["sum_deltas_procrustes"]
    if sum1:
        sums1.append(sum1)
    sum1_2 = values["sum_deltas_globalanchors"]
    if sum1_2:
        sums1_2.append(sum1_2)

print('Comparing', file0, file1)

comparisons = \
    [(distances0, distances1), (distances0_2, distances1_2), (sums0, sums1), (sums0_2, sums1_2)]
comparison_names = \
    ['Procrustes distances', 'Global Anchors distances',
     'Sum of Procrustes deltas', 'Sum of Global Anchors deltas']

for pair, name in zip(comparisons, comparison_names):
    print('=====')
    print(name)
    print('Averages: %.3f \t %.3f' % (np.average(pair[0]), np.average(pair[1])))
    print('T-test (difference and p-value): %.5f, %.5f' % test(pair[0], pair[1]))




