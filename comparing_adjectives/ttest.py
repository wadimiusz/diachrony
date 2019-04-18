# python3
# coding: utf-8

import sys
import numpy as np
from scipy.stats import ttest_ind as test

file0 = sys.argv[1]
file1 = sys.argv[2]

distances0 = []
distances1 = []

distances0_2 = []
distances1_2 = []

deviations0 = []
deviations1 = []

sums0 = []
sums1 = []

sums0_2 = []
sums1_2 = []


for line in open(file0, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[3])
    if distance:
        distances0.append(distance)
    distance0_2 = float(res[4])
    if distance0_2:
        distances0_2.append(distance0_2)
    deviation0 = float(res[5])
    if deviation0:
        deviations0.append(deviation0)
    sum0 = float(res[6])
    if sum0:
        sums0.append(sum0)
    sum0_2 = float(res[7])
    if sum0_2:
        sums0_2.append(sum0_2)

for line in open(file1, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[3])
    if distance:
        distances1.append(distance)
    distance1_2 = float(res[4])
    if distance1_2:
        distances1_2.append(distance1_2)
    deviation1 = float(res[5])
    if deviation1:
        deviations1.append(deviation1)
    sum1 = float(res[6])
    if sum1:
        sums1.append(sum1)
    sum1_2 = float(res[7])
    if sum1_2:
        sums1_2.append(sum1_2)

print('Comparing', file0, file1)

comparisons = \
    [(distances0, distances1), (distances0_2, distances1_2),
     (deviations0, deviations1), (sums0, sums1), (sums0_2, sums1_2)]
comparison_names = \
    ['Procrustes distances', 'Global Anchors distances', 'Standard deviations',
     'Sum of Procrustes deltas', 'Sum of Global Anchors deltas']

for pair, name in zip(comparisons, comparison_names):
    print('=====')
    print(name)
    print('Averages: %.3f \t %.3f' % (np.average(pair[0]), np.average(pair[1])))
    print('T-test (difference and p-value): %.5f, %.5f' % test(pair[0], pair[1]))




