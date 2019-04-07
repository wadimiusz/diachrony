# python3
# coding: utf-8

import sys
import numpy as np
from scipy.stats import ttest_ind as test

file0 = sys.argv[1]
file1 = sys.argv[2]

distances0 = []
distances1 = []

for line in open(file0, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[2])
    distances0.append(distance)

for line in open(file1, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[2])
    distances1.append(distance)

print('Comparing', file0, file1)

print('Distances')
print('Averages:', np.average(distances0), np.average(distances1))
print('T-test:', test(distances0, distances1))





