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


for line in open(file0, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[3])
    if distance:
        distances0.append(distance)
    distance0_2 = float(res[4])
    if distance0_2:
        distances0_2.append(distance0_2)

for line in open(file1, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[3])
    if distance:
        distances1.append(distance)
    distance1_2 = float(res[4])
    if distance1_2:
        distances1_2.append(distance1_2)

deviations0 = []
deviations1 = []

for line in open(file0, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[5])
    if distance:
        deviations0.append(distance)

for line in open(file1, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[5])
    if distance:
        deviations1.append(distance)

sums0 = []
sums1 = []

for line in open(file0, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[6])
    if distance:
        sums0.append(distance)

for line in open(file1, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[6])
    if distance:
        sums1.append(distance)

sums0_2 = []
sums1_2 = []

for line in open(file0, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[7])
    if distance:
        sums0_2.append(distance)

for line in open(file1, 'r').readlines()[1:]:
    res = line.strip().split(',')
    distance = float(res[7])
    if distance:
        sums1_2.append(distance)

print('Comparing', file0, file1)

print('=====')
print('Distances_procrustes')
print('Averages: %.3f \t %.3f' % (np.average(distances0), np.average(distances1)))
print('T-test: %.5f \t %.5f' % test(distances0, distances1))

print('=====')
print('Distances_globalanchors')
print('Averages: %.3f \t %.3f' % (np.average(distances0_2), np.average(distances1_2)))
print('T-test: %.5f \t %.5f' % test(distances0_2, distances1_2))

print('=====')
print('Standard deviations')
print('Averages: %.3f \t %.3f' % (np.average(deviations0), np.average(deviations1)))
print('T-test: %.5f \t %.5f' % test(deviations0, deviations1))

print('=====')
print('Sum of deltas procrustes')
print('Averages: {0:.3f} \t {0:.3f}'.format(np.average(sums0), np.average(sums1)))
print('T-test: %.5f \t %.5f' % test(sums0, sums1))

print('=====')
print('Sum of deltas globalanchors')
print('Averages: %.3f \t %.3f' % (np.average(sums0_2), np.average(sums1_2)))
print('T-test: %.5f \t %.5f' % test(sums0_2, sums1_2))




