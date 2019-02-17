#! python3
# coding: utf-8

import numpy as np
import krippendorff
import pandas as pd
import sys

# It is necessary to pip-install the krippendorff module
# (https://github.com/pln-fing-udelar/fast-krippendorff)

dataset = pd.read_csv(sys.argv[1])  # should contain assessor scores

scores = np.zeros((3, dataset.shape[0]))

for nr, annotator in enumerate(['ASSESSOR1', 'ASSESSOR2', 'ASSESSOR3']):
    scores[nr, :] = dataset[annotator].values

print('Raters:', scores.shape[0])
print('Instances:', scores.shape[1])


agreement = krippendorff.alpha(scores)

print('Krippendorff Alpha:', agreement)
