import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error


def read_files(data):
    samples = pd.read_csv(data)
    binary_samples = pd.read_csv(data)

    for index in binary_samples.index:
        if binary_samples.loc[index, 'ASSESSOR_LABEL'] == 2:
            result = binary_samples.set_value(index, 'ASSESSOR_LABEL', 1)
            binary_samples.update(result)
        else:
            continue

    return samples, binary_samples


def cohen_kappa(data1, data2):

    k = cohen_kappa_score(data1['ASSESSOR_LABEL'], data2['ASSESSOR_LABEL'])

    return k


def mean_sqr_error(data):

    mse = mean_squared_error(data['LABEL'],
                             data['ASSESSOR_LABEL'])

    return mse


for i in range(1, 3):  # 3 поменять на 4
    results, results_binary = read_files(sys.argv[i])
    print('Mean square error annotator(binary)/algorithm: ', mean_sqr_error(results_binary))

# Оптимизируйте плиз как-то это говно
i = 1
results, results_binary = read_files(sys.argv[i])
results2, results_binary2 = read_files(sys.argv[i+1])
#results3, results_binary3 = read_files(sys.argv[i+2])
print('Cohen\'s kappa scores: ', cohen_kappa(results, results2))
#print('Cohen\'s kappa scores: ', cohen_kappa(results, results2), '\n', cohen_kappa(results, results3),
#      '\n', cohen_kappa(results2, results3))
#print('Mean kappa: ', np.mean())

