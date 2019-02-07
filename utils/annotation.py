import csv
import os
import pandas as pd
import sys

corpus = pd.read_csv(sys.argv[1], sep = ',', index_col = ['ID'])
samples = pd.read_csv(sys.argv[2], sep = ',', index_col = ['ID'])


for index in corpus.index:
    print(corpus.loc[index, 'WORD'], '\n', corpus.loc[index, 'OLD_CONTEXTS'], '\n', corpus.loc[index, 'NEW_CONTEXTS'], '\n')

    answer = input(
      "Оцените, насколько изменилось значение/употребление слова от 0 (совсем не изменилось) до 2 (полностью изменилось): ")
    if answer == 'стоп':
        break

    result = samples.set_value(index, 'ASSESSOR_LABEL', answer)
    samples.update(result)

samples.to_csv('results.csv')
