#! python3
# coding: utf-8

import sys
import pandas as pd

word = sys.argv[1]
year = int(sys.argv[2])

dataset = pd.read_csv('corpus_ADJ.csv')

for ind, res in dataset.iterrows():
    if res.WORD == word and res.BASE_YEAR == year:
        print(res.WORD)
        print('\n'.join(res.OLD_CONTEXTS.replace("', '", " ").split('], [')))
        print('NEW==================NEW')
        print('\n'.join(res.NEW_CONTEXTS.replace("', '", " ").split('], [')))

