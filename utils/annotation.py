import csv
import os
import pandas as pd
import sys

corpus = pd.read_csv(sys.argv[1], sep = ',', index_col = ['ID'])
samples = pd.read_csv(sys.argv[2], sep = ',', index_col = ['ID'])


for index in corpus.index:
    if samples.loc[index, 'ASSESSOR_LABEL'] == -1:
        print('===============')
        print("{0} / 280, {1:0.2%}".format(index, index / 280))
        print(corpus.loc[index, 'WORD'])
        print('Старое:')
        for context in eval(corpus.loc[index, 'OLD_CONTEXTS']):
            print(*context, sep=' ')
            print('')
        print('')
        print('Новое:')
        for context in eval(corpus.loc[index, 'NEW_CONTEXTS']):
            print(*context, sep=' ')
            print('')

        answer = input("Оцените, насколько изменилось значение/употребление слова от 0 (совсем не изменилось) до 2 (полностью изменилось): ")
        if answer not in ["0","1","2","стоп"]:
            answer = input("Попробуйте ещё раз: ")
        if answer == 'стоп':
            break
    else:
        continue

    result = samples.set_value(index, 'ASSESSOR_LABEL', answer)
    samples.update(result)

samples.to_csv(sys.argv[2])
