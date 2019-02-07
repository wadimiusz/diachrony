import csv
import os
import pandas as pd
import sys

with open(sys.argv[1], 'r') as infile:
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]

samples = pd.read_csv(sys.argv[2])

words = list(zip(data['ID'], data['WORD'], data['OLD_CONTEXTS'], data['NEW_CONTEXTS']))

for item in words:
    print(item[1], '\n', item[2], '\n', item[3], '\n')
    answer = input(
        "Оцените, насколько изменилось значение/употребление слова от 0 (совсем не изменилось) до 3 (полностью изменилось): ")
    if answer == 'стоп':
        break
    result = samples.set_value(int(item[0]), 'ASSESSOR_LABEL', answer)
    samples.update(result)

samples.to_csv('results.csv')
