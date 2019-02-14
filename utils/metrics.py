import os, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from nltk import agreement

words = pd.read_csv('dataset/annotated/samples_ADJ_daria.csv')['WORD']
daria = pd.read_csv('dataset/annotated/samples_ADJ_daria.csv')['ASSESSOR_LABEL']
julia = pd.read_csv('dataset/annotated/samples_ADJ_julia.csv')['ASSESSOR_LABEL']
vadim = pd.read_csv('dataset/annotated/samples_ADJ_vadim.csv')['ASSESSOR_LABEL']
correct = pd.read_csv('dataset/annotated/samples_ADJ_vadim.csv')['LABEL']
mean = pd.concat([julia, daria, vadim], axis=1).mean(axis=1)
median = pd.concat([julia, daria, vadim], axis=1).median(axis=1)

print("Daria labels stats:", daria.value_counts(), sep='\n', end='\n\n')
print("Julia labels stats:", julia.value_counts(), sep='\n', end='\n\n')
print("Vadim labels stats:", vadim.value_counts(), sep='\n', end='\n\n')
print("========================================")
print('Agreement between Vadim and Julia (3 labels):', cohen_kappa_score(vadim, julia))
print('Agreement between Vadim and Daria (3 labels):', cohen_kappa_score(vadim, daria))
print('Agreement between Julia and Daria (3 labels):', cohen_kappa_score(julia, daria), end='\n\n')

print('Agreement between Vadim and Julia (2 labels):', cohen_kappa_score(vadim > 0, julia > 0))

print('Agreement between Vadim and Daria (2 labels):', cohen_kappa_score(vadim > 0, daria > 0))

print('Agreement between Julia and Daria (2 labels):', cohen_kappa_score(julia > 0, daria > 0), end='\n\n')

print("The cohen cappa of random guessing is 0.")
print("========================================")
print("Mean squared error:")
print("Label / 2:")
print("Julia:", mean_squared_error(correct, julia / 2))
print("Daria:", mean_squared_error(correct, daria / 2))
print("Vadim:", mean_squared_error(correct, vadim / 2))
print("Mean:", mean_squared_error(correct, mean / 2))
print("Median:", mean_squared_error(correct, median / 2), end='\n\n')

print("Label > 0:")
print("Julia:", mean_squared_error(correct, julia > 0))
print("Daria:", mean_squared_error(correct, daria > 0))
print("Vadim:", mean_squared_error(correct, vadim > 0))
print("Mean:", mean_squared_error(correct, mean > 0))
print("Median:", mean_squared_error(correct, median > 0), end='\n\n')

print("Label > 1:")
print("Julia:", mean_squared_error(correct, julia > 1))
print("Daria:", mean_squared_error(correct, daria > 1))
print("Vadim:", mean_squared_error(correct, vadim > 1))
print("Mean:", mean_squared_error(correct, mean > 1))
print("Median:", mean_squared_error(correct, median > 1), end='\n\n')

print("The MSE of random guessing here is 0.5.")
print("========================================")
print("Frequent words")
counts = words[correct == 1].value_counts()
print(counts[counts > 1])