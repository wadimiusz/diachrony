from matplotlib import pyplot
import pandas as pd
import numpy as np

evaluative_ga = pd.read_csv('globalanchors_eval_separated.csv')
rest_ga = pd.read_csv('globalanchors_rest_separated.csv')

evaluative_proc = pd.read_csv('procrustes_eval_separated.csv')
rest_proc = pd.read_csv('procrustes_rest_separated.csv')



x = [i for i in evaluative_ga['mean_dist1'].tolist()]
y = [i for i in rest_ga['mean_dist1'].tolist()]
a = [i for i in evaluative_ga['mean_dist2'].tolist()]
b = [i for i in rest_ga['mean_dist2'].tolist()]

x2 = [i for i in evaluative_proc['mean_dist1'].tolist()]
y2 = [i for i in rest_proc['mean_dist1'].tolist()]
a2 = [i for i in evaluative_proc['mean_dist2'].tolist()]
b2 = [i for i in rest_proc['mean_dist2'].tolist()]

bins = np.linspace(0, 1, 50)
'''
plt1 = pyplot
plt1.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
plt1.hist(y, bins, alpha=0.5, label='rest', ec='black')
plt1.legend(loc='upper right')
plt1.savefig('globalanchors_1_2.png')

plt2 = pyplot
plt2.hist(a, bins, alpha=0.5, label='evaluative', ec='black')
plt2.hist(b, bins, alpha=0.5, label='rest', ec='black')
plt2.legend(loc='upper right')
plt2.savefig('globalanchors_2_3.png')

plt3 = pyplot
plt3.hist(x2, bins, alpha=0.5, label='evaluative', ec='black')
plt3.hist(y2, bins, alpha=0.5, label='rest', ec='black')
plt3.legend(loc='upper right')
plt3.savefig('procrustes_1_2.png')
'''
plt4 = pyplot
plt4.hist(a2, bins, alpha=0.5, label='evaluative', ec='black')
plt4.hist(b2, bins, alpha=0.5, label='rest', ec='black')
plt4.legend(loc='upper right')
plt4.savefig('procrustes_2_3.png')