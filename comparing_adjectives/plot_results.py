import numpy as np
from matplotlib import pyplot
import pandas as pd

#evaluative = pd.read_csv("cos_dist_eval.csv")
#rest = pd.read_csv("cos_dist_rest.csv")
evaluative = pd.read_csv("globalanchors_eval.csv")
rest = pd.read_csv("globalanchors_rest.csv")


x = [i for i in evaluative['mean_ga'].tolist()]  # if i != 0.0]
y = [i for i in rest['mean_ga'].tolist()]  # if i != 0.0]

bins = np.linspace(0, 1, 50)

pyplot.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
pyplot.hist(y, bins, alpha=0.5, label='rest', ec='black')
pyplot.legend(loc='upper right')
#pyplot.show()
#pyplot.savefig('speedchange_cosdist.png')
pyplot.savefig('speedchange_globalanchors.png')