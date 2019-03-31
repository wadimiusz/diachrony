import numpy as np
from matplotlib import pyplot
import pandas as pd

evaluative = pd.read_csv("alignedvectors_result_soviet.csv")
rest = pd.read_csv("result_soviet_to_compare.csv")

x = [i for i in evaluative['norm_mean_diff_vec'].tolist() if i!=0.0]
y = [i for i in rest['norm_mean_diff_vec'].tolist() if i != 0.0]

bins = np.linspace(0, 1, 50)

pyplot.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
pyplot.hist(y, bins, alpha=0.5, label='rest', ec='black')
pyplot.legend(loc='upper right')
#pyplot.show()
pyplot.savefig('speedchange.png')