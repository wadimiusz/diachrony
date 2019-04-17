import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import pandas as pd
import sys

evaluative = pd.read_csv(sys.argv[1])
rest = pd.read_csv(sys.argv[2])


def plot_procrustes(evaluative, rest):
    x = [i for i in evaluative['mean_dist_procrustes'].tolist()]
    y = [i for i in rest['mean_dist_procrustes'].tolist()]

    bins = np.linspace(0, 1, 50)

    pyplot.xlabel('mean_cos_dist')
    pyplot.ylabel('number_of_adjectives')
    pyplot.title('Mean cosine distance of coherent time bins Procrustes Aligner')
    pyplot.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
    pyplot.hist(y, bins, alpha=0.5, label='rest', ec='black')
    pyplot.legend(loc='upper right')
    #pyplot.show()
    pyplot.savefig(sys.argv[3])


def plot_globalanchors(evaluative, rest):
    x = [i for i in evaluative['mean_dist_globalanchors'].tolist()]
    y = [i for i in rest['mean_dist_globalanchors'].tolist()]

    bins = np.linspace(0, 1, 50)

    pyplot.xlabel('mean_cos_dist')
    pyplot.ylabel('number_of_adjectives')
    pyplot.title('Mean cosine distance of coherent time bins Global Anchors')
    pyplot.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
    pyplot.hist(y, bins, alpha=0.5, label='rest', ec='black')
    pyplot.legend(loc='upper right')
    #pyplot.show()
    pyplot.savefig(sys.argv[3])


if sys.argv[3].endswith('procrustes.png'):
    plot_procrustes(evaluative, rest)
elif sys.argv[3].endswith('globalanchors.png'):
    plot_globalanchors(evaluative, rest)
