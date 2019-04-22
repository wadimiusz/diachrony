import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import pandas as pd
import sys

evaluative = pd.read_csv(sys.argv[1])
rest = pd.read_csv(sys.argv[2])


def plot_procrustes(evaluative, rest):
    x = [i for i in evaluative[sys.argv[3]].tolist()]
    y = [i for i in rest[sys.argv[3]].tolist()]

    bins = np.linspace(0, 1, 50)

    pyplot.xlabel(sys.argv[3])
    pyplot.ylabel('number_of_adjectives')
    pyplot.title('Procrustes Aligner')
    pyplot.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
    pyplot.hist(y, bins, alpha=0.5, label='rest', ec='black')
    pyplot.legend(loc='upper right')
    #pyplot.show()
    pyplot.savefig(sys.argv[4])


def plot_globalanchors(evaluative, rest):
    x = [i for i in evaluative[sys.argv[3]].tolist()]
    y = [i for i in rest[sys.argv[3]].tolist()]

    bins = np.linspace(0, 1, 50)

    pyplot.xlabel(sys.argv[3])
    pyplot.ylabel('number_of_adjectives')
    pyplot.title('Global Anchors')
    pyplot.hist(x, bins, alpha=0.5, label='evaluative', ec='black')
    pyplot.hist(y, bins, alpha=0.5, label='rest', ec='black')
    pyplot.legend(loc='upper right')
    #pyplot.show()
    pyplot.savefig(sys.argv[4])


if sys.argv[3].endswith('procrustes'):
    plot_procrustes(evaluative, rest)
else:
    plot_globalanchors(evaluative, rest)
