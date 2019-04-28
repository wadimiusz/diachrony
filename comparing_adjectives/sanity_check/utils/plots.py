import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset/annotated.csv')
means = df["ASSESSOR_MEAN"]
x = [0, 1/3, 2/3, 1, 4/3, 5/3, 2]
height = [sum(means == value) for value in x]
labels = "0, 1/3, 2/3, 1, 4/3, 5/3, 2".split(", ")
plt.bar(x=x, height=height, tick_label=labels, width=0.2)
plt.savefig('outputs/plots/annotator_mean_hist.png')
