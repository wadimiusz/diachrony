from scipy.stats import pearsonr
import pandas as pd
import sys

df1 = pd.read_csv(sys.argv[1])
df2 = pd.read_csv(sys.argv[2])

df = pd.concat([df1, df2])
#print(df.head(10))

print("Correlation frequency-mean_dist_jaccard: ", pearsonr(df['frequency'], df['mean_dist_jaccard']))
print("Correlation frequency-sum_deltas_jaccard: ", pearsonr(df['frequency'], df['sum_deltas_jaccard']))
print("Correlation frequency-mean_dist_globalanchors: ", pearsonr(df['frequency'], df['mean_dist_globalanchors']))
print("Correlation frequency-sum_deltas_globalanchors: ", pearsonr(df['frequency'], df['sum_deltas_globalanchors']))
print("Correlation frequency-mean_dist_procrustes: ", pearsonr(df['frequency'], df['mean_dist_procrustes']))
print("Correlation frequency-sum_deltas_procrustes: ", pearsonr(df['frequency'], df['sum_deltas_procrustes']))