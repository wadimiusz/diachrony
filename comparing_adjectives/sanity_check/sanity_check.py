from scipy.stats import pearsonr
import pandas as pd
import sys

sentprop_regular = pd.read_csv('sentprop_regular.csv')
sentprop_incremental = pd.read_csv('sentprop_incremental.csv')

jaccard_regular = pd.read_csv('jaccard_regular.csv')
jaccard_incremental = pd.read_csv('jaccard_incremental.csv')

anchors_regular = pd.read_csv('globalanchors_regular.csv')
anchors_incremental = pd.read_csv('globalanchors_incremental.csv')

cosdist_regular = pd.read_csv('procrustes_regular.csv')
cosdist_incremental = pd.read_csv('cosine_incremental.csv')


def mean_correlation(df1, df2):
    mean_corr = (pearsonr(df1['60-70'], df2['60-70'])[0] + pearsonr(df1['70-80'], df2['70-80'])[0] +
                 pearsonr(df1['80-90'], df2['80-90'])[0] + pearsonr(df1['90-00'], df2['90-00'])[0]) / 4
    mean_p = (pearsonr(df1['60-70'], df2['60-70'])[1] + pearsonr(df1['70-80'], df2['70-80'])[1] +
                 pearsonr(df1['80-90'], df2['80-90'])[1] + pearsonr(df1['90-00'], df2['90-00'])[1]) / 4

    return mean_corr, mean_p


print('REGULAR: ')
print('Jaccard: ', mean_correlation(sentprop_regular, jaccard_regular))
print('Procrustes: ', mean_correlation(sentprop_regular, cosdist_regular))
print('Global anchors: ', mean_correlation(sentprop_regular, anchors_regular))

print('INCREMENTAL: ')
print('Jaccard: ', mean_correlation(sentprop_incremental, jaccard_incremental))
print('Procrustes: ', mean_correlation(sentprop_incremental, cosdist_incremental))
print('Global anchors: ', mean_correlation(sentprop_incremental, anchors_incremental))