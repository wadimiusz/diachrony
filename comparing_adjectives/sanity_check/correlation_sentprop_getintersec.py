from scipy.stats import pearsonr
import pandas as pd
import sys

ours1 = pd.read_csv('../adjectives/eng_regular_filtered_0.csv')
ours2 = pd.read_csv('../adjectives/rest/eng/regular.csv')

ours3 = pd.read_csv('../adjectives/eng_incremental_filtered_0.csv')
ours4 = pd.read_csv('../adjectives/rest/eng/incremental.csv')

ours_regular = pd.concat([ours1, ours2])
#print(ours_regular.head())
ours_incremental = pd.concat([ours3, ours4])
#print(ours_incremental.head())

sentprop_60 = pd.read_csv('../../datasets/sentiment_lexicons/historical/1960.tsv', sep='\t')
sentprop_70 = pd.read_csv('../../datasets/sentiment_lexicons/historical/1970.tsv', sep='\t')
sentprop_80 = pd.read_csv('../../datasets/sentiment_lexicons/historical/1980.tsv', sep='\t')
sentprop_90 = pd.read_csv('../../datasets/sentiment_lexicons/historical/1990.tsv', sep='\t')
sentprop_00 = pd.read_csv('../../datasets/sentiment_lexicons/historical/2000.tsv', sep='\t')

ours_vocab_reg = ours_regular['WORD'].tolist()
print(len(ours_vocab_reg))
set_regular = set(ours_vocab_reg)
print(len(set_regular))
print()
ours_vocab_incr = ours_incremental['WORD'].tolist()
print(len(ours_vocab_incr))
set_incremental = set(ours_vocab_incr)
print(len(set_incremental))
print()

vocabs = []
for decade in [sentprop_60, sentprop_70, sentprop_80, sentprop_90, sentprop_00]:
    vocab = (decade[decade.iloc[:, 2] > 0.5].iloc[:, 0]).tolist()
    vocabs.append(vocab)

sentprop_all = set.intersection(*map(set, vocabs))

sentprop_vocab = [word+'_ADJ' for word in sentprop_all]
print(len(sentprop_vocab))
print()

intersec_regular = set_regular.intersection(set(sentprop_vocab))
print(len(intersec_regular))

intersec_incremental = set_incremental.intersection(set(sentprop_vocab))
print(len(intersec_incremental))

results_regular = pd.DataFrame()
results_regular['WORD'] = list(intersec_regular)

results_incremental = pd.DataFrame()
results_incremental['WORD'] = list(intersec_incremental)


def get_diff(decade1, decade2, wordlist):
    dic = {}
    for word in wordlist:
        dic[word] = decade2[decade2.iloc[:, 0] == word.split('_')[0]].iloc[:, 1].item() -\
                    decade1[decade1.iloc[:, 0] == word.split('_')[0]].iloc[:, 1].item()
        #print(dic[word])

    return dic


results_regular['60-70'] = results_regular['WORD'].map(get_diff(sentprop_60, sentprop_70, intersec_regular))
results_regular['70-80'] = results_regular['WORD'].map(get_diff(sentprop_70, sentprop_80, intersec_regular))
results_regular['80-90'] = results_regular['WORD'].map(get_diff(sentprop_80, sentprop_90, intersec_regular))
results_regular['90-00'] = results_regular['WORD'].map(get_diff(sentprop_90, sentprop_00, intersec_regular))

results_incremental['60-70'] = results_incremental['WORD'].map(get_diff(sentprop_60, sentprop_70, intersec_incremental))
results_incremental['70-80'] = results_incremental['WORD'].map(get_diff(sentprop_70, sentprop_80, intersec_incremental))
results_incremental['80-90'] = results_incremental['WORD'].map(get_diff(sentprop_80, sentprop_90, intersec_incremental))
results_incremental['90-00'] = results_incremental['WORD'].map(get_diff(sentprop_90, sentprop_00, intersec_incremental))

print()
print(results_regular.head())
print(results_incremental.head())

results_regular.to_csv('sentprop_regular.csv')
results_incremental.to_csv('sentprop_incremental.csv')