from models import ProcrustesAligner
from utils import load_model, intersection_align_gensim
from gensim import matutils
import pandas as pd
import numpy as np

results = pd.DataFrame()

model1 = load_model("wordvectors/macro/pre-soviet_incremental.model")
model2 = load_model("wordvectors/macro/soviet_incremental.model")
model3 = load_model("wordvectors/macro/post-soviet_incremental.model")
vocab1 = model1.wv.vocab
vocab2 = model2.wv.vocab
vocab3 = model3.wv.vocab

words = []
#adjs = open('comparing_adjectives/test_experiments/eval_adj_rus.txt', 'r', encoding='utf8')
adjs = open('comparing_adjectives/test_experiments/rest_adj_largescale.txt', 'r', encoding='utf8')

for line in adjs.read().splitlines():
    #word = line + '_ADJ'
    word = line
    if word in list((set(vocab1) & set(vocab2) & set(vocab3))):
        words.append(word)
results['word'] = words

model1_aligned, model2_aligned = intersection_align_gensim(m1=model1, m2=model2)
model1_aligned, model3_aligned = intersection_align_gensim(m1=model1, m2=model3)

mean_score = []
for word in words:
    scores = []

    score1 = ProcrustesAligner(w2v1=model1_aligned, w2v2=model2_aligned).get_score(word)
    scores.append(1 - score1)
    score2 = ProcrustesAligner(w2v1=model2_aligned, w2v2=model3_aligned).get_score(word)
    scores.append(1 - score2)

    mean_score.append(np.mean(scores, axis=0))

def mean_freq(word, vocab1, vocab2, vocab3):

    try:
        fr1 = vocab1[word].count / 44626840
    except KeyError:
        fr1 = 0
    try:
        fr2 = vocab2[word].count / 59579878
    except KeyError:
        fr2 = 0
    try:
        fr3 = vocab3[word].count / 54299964
    except KeyError:
        fr3 = 0

    #print(fr1, fr2, fr3)

    return (fr1 + fr2 + fr3) / 3

mean_freqs = []
for word in words:
    freq = mean_freq(word, vocab1, vocab2, vocab3)
    mean_freqs.append(freq)

results['mean_dist'] = mean_score
results['mean_freq'] = mean_freqs
#results.to_csv('cos_dist_eval_incremental.csv', encoding='utf8')
results.to_csv('cos_dist_rest_incremental.csv', encoding='utf8')

#sorted = results.sort_values('mean_dist', ascending=False)
#sorted.to_csv('cos_dist_eval_sorted.csv', encoding='utf8')
#sorted.to_csv('cos_dist_rest_sorted.csv', encoding='utf8')