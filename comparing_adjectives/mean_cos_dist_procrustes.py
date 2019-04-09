from models import smart_procrustes_align_gensim
from utils import load_model, intersection_align_gensim
from gensim import matutils
import pandas as pd
import numpy as np

results = pd.DataFrame()

model1 = load_model("wordvectors/soviet/pre-soviet.model")
model2 = load_model("wordvectors/soviet/soviet.model")
model3 = load_model("wordvectors/soviet/post-soviet.model")
vocab1 = model1.wv.vocab
vocab2 = model2.wv.vocab
vocab3 = model3.wv.vocab

words = []
#adjs = open('comparing_adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
adjs = open('comparing_adjectives/rest_adj_largescale.txt', 'r', encoding='utf8')

for line in adjs.read().splitlines():
    #word = line + '_ADJ'
    word = line
    if word in list((set(vocab1) & set(vocab2) & set(vocab3))):
        words.append(word)
results['word'] = words

model1_aligned, model2_aligned = intersection_align_gensim(m1=model1, m2=model2)
model2_procrustes = smart_procrustes_align_gensim(model1_aligned, model2_aligned)
model1_aligned, model3_aligned = intersection_align_gensim(m1=model1, m2=model3)
model3_procrustes = smart_procrustes_align_gensim(model1_aligned, model3_aligned)

mean_dist_vectors = []
for word in words:
    dists = []

    vec1 = model1_aligned[word]
    vec2 = model2_procrustes[word]
    vec3 = model3_procrustes[word]

    dists.append(1 - np.dot(matutils.unitvec(vec1), matutils.unitvec(vec2)))
    dists.append(1 - np.dot(matutils.unitvec(vec2), matutils.unitvec(vec3)))

    mean_dist_vectors.append(np.mean(dists, axis=0))

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

results['mean_dist'] = mean_dist_vectors
results['mean_freq'] = mean_freqs
#results.to_csv('cos_dist_eval.csv', encoding='utf8')
results.to_csv('cos_dist_rest.csv', encoding='utf8')

sorted = results.sort_values('mean_dist', ascending=False)
#sorted.to_csv('cos_dist_eval_sorted.csv', encoding='utf8')
sorted.to_csv('cos_dist_rest_sorted.csv', encoding='utf8')