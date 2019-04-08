from models import smart_procrustes_align_gensim
from utils import load_model, intersection_align_gensim
from gensim import matutils
import pandas as pd
import numpy as np

'''
Модели выравниваются по первой, сначала оставляем пересечение слов,
потом применяем прокрустово выравнивание, считаем разницу векторных представлений слова
для каждой пары, усредняем эти вектора и находим евклидову норму.
'''

results = pd.DataFrame()

words = []
adjs = open('comparing_adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
#adjs = open('comparing_adjectives/soviet_adjs.txt', 'r', encoding='utf8')
for line in adjs.read().splitlines():
    words.append(line + '_ADJ')
    #words.append(line)
results['word'] = words

model1 = load_model("wordvectors/soviet/pre-soviet.model")
model2 = load_model("wordvectors/soviet/soviet.model")
model3 = load_model("wordvectors/soviet/post-soviet.model")

model1_aligned, model2_aligned = intersection_align_gensim(m1=model1, m2=model2)
model2_procrustes = smart_procrustes_align_gensim(model1_aligned, model2_aligned)
model1_aligned, model3_aligned = intersection_align_gensim(m1=model1, m2=model3)
model3_procrustes = smart_procrustes_align_gensim(model1_aligned, model3_aligned)

mean_dist_vectors = []
for word in words:
    dists = []
    try:
        vec1 = model1_aligned[word]
        vec2 = model2_procrustes[word]
        dists.append(1 - np.dot(matutils.unitvec(vec1), matutils.unitvec(vec2)))
    except KeyError:
        dists.append(None)
    try:
        vec2 = model2_procrustes[word]
        vec3 = model3_procrustes[word]
        dists.append(1 - np.dot(matutils.unitvec(vec2), matutils.unitvec(vec3)))
    except KeyError:
        dists.append(None)
    try:
        mean_dist_vectors.append(np.nanmean(dists, axis=0))
    except TypeError:
        mean_dist_vectors.append(np.nan)

results['norm_mean_diff_vec'] = mean_dist_vectors
results = results.dropna()
results = results.reset_index(drop=True)
results.to_csv('cos_dist_result.csv', encoding='utf8')
#results.to_csv('cos_dist_to_compare.csv', encoding='utf8')