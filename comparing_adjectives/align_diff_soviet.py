from models import smart_procrustes_align_gensim
from utils import load_model, intersection_align_gensim
import pandas as pd
import numpy as np

'''
Модели выравниваются по первой, сначала оставляем пересечение слов,
потом применяем прокрустово выравнивание, считаем разницу векторных представлений слова
для каждой пары, усредняем эти вектора и находим евклидову норму.
'''

results = pd.DataFrame()

words = []
#adjs = open('adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
#adjs = open('adjectives/soviet_adjs.txt', 'r', encoding='utf8')
adjs = open('comparing_adjectives/soviet_adjs.txt', 'r', encoding='utf8')
for line in adjs.read().splitlines():
    #words.append(line + '_ADJ')
    words.append(line)
results['word'] = words

model1 = load_model("wordvectors/soviet/pre-soviet.model")
model2 = load_model("wordvectors/soviet/soviet.model")
model3 = load_model("wordvectors/soviet/post-soviet.model")

model1_aligned, model2_aligned = intersection_align_gensim(m1=model1, m2=model2)
model2_procrustes = smart_procrustes_align_gensim(model1_aligned, model2_aligned)
model1_aligned, model3_aligned = intersection_align_gensim(m1=model1, m2=model3)
model3_procrustes = smart_procrustes_align_gensim(model1_aligned, model3_aligned)

mean_diff_vectors = []
for word in words:
    diffs = []
    try:
        diffs.append(model2_procrustes[word] - model1_aligned[word])
    except KeyError:
        diffs.append(None)
    try:
        diffs.append(model3_procrustes[word] - model2_procrustes[word])
    except KeyError:
        diffs.append(None)
    try:
        mean_diff_vectors.append(np.nanmean(diffs, axis=0))
    except TypeError:
        mean_diff_vectors.append(0)

results['norm_mean_diff_vec'] = [np.linalg.norm(vec) for vec in mean_diff_vectors]
#results.to_csv('alignedvectors_result_soviet.csv', encoding='utf8')
results.to_csv('result_soviet_to_compare.csv', encoding='utf8')
