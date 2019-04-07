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
    except KeyError:
        vec1 = 1
    print(vec1)
    try:
        vec2 = model2_procrustes[word]
    except KeyError:
        vec2 = 1
    print(vec2)
    try:
        vec3 = model3_procrustes[word]
    except KeyError:
        vec3 = 1
    print(vec3)
    dists.append(1 - (np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    dists.append(1 - (np.sum(vec2 * vec3) / (np.linalg.norm(vec2) * np.linalg.norm(vec3))))
    mean_dist_vectors.append(np.mean(dists, axis=0))

results['norm_mean_diff_vec'] = mean_dist_vectors
results.to_csv('cos_dist_result.csv', encoding='utf8')
#results.to_csv('cos_dist_to_compare.csv', encoding='utf8')