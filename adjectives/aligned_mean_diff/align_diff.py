from models import smart_procrustes_align_gensim
from utils import load_model, intersection_align_gensim
import pandas as pd
import numpy as np

'''
Модели выравниваются по первой, сначала оставляем пересечение слов,
потом применяем прокрустово выравнивание, считаем разницу векторных представлений слова
для каждой пары лет, усредняем эти вектора и находим евклидову норму.
'''

results = pd.DataFrame()

words = []
adjs = open('adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
for line in adjs.read().splitlines():
    words.append(line + '_ADJ')
results['word'] = words

model2000 = load_model('wordvectors/2000.model')


def alignment(year):
    model2 = load_model('wordvectors/{year}.model'.format(year=year))

    model1_aligned, model2_aligned = intersection_align_gensim(m1=model2000, m2=model2)
    model2_procrustes = smart_procrustes_align_gensim(model1_aligned, model2_aligned)

    return model2_procrustes

# I love shitcoding
models = [alignment(2000), alignment(2001), alignment(2002), alignment(2003), alignment(2004), alignment(2005),
          alignment(2006), alignment(2007), alignment(2008), alignment(2009), alignment(2010), alignment(2011),
          alignment(2012), alignment(2013), alignment(2014)]

mean_diff_vectors = []
for word in words:
    diffs = []
    for i in range(1, len(models)):
        try:
            diffs.append(models[i][word] - models[i-1][word])
        except KeyError:
            diffs.append(None)

    try:
        mean_diff_vectors.append(np.nanmean(diffs, axis=0))
    except TypeError:
        mean_diff_vectors.append(0)

results['norm_mean_diff_vec'] = [np.linalg.norm(vec) for vec in mean_diff_vectors]
results.to_csv('alignedvectors_result.csv', encoding='utf8')
