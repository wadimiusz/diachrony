import pandas as pd
import numpy as np
from utils import log, format_time, intersection_align_gensim
from algos import smart_procrustes_align_gensim
import functools
import random
import sys
import time
import os
import gensim
import logging
from scipy import spatial
import conllu
from tqdm import tqdm


def get_instances(word: str, year: (str, int, np.int)):
    def gen():
        with open('corpora/{year}_texts.txt'.format(year=year)) as f:
            for line in f:
                if word in line.strip().split():
                    yield line.strip().split()

    return list(gen())


'''
def get_instances_conllu(word: str, year: (str, int, np.int)):
    contexts = []
    corpus = open('corpora/conllu/{year}.conllu'.format(year=year))
    
    for tokenlist in conllu.parse_incr(corpus):
        lemma = word.split('_')[0]
        tag = word.split('_')[1]
        tokens = {}
        for token in tokenlist:
            tokens.update({token['form']: token['upostag']})
            if lemma in list(tokens.keys()) and tokens.get(word) == tag:
                context = tokenlist.metadata
                contexts.append(context)

    return contexts
'''


def intersec_models(modeldict, intersec_vocab):
    for year, model in modeldict.items():
        if year != 2015:
            _, _ = intersection_align_gensim(m1=modeldict.get(2015), m2=model, words=intersec_vocab)

    return modeldict


def align_models(modeldict):
    for year, model in modeldict.items():
        if year != 2015:
            _ = smart_procrustes_align_gensim(modeldict.get(2015), model)

    return modeldict


models = {}
for year in tqdm(range(2015, 2020)):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'models/{year}_0_5.bin'.format(year=year), binary=True, unicode_errors='replace')
    model.init_sims(replace=True)

    models.update({year: model})

vocabs = [model.vocab for model in list(models.values())]
intersected_vocab = set.intersection(*map(set, vocabs))
intersected_models = intersec_models(models, intersected_vocab)
aligned_models = align_models(intersected_models)


def avg_feature_vector(sentence, model, num_features):
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in sentence:
        if word in model.vocab:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)

    return feature_vec


def main():
    old_contexts = list()
    new_contexts = list()

    base_years = list()
    new_years = list()

    word = random.sample(intersected_vocab, 1)[0]

    '''
    old_samples = get_instances(word, 2015)
    new_samples = get_instances(word, 2016)

    if len(old_samples) > 5:
        five_old_samples = random.sample(old_samples, 5)
    else:
        five_old_samples = old_samples

    if len(new_samples) > 5:
        five_new_samples = random.sample(new_samples, 5)
    else:
        five_new_samples = new_samples

    old_contexts.append(five_old_samples)
    new_contexts.append(five_new_samples)
    '''

    start = time.time()

    all_samples = {}

    years = []
    for year in tqdm(range(2015, 2020)):
        years.append(year)

        try:
            samples = get_instances(word, year)
            all_samples.update({year: samples})

        except ValueError:
            raise ValueError("Problem with", word, year, "because not enough samples found")

    pairs = [[years[y1], years[y2]] for y1 in range(len(years)) for y2 in range(y1 + 1, len(years))]

    for pair in tqdm(pairs):
        model1 = aligned_models.get(pair[0])
        model2 = aligned_models.get(pair[1])

        old_samples = all_samples.get(pair[0])
        new_samples = all_samples.get(pair[1])

        old_samples_vec = []
        new_samples_vec = []

        for sample in old_samples:
            sample_vec = avg_feature_vector(sample, model=model1, num_features=300)

            sample_dict = {'vec': sample_vec, 'sent': sample}
            old_samples_vec.append(sample_dict)

        for sample in new_samples:
            sample_vec = avg_feature_vector(sample, model=model2, num_features=300)
            sample_dict = {'vec': sample_vec, 'sent': sample}
            new_samples_vec.append(sample_dict)

        similarities = {}
        for dict1 in old_samples_vec:
            for dict2 in new_samples_vec:
                vec1 = dict1.get('vec')
                vec2 = dict2.get('vec')
                sim = spatial.distance.cosine(vec1, vec2)
                similarities.update({sim: [dict1.get('sent'), dict2.get('sent')]})
        # print("Distances dict lengths: ", len(similarities))

        five_old_samples = []
        five_new_samples = []

        for k, v in sorted(similarities.items(), reverse=True):
            old = [word.split('_')[0] for word in v[0]]
            new = [word.split('_')[0] for word in v[1]]
            if (old not in five_old_samples) and (new not in five_new_samples):
                print(k, v)
                five_old_samples.append(old)
                five_new_samples.append(new)
                if len(five_new_samples) == 5:
                    break

        old_contexts.append(five_old_samples)
        new_contexts.append(five_new_samples)
        base_years.append(pair[0])
        new_years.append(pair[1])

    log("")
    log("This took", format_time(time.time() - start))
    output_df = pd.DataFrame({"WORD": word, "BASE_YEAR": base_years,
                              "OLD_CONTEXTS": old_contexts, "NEW_YEAR": new_years, "NEW_CONTEXTS": new_contexts})
    output_df.index.names = ['ID']
    output_df.to_csv('contexts_by_year.csv')


if __name__ == "__main__":
    main()

