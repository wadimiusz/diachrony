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


class GetExamples:
    def __init__(self, word):
        self.word = word

    def get_corpuses(self):
        corpuses = {}
        for year in range(2015, 2020):
            df = pd.read_csv('corpora/tables/{year}_contexts.csv'.format(year=year), index_col='ID')
            corpuses.update({year: df})

        return corpuses

    def intersec_models(self, modeldict, intersec_vocab):
        for year, model in modeldict.items():
            if year != 2015:
                _, _ = intersection_align_gensim(m1=modeldict.get(2015), m2=model, words=intersec_vocab)

        return modeldict

    def align_models(self, modeldict):
        for year, model in modeldict.items():
            if year != 2015:
                _ = smart_procrustes_align_gensim(modeldict.get(2015), model)

        return modeldict

    def avg_feature_vector(self, sentence, model, num_features):
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in sentence:
            if word in model.vocab:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)

        return feature_vec

    def create_examples(self):
        models = {}
        for year in tqdm(range(2015, 2020)):
            model = gensim.models.KeyedVectors.load_word2vec_format(
                'models/{year}_0_5.bin'.format(year=year), binary=True, unicode_errors='replace')
            model.init_sims(replace=True)

            models.update({year: model})

        vocabs = [model.vocab for model in list(models.values())]
        intersected_vocab = set.intersection(*map(set, vocabs))
        intersected_models = GetExamples.intersec_models(self, models, intersected_vocab)
        aligned_models = GetExamples.align_models(self, intersected_models)

        corpora = GetExamples.get_corpuses(self)

        old_contexts = list()
        new_contexts = list()

        base_years = list()
        new_years = list()

        word = self.word

        start = time.time()

        all_samples = {}

        years = []
        for year in tqdm(range(2015, 2020)):
            years.append(year)
            corpus = corpora.get(year)
            samples = []

            try:
                for idx, lemmas, raw in corpus[['LEMMAS', 'RAW']].itertuples():
                    if word in lemmas:
                        samples.append([lemmas.split(), raw])
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
                sample_vec = GetExamples.avg_feature_vector(self, sample[0], model=model1, num_features=300)
                sample_dict = {'vec': sample_vec, 'sent': sample[1]}
                old_samples_vec.append(sample_dict)

            for sample in new_samples:
                sample_vec = GetExamples.avg_feature_vector(self, sample[0], model=model2, num_features=300)
                sample_dict = {'vec': sample_vec, 'sent': sample[1]}
                new_samples_vec.append(sample_dict)

            similarities = {}
            for dict1 in old_samples_vec:
                for dict2 in new_samples_vec:
                    vec1 = dict1.get('vec')
                    vec2 = dict2.get('vec')
                    sim = spatial.distance.cosine(vec1, vec2)
                    similarities.update({sim: [dict1.get('sent'), dict2.get('sent')]})

            five_old_samples = []
            five_new_samples = []

            for k, v in sorted(similarities.items(), reverse=True):
                if (v[0] not in five_old_samples) and (v[1] not in five_new_samples):
                    print(k, v)
                    five_old_samples.append(v[0])
                    five_new_samples.append(v[1])
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
        output_df.index.names = ["ID"]
        output_df.to_csv('contexts_by_year.csv')

