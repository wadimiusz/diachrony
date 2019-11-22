import pandas as pd
import numpy as np
from utils import log, format_time, intersection_align_gensim
import random
import sys
import time
import os
import gensim
import logging
from scipy import spatial
from tqdm import tqdm
from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder


def get_corpuses():
    corpuses = {}
    for year in range(2015, 2020):
        df = pd.read_csv('corpora/tables/{year}_contexts.csv'.format(year=year), index_col='ID')
        corpuses.update({year: df})

    return corpuses


def main():
    elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz")
    corpora = get_corpuses()

    old_contexts = list()
    new_contexts = list()

    base_years = list()
    new_years = list()

    word = input()

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
                    lemmatized = [word.split('_')[0] for word in lemmas.split()]
                    samples.append([lemmatized, raw])
            all_samples.update({year: samples})

        except ValueError:
            raise ValueError("Problem with", word, year, "because not enough samples found")

    pairs = [[years[y1], years[y2]] for y1 in range(len(years)) for y2 in range(y1 + 1, len(years))]

    for pair in tqdm(pairs):
        old_samples = all_samples.get(pair[0])
        new_samples = all_samples.get(pair[1])

        old_samples_vec = []
        new_samples_vec = []

        for sample in old_samples:
            sample_vec = elmo([sample[0]])
            sample_dict = {'vec': sample_vec, 'sent': sample[1]}
            old_samples_vec.append(sample_dict)

        for sample in new_samples:
            sample_vec = elmo([sample[0]])
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
    output_df.to_csv('contexts_by_year_elmo.csv')


if __name__ == "__main__":
    main()

