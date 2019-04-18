# python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from gensim import matutils
from get_adjs import get_models_by_decade
from models import ProcrustesAligner, GlobalAnchors
from utils import intersection_align_gensim


def align_models(modellist):
    # aligned_models = []
    for model in modellist[1:]:
        _, _ = intersection_align_gensim(m1=modellist[0], m2=model)
        # aligned_models.append(model)
    # aligned_models.insert(0, model1_aligned)

    return modellist


def get_mean_dist_procrustes(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = []
        for i in range(len(modellist) - 1):
            score = ProcrustesAligner(w2v1=modellist[i], w2v2=modellist[i + 1]).get_score(word)
            scores.append(1 - score)
        mean_scores[word] = np.mean(scores)

    return mean_scores


def get_mean_dist_globalanchors(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = []
        for i in range(len(modellist) - 1):
            score = GlobalAnchors(w2v1=modellist[i], w2v2=modellist[i + 1]).get_score(word)
            scores.append(1 - score)
        mean_scores[word] = np.mean(scores)

    return mean_scores


def get_move_from_initial_procrustes(wordlist, modellist):
    move_from_init = {}
    for word in wordlist:
        deltas = []
        current = 0
        for i in range(1, len(modellist)):
            delta = \
                ProcrustesAligner(w2v1=modellist[0], w2v2=modellist[i]).get_score(word) - current
            current = ProcrustesAligner(w2v1=modellist[0], w2v2=modellist[i]).get_score(word)
            if i == 1:
                deltas.append(1 - delta)
            else:
                deltas.append(- delta)
        move_from_init[word] = np.sum(deltas)

    return move_from_init


def get_move_from_initial_globalanchors(wordlist, modellist):
    move_from_init = {}
    for word in wordlist:
        deltas = []
        current = 0
        for i in range(1, len(modellist)):
            delta = GlobalAnchors(w2v1=modellist[0], w2v2=modellist[i]).get_score(word) - current
            current = GlobalAnchors(w2v1=modellist[0], w2v2=modellist[i]).get_score(word)
            if i == 1:
                deltas.append(1 - delta)
            else:
                deltas.append(- delta)
        move_from_init[word] = np.sum(deltas)

    return move_from_init


def get_deviation(wordlist, modellist):
    mean_vectors = {}
    all_vectors = {}
    for word in wordlist:
        vectors = []
        for i in range(len(modellist) - 1):
            vectors.append(modellist[i][word])
        mean_vectors[word] = np.mean(vectors, axis=0)
        all_vectors[word] = vectors

    deviations = {}
    for word in wordlist:
        dists = []
        for vector in all_vectors[word]:
            dist = np.dot(matutils.unitvec(mean_vectors[word]), matutils.unitvec(vector))
            dists.append(dist)
        deviations[word] = np.std(dists)

    return deviations


def get_freqdict(wordlist, vocablist, corpus_size):
    all_freqs = []
    word_freq = {}
    for word in wordlist:
        counts = [vocab[word].count for vocab in vocablist]
        frequency = [counts[i] / corpus_size[i] for i in range(len(vocablist))]
        mean_frequency = sum(frequency) / len(frequency)
        all_freqs.append(mean_frequency)
        word_freq[word] = mean_frequency

    return word_freq


if __name__ == '__main__':
    models_regular = []
    models_incremental = []
    for decade in range(1960, 2010, 10):
        model_regular = get_models_by_decade(decade, 'regular')
        model_incremental = get_models_by_decade(decade, 'incremental')

        models_regular.append(model_regular)
        models_incremental.append(model_incremental)

    vocabs_regular = [model.wv.vocab for model in models_regular]
    vocabs_incremental = [model.wv.vocab for model in models_incremental]

    intersec_regular = set.intersection(*map(set, vocabs_regular))
    intersec_incremental = set.intersection(*map(set, vocabs_incremental))

    # print(len(intersec_regular), len(intersec_incremental))

    words_regular = []
    words_incremental = []
    eval_adj_regular = pd.read_csv(sys.argv[1])
    eval_adj_incremental = pd.read_csv(sys.argv[2])
    for source_word in eval_adj_regular['WORD']:
        words_regular.append(source_word)
    for source_word in eval_adj_incremental['WORD']:
        words_incremental.append(source_word)

    results_eval_regular = pd.DataFrame()
    results_eval_regular['WORD'] = words_regular

    results_eval_incremental = pd.DataFrame()
    results_eval_incremental['WORD'] = words_incremental

    rest_adj_regular = pd.read_csv(sys.argv[3])
    rest_adj_incremental = pd.read_csv(sys.argv[4])

    rest_regular = []
    rest_incremental = []
    for line in rest_adj_regular['WORD']:
        source_word = line
        if source_word in intersec_regular:
            rest_regular.append(source_word)
    for line in rest_adj_incremental['WORD']:
        source_word = line
        if source_word in intersec_incremental:
            rest_incremental.append(source_word)

    results_rest_regular = pd.DataFrame()
    results_rest_regular['WORD'] = rest_regular

    results_rest_incremental = pd.DataFrame()
    results_rest_incremental['WORD'] = rest_incremental

    aligned_models_regular = align_models(models_regular)
    aligned_models_incremental = align_models(models_incremental)

    corpus_len = [int(i) for i in sys.argv[9:]]

    wordfreq_eval_regular = get_freqdict(words_regular, vocabs_regular, corpus_len)
    wordfreq_eval_incremental = get_freqdict(words_incremental, vocabs_incremental, corpus_len)
    wordfreq_rest_regular = get_freqdict(rest_regular, vocabs_regular, corpus_len)
    wordfreq_rest_incremental = get_freqdict(rest_incremental, vocabs_incremental, corpus_len)

    results_eval_regular['frequency'] = results_eval_regular['WORD'].map(wordfreq_eval_regular)
    results_eval_incremental['frequency'] = \
        results_eval_incremental['WORD'].map(wordfreq_eval_incremental)
    results_rest_regular['frequency'] = results_rest_regular['WORD'].map(wordfreq_rest_regular)
    results_rest_incremental['frequency'] = \
        results_rest_incremental['WORD'].map(wordfreq_rest_incremental)

    eval_reg_proc = get_mean_dist_procrustes(words_regular, aligned_models_regular)
    eval_reg_ga = get_mean_dist_globalanchors(words_regular, aligned_models_regular)
    eval_incr_proc = get_mean_dist_procrustes(words_incremental, aligned_models_incremental)
    eval_incr_ga = get_mean_dist_globalanchors(words_incremental, aligned_models_incremental)
    rest_reg_proc = get_mean_dist_procrustes(rest_regular, aligned_models_regular)
    rest_reg_ga = get_mean_dist_globalanchors(rest_regular, aligned_models_regular)
    rest_incr_proc = get_mean_dist_procrustes(rest_incremental, aligned_models_incremental)
    rest_incr_ga = get_mean_dist_globalanchors(rest_incremental, aligned_models_incremental)

    results_eval_regular['mean_dist_procrustes'] = results_eval_regular['WORD'].map(eval_reg_proc)
    results_eval_regular['mean_dist_globalanchors'] = results_eval_regular['WORD'].map(eval_reg_ga)

    results_eval_incremental['mean_dist_procrustes'] = \
        results_eval_incremental['WORD'].map(eval_incr_proc)
    results_eval_incremental['mean_dist_globalanchors'] = \
        results_eval_incremental['WORD'].map(eval_incr_ga)

    results_rest_regular['mean_dist_procrustes'] = results_rest_regular['WORD'].map(rest_reg_proc)
    results_rest_regular['mean_dist_globalanchors'] = results_rest_regular['WORD'].map(rest_reg_ga)

    results_rest_incremental['mean_dist_procrustes'] = \
        results_rest_incremental['WORD'].map(rest_incr_proc)
    results_rest_incremental['mean_dist_globalanchors'] = \
        results_rest_incremental['WORD'].map(rest_incr_ga)

    eval_deviation_regular = get_deviation(words_regular, aligned_models_regular)
    eval_deviation_incremental = get_deviation(words_incremental, aligned_models_incremental)
    rest_deviation_regular = get_deviation(rest_regular, aligned_models_regular)
    rest_deviation_incremental = get_deviation(rest_incremental, aligned_models_incremental)

    results_eval_regular['std_from_meanvec'] = results_eval_regular['WORD'].map(
        eval_deviation_regular)
    results_eval_incremental['std_from_meanvec'] = \
        results_eval_incremental['WORD'].map(eval_deviation_incremental)
    results_rest_regular['std_from_meanvec'] = results_rest_regular['WORD'].map(
        rest_deviation_regular)
    results_rest_incremental['std_from_meanvec'] = \
        results_rest_incremental['WORD'].map(rest_deviation_incremental)

    move_eval_reg_proc = get_move_from_initial_procrustes(words_regular, aligned_models_regular)
    move_eval_reg_ga = get_move_from_initial_globalanchors(words_regular, aligned_models_regular)
    move_eval_incr_proc = get_move_from_initial_procrustes(
        words_incremental, aligned_models_incremental)
    move_eval_incr_ga = get_move_from_initial_globalanchors(
        words_incremental, aligned_models_incremental)
    move_rest_reg_proc = get_move_from_initial_procrustes(rest_regular, aligned_models_regular)
    move_rest_reg_ga = get_move_from_initial_globalanchors(rest_regular, aligned_models_regular)
    move_rest_incr_proc = get_move_from_initial_procrustes(
        rest_incremental, aligned_models_incremental)
    move_rest_incr_ga = get_move_from_initial_globalanchors(
        rest_incremental, aligned_models_incremental)

    results_eval_regular['sum_deltas_procrustes'] = \
        results_eval_regular['WORD'].map(move_eval_reg_proc)
    results_eval_regular['sum_deltas_globalanchors'] = \
        results_eval_regular['WORD'].map(move_eval_reg_ga)

    results_eval_incremental['sum_deltas_procrustes'] = \
        results_eval_incremental['WORD'].map(move_eval_incr_proc)
    results_eval_incremental['sum_deltas_globalanchors'] = \
        results_eval_incremental['WORD'].map(move_eval_incr_ga)

    results_rest_regular['sum_deltas_procrustes'] = \
        results_rest_regular['WORD'].map(move_rest_reg_proc)
    results_rest_regular['sum_deltas_globalanchors'] = \
        results_rest_regular['WORD'].map(move_rest_reg_ga)

    results_rest_incremental['sum_deltas_procrustes'] = \
        results_rest_incremental['WORD'].map(move_rest_incr_proc)
    results_rest_incremental['sum_deltas_globalanchors'] = \
        results_rest_incremental['WORD'].map(move_rest_incr_ga)

    results_eval_regular.to_csv(sys.argv[5])
    results_eval_incremental.to_csv(sys.argv[6])
    results_rest_regular.to_csv(sys.argv[7])
    results_rest_incremental.to_csv(sys.argv[8])