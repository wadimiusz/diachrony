# python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
from gensim.matutils import unitvec
from get_adjs import get_models_by_decade, get_len
from models import ProcrustesAligner, GlobalAnchors, smart_procrustes_align_gensim
from utils import intersection_align_gensim


def intersec_models(modellist, intersec_vocab):
    for model in modellist[1:]:
        _, _ = intersection_align_gensim(m1=modellist[0], m2=model, words=intersec_vocab)

    return modellist


def align_models(modellist):
    for model in modellist[1:]:
        _ = smart_procrustes_align_gensim(modellist[0], model)

    return modellist


def get_mean_dist_procrustes(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = 0
        for i in range(len(modellist)-1):
            score = np.dot(modellist[i][word], modellist[i + 1][word])
            scores += (1 - score)
        mean_scores[word] = scores/(len(modellist)-1)

    return mean_scores


def get_mean_dist_globalanchors(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = 0
        for i in range(len(modellist)-1):
            score = GlobalAnchors(w2v1=modellist[i], w2v2=modellist[i + 1], assume_vocabs_are_identical=True).get_score(word)
            scores += (1 - score)
        mean_scores[word] = scores/(len(modellist)-1)

    return mean_scores


def get_move_from_initial_procrustes(wordlist, modellist):
    move_from_init = {}
    for word in wordlist:
        deltas = 0
        previous = np.dot(modellist[0][word], modellist[1][word])
        for i in range(2, len(modellist)):
            similarity = np.dot(modellist[0][word], modellist[i][word])
            delta = similarity - previous
            if delta > 0:
                deltas -= 1
            elif delta < 0:
                deltas += 1
            previous = similarity
            
        move_from_init[word] = deltas / (len(modellist) - 2)

    return move_from_init


def get_move_from_initial_globalanchors(wordlist, modellist):
    move_from_init = {}
    for word in wordlist:
        deltas = 0
        previous = GlobalAnchors(w2v1=modellist[0], w2v2=modellist[1], assume_vocabs_are_identical=True).get_score(word)
        for i in range(2, len(modellist)):
            similarity = \
                GlobalAnchors(w2v1=modellist[0], w2v2=modellist[i], assume_vocabs_are_identical=True).get_score(word)
            delta = similarity - previous
            if delta > 0:
                deltas -= 1
            elif delta < 0:
                deltas += 1
            previous = similarity
            
        move_from_init[word] = deltas / (len(modellist) - 2)

    return move_from_init


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
    corpus_lens = []
    for decade in range(1960, 2010, 10):
        model_regular = get_models_by_decade(decade, 'regular')
        model_incremental = get_models_by_decade(decade, 'incremental')
        corpus_len = get_len(decade, sys.argv[9])

        corpus_lens.append(corpus_len)
        models_regular.append(model_regular)
        models_incremental.append(model_incremental)

    vocabs_regular = [model.vocab for model in models_regular]
    vocabs_incremental = [model.vocab for model in models_incremental]

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

    wordfreq_eval_regular, wordfreq_rest_regular = [get_freqdict(words, vocabs_regular, corpus_lens)
                                                    for words in [words_regular, rest_regular]]

    wordfreq_eval_incremental, wordfreq_rest_incremental = \
        [get_freqdict(words, vocabs_incremental, corpus_lens)
         for words in [words_incremental, rest_incremental]]

    results_eval_regular['frequency'] = results_eval_regular['WORD'].map(wordfreq_eval_regular)
    results_eval_incremental['frequency'] = \
        results_eval_incremental['WORD'].map(wordfreq_eval_incremental)
    results_rest_regular['frequency'] = results_rest_regular['WORD'].map(wordfreq_rest_regular)
    results_rest_incremental['frequency'] = \
        results_rest_incremental['WORD'].map(wordfreq_rest_incremental)

    intersec_models_regular = intersec_models(models_regular, intersec_regular)
    intersec_models_incremental = intersec_models(models_incremental, intersec_incremental)

    eval_reg_ga = get_mean_dist_globalanchors(words_regular, intersec_models_regular)
    eval_incr_proc = get_mean_dist_procrustes(words_incremental, intersec_models_incremental)
    eval_incr_ga = get_mean_dist_globalanchors(words_incremental, intersec_models_incremental)
    rest_reg_ga = get_mean_dist_globalanchors(rest_regular, intersec_models_regular)
    rest_incr_proc = get_mean_dist_procrustes(rest_incremental, intersec_models_incremental)
    rest_incr_ga = get_mean_dist_globalanchors(rest_incremental, intersec_models_incremental)

    results_eval_regular['mean_dist_globalanchors'] = results_eval_regular['WORD'].map(eval_reg_ga)
    results_eval_incremental['mean_dist_procrustes'] = \
        results_eval_incremental['WORD'].map(eval_incr_proc)
    results_eval_incremental['mean_dist_globalanchors'] = \
        results_eval_incremental['WORD'].map(eval_incr_ga)
    results_rest_regular['mean_dist_globalanchors'] = results_rest_regular['WORD'].map(rest_reg_ga)
    results_rest_incremental['mean_dist_procrustes'] = \
        results_rest_incremental['WORD'].map(rest_incr_proc)
    results_rest_incremental['mean_dist_globalanchors'] = \
        results_rest_incremental['WORD'].map(rest_incr_ga)

    move_eval_reg_ga = get_move_from_initial_globalanchors(words_regular, intersec_models_regular)
    move_eval_incr_proc = get_move_from_initial_procrustes(
        words_incremental, intersec_models_incremental)
    move_eval_incr_ga = get_move_from_initial_globalanchors(
        words_incremental, intersec_models_incremental)
    move_rest_reg_ga = get_move_from_initial_globalanchors(rest_regular, intersec_models_regular)
    move_rest_incr_proc = get_move_from_initial_procrustes(
        rest_incremental, intersec_models_incremental)
    move_rest_incr_ga = get_move_from_initial_globalanchors(
        rest_incremental, intersec_models_incremental)

    results_eval_regular['sum_deltas_globalanchors'] = \
        results_eval_regular['WORD'].map(move_eval_reg_ga)
    results_eval_incremental['sum_deltas_procrustes'] = \
        results_eval_incremental['WORD'].map(move_eval_incr_proc)
    results_eval_incremental['sum_deltas_globalanchors'] = \
        results_eval_incremental['WORD'].map(move_eval_incr_ga)
    results_rest_regular['sum_deltas_globalanchors'] = \
        results_rest_regular['WORD'].map(move_rest_reg_ga)
    results_rest_incremental['sum_deltas_procrustes'] = \
        results_rest_incremental['WORD'].map(move_rest_incr_proc)
    results_rest_incremental['sum_deltas_globalanchors'] = \
        results_rest_incremental['WORD'].map(move_rest_incr_ga)

    aligned_models_regular = align_models(intersec_models_regular)

    eval_reg_proc = get_mean_dist_procrustes(words_regular, aligned_models_regular)
    rest_reg_proc = get_mean_dist_procrustes(rest_regular, aligned_models_regular)

    results_eval_regular['mean_dist_procrustes'] = results_eval_regular['WORD'].map(eval_reg_proc)
    results_rest_regular['mean_dist_procrustes'] = results_rest_regular['WORD'].map(rest_reg_proc)

    move_eval_reg_proc = get_move_from_initial_procrustes(words_regular, aligned_models_regular)
    move_rest_reg_proc = get_move_from_initial_procrustes(rest_regular, aligned_models_regular)

    results_eval_regular['sum_deltas_procrustes'] = \
        results_eval_regular['WORD'].map(move_eval_reg_proc)
    results_rest_regular['sum_deltas_procrustes'] = \
        results_rest_regular['WORD'].map(move_rest_reg_proc)

    results_eval_regular.to_csv(sys.argv[5])
    results_eval_incremental.to_csv(sys.argv[6])
    results_rest_regular.to_csv(sys.argv[7])
    results_rest_incremental.to_csv(sys.argv[8])
