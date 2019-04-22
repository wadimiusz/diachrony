# python3
# coding: utf-8

import sys
import numpy as np
import pandas as pd
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
    models = []
    corpus_lens = []
    for decade in range(1960, 2010, 10):
        model = get_models_by_decade(decade, sys.argv[3])
        corpus_len = get_len(decade, sys.argv[6])

        corpus_lens.append(corpus_len)
        models.append(model)

    vocabs = [model.vocab for model in models]

    intersec = set.intersection(*map(set, vocabs))

    words = []
    eval_adjs = pd.read_csv(sys.argv[1])
    for word in eval_adjs['WORD']:
        words.append(word)

    results_eval = pd.DataFrame()
    results_eval['WORD'] = words

    rest_adjs = pd.read_csv(sys.argv[2])

    rest = []
    for line in rest_adjs['WORD']:
        if line in intersec:
            rest.append(line)

    results_rest = pd.DataFrame()
    results_rest['WORD'] = rest

    wordfreq_eval, wordfreq_rest = [get_freqdict(words, vocabs, corpus_lens)
                                                    for words in [words, rest]]

    results_eval['frequency'] = results_eval['WORD'].map(wordfreq_eval)
    results_rest['frequency'] = results_rest['WORD'].map(wordfreq_rest)

    intersected_models = intersec_models(models, intersec)

    eval_ga = get_mean_dist_globalanchors(words, intersected_models)
    rest_ga = get_mean_dist_globalanchors(rest, intersected_models)

    results_eval['mean_dist_globalanchors'] = results_eval['WORD'].map(eval_ga)
    results_rest['mean_dist_globalanchors'] = results_rest['WORD'].map(rest_ga)

    move_eval_ga = get_move_from_initial_globalanchors(words, intersected_models)
    move_rest_ga = get_move_from_initial_globalanchors(rest, intersected_models)

    results_eval['sum_deltas_globalanchors'] = results_eval['WORD'].map(move_eval_ga)
    results_rest['sum_deltas_globalanchors'] = results_rest['WORD'].map(move_rest_ga)

    if sys.argv[3] == 'regular':
        aligned_models = align_models(intersected_models)

        eval_proc = get_mean_dist_procrustes(words, aligned_models)
        rest_proc = get_mean_dist_procrustes(rest, aligned_models)

        move_eval_proc = get_move_from_initial_procrustes(words, aligned_models)
        move_rest_proc = get_move_from_initial_procrustes(rest, aligned_models)
    else:
        eval_proc = get_mean_dist_procrustes(words, intersected_models)
        rest_proc = get_mean_dist_procrustes(rest, intersected_models)

        move_eval_proc = get_move_from_initial_procrustes(words, intersected_models)
        move_rest_proc = get_move_from_initial_procrustes(rest, intersected_models)

    results_eval['mean_dist_procrustes'] = results_eval['WORD'].map(eval_proc)
    results_rest['mean_dist_procrustes'] = results_rest['WORD'].map(rest_proc)

    results_eval['sum_deltas_procrustes'] = results_eval['WORD'].map(move_eval_proc)
    results_rest['sum_deltas_procrustes'] = results_rest['WORD'].map(move_rest_proc)

    results_eval.to_csv(sys.argv[5]+'eval_'+sys.argv[3]+'_'+sys.argv[4]+'.csv')
    results_rest.to_csv(sys.argv[5]+'rest_'+sys.argv[3]+'_'+sys.argv[4]+'.csv')
