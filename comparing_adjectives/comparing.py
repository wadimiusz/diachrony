# python3
# coding: utf-8

import argparse
import sys
import numpy as np
import pandas as pd
from get_adjs import get_models_by_decade, get_len, get_freqdict
from models import GlobalAnchors, smart_procrustes_align_gensim, Jaccard
from utils import intersection_align_gensim
import percache

cache = percache.Cache('my-cache')


def intersec_models(modellist, intersec_vocab):
    for model in modellist[1:]:
        _, _ = intersection_align_gensim(m1=modellist[0], m2=model, words=intersec_vocab)

    return modellist


def align_models(modellist):
    for model in modellist[1:]:
        _ = smart_procrustes_align_gensim(modellist[0], model)

    return modellist


@cache
def get_anchor(word, model):
    model_anchor = GlobalAnchors(
        w2v1=model, w2v2=model, assume_vocabs_are_identical=True).get_global_anchors(
        word, w2v=model)
    return model_anchor


def get_mean_dist_procrustes(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = 0
        for i in range(len(modellist) - 1):
            score = np.dot(modellist[i][word], modellist[i + 1][word])
            scores += (1 - score)
        mean_scores[word] = scores / (len(modellist) - 1)

    return mean_scores


def get_mean_dist_jaccard(wordlist, modellist, top_n_neighbors):
    mean_scores = {}
    for word in wordlist:
        scores = 0
        for i in range(len(modellist) - 1):
            score = Jaccard(w2v1=modellist[i], w2v2=modellist[i + 1],
                            top_n_neighbors=top_n_neighbors).get_score(word)
            scores += (1 - score)
        mean_scores[word] = scores / (len(modellist) - 1)

    return mean_scores


'''
def get_mean_dist_globalanchors(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = 0
        for i in range(len(modellist) - 1):
            score = GlobalAnchors(w2v1=modellist[i], w2v2=modellist[i + 1],
                                  assume_vocabs_are_identical=True).get_score(word)
            scores += (1 - score)
        mean_scores[word] = scores / (len(modellist) - 1)

    return mean_scores
'''


def get_mean_dist_globalanchors(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = 0
        for i in range(len(modellist) - 1):
            score = np.dot(get_anchor(word, modellist[i]), get_anchor(word, modellist[i + 1]))
            scores += (1 - score)
        mean_scores[word] = scores / (len(modellist) - 1)

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


'''
def get_move_from_initial_globalanchors(wordlist, modellist):
    move_from_init = {}
    for word in wordlist:
        deltas = 0
        previous = GlobalAnchors(w2v1=modellist[0], w2v2=modellist[1],
                                 assume_vocabs_are_identical=True).get_score(word)
        for i in range(2, len(modellist)):
            similarity = \
                GlobalAnchors(w2v1=modellist[0], w2v2=modellist[i],
                              assume_vocabs_are_identical=True).get_score(word)
            delta = similarity - previous
            if delta > 0:
                deltas -= 1
            elif delta < 0:
                deltas += 1
            previous = similarity

        move_from_init[word] = deltas / (len(modellist) - 2)

    return move_from_init
'''


def get_move_from_initial_globalanchors(wordlist, modellist):
    move_from_init = {}
    for word in wordlist:
        deltas = 0
        previous = np.dot(get_anchor(word, modellist[0]), get_anchor(word, modellist[1]))
        for i in range(2, len(modellist)):
            similarity = np.dot(get_anchor(word, modellist[0]), get_anchor(word, modellist[i]))
            delta = similarity - previous
            if delta > 0:
                deltas -= 1
            elif delta < 0:
                deltas += 1
            previous = similarity

        move_from_init[word] = deltas / (len(modellist) - 2)

    return move_from_init


def get_move_from_initial_jaccard(wordlist, modellist, top_n_neighbors):
    move_from_init = {}
    for word in wordlist:
        deltas = 0
        previous = Jaccard(modellist[0], modellist[1],
                           top_n_neighbors=top_n_neighbors).get_score(word)
        for i in range(2, len(modellist)):
            similarity = Jaccard(modellist[0], modellist[i],
                                 top_n_neighbors=top_n_neighbors).get_score(word)
            delta = similarity - previous
            if delta > 0:
                deltas -= 1
            elif delta < 0:
                deltas += 1
            previous = similarity

        move_from_init[word] = deltas / (len(modellist) - 2)

    return move_from_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', '-l', dest='lexicon', choices=['rus', 'eng', 'nor'])
    parser.add_argument('--kind', '-k', dest='kind', choices=['regular', 'incremental'])
    parser.add_argument('--min-freq', '-mf', dest='min_freq', type=int)
    parser.add_argument('--distrib', help="Use filler sets with distribution reproduction?",
                        action="store_true")
    parser.add_argument('--lengths', '-lg', dest='lengths', default="datasets/corpus_lengths.json")
    parser.add_argument('--root', '-r', default="comparing_adjectives/adjectives/")
    parser.add_argument('--out_root', '-or', default="comparing_adjectives/outputs/")
    parser.add_argument('--top-n-neighbors', '-n', type=int, default=50, dest='n',
                        help="Parameter of the Jaccard model: number of associates to consider")
    args = parser.parse_args()

    root = args.root
    output_root = args.out_root

    evaluative_adj_path = "{}{}_{}_filtered_{}.csv".format(
        root, args.lexicon, args.kind, args.min_freq)

    if args.distrib:
        rest_adj_path = "{}rest/{}/with_distribution/{}_filtered_{}.csv".format(
            root, args.lexicon, args.kind, args.min_freq)
    else:
        rest_adj_path = "{}rest/{}/{}_filtered_{}.csv".format(
            root, args.lexicon, args.kind, args.min_freq)

    models = []
    corpus_lens = []
    for decade in range(1960, 2010, 10):
        model = get_models_by_decade(decade, args.kind, lang=args.lexicon)
        corpus_len = get_len(str(decade), args.lengths, lang=args.lexicon)

        corpus_lens.append(corpus_len)
        models.append(model)

    vocabs = [model.vocab for model in models]

    intersec = set.intersection(*map(set, vocabs))

    print('Size of shared vocabulary:', len(intersec), file=sys.stderr)

    words = set()
    eval_adjs = pd.read_csv(evaluative_adj_path)
    for word in eval_adjs['WORD']:
        if word in intersec:
            words.add(word)

    results_eval = pd.DataFrame()
    results_eval['WORD'] = list(words)

    rest_adjs = pd.read_csv(rest_adj_path)

    rest = set()
    for word in rest_adjs['WORD']:
        if word in intersec:
            rest.add(word)

    results_rest = pd.DataFrame()
    results_rest['WORD'] = list(rest)

    print('Calculating frequencies...', file=sys.stderr)
    wordfreq_eval, wordfreq_rest = [get_freqdict(
        words, vocabs, corpus_lens, return_percentiles=False) for words in [words, rest]]

    results_eval['frequency'] = results_eval['WORD'].map(wordfreq_eval)
    results_rest['frequency'] = results_rest['WORD'].map(wordfreq_rest)

    # Don't need any alignment for Jaccard:
    print('Calculating Jaccard...', file=sys.stderr)
    eval_jaccard = get_mean_dist_jaccard(words, models, top_n_neighbors=args.n)
    rest_jaccard = get_mean_dist_jaccard(rest, models, top_n_neighbors=args.n)

    results_eval['mean_dist_jaccard'] = results_eval['WORD'].map(eval_jaccard)
    results_rest['mean_dist_jaccard'] = results_rest['WORD'].map(rest_jaccard)

    move_eval_jaccard = get_move_from_initial_jaccard(words, models, top_n_neighbors=args.n)
    move_rest_jaccard = get_move_from_initial_jaccard(rest, models, top_n_neighbors=args.n)

    results_eval["sum_deltas_jaccard"] = results_eval["WORD"].map(move_eval_jaccard)
    results_rest["sum_deltas_jaccard"] = results_rest["WORD"].map(move_rest_jaccard)

    print('Intersecting vocabularies...', file=sys.stderr)
    intersected_models = intersec_models(models, intersec)

    print('Calculating Global Anchors...', file=sys.stderr)
    eval_ga = get_mean_dist_globalanchors(words, intersected_models)
    rest_ga = get_mean_dist_globalanchors(rest, intersected_models)

    results_eval['mean_dist_globalanchors'] = results_eval['WORD'].map(eval_ga)
    results_rest['mean_dist_globalanchors'] = results_rest['WORD'].map(rest_ga)

    move_eval_ga = get_move_from_initial_globalanchors(words, intersected_models)
    move_rest_ga = get_move_from_initial_globalanchors(rest, intersected_models)

    results_eval['sum_deltas_globalanchors'] = results_eval['WORD'].map(move_eval_ga)
    results_rest['sum_deltas_globalanchors'] = results_rest['WORD'].map(move_rest_ga)

    if args.kind == 'regular':
        print('Procrustes aligning...', file=sys.stderr)
        aligned_models = align_models(intersected_models)

        print('Calculating distances...', file=sys.stderr)
        eval_proc = get_mean_dist_procrustes(words, aligned_models)
        rest_proc = get_mean_dist_procrustes(rest, aligned_models)

        move_eval_proc = get_move_from_initial_procrustes(words, aligned_models)
        move_rest_proc = get_move_from_initial_procrustes(rest, aligned_models)
    else:
        print('Calculating distances...', file=sys.stderr)
        eval_proc = get_mean_dist_procrustes(words, intersected_models)
        rest_proc = get_mean_dist_procrustes(rest, intersected_models)

        move_eval_proc = get_move_from_initial_procrustes(words, intersected_models)
        move_rest_proc = get_move_from_initial_procrustes(rest, intersected_models)

    results_eval['mean_dist_procrustes'] = results_eval['WORD'].map(eval_proc)
    results_rest['mean_dist_procrustes'] = results_rest['WORD'].map(rest_proc)

    results_eval['sum_deltas_procrustes'] = results_eval['WORD'].map(move_eval_proc)
    results_rest['sum_deltas_procrustes'] = results_rest['WORD'].map(move_rest_proc)

    print('Saving output files...', file=sys.stderr)
    results_eval.to_csv("{}{}/eval_{}_{}.csv".format(
        output_root, args.lexicon, args.kind, args.min_freq))
    if args.distrib:
        results_rest.to_csv("{}{}/rest_{}_{}_distrib.csv".format(
            output_root, args.lexicon, args.kind, args.min_freq))
    else:
        results_rest.to_csv("{}{}/rest_{}_{}.csv".format(
            output_root, args.lexicon, args.kind, args.min_freq))


if __name__ == "__main__":
    main()
