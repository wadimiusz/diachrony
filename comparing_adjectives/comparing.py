from models import ProcrustesAligner, GlobalAnchors
from utils import load_model, intersection_align_gensim
from gensim import matutils
import pandas as pd
import numpy as np
import sys


def get_models_by_decade(decade: int, kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model = load_model('data/models/{decade}.model'.format(decade=decade))
    else:
        model = load_model('data/models/{decade}_incremental.model'.format(decade=decade))

    return model


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

#print(len(intersec_regular), len(intersec_incremental))


words_regular = []
words_incremental = []
eval_adj_regular = pd.read_csv(sys.argv[1])
eval_adj_incremental = pd.read_csv(sys.argv[2])
for word in eval_adj_regular['WORD']:
    words_regular.append(word)
for word in eval_adj_incremental['WORD']:
    words_incremental.append(word)

results_eval_regular = pd.DataFrame()
results_eval_regular['WORD'] = words_regular

results_eval_incremental = pd.DataFrame()
results_eval_incremental['WORD'] = words_incremental

rest_adj_regular = pd.read_csv(sys.argv[3])
rest_adj_incremental = pd.read_csv(sys.argv[4])

rest_regular = []
rest_incremental = []
for line in rest_adj_regular['WORD']:
    word = line
    if word in intersec_regular:
        rest_regular.append(word)
for line in rest_adj_incremental['WORD']:
    word = line
    if word in intersec_incremental:
        rest_incremental.append(word)

results_rest_regular = pd.DataFrame()
results_rest_regular['WORD'] = rest_regular

results_rest_incremental = pd.DataFrame()
results_rest_incremental['WORD'] = rest_incremental


def align_models(modellist):
    aligned_models = []
    for model in modellist[1:]:
        model1_aligned, model2_aligned = intersection_align_gensim(m1=modellist[0], m2=model)
        aligned_models.append(model2_aligned)
    aligned_models.insert(0, model1_aligned)

    return aligned_models


aligned_models_regular = align_models(models_regular)
aligned_models_incremental = align_models(models_incremental)


def get_mean_dist_procrustes(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = []
        for i in range(len(modellist)-1):
            score = ProcrustesAligner(w2v1=modellist[i], w2v2=modellist[i+1]).get_score(word)
            scores.append(1 - score)
        mean_scores[word] = np.mean(scores)

    return mean_scores


def get_mean_dist_globalanchors(wordlist, modellist):
    mean_scores = {}
    for word in wordlist:
        scores = []
        for i in range(len(modellist)-1):
            score = GlobalAnchors(w2v1=modellist[i], w2v2=modellist[i + 1]).get_score(word)
            scores.append(1 - score)
        mean_scores[word] = np.mean(scores)

    return mean_scores


def get_freqdict(wordlist, vocablist, corpus_len):
    all_freqs = []
    word_freq = {}
    for word in wordlist:
        counts = [vocab[word].count for vocab in vocablist]
        frequency = [counts[i] / corpus_len[i] for i in range(len(vocablist))]
        mean_frequency = sum(frequency)/len(frequency)
        all_freqs.append(mean_frequency)
        word_freq[word] = mean_frequency

    return word_freq


corpus_len = [int(i) for i in sys.argv[9:]]

wordfreq_eval_regular = get_freqdict(words_regular, vocabs_regular, corpus_len)
wordfreq_eval_incremental = get_freqdict(words_incremental, vocabs_incremental, corpus_len)
wordfreq_rest_regular = get_freqdict(rest_regular, vocabs_regular, corpus_len)
wordfreq_rest_incremental = get_freqdict(rest_incremental, vocabs_incremental, corpus_len)

results_eval_regular['frequency'] = results_eval_regular['WORD'].map(wordfreq_eval_regular)
results_eval_incremental['frequency'] = results_eval_incremental['WORD'].map(wordfreq_eval_incremental)
results_rest_regular['frequency'] = results_rest_regular['WORD'].map(wordfreq_rest_regular)
results_rest_incremental['frequency'] = results_rest_incremental['WORD'].map(wordfreq_rest_incremental)

eval_reg_proc = get_mean_dist_procrustes(words_regular, models_regular)
eval_reg_ga = get_mean_dist_globalanchors(words_regular, models_regular)
eval_incr_proc = get_mean_dist_procrustes(words_incremental, models_incremental)
eval_incr_ga = get_mean_dist_globalanchors(words_incremental, models_incremental)
rest_reg_proc = get_mean_dist_procrustes(rest_regular, models_regular)
rest_reg_ga = get_mean_dist_globalanchors(rest_regular, models_regular)
rest_incr_proc = get_mean_dist_procrustes(rest_incremental, models_incremental)
rest_incr_ga = get_mean_dist_globalanchors(rest_incremental, models_incremental)

results_eval_regular['mean_dist_procrustes'] = results_eval_regular['WORD'].map(eval_reg_proc)
results_eval_regular['mean_dist_globalanchors'] = results_eval_regular['WORD'].map(eval_reg_ga)

results_eval_incremental['mean_dist_procrustes'] = results_eval_incremental['WORD'].map(eval_incr_proc)
results_eval_incremental['mean_dist_globalanchors'] = results_eval_incremental['WORD'].map(eval_incr_ga)

results_rest_regular['mean_dist_procrustes'] = results_rest_regular['WORD'].map(rest_reg_proc)
results_rest_regular['mean_dist_globalanchors'] = results_rest_regular['WORD'].map(rest_reg_ga)

results_rest_incremental['mean_dist_procrustes'] = results_rest_incremental['WORD'].map(rest_incr_proc)
results_rest_incremental['mean_dist_globalanchors'] = results_rest_incremental['WORD'].map(rest_incr_ga)


results_eval_regular.to_csv(sys.argv[5])
results_eval_incremental.to_csv(sys.argv[6])
results_rest_regular.to_csv(sys.argv[7])
results_rest_incremental.to_csv(sys.argv[8])