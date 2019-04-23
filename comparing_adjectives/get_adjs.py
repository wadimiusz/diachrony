import os
import sys
import json
import pandas as pd
import gensim
import logging
import random
from scipy.stats import percentileofscore


def load_model(embeddings_file):
    """
    This function, unifies various standards of word embedding files.
    It automatically determines the format by the file extension
    and loads it from the disk correspondingly.
    :param embeddings_file: path to the file
    :return: the loaded model
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if not os.path.isfile(embeddings_file):
        raise FileNotFoundError("No file called {file}".format(file=embeddings_file))
    # Determine the model format by the file extension
    # Binary word2vec file:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    # Text word2vec file:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)

    return emb_model


def get_models_by_decade(cur_decade: int, kind: str, lang='rus'):
    if kind not in ['regular', 'incremental']:
        raise ValueError
    if lang not in ['rus', 'nor', 'eng']:
        raise ValueError

    if kind == "regular":
        model = load_model('data/models/{lang}/{decade}.model'.format(lang=lang, decade=cur_decade))
    else:
        model = load_model('data/models/{lang}/{decade}_incremental.model'.format(
            lang=lang, decade=cur_decade))

    return model


def delete_lowfrequent(wordlist, threshold, vocablist):
    newlist = set()  # sets are generally faster than lists, and we don't need order here
    for word in wordlist:
        hit = False
        for vocab in vocablist:
            if vocab[word].count < threshold:
                hit = True
                break
        if hit:
            continue
        newlist.add(word)

    return newlist


def get_freqdict(wordlist, vocablist, corpus_size, return_percentiles=True):
    all_freqs = []
    word_freq = {}
    for word in wordlist:
        counts = [vocab[word].count for vocab in vocablist]
        frequency = [counts[i] / corpus_size[i] for i in range(len(vocablist))]
        mean_frequency = sum(frequency) / len(frequency)
        all_freqs.append(mean_frequency)
        word_freq[word] = mean_frequency

    if return_percentiles:
        percentiles = {}
        for word in wordlist:
            percentiles[word] = int(percentileofscore(all_freqs, word_freq[word]))
        return percentiles

    else:
        return word_freq


def output_results(evaluative_dict, rest_dict):
    rest_dict_inv = {}

    for key, value in rest_dict.items():
        rest_dict_inv.setdefault(value, []).append(key)

    df = pd.DataFrame()
    finallist = set()
    missing_perc = []

    for i in evaluative_dict:
        perc = evaluative_dict[i]
        try:
            sl = random.sample(rest_dict_inv[perc], 2)
        except ValueError or KeyError:
            missing_perc.append(evaluative_dict[i])
            continue
        for l in sl:
            finallist.add(l)
            rest_dict_inv[perc].remove(l)

    df['WORD'] = list(finallist)

    return df  # , missing_perc


def get_len(current_corpus, lens_file, lang='rus'):
    if lang not in ['rus', 'nor', 'eng']:
        raise ValueError

    with open(lens_file) as f:
        length_data = json.load(f)

    corpus_size = length_data[lang][current_corpus]

    return corpus_size


if __name__ == '__main__':
    root = 'comparing_adjectives/adjectives/'  # can be parameterized

    models_regular = []
    models_incremental = []
    corpus_lens = []

    language = sys.argv[1]  # one of ['eng', 'rus', 'nor']

    corpora_sizes_file = "datasets/corpus_lengths.json"  # will hardly ever change

    for decade in range(1960, 2010, 10):
        model_regular = get_models_by_decade(decade, 'regular')
        model_incremental = get_models_by_decade(decade, 'incremental')
        corpus_len = get_len(decade, corpora_sizes_file)

        corpus_lens.append(corpus_len)
        models_regular.append(model_regular)
        models_incremental.append(model_incremental)

    vocabs_regular = [model.vocab for model in models_regular]
    vocabs_incremental = [model.vocab for model in models_incremental]

    intersec_regular = set.intersection(*map(set, vocabs_regular))
    intersec_incremental = set.intersection(*map(set, vocabs_incremental))

    words_regular = []
    words_incremental = []
    eval_adj = pd.read_csv('datasets/sentiment_lexicons/{}/{}_sentiment.csv'.format(
        language, language))

    tag = 'ADJ'  # We work only with adjectives, no need to specify it each time

    for line in eval_adj['WORD']:
        voc_word = line + '_' + tag
        if voc_word in intersec_regular:
            words_regular.append(voc_word)
        if voc_word in intersec_incremental:
            words_incremental.append(voc_word)

    # print(len(words_regular), len(words_incremental))
    # print(words_regular[:10], words_intersec[:10])

    words_regular_filtered = delete_lowfrequent(words_regular, int(sys.argv[2]), vocabs_regular)
    words_incremental_filtered = delete_lowfrequent(words_incremental, int(sys.argv[2]),
                                                    vocabs_incremental)

    # print(len(words_regular_filtered), len(words_incremental_filtered))
    # print(words_regular_filtered[:10], words_incremental_filtered[:10])

    wordfreq_regular = get_freqdict(words_regular, vocabs_regular, corpus_lens)
    wordfreq_incremental = get_freqdict(words_incremental, vocabs_incremental, corpus_lens)
    wordfreq_regular_filtered = get_freqdict(words_regular_filtered, vocabs_regular, corpus_lens)
    wordfreq_incremental_filtered = \
        get_freqdict(words_incremental_filtered, vocabs_incremental, corpus_lens)

    '''
    for x in list(wordfreq_regular)[0:5]:
        print("key {}, value {} ".format(x, wordfreq_regular[x]))
    
    for x in list(wordfreq_incremental_filtered)[0:5]:
        print("key {}, value {} ".format(x, wordfreq_incremental_filtered[x]))
    '''

    rest_regular = []
    rest_incremental = []
    for voc_word in intersec_regular:
        if voc_word.endswith(tag) and voc_word not in words_regular:
            rest_regular.append(voc_word)
    for voc_word in intersec_incremental:
        if voc_word.endswith(tag) and voc_word not in words_incremental:
            rest_incremental.append(voc_word)

    rest_regular_filtered = delete_lowfrequent(rest_regular, int(sys.argv[2]), vocabs_regular)
    rest_incremental_filtered = delete_lowfrequent(rest_incremental, int(sys.argv[2]),
                                                   vocabs_incremental)

    restfreq_regular = get_freqdict(rest_regular, vocabs_regular, corpus_lens)
    restfreq_incremental = get_freqdict(rest_incremental, vocabs_incremental, corpus_lens)
    restfreq_regular_filtered = get_freqdict(rest_regular_filtered, vocabs_regular, corpus_lens)
    restfreq_incremental_filtered = get_freqdict(rest_incremental_filtered, vocabs_incremental,
                                                 corpus_lens)

    '''
    for x in list(restfreq_regular)[0:5]:
        print("key {}, value {} ".format(x, restfreq_regular[x]))
    
    for x in list(restfreq_incremental_filtered)[0:5]:
        print("key {}, value {} ".format(x, restfreq_incremental_filtered[x]))
    '''
    if sys.argv[3] == 'with_distribution':
        output_results(wordfreq_regular, restfreq_regular).to_csv(root + 'rest/'+language + '/'
                                                                  + sys.argv[3]+'/regular.csv')
        output_results(wordfreq_incremental, restfreq_incremental).to_csv(
            root + 'rest/'+language + '/' + sys.argv[3]+'/incremental.csv')
        output_results(wordfreq_regular_filtered, restfreq_regular_filtered).to_csv(
            root + 'rest/'+language + '/' + sys.argv[3]+'/regular_filtered_'+sys.argv[2]+'.csv')
        output_results(wordfreq_incremental_filtered, restfreq_incremental_filtered).to_csv(
            root + 'rest/'+language + '/' + sys.argv[3]+'/incremental_filtered_'+sys.argv[2]+'.csv')
    else:
        rest_reg_df = pd.DataFrame()
        rest_reg_df['WORD'] = rest_regular
        rest_reg_df.to_csv(root + 'rest/'+language+'/regular.csv')
        rest_incr_df = pd.DataFrame()
        rest_incr_df['WORD'] = rest_incremental
        rest_incr_df.to_csv(root + 'rest/'+language+'/incremental.csv')
        rest_reg_fil_df = pd.DataFrame()
        rest_reg_fil_df['WORD'] = rest_regular_filtered
        rest_reg_fil_df.to_csv(root + 'rest/'+language+'regular_filtered_'+sys.argv[2]+'.csv')
        rest_incr_fil_df = pd.DataFrame()
        rest_incr_fil_df['WORD'] = rest_incremental_filtered
        rest_incr_fil_df.to_csv(root + 'rest/'+language+'incremental_filtered_'+sys.argv[2]+'.csv')
    
    eval_filtered_regular = pd.DataFrame()
    eval_filtered_incremental = pd.DataFrame()
    eval_filtered_regular['WORD'] = words_regular_filtered
    eval_filtered_regular.to_csv('{}{}_regular_filtered_{}.csv'.format(root, language, sys.argv[2]))

    eval_filtered_incremental['WORD'] = words_incremental_filtered
    eval_filtered_incremental.to_csv('{}{}_incremental_filtered_{}.csv'.format(
        root, language, sys.argv[2]))
