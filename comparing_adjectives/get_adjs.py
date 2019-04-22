import os
import sys
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


def get_models_by_decade(cur_decade: int, kind: str):
    if kind not in ['regular', 'incremental']:
        raise ValueError

    if kind == "regular":
        model = load_model('data/models/{decade}.model'.format(decade=cur_decade))
    else:
        model = load_model('data/models/{decade}_incremental.model'.format(decade=cur_decade))

    return model


def delete_lowfreqent(wordlist, treshold, vocablist):
    newlist = []
    for word in wordlist:
        counts = []
        for vocab in vocablist:
            counts.append(vocab[word].count)

        for count in counts:
            if count < treshold:
                counts.remove(count)

        if len(counts) == len(vocablist):
            newlist.append(word)

    return newlist


def get_freqdict(wordlist, vocablist, corpus_size):
    all_freqs = []
    word_freq = {}
    for word in wordlist:
        counts = [vocab[word].count for vocab in vocablist]
        frequency = [counts[i] / corpus_size[i] for i in range(len(vocablist))]
        mean_frequency = sum(frequency) / len(frequency)
        all_freqs.append(mean_frequency)
        word_freq[word] = mean_frequency

    percentiles = {}
    for word in wordlist:
        percentiles[word] = int(percentileofscore(all_freqs, word_freq[word]))

    return percentiles


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
        except ValueError:
            missing_perc.append(evaluative_dict[i])
            continue
        except KeyError:
            missing_perc.append(evaluative_dict[i])
            continue
        for l in sl:
            finallist.add(l)
            rest_dict_inv[perc].remove(l)

    df['WORD'] = list(finallist)

    return df  # , missing_perc


def get_len(decade, lens_file):
    df = pd.read_csv(lens_file, sep='\t', index_col=False)

    return int(df.loc[df['CORPUS'] == decade, 'LENGTH'].iloc[0])


if __name__ == '__main__':
    models_regular = []
    models_incremental = []
    corpus_lens = []
    for decade in range(1960, 2010, 10):
        model_regular = get_models_by_decade(decade, 'regular')
        model_incremental = get_models_by_decade(decade, 'incremental')
        corpus_len = get_len(decade, sys.argv[4])

        corpus_lens.append(corpus_len)
        models_regular.append(model_regular)
        models_incremental.append(model_incremental)

    vocabs_regular = [model.vocab for model in models_regular]
    vocabs_incremental = [model.vocab for model in models_incremental]

    intersec_regular = set.intersection(*map(set, vocabs_regular))
    intersec_incremental = set.intersection(*map(set, vocabs_incremental))

    words_regular = []
    words_incremental = []
    eval_adj = pd.read_csv(sys.argv[1])
    tag = sys.argv[2]
    for line in eval_adj['WORD']:
        voc_word = line + '_' + tag
        if voc_word in intersec_regular:
            words_regular.append(voc_word)
        if voc_word in intersec_incremental:
            words_incremental.append(voc_word)

    # print(len(words_regular), len(words_incremental))
    # print(words_regular[:10], words_intersec[:10])

    words_regular_filtered = delete_lowfreqent(words_regular, int(sys.argv[3]), vocabs_regular)
    words_incremental_filtered = delete_lowfreqent(words_incremental, int(sys.argv[3]),
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

    rest_regular_filtered = delete_lowfreqent(rest_regular, int(sys.argv[3]), vocabs_regular)
    rest_incremental_filtered = delete_lowfreqent(rest_incremental, int(sys.argv[3]),
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
    if sys.argv[6] == 'with_distribution':
        output_results(wordfreq_regular, restfreq_regular).to_csv(sys.argv[5]+sys.argv[6]+'/regular.csv')
        output_results(wordfreq_incremental, restfreq_incremental).to_csv(sys.argv[5]+sys.argv[6]+'/incremental.csv')
        output_results(wordfreq_regular_filtered, restfreq_regular_filtered)\
            .to_csv(sys.argv[5]+sys.argv[6]+'/regular_filtered_'+sys.argv[3]+'.csv')
        output_results(wordfreq_incremental_filtered, restfreq_incremental_filtered)\
            .to_csv(sys.argv[5]+sys.argv[6]+'/incremental_filtered_'+sys.argv[3]+'.csv')
    else:
        rest_reg_df = pd.DataFrame()
        rest_reg_df['WORD'] = rest_regular
        rest_reg_df.to_csv(sys.argv[5]+'regular.csv')
        rest_incr_df = pd.DataFrame()
        rest_incr_df['WORD'] = rest_incremental
        rest_incr_df.to_csv(sys.argv[5]+'incremental.csv')
        rest_reg_fil_df = pd.DataFrame()
        rest_reg_fil_df['WORD'] = rest_regular_filtered
        rest_reg_fil_df.to_csv(sys.argv[5]+'regular_filtered_'+sys.argv[3]+'.csv')
        rest_incr_fil_df = pd.DataFrame()
        rest_incr_fil_df['WORD'] = rest_incremental_filtered
        rest_incr_fil_df.to_csv(sys.argv[5]+'incremental_filtered_'+sys.argv[3]+'.csv')
    
    eval_filtered_regular = pd.DataFrame()
    eval_filtered_incremental = pd.DataFrame()
    eval_filtered_regular['WORD'] = words_regular_filtered
    eval_filtered_regular.to_csv(sys.argv[1].split('.')[0]+'_regular_filtered_'+sys.argv[3]+'.csv')
    eval_filtered_incremental['WORD'] = words_incremental_filtered
    eval_filtered_incremental.to_csv(sys.argv[1].split('.')[0]+'_incremental_filtered_'+sys.argv[3]+'.csv')
