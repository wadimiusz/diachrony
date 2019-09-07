import os
import sys
import json
import pandas as pd
import gensim
import logging
import random
from scipy.stats import percentileofscore


class LoadingModels(object):
    def __init__(self, kind, lang):
        self.kind = kind
        self.lang = lang

    def load_model(self, embeddings_file):
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

    def get_models_by_decade(self, decade:int):
        if self.kind not in ['regular', 'incremental']:
            raise ValueError
        if self.lang not in ['rus', 'nor', 'eng']:
            raise ValueError

        if self.kind == "regular":
            model = self.load_model('models/{lang}/{decade}.model'.format(lang=self.lang, decade=decade))
        else:
            model = self.load_model('models/{lang}/{decade}_incremental.model'.format(
                lang=self.lang, decade=decade))

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


class GetAdjectives(object):
    def __init__(self, wordlist, restlist, vocablist, return_percentiles):
        self.evaluative = wordlist
        self.rest = restlist
        self.vocab = vocablist
        self.return_percentiles = return_percentiles

    def get_freqdict(self, corpus_size):
        all_freqs = []
        word_freq = {}
        for word in self.evaluative:
            counts = [vocab[word].count for vocab in self.vocab]
            frequency = [counts[i] / corpus_size[i] for i in range(len(self.vocab))]
            mean_frequency = sum(frequency) / len(frequency)
            all_freqs.append(mean_frequency)
            word_freq[word] = mean_frequency

        all_freqs_rest = []
        rest_freq = {}
        for word in self.rest:
            counts = [vocab[word].count for vocab in self.vocab]
            frequency = [counts[i] / corpus_size[i] for i in range(len(self.vocab))]
            mean_frequency = sum(frequency) / len(frequency)
            all_freqs_rest.append(mean_frequency)
            rest_freq[word] = mean_frequency

        if self.return_percentiles:
            percentiles_ev = {}
            percentiles_rest = {}
            for word in self.evaluative:
                percentiles_ev[word] = int(percentileofscore(all_freqs, word_freq[word]))
            for word in self.rest:
                percentiles_rest[word] = int(percentileofscore(all_freqs_rest, rest_freq[word]))
            return percentiles_ev, percentiles_rest

        else:
            return word_freq, rest_freq

    def output_results(self, corpus_size):
        evaluative_dict = self.get_freqdict(corpus_size=corpus_size)[0]
        rest_dict = self.get_freqdict(corpus_size=corpus_size)[1]
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
            except (ValueError, KeyError):
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
    root = 'adjectives/'

    models_regular = []
    models_incremental = []
    corpus_lens = []

    language = sys.argv[1]  # one of ['eng', 'rus', 'nor']

    corpora_sizes_file = sys.argv[4]

    for decade in range(1960, 2010, 10):
        model_regular = LoadingModels(kind='regular', lang=language).get_models_by_decade(decade)
        model_incremental = LoadingModels(kind='incremental', lang=language).get_models_by_decade(decade)
        corpus_len = get_len(str(decade), corpora_sizes_file)

        corpus_lens.append(corpus_len)
        models_regular.append(model_regular)
        models_incremental.append(model_incremental)

    vocabs_regular = [model.vocab for model in models_regular]
    vocabs_incremental = [model.vocab for model in models_incremental]

    intersec_regular = set.intersection(*map(set, vocabs_regular))
    print('Size of shared vocabulary, regular:', len(intersec_regular), file=sys.stderr)
    intersec_incremental = set.intersection(*map(set, vocabs_incremental))
    print('Size of shared vocabulary, incremental:', len(intersec_incremental), file=sys.stderr)

    print('Loading evaluative vocabulary...', file=sys.stderr)
    words_regular = []
    words_incremental = []
    eval_adj = pd.read_csv('datasets/{}/{}_sentiment.csv'.format(
        language, language))

    tag = 'ADJ'  # We work only with adjectives, no need to specify it each time

    all_eval_adj = set()  # All evaluative adjectives, independent of models

    for line in eval_adj['WORD']:
        voc_word = line + '_' + tag
        all_eval_adj.add(voc_word)
        if voc_word in intersec_regular:
            words_regular.append(voc_word)
        if voc_word in intersec_incremental:
            words_incremental.append(voc_word)

    print('Filtering by frequency...', file=sys.stderr)
    words_regular_filtered = delete_lowfrequent(words_regular, int(sys.argv[2]), vocabs_regular)
    words_incremental_filtered = delete_lowfrequent(words_incremental, int(sys.argv[2]),
                                                    vocabs_incremental)

    print('Generating fillers...', file=sys.stderr)
    rest_regular = set()
    rest_incremental = set()
    for voc_word in intersec_regular:
        if voc_word.endswith(tag) and voc_word not in all_eval_adj:
            rest_regular.add(voc_word)
    for voc_word in intersec_incremental:
        if voc_word.endswith(tag) and voc_word not in all_eval_adj:
            rest_incremental.add(voc_word)

    rest_regular_filtered = delete_lowfrequent(rest_regular, int(sys.argv[2]), vocabs_regular)
    rest_incremental_filtered = delete_lowfrequent(rest_incremental, int(sys.argv[2]),
                                                   vocabs_incremental)

    if sys.argv[3] == 'with_distribution':
        print('Sampling proper distribution...', file=sys.stderr)

        rest_reg = GetAdjectives(words_regular, rest_regular, vocabs_regular, return_percentiles=True)\
            .output_results(corpus_lens)
        rest_reg.to_csv(root + 'rest/' + language + '/' + sys.argv[3] + '/regular.csv')

        rest_incr = GetAdjectives(words_incremental, rest_incremental, vocabs_incremental, return_percentiles=True)\
            .output_results(corpus_lens)
        rest_incr.to_csv(root + 'rest/' + language + '/' + sys.argv[3] + '/incremental.csv')

        rest_reg_fil = GetAdjectives(words_regular_filtered, rest_regular_filtered, vocabs_regular,
                                     return_percentiles=True).output_results(corpus_lens)
        rest_reg_fil.to_csv(root + 'rest/' + language + '/' + sys.argv[3] + '/regular_filtered_' + sys.argv[2] + '.csv')

        rest_incr_fil = GetAdjectives(words_incremental_filtered, rest_incremental_filtered, vocabs_incremental,
                                      return_percentiles=True).output_results(corpus_lens)
        rest_incr_fil.to_csv(root + 'rest/' + language + '/' + sys.argv[3] + '/incremental_filtered_'
                             + sys.argv[2] + '.csv')
    else:
        rest_reg_df = pd.DataFrame()
        rest_reg_df['WORD'] = sorted(list(rest_regular))
        rest_reg_df.to_csv(root + 'rest/' + language + '/regular.csv')
        rest_incr_df = pd.DataFrame()
        rest_incr_df['WORD'] = sorted(list(rest_incremental))
        rest_incr_df.to_csv(root + 'rest/' + language + '/incremental.csv')
        rest_reg_fil_df = pd.DataFrame()
        rest_reg_fil_df['WORD'] = sorted(list(rest_regular_filtered))
        rest_reg_fil_df.to_csv(
            root + 'rest/' + language + '/regular_filtered_' + sys.argv[2] + '.csv')
        rest_incr_fil_df = pd.DataFrame()
        rest_incr_fil_df['WORD'] = sorted(list(rest_incremental_filtered))
        rest_incr_fil_df.to_csv(
            root + 'rest/' + language + '/incremental_filtered_' + sys.argv[2] + '.csv')

    eval_filtered_regular = pd.DataFrame()
    eval_filtered_incremental = pd.DataFrame()
    eval_filtered_regular['WORD'] = sorted(list(words_regular_filtered))
    eval_filtered_regular.to_csv('{}{}_regular_filtered_{}.csv'.format(root, language, sys.argv[2]))

    eval_filtered_incremental['WORD'] = sorted(list(words_incremental_filtered))
    eval_filtered_incremental.to_csv('{}{}_incremental_filtered_{}.csv'.format(
        root, language, sys.argv[2]))
