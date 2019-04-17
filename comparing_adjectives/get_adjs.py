import os
import sys
import pandas as pd
import gensim
import logging
from scipy.stats import percentileofscore


def load_model(embeddings_file):
    """
    This function, written by github.com/akutuzov, unifies various standards of word embedding files.
    It automatically determines the format by the file extension and loads it from the disk correspondingly.
    :param embeddings_file: path to the file
    :return: the loaded model
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if not os.path.isfile(embeddings_file):
        raise FileNotFoundError("No file called {file}".format(file=embeddings_file))
    # Determine the model format by the file extension
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):  # Binary word2vec file
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=True, unicode_errors='replace')
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):  # Text word2vec file
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(embeddings_file)
    emb_model.init_sims(replace=True)

    return emb_model


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
eval_adj = pd.read_csv(sys.argv[1])
tag = sys.argv[2]
for line in eval_adj['WORD']:
    word = line + '_' + tag
    if word in intersec_regular:
        words_regular.append(word)
    if word in intersec_incremental:
        words_incremental.append(word)

#print(len(words_regular), len(words_incremental))
#print(words_regular[:10], words_intersec[:10])


def delete_lowfreqent(wordlist, treshold, vocablist):
    newlist = []
    for word in wordlist:
        counts = []
        for vocab in vocablist:
            counts.append(vocab[word].count)
        if sum(counts) >= treshold*len(vocablist):
            newlist.append(word)

    return newlist


words_regular_filtered = delete_lowfreqent(words_regular, int(sys.argv[3]), vocabs_regular)
words_incremental_filtered = delete_lowfreqent(words_incremental, int(sys.argv[3]), vocabs_incremental)

#print(len(words_regular_filtered), len(words_incremental_filtered))
#print(words_regular_filtered[:10], words_incremental_filtered[:10])


def get_freqdict(wordlist, vocablist, corpus_len):
    all_freqs = []
    word_freq = {}
    for word in wordlist:
        counts = [vocab[word].count for vocab in vocablist]
        frequency = [counts[i] / corpus_len[i] for i in range(len(vocablist))]
        mean_frequency = sum(frequency)/len(frequency)
        all_freqs.append(mean_frequency)
        word_freq[word] = mean_frequency

    percentiles = {}
    for word in wordlist:
        percentiles[word] = int(percentileofscore(all_freqs, word_freq[word]))

    return percentiles


corpus_len = [int(i) for i in sys.argv[4:8+1]]

wordfreq_regular = get_freqdict(words_regular, vocabs_regular, corpus_len)
wordfreq_incremental = get_freqdict(words_incremental, vocabs_incremental, corpus_len)
wordfreq_regular_filtered = get_freqdict(words_regular_filtered, vocabs_regular, corpus_len)
wordfreq_incremental_filtered = get_freqdict(words_incremental_filtered, vocabs_incremental, corpus_len)

'''
for x in list(wordfreq_regular)[0:5]:
    print("key {}, value {} ".format(x, wordfreq_regular[x]))

for x in list(wordfreq_incremental_filtered)[0:5]:
    print("key {}, value {} ".format(x, wordfreq_incremental_filtered[x]))
'''

rest_regular = []
rest_incremental = []
for word in intersec_regular:
    if word.endswith(tag) and word not in words_regular:
        rest_regular.append(word)
for word in intersec_incremental:
    if word.endswith(tag) and word not in words_incremental:
        rest_incremental.append(word)

rest_regular_filtered = delete_lowfreqent(rest_regular, int(sys.argv[3]), vocabs_regular)
rest_incremental_filtered = delete_lowfreqent(rest_incremental, int(sys.argv[3]), vocabs_incremental)

restfreq_regular = get_freqdict(rest_regular, vocabs_regular, corpus_len)
restfreq_incremental = get_freqdict(rest_incremental, vocabs_incremental, corpus_len)
restfreq_regular_filtered = get_freqdict(rest_regular_filtered, vocabs_regular, corpus_len)
restfreq_incremental_filtered = get_freqdict(rest_incremental_filtered, vocabs_incremental, corpus_len)

'''
for x in list(restfreq_regular)[0:5]:
    print("key {}, value {} ".format(x, restfreq_regular[x]))

for x in list(restfreq_incremental_filtered)[0:5]:
    print("key {}, value {} ".format(x, restfreq_incremental_filtered[x]))
'''


def output_results(evaluative_dict, rest_dict):
    df = pd.DataFrame()
    finallist = []
    temp = []
    for i in evaluative_dict:
        l = 0
        for j in rest_dict:
            if evaluative_dict[i] == rest_dict[j] and l < 2 and j not in temp:
                finallist.append(j)
                l += 1
                temp.append(j)
    df['WORD'] = finallist

    return df


output_results(wordfreq_regular, restfreq_regular).to_csv(sys.argv[9])
output_results(wordfreq_incremental, restfreq_incremental).to_csv(sys.argv[10])
output_results(wordfreq_regular_filtered, restfreq_regular_filtered).to_csv(sys.argv[11])
output_results(wordfreq_incremental_filtered, restfreq_incremental_filtered).to_csv(sys.argv[12])

eval_filteres_regular = pd.DataFrame()
eval_filteres_incremental = pd.DataFrame()
eval_filteres_regular['WORD'] = words_regular_filtered
eval_filteres_regular.to_csv(sys.argv[13])
eval_filteres_incremental['WORD'] = words_incremental_filtered
eval_filteres_incremental.to_csv(sys.argv[14])