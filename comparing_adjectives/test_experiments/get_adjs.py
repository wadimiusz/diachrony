import os
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

model1 = load_model("../wordvectors/soviet/pre-soviet.model")
model2 = load_model("../wordvectors/soviet/soviet.model")
model3 = load_model("../wordvectors/soviet/post-soviet.model")
vocab1 = model1.wv.vocab
vocab2 = model2.wv.vocab
vocab3 = model3.wv.vocab

words = []
ruscorp = open('eval_adj_rus.txt', 'r', encoding='utf8')
for line in ruscorp.read().splitlines():
    word = line + '_ADJ'
    if word in list((set(vocab1) & set(vocab2) & set(vocab3))):
        words.append(word)

def mean_freq(word, vocab1, vocab2, vocab3):

    try:
        fr1 = vocab1[word].count / 44626840
    except KeyError:
        fr1 = 0
    try:
        fr2 = vocab2[word].count / 59579878
    except KeyError:
        fr2 = 0
    try:
        fr3 = vocab3[word].count / 54299964
    except KeyError:
        fr3 = 0

    #print(fr1, fr2, fr3)

    return (fr1 + fr2 + fr3) / 3


all_freqs = []
for word in words:
    freq = mean_freq(word, vocab1, vocab2, vocab3)
    all_freqs.append(freq)

word_freq = {}
for word in words:
    word_freq[word] = int(percentileofscore(all_freqs, mean_freq(word, vocab1, vocab2, vocab3)))

# print(all_freqs[:10])
# for x in list(word_freq)[0:10]:
#    print("key {}, value {} ".format(x, word_freq[x]))

corpus_freqs = []
for word in list((set(vocab1) & set(vocab2) & set(vocab3))):
    if word.endswith('_ADJ') and word not in words:
        freq = mean_freq(word, vocab1, vocab2, vocab3)
        corpus_freqs.append(freq)

all_adjs = {}
for word in list((set(vocab1) & set(vocab2) & set(vocab3))):
    if word.endswith('_ADJ') and word not in words:
        all_adjs[word] = int(percentileofscore(corpus_freqs, mean_freq(word, vocab1, vocab2, vocab3)))

# for x in list(all_adjs)[0:10]:
#    print("key {}, value {} ".format(x, all_adjs[x]))

f = open('rest_adj_largescale.txt', 'a', encoding='utf-8')
temp = []
for i in word_freq:
    l = 0
    for j in all_adjs:
        if word_freq[i] == all_adjs[j] and l < 6 and j not in temp:
            f.write(j+'\n')
            #print(word_freq[i], i, j)
            l += 1
            temp.append(j)
    f.write('\n')
f.close()

#print(len(words))
#print(len(word_freq))