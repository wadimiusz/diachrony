from utils import load_model
from collections import Counter
import pandas as pd
from scipy.stats import percentileofscore

words = []
ruscorp = open('adjectives/eval_adj_rus.txt', 'r', encoding='utf8')
for line in ruscorp.read().splitlines():
    words.append(line + '_ADJ')

model1 = load_model("wordvectors/soviet/pre-soviet.model")
model2 = load_model("wordvectors/soviet/soviet.model")
model3 = load_model("wordvectors/soviet/post-soviet.model")
vocab1 = model1.wv.vocab
vocab2 = model2.wv.vocab
vocab3 = model3.wv.vocab


def mean_freq(word, vocab1, vocab2, vocab3):
    try:
        fr1 = vocab1[word].count / len(vocab1)
    except KeyError:
        fr1 = 0
    try:
        fr2 = vocab2[word].count / len(vocab2)
    except KeyError:
        fr2 = 0
    try:
        fr3 = vocab3[word].count / len(vocab3)
    except KeyError:
        fr3 = 0

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
for word in list(set().union(vocab1, vocab2, vocab3)):
    if word.endswith('_ADJ') and word not in words:
        freq = mean_freq(word, vocab1, vocab2, vocab3)
        corpus_freqs.append(freq)

all_adjs = {}
for word in list(set().union(vocab1, vocab2, vocab3)):
    if word.endswith('_ADJ') and word not in words:
        all_adjs[word] = int(percentileofscore(corpus_freqs, mean_freq(word, vocab1, vocab2, vocab3)))

# for x in list(all_adjs)[0:10]:
#    print("key {}, value {} ".format(x, all_adjs[x]))

f = open('soviet_adjs.txt', 'a', encoding='utf-8')
temp = []
for i in word_freq:
    l = 0
    for j in all_adjs:
        if word_freq[i] == all_adjs[j] and l <= 2 and j not in temp:
            f.write(j+'\n')
            #print(word_freq[i], i, j)
            l += 1
            temp.append(j)
    f.write('\n')
f.close()

#print(len(words))
#print(len(word_freq))